"""
운동학(Dynamics) 시뮬레이션 모듈
- 설계서 9.3절의 단순 운동학 모델 구현
- hold 모드 로직
- 속도/고도/기수각 제한값 적용
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional

from src.sim.state import AircraftState


# ------------------------------------------------------------------
# 기본 물리 상수
# ------------------------------------------------------------------

V_MIN: float = 0.0       # 최소 수평 속도 [m/s]
V_MAX: float = 70.0      # 최대 수평 속도 [m/s]
V_HOLD: float = 5.0      # hold 모드 목표 속도 [m/s]
VZ_MIN: float = -6.0     # 최대 하강 속도 [m/s]
VZ_MAX: float = 6.0      # 최대 상승 속도 [m/s]
MAX_TURN_RAD: float = np.radians(12.0)  # 최대 선회율 [rad/s]
ACCEL_MAX: float = 5.0   # 최대 가속도 [m/s^2]
HOLD_SPEED_THRESHOLD: float = 0.5  # hold 활성화 판정 action 임계값


def kinematic_step(
    state: AircraftState,
    action: np.ndarray,
    dt: float = 1.0,
    config: Optional[Dict[str, Any]] = None,
) -> AircraftState:
    """
    단일 시간 스텝 운동학 업데이트.

    설계서 9.3절 운동학:
        x_{t+1} = x_t + v_t * cos(psi_t) * dt
        y_{t+1} = y_t + v_t * sin(psi_t) * dt
        z_{t+1} = z_t + vz_t * dt
        psi_{t+1} = psi_t + yaw_rate_cmd * dt
        v_{t+1} = clip(v_t + accel_cmd * dt, v_min, v_max)
        vz_{t+1} = clip(vz_cmd, vz_min, vz_max)

    Action 해석 (Box(4,), 범위 [-1, 1]):
        action[0]: yaw_rate_cmd → [-max_turn, +max_turn] rad/s
        action[1]: vz_cmd       → [vz_min, vz_max] m/s
        action[2]: accel_cmd    → [-accel_max, +accel_max] m/s^2
        action[3]: hold_cmd
            > +0.5 : hold 요청
            < -0.5 : hold 해제
            else   : 유지

    Args:
        state: 현재 항공기 상태
        action: 4차원 연속 액션 벡터 (모두 [-1,1] 범위)
        dt: 시간 간격 [s]
        config: 파라미터 오버라이드 dict

    Returns:
        새 AircraftState 인스턴스
    """
    cfg = _resolve_config(config)

    # action 클램프
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    a_yaw, a_vz, a_accel, a_hold = float(action[0]), float(action[1]), float(action[2]), float(action[3])

    # --- hold 모드 업데이트 ---
    new_hold_mode = state.hold_mode
    new_hold_timer = state.hold_timer

    if a_hold > HOLD_SPEED_THRESHOLD:
        new_hold_mode = True
    elif a_hold < -HOLD_SPEED_THRESHOLD:
        new_hold_mode = False

    if new_hold_mode:
        new_hold_timer = state.hold_timer + dt
    else:
        new_hold_timer = 0.0

    # --- 기수각 업데이트 ---
    yaw_rate = a_yaw * cfg["max_turn_rad"]
    new_psi = state.psi + yaw_rate * dt
    # 각도 래핑 (-pi ~ pi)
    new_psi = float((new_psi + np.pi) % (2 * np.pi) - np.pi)

    # --- 수평 속도 업데이트 ---
    if new_hold_mode:
        # hold 모드: 목표 속도 V_HOLD로 수렴
        v_target = cfg["v_hold"]
        v_err = v_target - state.v
        # 최대 decel: 10 m/s^2
        v_delta = np.clip(v_err, -10.0 * dt, 10.0 * dt)
        new_v = float(np.clip(state.v + v_delta, cfg["v_min"], cfg["v_max"]))
    else:
        accel_cmd = a_accel * cfg["accel_max"]
        new_v = float(np.clip(state.v + accel_cmd * dt, cfg["v_min"], cfg["v_max"]))

    # --- 수직 속도 업데이트 ---
    new_vz = float(np.clip(
        a_vz * cfg["vz_max"] if a_vz >= 0 else a_vz * abs(cfg["vz_min"]),
        cfg["vz_min"], cfg["vz_max"]
    ))

    # hold 모드에서는 수직 속도도 감쇠
    if new_hold_mode:
        new_vz = new_vz * 0.5  # hold 중 상승/하강 완화

    # --- 위치 업데이트 ---
    new_x = state.x + new_v * np.cos(new_psi) * dt
    new_y = state.y + new_v * np.sin(new_psi) * dt
    new_z = state.z + new_vz * dt

    # 최소 고도 하드 클램프 (지형 고려는 reward/termination에서)
    new_z = float(max(new_z, 0.0))

    return AircraftState(
        x=float(new_x),
        y=float(new_y),
        z=float(new_z),
        v=new_v,
        psi=new_psi,
        vz=new_vz,
        hold_mode=new_hold_mode,
        hold_timer=new_hold_timer,
    )


def _resolve_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    파라미터 config 처리. 없으면 기본값 반환.

    Args:
        config: 사용자 제공 config dict (None 가능)

    Returns:
        완전한 파라미터 dict
    """
    defaults = {
        "v_min": V_MIN,
        "v_max": V_MAX,
        "v_hold": V_HOLD,
        "vz_min": VZ_MIN,
        "vz_max": VZ_MAX,
        "max_turn_rad": MAX_TURN_RAD,
        "accel_max": ACCEL_MAX,
        "dt": 1.0,
    }
    if config is None:
        return defaults
    return {**defaults, **config}
