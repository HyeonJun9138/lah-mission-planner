"""
보상 함수(Reward Function) 모듈
- 설계서 12절의 다중 요소 보상 구조 구현
- progress, cross_track, risk, clearance, smoothness, hold 등 모든 요소 포함
- compute_reward(): 스칼라 보상 반환
- compute_reward_breakdown(): 각 항목을 dict로 반환
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, TYPE_CHECKING

from src.sim.state import AircraftState

if TYPE_CHECKING:
    from src.terrain.synthetic import SyntheticTerrain
    from src.route.ref_path import RefPath
    from src.route.corridor import Corridor


# ------------------------------------------------------------------
# 기본 보상 가중치
# ------------------------------------------------------------------

DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {
    "w_prog": 2.0,        # ref path 진행 보상
    "w_goal": 0.5,        # 목표 거리 감소 보상
    "w_d": 0.8,           # cross-track 이탈 페널티
    "w_zref": 0.2,        # 권장 고도 오차 페널티
    "w_risk": 1.0,        # 현재 위치 위험도 페널티
    "w_risk_a": 0.7,      # 앞쪽 위험도 페널티
    "w_clear": 3.0,       # 지형 여유 부족 페널티
    "w_smooth": 0.05,     # 제어 변화(jerk) 페널티
    "w_energy": 0.05,     # 과도한 속도/상승 페널티
    "w_hold": 0.03,       # hold 남용 페널티
    "w_hold_good": 0.02,  # 앞 위험 높을 때 hold 보너스
    "terminal_success": 150.0,    # 성공 보너스
    "terminal_collision": -150.0, # 충돌 페널티
    "terminal_out": -100.0,       # 이탈 페널티
    "terminal_timeout": -30.0,    # 타임아웃 페널티
}

# 안전 고도 기준
AGL_SAFE_MIN: float = 120.0   # 절대 최소 AGL [m]
AGL_PREF: float = 180.0       # 권장 AGL [m]
MAX_HOLD_TIME: float = 60.0   # 최대 hold 시간 [s]
GOAL_RADIUS: float = 200.0    # 목표 도달 반경 [m]
AHEAD_RISK_HIGH_THRESH: float = 0.6  # hold 보너스 임계 위험도


def compute_reward(
    state: AircraftState,
    prev_state: AircraftState,
    action: np.ndarray,
    prev_action: np.ndarray,
    terrain: "SyntheticTerrain",
    ref_path: "RefPath",
    corridor: "Corridor",
    config: Optional[Dict[str, Any]] = None,
) -> float:
    """
    단일 스텝 보상 계산 (스칼라).

    Args:
        state: 현재 상태
        prev_state: 이전 상태
        action: 현재 액션 (4,)
        prev_action: 이전 액션 (4,)
        terrain: 지형 객체
        ref_path: 참조 경로
        corridor: corridor 정의
        config: 보상 파라미터 오버라이드

    Returns:
        보상값 (float)
    """
    breakdown = compute_reward_breakdown(
        state, prev_state, action, prev_action,
        terrain, ref_path, corridor, config
    )
    return float(breakdown["total"])


def compute_reward_breakdown(
    state: AircraftState,
    prev_state: AircraftState,
    action: np.ndarray,
    prev_action: np.ndarray,
    terrain: "SyntheticTerrain",
    ref_path: "RefPath",
    corridor: "Corridor",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    보상 각 항목을 dict로 반환.

    Args:
        state: 현재 상태
        prev_state: 이전 상태
        action: 현재 액션 (4,)
        prev_action: 이전 액션 (4,)
        terrain: 지형 객체
        ref_path: 참조 경로
        corridor: corridor 정의
        config: 보상 파라미터 오버라이드

    Returns:
        {
            "progress": ...,
            "goal": ...,
            "cross_track": ...,
            "z_ref": ...,
            "risk": ...,
            "risk_ahead": ...,
            "clearance": ...,
            "smoothness": ...,
            "energy": ...,
            "hold": ...,
            "hold_good": ...,
            "total": ...,
        }
    """
    from src.route.frenet import project_to_path, get_z_ref_error, compute_heading_error

    w = {**DEFAULT_REWARD_WEIGHTS, **(config or {})}

    # --- Frenet 좌표 계산 ---
    s_curr, d_curr, _ = project_to_path(state.x, state.y, ref_path)
    s_prev, d_prev, _ = project_to_path(prev_state.x, prev_state.y, ref_path)

    # --- 지형 고도 및 AGL ---
    terrain_h = terrain.get_elevation(state.x, state.y)
    agl = state.z - terrain_h

    # --- 1) 진행 보상 (progress) ---
    delta_s = s_curr - s_prev
    r_prog = w["w_prog"] * float(np.clip(delta_s / 50.0, -1.0, 2.0))

    # --- 2) 목표 거리 감소 보상 (goal) ---
    goal = ref_path.end_point[:2]
    dist_curr = float(np.linalg.norm(np.array([state.x, state.y]) - goal))
    dist_prev = float(np.linalg.norm(np.array([prev_state.x, prev_state.y]) - goal))
    delta_dist = dist_prev - dist_curr
    r_goal = w["w_goal"] * float(np.clip(delta_dist / 100.0, -1.0, 1.0))

    # --- 3) cross-track 페널티 ---
    d_norm = abs(d_curr) / max(corridor.hard_width, 1.0)
    r_cross = -w["w_d"] * float(np.clip(d_norm, 0.0, 3.0))

    # soft corridor 밖이면 추가 페널티
    if abs(d_curr) > corridor.soft_width:
        excess = (abs(d_curr) - corridor.soft_width) / max(
            corridor.hard_width - corridor.soft_width, 1.0
        )
        r_cross -= w["w_d"] * float(np.clip(excess, 0.0, 2.0))

    # --- 4) 권장 고도 오차 페널티 ---
    z_err = get_z_ref_error(state.z, s_curr, ref_path)
    z_err_norm = abs(z_err) / max(AGL_PREF, 1.0)
    r_zref = -w["w_zref"] * float(np.clip(z_err_norm, 0.0, 2.0))

    # --- 5) 현재 위치 위험도 ---
    slope = terrain.get_slope(state.x, state.y)
    roughness = terrain.get_roughness(state.x, state.y)
    local_risk = float(np.clip(
        0.5 * (slope / (np.pi / 4)) + 0.5 * roughness,
        0.0, 1.0
    ))
    r_risk = -w["w_risk"] * local_risk

    # --- 6) 앞쪽 위험도 ---
    risk_ahead = _compute_ahead_risk(state, terrain, ref_path, n_steps=5, spacing=50.0)
    r_risk_a = -w["w_risk_a"] * risk_ahead

    # --- 7) 지형 여유(clearance) 페널티 ---
    # relu(h_safe_min - agl) / h_safe_min
    clearance_deficit = max(0.0, AGL_SAFE_MIN - agl)
    r_clear = -w["w_clear"] * float(clearance_deficit / max(AGL_SAFE_MIN, 1.0))

    # AGL이 권장값보다 낮으면 추가 패널티
    if agl < AGL_PREF:
        pref_deficit = (AGL_PREF - agl) / max(AGL_PREF - AGL_SAFE_MIN, 1.0)
        r_clear -= w["w_clear"] * 0.3 * float(np.clip(pref_deficit, 0.0, 1.0))

    # --- 8) 제어 부드러움 (smoothness) ---
    if prev_action is not None and len(prev_action) >= 4:
        action_diff = np.asarray(action) - np.asarray(prev_action)
        action_jerk = float(np.sum(action_diff ** 2))
    else:
        action_jerk = 0.0
    r_smooth = -w["w_smooth"] * float(np.clip(action_jerk, 0.0, 4.0))

    # --- 9) 에너지 페널티 (과도한 속도/상승) ---
    v_excess = max(0.0, state.v - 55.0) / 15.0  # 55 m/s 초과분
    vz_excess = max(0.0, abs(state.vz) - 4.0) / 2.0  # 4 m/s 초과분
    r_energy = -w["w_energy"] * float(np.clip(v_excess + vz_excess, 0.0, 2.0))

    # --- 10) hold 관련 ---
    # hold 남용 페널티
    hold_time_norm = state.hold_timer / max(MAX_HOLD_TIME, 1.0)
    r_hold = -w["w_hold"] * float(np.clip(hold_time_norm, 0.0, 1.0)) if state.hold_mode else 0.0

    # 앞 위험 높을 때 짧은 hold 보너스
    r_hold_good = 0.0
    if state.hold_mode and risk_ahead > AHEAD_RISK_HIGH_THRESH and state.hold_timer < 30.0:
        r_hold_good = w["w_hold_good"]

    # --- 합산 ---
    total = (
        r_prog + r_goal + r_cross + r_zref
        + r_risk + r_risk_a + r_clear
        + r_smooth + r_energy
        + r_hold + r_hold_good
    )

    return {
        "progress": float(r_prog),
        "goal": float(r_goal),
        "cross_track": float(r_cross),
        "z_ref": float(r_zref),
        "risk": float(r_risk),
        "risk_ahead": float(r_risk_a),
        "clearance": float(r_clear),
        "smoothness": float(r_smooth),
        "energy": float(r_energy),
        "hold": float(r_hold),
        "hold_good": float(r_hold_good),
        "total": float(total),
        # 진단 정보
        "_s": float(s_curr),
        "_d": float(d_curr),
        "_agl": float(agl),
        "_local_risk": float(local_risk),
        "_risk_ahead": float(risk_ahead),
    }


def _compute_ahead_risk(
    state: AircraftState,
    terrain: "SyntheticTerrain",
    ref_path: "RefPath",
    n_steps: int = 5,
    spacing: float = 50.0,
) -> float:
    """
    앞쪽 N스텝 평균 위험도 계산 (간이 버전).

    Args:
        state: 현재 상태
        terrain: 지형 객체
        ref_path: 참조 경로
        n_steps: 앞으로 볼 스텝 수
        spacing: 스텝 간격 [m]

    Returns:
        평균 위험도 [0, 1]
    """
    from src.route.frenet import project_to_path

    s_curr, _, _ = project_to_path(state.x, state.y, ref_path)
    risks = []

    for i in range(1, n_steps + 1):
        s_target = min(s_curr + i * spacing, ref_path.total_length)
        pt = ref_path.get_point_at_s(s_target)
        slope = terrain.get_slope(float(pt[0]), float(pt[1]))
        rough = terrain.get_roughness(float(pt[0]), float(pt[1]))
        r = float(np.clip(0.5 * (slope / (np.pi / 4)) + 0.5 * rough, 0.0, 1.0))
        risks.append(r)

    if risks:
        return float(np.mean(risks))
    return 0.0
