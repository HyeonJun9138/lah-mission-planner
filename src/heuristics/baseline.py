"""
휴리스틱 기준 정책 (Heuristic Baseline Policies)
- BaselineTracker: ref path 단순 추종
- CorridorShiftPolicy: 좌/중/우 band 위험 비교 후 이동
- HoldPolicy: 앞 위험 높으면 hold
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.route.ref_path import RefPath
    from src.terrain.synthetic import SyntheticTerrain
    from src.route.corridor import Corridor


class BaselineTracker:
    """
    ref path 단순 추종 기준 정책.

    ref path 방향으로 기수를 맞추고,
    지형 + safety margin 고도를 유지하며 전진한다.

    Attributes:
        v_target (float): 목표 속도 [m/s]
        agl_target (float): 목표 AGL [m]
        k_heading (float): heading 제어 게인
        k_altitude (float): 고도 제어 게인
    """

    def __init__(
        self,
        v_target: float = 40.0,
        agl_target: float = 180.0,
        k_heading: float = 0.5,
        k_altitude: float = 0.3,
    ) -> None:
        """
        Args:
            v_target: 목표 순항 속도 [m/s]
            agl_target: 목표 AGL [m]
            k_heading: 기수각 제어 게인 (0~1)
            k_altitude: 고도 제어 게인 (0~1)
        """
        self.v_target = v_target
        self.agl_target = agl_target
        self.k_heading = k_heading
        self.k_altitude = k_altitude

    def get_action(
        self,
        obs: Dict[str, np.ndarray],
        ref_path: Optional["RefPath"] = None,
        terrain: Optional["SyntheticTerrain"] = None,
    ) -> np.ndarray:
        """
        observation에서 액션 계산.

        Args:
            obs: Gymnasium observation dict
                state_vec[5] = s_norm, state_vec[6] = d_norm,
                state_vec[7..8] = heading_error_sin/cos,
                state_vec[4] = agl_norm
            ref_path: 참조 경로 (없으면 state_vec만 사용)
            terrain: 지형 객체 (없으면 state_vec만 사용)

        Returns:
            action (4,) [yaw_rate, vz, accel, hold]
        """
        state_vec = obs.get("state_vec", np.zeros(13, dtype=np.float32))

        # --- 1) 기수각 제어 (heading error 기반) ---
        # state_vec[7] = sin(heading_error), state_vec[8] = cos(heading_error)
        he_sin = float(state_vec[7]) if len(state_vec) > 8 else 0.0
        he_cos = float(state_vec[8]) if len(state_vec) > 8 else 1.0
        heading_err = float(np.arctan2(he_sin, he_cos))

        # yaw_rate_cmd: heading error를 줄이는 방향
        yaw_cmd = float(np.clip(-self.k_heading * heading_err / np.pi, -1.0, 1.0))

        # --- 2) 고도 제어 ---
        # state_vec[4] = agl_norm (0 = 지면, 1 = pref*2)
        agl_norm = float(state_vec[4]) if len(state_vec) > 4 else 0.5
        # 목표 agl_norm 추정 (agl_pref / (agl_pref * 2))
        agl_target_norm = 0.5

        agl_err = agl_target_norm - agl_norm
        vz_cmd = float(np.clip(self.k_altitude * agl_err * 4, -1.0, 1.0))

        # --- 3) 속도 제어 ---
        # state_vec[0] = v_norm
        v_norm = float(state_vec[0]) if len(state_vec) > 0 else 0.5
        v_target_norm = self.v_target / 70.0  # v_max=70
        v_err = v_target_norm - v_norm
        accel_cmd = float(np.clip(v_err * 2.0, -1.0, 1.0))

        # --- 4) hold 미사용 ---
        hold_cmd = -1.0  # hold 해제

        return np.array([yaw_cmd, vz_cmd, accel_cmd, hold_cmd], dtype=np.float32)

    def __repr__(self) -> str:
        return f"BaselineTracker(v={self.v_target}, agl={self.agl_target})"


class CorridorShiftPolicy:
    """
    좌/중/우 corridor band 위험도 비교 후 최소 위험 band로 이동하는 정책.

    lookahead_ref에서 위험도(risk) 정보를 읽어,
    왼쪽/중앙/오른쪽 중 가장 위험이 낮은 방향으로 유도한다.

    Attributes:
        base_tracker (BaselineTracker): 기본 추종 정책 (고도/속도 제어용)
        risk_threshold (float): 우회 시작 위험도 임계값
        shift_strength (float): 우회 강도 (yaw 명령 보정값)
    """

    def __init__(
        self,
        v_target: float = 40.0,
        agl_target: float = 180.0,
        risk_threshold: float = 0.5,
        shift_strength: float = 0.3,
    ) -> None:
        """
        Args:
            v_target: 목표 속도 [m/s]
            agl_target: 목표 AGL [m]
            risk_threshold: 우회 판단 위험도 임계값 [0~1]
            shift_strength: 우회 시 yaw 명령 보정 강도
        """
        self.base_tracker = BaselineTracker(v_target, agl_target)
        self.risk_threshold = risk_threshold
        self.shift_strength = shift_strength

    def get_action(
        self,
        obs: Dict[str, np.ndarray],
        ref_path: Optional["RefPath"] = None,
        terrain: Optional["SyntheticTerrain"] = None,
    ) -> np.ndarray:
        """
        corridor band 위험도 기반 액션 계산.

        Args:
            obs: Gymnasium observation dict
                lookahead_ref[i, 3] = 해당 포인트 위험도
            ref_path: 참조 경로 (사용 안 함)
            terrain: 지형 (사용 안 함)

        Returns:
            action (4,) [yaw_rate, vz, accel, hold]
        """
        # 기본 추종 액션
        action = self.base_tracker.get_action(obs)

        # lookahead_ref에서 위험도 정보 추출
        lookahead = obs.get("lookahead_ref", None)
        if lookahead is None or lookahead.size == 0:
            return action

        # state_vec에서 cross-track 정보
        state_vec = obs.get("state_vec", np.zeros(13))
        d_norm = float(state_vec[6]) if len(state_vec) > 6 else 0.0

        # 앞쪽 lookahead 포인트의 평균 위험도
        n_pts = min(5, len(lookahead))
        ahead_risks = lookahead[:n_pts, 3] if lookahead.shape[1] > 3 else np.zeros(n_pts)
        mean_ahead_risk = float(np.mean(ahead_risks))

        if mean_ahead_risk < self.risk_threshold:
            # 위험 낮음 → 기본 추종으로 충분
            return action

        # 위험이 높으면 현재 d_norm을 줄이는 방향으로 이동
        # d_norm 양수 = 좌측, 음수 = 우측
        # 좌측으로 이동: yaw 음수 (오른쪽으로 틀기)
        # 우측으로 이동: yaw 양수 (왼쪽으로 틀기)
        risk_factor = float(np.clip(
            (mean_ahead_risk - self.risk_threshold) / (1.0 - self.risk_threshold + 1e-8),
            0.0, 1.0
        ))

        # 현재 d_norm이 중앙(0)보다 우측이면 더 우측으로, 좌측이면 더 좌측으로
        # (위험 구역이 중앙에 있다면 어느 쪽이든 피해야 하므로 단순화)
        # 여기서는 좌측 우회 우선 (d_norm > 0 방향)
        if d_norm < 0.3:  # 아직 좌측으로 여유 있음
            shift = self.shift_strength * risk_factor
        else:
            shift = -self.shift_strength * risk_factor  # 우측 우회

        yaw_cmd = float(np.clip(float(action[0]) + shift, -1.0, 1.0))
        action = action.copy()
        action[0] = yaw_cmd

        return action

    def __repr__(self) -> str:
        return f"CorridorShiftPolicy(threshold={self.risk_threshold})"


class HoldPolicy:
    """
    앞쪽 위험이 높으면 hold(체공)하고, 낮아지면 재출발하는 정책.

    lookahead_ref의 위험도를 보고 threshold 초과 시 hold 명령,
    위험이 낮아지면 hold 해제 후 BaselineTracker로 재진행.

    Attributes:
        base_tracker (BaselineTracker): 기본 추종 정책
        hold_threshold (float): hold 시작 위험도 임계값
        resume_threshold (float): hold 해제 위험도 임계값
        n_lookahead_for_hold (int): hold 판단에 사용할 lookahead 수
        _in_hold (bool): 현재 hold 상태
    """

    def __init__(
        self,
        v_target: float = 40.0,
        agl_target: float = 180.0,
        hold_threshold: float = 0.7,
        resume_threshold: float = 0.4,
        n_lookahead_for_hold: int = 5,
    ) -> None:
        """
        Args:
            v_target: 목표 속도 [m/s]
            agl_target: 목표 AGL [m]
            hold_threshold: hold 시작 위험도 임계값
            resume_threshold: hold 해제 위험도 임계값 (< hold_threshold)
            n_lookahead_for_hold: hold 판단 lookahead 포인트 수
        """
        self.base_tracker = BaselineTracker(v_target, agl_target)
        self.hold_threshold = hold_threshold
        self.resume_threshold = min(resume_threshold, hold_threshold)
        self.n_lookahead_for_hold = n_lookahead_for_hold
        self._in_hold: bool = False

    def get_action(
        self,
        obs: Dict[str, np.ndarray],
        ref_path: Optional["RefPath"] = None,
        terrain: Optional["SyntheticTerrain"] = None,
    ) -> np.ndarray:
        """
        앞 위험도 기반 hold/재출발 제어.

        Args:
            obs: Gymnasium observation dict
            ref_path: 참조 경로 (사용 안 함)
            terrain: 지형 (사용 안 함)

        Returns:
            action (4,) [yaw_rate, vz, accel, hold]
        """
        # 앞쪽 위험도 계산
        lookahead = obs.get("lookahead_ref", None)
        if lookahead is not None and lookahead.size > 0:
            n_pts = min(self.n_lookahead_for_hold, len(lookahead))
            if lookahead.shape[1] > 3:
                ahead_risks = lookahead[:n_pts, 3]
            else:
                ahead_risks = np.zeros(n_pts)
            mean_risk = float(np.mean(ahead_risks))
        else:
            mean_risk = 0.0

        # hold 상태 업데이트 (히스테리시스)
        if not self._in_hold and mean_risk >= self.hold_threshold:
            self._in_hold = True
        elif self._in_hold and mean_risk <= self.resume_threshold:
            self._in_hold = False

        if self._in_hold:
            # hold 모드: 속도 최소화, hold 명령
            action = self.base_tracker.get_action(obs)
            # hold 명령 활성화
            action = action.copy()
            action[2] = -1.0  # 감속
            action[3] = 1.0   # hold 활성화
            return action
        else:
            # 일반 추종 모드
            action = self.base_tracker.get_action(obs)
            action = action.copy()
            action[3] = -1.0  # hold 해제
            return action

    @property
    def in_hold(self) -> bool:
        """현재 hold 상태 여부."""
        return self._in_hold

    def reset(self) -> None:
        """에피소드 시작 시 상태 초기화."""
        self._in_hold = False

    def __repr__(self) -> str:
        return (
            f"HoldPolicy("
            f"hold_thresh={self.hold_threshold}, "
            f"resume_thresh={self.resume_threshold}, "
            f"in_hold={self._in_hold})"
        )
