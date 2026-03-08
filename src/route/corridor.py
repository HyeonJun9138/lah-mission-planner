"""
비행 corridor 관리 모듈
- ref path 중심 soft/hard corridor 정의
- 좌/중/우 band 분류
- corridor 내 위험도 프로파일 계산
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.route.ref_path import RefPath
    from src.terrain.synthetic import SyntheticTerrain


class Corridor:
    """
    ref path 기준 비행 corridor 관리 클래스.

    ref path를 중심으로 soft corridor와 hard corridor를 정의한다.
    - hard corridor 이탈: 강한 페널티 + 종료 조건
    - soft corridor 이탈: 완만한 페널티

    Band 정의 (ref path 중심):
    - center band: ±center_half_width_m
    - left band: center_half_width_m ~ side_outer_m (왼쪽, 양수 방향)
    - right band: center_half_width_m ~ side_outer_m (오른쪽, 음수 방향)

    Attributes:
        soft_width (float): soft corridor 반폭 [m]
        hard_width (float): hard corridor 반폭 [m]
        center_half_width (float): center band 반폭 [m]
        side_outer_width (float): side band 외부 경계 [m]
    """

    def __init__(
        self,
        soft_width: float = 400.0,
        hard_width: float = 1000.0,
        center_half_width: float = 100.0,
        side_outer_width: float = 500.0,
    ) -> None:
        """
        Args:
            soft_width: soft corridor 반폭 [m] (ref path 중심 기준)
            hard_width: hard corridor 반폭 [m]
            center_half_width: center band 반폭 [m]
            side_outer_width: left/right band 외부 경계 [m]
        """
        self.soft_width = soft_width
        self.hard_width = hard_width
        self.center_half_width = center_half_width
        self.side_outer_width = side_outer_width

    # ------------------------------------------------------------------
    # 위치 판별 메서드
    # ------------------------------------------------------------------

    def is_in_hard_corridor(self, d: float) -> bool:
        """
        cross-track 거리 d가 hard corridor 안에 있는지 확인.

        Args:
            d: cross-track 거리 [m]

        Returns:
            True if in hard corridor
        """
        return abs(d) <= self.hard_width

    def is_in_soft_corridor(self, d: float) -> bool:
        """
        cross-track 거리 d가 soft corridor 안에 있는지 확인.

        Args:
            d: cross-track 거리 [m]

        Returns:
            True if in soft corridor
        """
        return abs(d) <= self.soft_width

    def is_in_corridor(self, x: float, y: float, ref_path: "RefPath") -> bool:
        """
        월드 좌표 (x, y)가 hard corridor 내에 있는지 확인.

        Args:
            x: 동서 좌표 [m]
            y: 남북 좌표 [m]
            ref_path: 참조 경로

        Returns:
            True if in hard corridor
        """
        from src.route.frenet import project_to_path
        _, d, _ = project_to_path(x, y, ref_path)
        return self.is_in_hard_corridor(d)

    def get_band(self, d: float) -> str:
        """
        cross-track 거리 d에 해당하는 band 이름 반환.

        Args:
            d: cross-track 거리 [m] (양수=좌, 음수=우)

        Returns:
            "left", "center", "right", "out_left", "out_right"
        """
        abs_d = abs(d)
        if abs_d <= self.center_half_width:
            return "center"
        elif d > 0:
            if abs_d <= self.side_outer_width:
                return "left"
            else:
                return "out_left"
        else:
            if abs_d <= self.side_outer_width:
                return "right"
            else:
                return "out_right"

    def get_cross_track_penalty(self, d: float) -> float:
        """
        cross-track 이탈 거리에 따른 패널티 계수 반환.

        soft corridor 내: 선형 증가
        hard corridor 밖: 급격히 큰 패널티

        Args:
            d: cross-track 거리 [m]

        Returns:
            패널티 값 [0, ...)
        """
        abs_d = abs(d)

        if abs_d <= self.soft_width:
            # soft corridor 내: 0 ~ 1 선형
            return abs_d / self.soft_width
        elif abs_d <= self.hard_width:
            # soft ~ hard 사이: 1 ~ 3 선형
            frac = (abs_d - self.soft_width) / (self.hard_width - self.soft_width + 1e-8)
            return 1.0 + 2.0 * frac
        else:
            # hard corridor 밖: 강한 패널티
            excess = abs_d - self.hard_width
            return 3.0 + excess / 100.0  # 100m 초과당 1씩 증가

    # ------------------------------------------------------------------
    # corridor band 위험도 프로파일
    # ------------------------------------------------------------------

    def get_band_risks(
        self,
        s: float,
        terrain: "SyntheticTerrain",
        ref_path: "RefPath",
        n_lookahead: int = 20,
        lookahead_spacing: float = 50.0,
    ) -> np.ndarray:
        """
        ref path 앞쪽 K-step에 대한 좌/중/우 band별 위험도 프로파일 계산.

        각 lookahead step i, 각 band (left=0, center=1, right=2),
        각 feature (mean_risk, p95_risk, max_elev, landing_score, no_fly_frac)

        Args:
            s: 현재 경로 진행 거리 [m]
            terrain: 지형 객체 (SyntheticTerrain)
            ref_path: 참조 경로
            n_lookahead: 앞으로 볼 스텝 수 (K)
            lookahead_spacing: 스텝 간 간격 [m]

        Returns:
            shape (K, 3, 5) 위험도 프로파일
            axis0: lookahead step
            axis1: band [left=0, center=1, right=2]
            axis2: feature [mean_risk, p95_risk, max_elev_norm, landing, no_fly_frac]
        """
        # 지형 특징 맵 (지연 계산)
        from src.terrain.features import compute_risk_map, compute_landing_suitability

        if not hasattr(self, "_terrain_features_cache"):
            self._terrain_features_cache = {}

        profile = np.zeros((n_lookahead, 3, 5), dtype=np.float32)

        # band 중심 거리: left=+200m, center=0m, right=-200m
        band_centers = np.array([
            (self.center_half_width + self.side_outer_width) / 2.0,   # left
            0.0,                                                        # center
            -(self.center_half_width + self.side_outer_width) / 2.0,  # right
        ])
        band_half_widths = np.array([
            (self.side_outer_width - self.center_half_width) / 2.0,  # left
            self.center_half_width,                                    # center
            (self.side_outer_width - self.center_half_width) / 2.0,  # right
        ])

        # 샘플 포인트 수 (band 내 샘플링)
        n_band_samples = 5

        for k in range(n_lookahead):
            s_k = s + (k + 1) * lookahead_spacing
            s_k = min(s_k, ref_path.total_length)

            ref_pt = ref_path.get_point_at_s(s_k)
            s_idx = int(np.clip(
                np.searchsorted(ref_path.cumulative_s, s_k),
                0, ref_path.n_points - 1
            ))
            normal = ref_path.normals[s_idx]

            for b, (d_center, d_half) in enumerate(zip(band_centers, band_half_widths)):
                # band 내에서 여러 샘플 포인트 추출
                d_samples = np.linspace(
                    d_center - d_half,
                    d_center + d_half,
                    n_band_samples,
                )

                elevations = []
                risks = []
                landings = []

                for d_s in d_samples:
                    wx = float(ref_pt[0] + d_s * normal[0])
                    wy = float(ref_pt[1] + d_s * normal[1])

                    elev = terrain.get_elevation(wx, wy)
                    slope = terrain.get_slope(wx, wy)
                    rough = terrain.get_roughness(wx, wy)

                    # 간이 위험도 계산
                    r = float(np.clip(
                        0.5 * (slope / (np.pi / 4)) + 0.5 * rough,
                        0.0, 1.0
                    ))

                    # 간이 착륙 적합도
                    land = float(np.clip(
                        1.0 - 0.6 * (slope / (np.pi / 4)) - 0.4 * rough,
                        0.0, 1.0
                    ))

                    elevations.append(elev)
                    risks.append(r)
                    landings.append(land)

                elevations = np.array(elevations, dtype=np.float32)
                risks = np.array(risks, dtype=np.float32)
                landings = np.array(landings, dtype=np.float32)

                # 고도 정규화 (최대 고도 대비)
                max_possible_elev = terrain.max_elevation
                elev_norm = float(np.max(elevations) / max(max_possible_elev, 1.0))

                profile[k, b, 0] = float(np.mean(risks))         # mean_risk
                profile[k, b, 1] = float(np.percentile(risks, 95))  # p95_risk
                profile[k, b, 2] = elev_norm                      # max_elev_norm
                profile[k, b, 3] = float(np.mean(landings))       # landing_score
                profile[k, b, 4] = 0.0                            # no_fly_frac (확장용)

        return profile

    # ------------------------------------------------------------------
    # 속성
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Corridor(soft={self.soft_width}m, hard={self.hard_width}m, "
            f"center_half={self.center_half_width}m)"
        )
