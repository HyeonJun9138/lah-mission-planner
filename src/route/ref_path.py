"""
참조 경로(Reference Path) 관리 모듈
- 운용자가 입력한 piecewise-linear polyline을 등간격 리샘플링
- 각 샘플 포인트의 접선벡터(tangent), 법선벡터(normal), 누적 거리(s) 계산
- Frenet 좌표계를 위한 투영 기능 제공
"""

from __future__ import annotations

import json
import numpy as np
from typing import List, Tuple, Optional


class RefPath:
    """
    운용자가 입력한 참조 경로(ref path) 클래스.

    waypoints를 등간격으로 리샘플링하고 Frenet feature 계산에
    필요한 접선/법선/누적 거리를 제공한다.

    Attributes:
        points (np.ndarray): 리샘플된 경로 포인트 shape (N, 3) [x, y, z]
        tangents (np.ndarray): 각 포인트의 접선 단위벡터 shape (N, 2) [tx, ty]
        normals (np.ndarray): 각 포인트의 법선 단위벡터 shape (N, 2) [nx, ny]
        cumulative_s (np.ndarray): 누적 호 거리 shape (N,) [m]
        total_length (float): 경로 전체 길이 [m]
    """

    def __init__(
        self,
        points: np.ndarray,
        tangents: np.ndarray,
        normals: np.ndarray,
        cumulative_s: np.ndarray,
    ) -> None:
        """
        직접 생성자 (from_waypoints() 또는 load_json() 사용 권장).

        Args:
            points: 리샘플된 경로 포인트 (N, 3)
            tangents: 접선 벡터 (N, 2)
            normals: 법선 벡터 (N, 2)
            cumulative_s: 누적 거리 (N,)
        """
        self.points = points.astype(np.float32)
        self.tangents = tangents.astype(np.float32)
        self.normals = normals.astype(np.float32)
        self.cumulative_s = cumulative_s.astype(np.float32)

    # ------------------------------------------------------------------
    # 팩토리 메서드
    # ------------------------------------------------------------------

    @classmethod
    def from_waypoints(
        cls,
        waypoints: List[Tuple[float, float, float]],
        step_m: float = 50.0,
    ) -> "RefPath":
        """
        waypoints 리스트에서 RefPath 생성.

        Args:
            waypoints: [(x0,y0,z0), (x1,y1,z1), ...] 형태의 경유점 리스트 [m]
            step_m: 리샘플 간격 [m] (기본 50m)

        Returns:
            RefPath 인스턴스

        Raises:
            ValueError: waypoints가 2개 미만일 때
        """
        if len(waypoints) < 2:
            raise ValueError("waypoints는 최소 2개 이상이어야 합니다.")

        wp_arr = np.array(waypoints, dtype=np.float32)  # (M, 3)
        if wp_arr.shape[1] == 2:
            # z 없으면 0으로 패딩
            wp_arr = np.hstack([wp_arr, np.zeros((len(wp_arr), 1), dtype=np.float32)])

        # 원본 경로 누적 거리 계산
        diffs = np.diff(wp_arr[:, :2], axis=0)  # (M-1, 2)
        seg_lens = np.linalg.norm(diffs, axis=1)  # (M-1,)
        orig_s = np.concatenate([[0.0], np.cumsum(seg_lens)])  # (M,)
        total_len = float(orig_s[-1])

        if total_len < 1e-6:
            raise ValueError("경로 길이가 너무 짧습니다 (< 1m).")

        # 리샘플 거리 배열 생성 (0 ~ total_len, step_m 간격)
        n_pts = max(2, int(np.ceil(total_len / step_m)) + 1)
        new_s = np.linspace(0.0, total_len, n_pts)

        # x, y, z 각각 1D 보간
        new_x = np.interp(new_s, orig_s, wp_arr[:, 0])
        new_y = np.interp(new_s, orig_s, wp_arr[:, 1])
        new_z = np.interp(new_s, orig_s, wp_arr[:, 2])

        new_pts = np.stack([new_x, new_y, new_z], axis=1)  # (N, 3)

        # 접선 벡터 계산 (중앙차분)
        tangents = _compute_tangents(new_pts[:, :2])

        # 법선 벡터 (접선을 90도 반시계 회전)
        normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

        return cls(
            points=new_pts,
            tangents=tangents,
            normals=normals,
            cumulative_s=new_s.astype(np.float32),
        )

    @classmethod
    def from_json(cls, filepath: str, step_m: float = 50.0) -> "RefPath":
        """
        JSON 파일에서 RefPath 로드.

        JSON 형식: [[x0,y0,z0], [x1,y1,z1], ...] 또는
                   {"waypoints": [[x0,y0,z0], ...]}

        Args:
            filepath: JSON 파일 경로
            step_m: 리샘플 간격 [m]

        Returns:
            RefPath 인스턴스
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            waypoints = [tuple(pt) for pt in data]
        elif isinstance(data, dict) and "waypoints" in data:
            waypoints = [tuple(pt) for pt in data["waypoints"]]
        else:
            raise ValueError(f"지원하지 않는 JSON 형식: {filepath}")

        return cls.from_waypoints(waypoints, step_m=step_m)

    @classmethod
    def make_straight(
        cls,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        step_m: float = 50.0,
    ) -> "RefPath":
        """
        직선 경로 생성 (테스트/Stage0 용도).

        Args:
            start: 시작 위치 (x, y, z)
            end: 끝 위치 (x, y, z)
            step_m: 리샘플 간격 [m]

        Returns:
            RefPath 인스턴스
        """
        return cls.from_waypoints([start, end], step_m=step_m)

    # ------------------------------------------------------------------
    # 리샘플링
    # ------------------------------------------------------------------

    def resample(self, step_m: float = 50.0) -> "RefPath":
        """
        현재 경로를 다른 간격으로 재리샘플링.

        Args:
            step_m: 새 리샘플 간격 [m]

        Returns:
            새 RefPath 인스턴스
        """
        waypoints = [tuple(pt) for pt in self.points.tolist()]
        return RefPath.from_waypoints(waypoints, step_m=step_m)

    # ------------------------------------------------------------------
    # 쿼리 메서드
    # ------------------------------------------------------------------

    def get_reference_altitude(self, s: float) -> float:
        """
        경로 좌표 s에서 참조 고도 반환 (선형 보간).

        Args:
            s: 경로 좌표 [m] (누적 거리)

        Returns:
            참조 고도 z_ref [m]
        """
        s_clip = float(np.clip(s, 0.0, self.total_length))
        z_ref = float(np.interp(s_clip, self.cumulative_s, self.points[:, 2]))
        return z_ref

    def get_point_at_s(self, s: float) -> np.ndarray:
        """
        경로 좌표 s에서 (x, y, z) 포인트 반환.

        Args:
            s: 경로 좌표 [m]

        Returns:
            (x, y, z) 배열
        """
        s_clip = float(np.clip(s, 0.0, self.total_length))
        x = float(np.interp(s_clip, self.cumulative_s, self.points[:, 0]))
        y = float(np.interp(s_clip, self.cumulative_s, self.points[:, 1]))
        z = self.get_reference_altitude(s_clip)
        return np.array([x, y, z], dtype=np.float32)

    def get_tangent_at_s(self, s: float) -> np.ndarray:
        """
        경로 좌표 s에서 접선 단위벡터 반환.

        Args:
            s: 경로 좌표 [m]

        Returns:
            (tx, ty) 단위벡터
        """
        s_clip = float(np.clip(s, 0.0, self.total_length))
        tx = float(np.interp(s_clip, self.cumulative_s, self.tangents[:, 0]))
        ty = float(np.interp(s_clip, self.cumulative_s, self.tangents[:, 1]))
        norm = np.sqrt(tx ** 2 + ty ** 2)
        if norm > 1e-8:
            return np.array([tx / norm, ty / norm], dtype=np.float32)
        return np.array([1.0, 0.0], dtype=np.float32)

    def get_lookahead_points(
        self,
        s_current: float,
        n_points: int = 16,
        spacing_m: float = 50.0,
    ) -> np.ndarray:
        """
        현재 진행 거리 s에서 앞쪽 N개 참조 포인트 반환.

        Args:
            s_current: 현재 경로 진행 거리 [m]
            n_points: 반환할 포인트 수
            spacing_m: 포인트 간격 [m]

        Returns:
            shape (n_points, 3) [x, y, z_ref]
        """
        lookahead = np.zeros((n_points, 3), dtype=np.float32)
        for i in range(n_points):
            s_target = s_current + (i + 1) * spacing_m
            lookahead[i] = self.get_point_at_s(s_target)
        return lookahead

    def get_closest_s(self, x: float, y: float) -> Tuple[float, int]:
        """
        월드 좌표 (x, y)에서 가장 가까운 경로 포인트의 s값과 인덱스 반환.

        Args:
            x: 동서 좌표 [m]
            y: 남북 좌표 [m]

        Returns:
            (s_closest, idx_closest)
        """
        pos = np.array([x, y], dtype=np.float32)
        dists = np.linalg.norm(self.points[:, :2] - pos, axis=1)
        idx = int(np.argmin(dists))
        return float(self.cumulative_s[idx]), idx

    # ------------------------------------------------------------------
    # 속성
    # ------------------------------------------------------------------

    @property
    def total_length(self) -> float:
        """경로 전체 길이 [m]."""
        return float(self.cumulative_s[-1])

    @property
    def n_points(self) -> int:
        """리샘플된 포인트 수."""
        return len(self.points)

    @property
    def start_point(self) -> np.ndarray:
        """시작 포인트 (x, y, z)."""
        return self.points[0]

    @property
    def end_point(self) -> np.ndarray:
        """끝 포인트 (x, y, z)."""
        return self.points[-1]

    def __repr__(self) -> str:
        return (
            f"RefPath(n_pts={self.n_points}, "
            f"length={self.total_length:.0f}m, "
            f"start={self.start_point[:2].tolist()}, "
            f"end={self.end_point[:2].tolist()})"
        )


# ------------------------------------------------------------------
# 내부 헬퍼 함수
# ------------------------------------------------------------------

def _compute_tangents(xy: np.ndarray) -> np.ndarray:
    """
    2D 경로의 접선 단위벡터 계산 (중앙차분 + 경계처리).

    Args:
        xy: 경로 포인트 (N, 2)

    Returns:
        접선 단위벡터 (N, 2)
    """
    N = len(xy)
    tangents = np.zeros((N, 2), dtype=np.float32)

    # 내부: 중앙차분
    tangents[1:-1] = xy[2:] - xy[:-2]

    # 경계: 전방/후방 차분
    tangents[0] = xy[1] - xy[0]
    tangents[-1] = xy[-1] - xy[-2]

    # 정규화
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    tangents = tangents / norms

    return tangents
