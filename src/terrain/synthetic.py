"""
합성 지형 생성 모듈 (SyntheticTerrain)
- DEM 없이 numpy 기반 sin/cos 조합 및 Perlin-like noise로 합성 지형 생성
- 능선/계곡 패턴 포함
- 30m 기본 해상도
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


class SyntheticTerrain:
    """
    합성 지형 생성 클래스.

    numpy의 sin/cos 조합 및 Perlin-like 노이즈로 실제 산악 지형을 모사한다.
    DEM 파일이 없을 때 기본값으로 사용된다.

    Attributes:
        size_x (int): 동서 방향 격자 수
        size_y (int): 남북 방향 격자 수
        resolution (float): 격자 해상도 [m]
        seed (int): 난수 시드
        heightmap (np.ndarray): 생성된 고도 맵 [m] shape=(size_y, size_x)
        slope_map (np.ndarray): 경사도 맵 [rad]
        roughness_map (np.ndarray): 거칠기 맵
    """

    def __init__(
        self,
        size_x: int = 500,
        size_y: int = 500,
        resolution: float = 30.0,
        base_elevation: float = 200.0,
        max_elevation: float = 1200.0,
        seed: int = 42,
    ) -> None:
        """
        Args:
            size_x: 동서 방향 격자 수 (픽셀)
            size_y: 남북 방향 격자 수 (픽셀)
            resolution: 격자 해상도 [m] (기본 30m)
            base_elevation: 기반 고도 [m]
            max_elevation: 최대 고도 [m]
            seed: 난수 시드
        """
        self.size_x = size_x
        self.size_y = size_y
        self.resolution = resolution
        self.base_elevation = base_elevation
        self.max_elevation = max_elevation
        self.seed = seed

        # 고도 맵 생성
        self.heightmap = self.generate_heightmap(size_x, size_y, resolution)

        # 파생 특징 맵 (지연 계산)
        self._slope_map: np.ndarray | None = None
        self._roughness_map: np.ndarray | None = None

    # ------------------------------------------------------------------
    # 합성 지형 생성 핵심 메서드
    # ------------------------------------------------------------------

    def generate_heightmap(
        self,
        size_x: int,
        size_y: int,
        resolution: float = 30.0,
    ) -> np.ndarray:
        """
        Perlin-like 노이즈 + 능선/계곡 패턴으로 합성 고도 맵 생성.

        Args:
            size_x: 동서 방향 픽셀 수
            size_y: 남북 방향 픽셀 수
            resolution: 격자 해상도 [m]

        Returns:
            shape (size_y, size_x)의 고도 배열 [m]
        """
        rng = np.random.default_rng(self.seed)

        # 좌표 격자 생성
        x = np.linspace(0, 2 * np.pi, size_x)
        y = np.linspace(0, 2 * np.pi, size_y)
        X, Y = np.meshgrid(x, y)

        # --- 1) 저주파 기저 지형 (큰 산악 구조) ---
        base = (
            0.45 * np.sin(1.2 * X + 0.3) * np.cos(0.8 * Y)
            + 0.35 * np.cos(0.7 * X) * np.sin(1.5 * Y + 0.7)
            + 0.20 * np.sin(0.5 * X + 0.9 * Y)
        )

        # --- 2) 중간 주파수 지형 변화 (능선/계곡) ---
        mid_freq = (
            0.25 * np.sin(2.5 * X + rng.uniform(0, np.pi))
            * np.cos(2.1 * Y + rng.uniform(0, np.pi))
            + 0.15 * np.cos(3.0 * X) * np.sin(2.8 * Y)
            + 0.10 * np.sin(4.0 * X + 1.5 * Y + rng.uniform(0, np.pi))
        )

        # --- 3) 고주파 노이즈 (국지 지형 거칠기) ---
        # 다중 스케일 노이즈를 합산하여 Perlin-like 효과
        high_freq = np.zeros((size_y, size_x), dtype=np.float64)
        for octave in range(4):
            scale = 2 ** (octave + 3)
            amp = 0.5 ** (octave + 2)
            freq_x = rng.uniform(scale * 0.8, scale * 1.2)
            freq_y = rng.uniform(scale * 0.8, scale * 1.2)
            phase_x = rng.uniform(0, 2 * np.pi)
            phase_y = rng.uniform(0, 2 * np.pi)
            high_freq += amp * np.sin(freq_x * X / (2 * np.pi) + phase_x) * np.cos(
                freq_y * Y / (2 * np.pi) + phase_y
            )

        # --- 4) 능선 패턴 (ridge) ---
        # abs를 이용하여 능선 모양 생성
        ridge = (
            0.20 * (1.0 - np.abs(np.sin(1.8 * X + 0.4 * Y)))
            + 0.15 * (1.0 - np.abs(np.sin(0.9 * X - 1.1 * Y + 0.5)))
        )

        # --- 5) 계곡 패턴 (valley) ---
        valley = (
            -0.15 * np.exp(-((X - np.pi) ** 2 + (Y - np.pi) ** 2) / 2.0)
            - 0.10 * np.exp(-((X - 0.5) ** 2 + (Y - 1.5) ** 2) / 1.5)
        )

        # --- 6) 합성 및 정규화 ---
        combined = base + 0.5 * mid_freq + 0.3 * high_freq + ridge + valley

        # 0~1 정규화
        combined_min = combined.min()
        combined_max = combined.max()
        if combined_max - combined_min > 1e-6:
            combined = (combined - combined_min) / (combined_max - combined_min)
        else:
            combined = np.zeros_like(combined)

        # 목표 고도 범위로 스케일
        heightmap = self.base_elevation + combined * (self.max_elevation - self.base_elevation)

        return heightmap.astype(np.float32)

    # ------------------------------------------------------------------
    # 좌표계 변환 유틸
    # ------------------------------------------------------------------

    def _world_to_grid(self, x: float, y: float) -> Tuple[float, float]:
        """
        월드 좌표 (x, y) [m] → 격자 인덱스 (col, row) 변환.

        Args:
            x: 동서 좌표 [m]
            y: 남북 좌표 [m]

        Returns:
            (col, row) 실수 인덱스
        """
        col = x / self.resolution
        row = y / self.resolution
        return col, row

    def _clamp_index(self, col: float, row: float) -> Tuple[int, int]:
        """인덱스를 heightmap 범위로 클램프."""
        col_int = int(np.clip(col, 0, self.size_x - 1))
        row_int = int(np.clip(row, 0, self.size_y - 1))
        return col_int, row_int

    # ------------------------------------------------------------------
    # 지형 쿼리 인터페이스
    # ------------------------------------------------------------------

    def get_elevation(self, x: float, y: float) -> float:
        """
        월드 좌표 (x, y)에서 고도값 반환 (쌍선형 보간).

        Args:
            x: 동서 좌표 [m]
            y: 남북 좌표 [m]

        Returns:
            고도값 [m]
        """
        col, row = self._world_to_grid(x, y)

        # 경계 클램프
        col = np.clip(col, 0, self.size_x - 1)
        row = np.clip(row, 0, self.size_y - 1)

        # 쌍선형 보간
        col0, row0 = int(col), int(row)
        col1 = min(col0 + 1, self.size_x - 1)
        row1 = min(row0 + 1, self.size_y - 1)

        dc = col - col0
        dr = row - row0

        h00 = float(self.heightmap[row0, col0])
        h10 = float(self.heightmap[row0, col1])
        h01 = float(self.heightmap[row1, col0])
        h11 = float(self.heightmap[row1, col1])

        elev = (
            h00 * (1 - dc) * (1 - dr)
            + h10 * dc * (1 - dr)
            + h01 * (1 - dc) * dr
            + h11 * dc * dr
        )
        return float(elev)

    def get_elevation_batch(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        배치 좌표에서 고도값 반환.

        Args:
            xs: 동서 좌표 배열 [m]
            ys: 남북 좌표 배열 [m]

        Returns:
            고도값 배열 [m]
        """
        cols = np.clip(xs / self.resolution, 0, self.size_x - 1)
        rows = np.clip(ys / self.resolution, 0, self.size_y - 1)

        col0 = np.floor(cols).astype(int)
        row0 = np.floor(rows).astype(int)
        col1 = np.minimum(col0 + 1, self.size_x - 1)
        row1 = np.minimum(row0 + 1, self.size_y - 1)

        dc = cols - col0
        dr = rows - row0

        h00 = self.heightmap[row0, col0]
        h10 = self.heightmap[row0, col1]
        h01 = self.heightmap[row1, col0]
        h11 = self.heightmap[row1, col1]

        return (
            h00 * (1 - dc) * (1 - dr)
            + h10 * dc * (1 - dr)
            + h01 * (1 - dc) * dr
            + h11 * dc * dr
        ).astype(np.float32)

    def get_slope(self, x: float, y: float) -> float:
        """
        월드 좌표 (x, y)에서 경사도 반환 [rad].

        Args:
            x: 동서 좌표 [m]
            y: 남북 좌표 [m]

        Returns:
            경사도 [rad]
        """
        if self._slope_map is None:
            from src.terrain.features import compute_slope
            self._slope_map = compute_slope(self.heightmap, self.resolution)

        col, row = self._clamp_index(*self._world_to_grid(x, y))
        return float(self._slope_map[row, col])

    def get_roughness(self, x: float, y: float) -> float:
        """
        월드 좌표 (x, y)에서 거칠기 반환.

        Args:
            x: 동서 좌표 [m]
            y: 남북 좌표 [m]

        Returns:
            거칠기값 (표고 표준편차 정규화)
        """
        if self._roughness_map is None:
            from src.terrain.features import compute_roughness
            self._roughness_map = compute_roughness(self.heightmap)

        col, row = self._clamp_index(*self._world_to_grid(x, y))
        return float(self._roughness_map[row, col])

    def get_local_patch(
        self,
        x: float,
        y: float,
        patch_size: int = 64,
        patch_resolution: float = 30.0,
    ) -> np.ndarray:
        """
        항공기 위치 중심의 로컬 패치 추출.

        Args:
            x: 항공기 동서 좌표 [m]
            y: 항공기 남북 좌표 [m]
            patch_size: 패치 픽셀 크기 (정사각형)
            patch_resolution: 패치 해상도 [m]

        Returns:
            shape (patch_size, patch_size) 고도 패치 [m]
        """
        half = patch_size // 2
        # 패치 중심 격자 인덱스
        cx, cy = self._world_to_grid(x, y)
        cx, cy = int(cx), int(cy)

        # 패치 범위
        r0 = cy - half
        r1 = cy + half
        c0 = cx - half
        c1 = cx + half

        # 패딩 처리 (경계 바깥은 최솟값으로 패딩)
        padded = np.pad(
            self.heightmap,
            pad_width=half,
            mode="edge",
        )
        patch = padded[
            r0 + half : r1 + half,
            c0 + half : c1 + half,
        ]

        # 크기 보정 (경계 처리 후)
        if patch.shape != (patch_size, patch_size):
            patch = np.zeros((patch_size, patch_size), dtype=np.float32)

        return patch.astype(np.float32)

    # ------------------------------------------------------------------
    # 속성
    # ------------------------------------------------------------------

    @property
    def world_extent(self) -> Tuple[float, float, float, float]:
        """월드 좌표 범위 (x_min, x_max, y_min, y_max) [m]."""
        return (
            0.0,
            self.size_x * self.resolution,
            0.0,
            self.size_y * self.resolution,
        )

    @property
    def slope_map(self) -> np.ndarray:
        """전체 경사도 맵 (지연 계산)."""
        if self._slope_map is None:
            from src.terrain.features import compute_slope
            self._slope_map = compute_slope(self.heightmap, self.resolution)
        return self._slope_map

    @property
    def roughness_map(self) -> np.ndarray:
        """전체 거칠기 맵 (지연 계산)."""
        if self._roughness_map is None:
            from src.terrain.features import compute_roughness
            self._roughness_map = compute_roughness(self.heightmap)
        return self._roughness_map

    def __repr__(self) -> str:
        return (
            f"SyntheticTerrain(size=({self.size_x},{self.size_y}), "
            f"res={self.resolution}m, "
            f"elev=[{self.heightmap.min():.0f}, {self.heightmap.max():.0f}]m)"
        )
