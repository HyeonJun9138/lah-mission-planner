"""
DEM 지형 로더 모듈 (DEMTerrain)
- GeoTIFF DEM 파일을 로드하여 SyntheticTerrain과 동일한 인터페이스 제공
- SRTM 1-arc-second 타일 자동 모자이크 지원
- UTM52N(EPSG:32652)으로 재투영하여 미터 단위 좌표계 통일
- 중심 위경도 + 반경(km) 지정으로 관심 영역 크롭 지원
- nodata(-9999, -32767) 처리 및 보간
"""

from __future__ import annotations

import os
import glob as _glob
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# 유틸리티 함수
# ──────────────────────────────────────────────────────────────────────

def list_available_dems(resource_dir: Union[str, Path]) -> List[Dict]:
    """
    지정 디렉토리의 사용 가능한 DEM 파일 목록 반환.

    Args:
        resource_dir: DEM 파일이 있는 디렉토리 경로

    Returns:
        각 DEM 파일 정보 dict 리스트:
            {
                "name": 파일명 (확장자 제외),
                "path": 전체 경로,
                "type": "srtm_1arc" | "custom_utm",
                "crs": 좌표계 문자열,
                "bounds": (left, bottom, right, top),
                "resolution_m": 해상도 [m],
                "shape": (rows, cols),
                "nodata": nodata 값,
                "elev_min": 최솟값,
                "elev_max": 최댓값,
            }
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio가 필요합니다: pip install rasterio")

    resource_dir = Path(resource_dir)
    result = []

    tif_files = sorted(resource_dir.glob("*.tif")) + sorted(resource_dir.glob("*.tiff"))
    # 임시 파일(_로 시작하는 파일) 제외
    tif_files = [f for f in tif_files if not f.name.startswith('_')]

    for tif_path in tif_files:
        try:
            with rasterio.open(str(tif_path)) as ds:
                data = ds.read(1)
                nodata = ds.nodata

                # nodata 마스킹
                valid_mask = np.ones(data.shape, dtype=bool)
                if nodata is not None:
                    valid_mask = np.abs(data - nodata) > 1.0
                valid_data = data[valid_mask]

                elev_min = float(valid_data.min()) if valid_data.size > 0 else 0.0
                elev_max = float(valid_data.max()) if valid_data.size > 0 else 0.0

                # 타입 판별
                crs_epsg = ds.crs.to_epsg() if ds.crs else None
                if crs_epsg == 4326:
                    dem_type = "srtm_1arc"
                    # EPSG:4326 해상도를 미터로 근사 변환 (위도 38도 기준)
                    res_deg = abs(ds.res[0])
                    lat_mid = (ds.bounds.bottom + ds.bounds.top) / 2
                    res_m = res_deg * 111320 * math.cos(math.radians(lat_mid))
                elif crs_epsg == 32652:
                    dem_type = "custom_utm"
                    res_m = abs(ds.res[0])
                else:
                    dem_type = "unknown"
                    res_m = abs(ds.res[0])

                result.append({
                    "name": tif_path.stem,
                    "path": str(tif_path),
                    "type": dem_type,
                    "crs": str(ds.crs),
                    "crs_epsg": crs_epsg,
                    "bounds": tuple(ds.bounds),
                    "resolution_m": round(res_m, 1),
                    "shape": ds.shape,
                    "nodata": nodata,
                    "elev_min": round(elev_min, 1),
                    "elev_max": round(elev_max, 1),
                })
        except Exception as e:
            logger.warning(f"DEM 파일 열기 실패 {tif_path}: {e}")

    return result


# ──────────────────────────────────────────────────────────────────────
# DEMTerrain 클래스
# ──────────────────────────────────────────────────────────────────────

class DEMTerrain:
    """
    GeoTIFF DEM 파일 기반 지형 클래스.

    SyntheticTerrain과 완전히 동일한 인터페이스를 제공하여
    모든 하위 코드(reward, termination, observation 등)가
    변경 없이 동작한다.

    좌표계:
        - 내부 좌표: UTM52N (EPSG:32652) 기반 로컬 미터 좌표
        - origin_utm_x, origin_utm_y: UTM 절대 좌표 원점
        - 로컬 (x, y): (UTM_x - origin_utm_x, UTM_y - origin_utm_y)

    Attributes:
        size_x (int): 동서 방향 격자 수 (cols)
        size_y (int): 남북 방향 격자 수 (rows)
        resolution (float): 격자 해상도 [m]
        heightmap (np.ndarray): 고도 맵 shape=(size_y, size_x) [m]
        slope_map (np.ndarray): 경사도 맵 [rad]
        roughness_map (np.ndarray): 거칠기 맵 [0~1]
        origin_utm_x (float): UTM X 원점 [m]
        origin_utm_y (float): UTM Y 원점 [m]
    """

    # SRTM 타일 파일명 패턴 매핑 (lat_floor, lon_floor → 파일명)
    _SRTM_PATTERN = "n{lat}_e{lon}_1arc_v3.tif"

    def __init__(self) -> None:
        """내부 초기화 전용. 팩토리 메서드를 사용하세요."""
        self.heightmap: np.ndarray = np.zeros((100, 100), dtype=np.float32)
        self.size_x: int = 100
        self.size_y: int = 100
        self.resolution: float = 30.0
        self.origin_utm_x: float = 0.0
        self.origin_utm_y: float = 0.0
        self._crs: Optional[object] = None
        self._src_crs: Optional[object] = None
        self._slope_map: Optional[np.ndarray] = None
        self._roughness_map: Optional[np.ndarray] = None
        self._source_file: str = ""
        self._interp: Optional[RegularGridInterpolator] = None

    # ──────────────────────────────────────────────────────────────
    # 팩토리 메서드
    # ──────────────────────────────────────────────────────────────

    @classmethod
    def from_file(
        cls,
        tif_path: Union[str, Path],
        crop_size_km: Optional[float] = None,
        crop_center_lat: Optional[float] = None,
        crop_center_lon: Optional[float] = None,
        target_resolution_m: Optional[float] = None,
    ) -> "DEMTerrain":
        """
        단일 GeoTIFF 파일에서 DEMTerrain 로드.

        Args:
            tif_path: GeoTIFF 파일 경로
            crop_size_km: 크롭 크기 [km]. None이면 전체 파일 사용.
            crop_center_lat: 크롭 중심 위도 (crop_size_km 사용 시)
            crop_center_lon: 크롭 중심 경도 (crop_size_km 사용 시)
            target_resolution_m: 목표 해상도 [m]. None이면 원본 해상도.

        Returns:
            DEMTerrain 인스턴스
        """
        try:
            import rasterio
            from rasterio.warp import reproject, Resampling, calculate_default_transform
            from rasterio.crs import CRS
        except ImportError:
            raise ImportError("rasterio가 필요합니다: pip install rasterio")

        tif_path = str(tif_path)
        terrain = cls()
        terrain._source_file = tif_path

        logger.info(f"DEM 로딩: {tif_path}")

        with rasterio.open(tif_path) as ds:
            src_crs = ds.crs
            terrain._src_crs = src_crs
            dst_crs = CRS.from_epsg(32652)
            terrain._crs = dst_crs

            # UTM52N으로 재투영 변환 행렬 계산
            if target_resolution_m:
                dst_res = target_resolution_m
            else:
                # 원본 해상도를 미터로 변환
                if src_crs.to_epsg() == 4326:
                    # 위도 중심값으로 미터 환산
                    lat_mid = (ds.bounds.bottom + ds.bounds.top) / 2
                    dst_res = abs(ds.res[0]) * 111320 * math.cos(math.radians(lat_mid))
                    dst_res = round(dst_res / 10) * 10  # 10m 단위 반올림
                    dst_res = max(dst_res, 10.0)
                else:
                    dst_res = abs(ds.res[0])

            transform, width, height = calculate_default_transform(
                src_crs, dst_crs, ds.width, ds.height,
                *ds.bounds,
                resolution=dst_res,
            )

            # 재투영 실행
            data = np.zeros((height, width), dtype=np.float32)
            reproject(
                source=rasterio.band(ds, 1),
                destination=data,
                src_transform=ds.transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=ds.nodata,
                dst_nodata=-9999.0,
            )

            # nodata 처리
            data = cls._fill_nodata(data, nodata_val=-9999.0)

            # UTM 원점 (왼쪽 상단 = 최솟값 X, 최솟값 Y)
            # rasterio transform: (x_origin, x_res, 0, y_origin, 0, y_res)
            # y_origin은 북쪽(최대) 좌표이므로, 남쪽(최소) Y를 계산
            utm_x_origin = transform.c                   # 왼쪽 경계 UTM X
            utm_y_origin = transform.f + transform.e * height  # 아래쪽 경계 UTM Y (transform.e < 0)

            resolution = dst_res

            # 크롭 처리
            if crop_size_km is not None:
                data, utm_x_origin, utm_y_origin = cls._crop_utm(
                    data, utm_x_origin, utm_y_origin, resolution,
                    crop_size_km, crop_center_lat, crop_center_lon, dst_crs,
                )

            terrain.heightmap = data.astype(np.float32)
            terrain.size_x = data.shape[1]
            terrain.size_y = data.shape[0]
            terrain.resolution = float(resolution)
            terrain.origin_utm_x = float(utm_x_origin)
            terrain.origin_utm_y = float(utm_y_origin)

        # 보간기 초기화
        terrain._build_interpolator()

        logger.info(
            f"DEM 로드 완료: {terrain.size_x}x{terrain.size_y} px, "
            f"res={terrain.resolution:.1f}m, "
            f"elev=[{terrain.heightmap.min():.1f}, {terrain.heightmap.max():.1f}]m"
        )
        return terrain

    @classmethod
    def from_region(
        cls,
        resource_dir: Union[str, Path],
        center_lat: float,
        center_lon: float,
        size_km: float,
        target_resolution_m: Optional[float] = None,
    ) -> "DEMTerrain":
        """
        위경도 중심 + 크기 지정으로 관심 지역 DEM 로드.

        SRTM 타일과 커스텀 UTM 타일 중 가장 적합한 것을 자동 선택.
        인접 SRTM 타일이 필요하면 자동 모자이크.

        Args:
            resource_dir: DEM 파일 디렉토리
            center_lat: 중심 위도 [°]
            center_lon: 중심 경도 [°]
            size_km: 관심 영역 크기 [km] (정사각형)
            target_resolution_m: 목표 해상도 [m]

        Returns:
            DEMTerrain 인스턴스
        """
        resource_dir = Path(resource_dir)

        # 먼저 커스텀 UTM 타일에서 검색 (더 정확)
        utm_candidates = cls._find_utm_tile(resource_dir, center_lat, center_lon, size_km)
        if utm_candidates:
            tif_path = utm_candidates[0]
            logger.info(f"커스텀 UTM 타일 선택: {tif_path}")
            return cls.from_file(
                tif_path,
                crop_size_km=size_km,
                crop_center_lat=center_lat,
                crop_center_lon=center_lon,
                target_resolution_m=target_resolution_m,
            )

        # SRTM 타일 모자이크 시도
        srtm_files = cls._find_srtm_tiles(resource_dir, center_lat, center_lon, size_km)
        if not srtm_files:
            raise FileNotFoundError(
                f"위도 {center_lat:.2f}, 경도 {center_lon:.2f} 주변에 "
                f"적합한 DEM 파일을 찾을 수 없습니다. 디렉토리: {resource_dir}"
            )

        if len(srtm_files) == 1:
            return cls.from_file(
                srtm_files[0],
                crop_size_km=size_km,
                crop_center_lat=center_lat,
                crop_center_lon=center_lon,
                target_resolution_m=target_resolution_m,
            )

        # 복수 타일 모자이크
        mosaic_path = cls._mosaic_srtm(srtm_files, resource_dir)
        return cls.from_file(
            mosaic_path,
            crop_size_km=size_km,
            crop_center_lat=center_lat,
            crop_center_lon=center_lon,
            target_resolution_m=target_resolution_m,
        )

    # ──────────────────────────────────────────────────────────────
    # 좌표 변환 메서드
    # ──────────────────────────────────────────────────────────────

    def latlon_to_local(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        WGS84 위경도 → 로컬 미터 좌표 변환.

        Args:
            lat: 위도 [°]
            lon: 경도 [°]

        Returns:
            (x, y) 로컬 미터 좌표 [m]
        """
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32652", always_xy=True)
        utm_x, utm_y = transformer.transform(lon, lat)
        x = utm_x - self.origin_utm_x
        y = utm_y - self.origin_utm_y
        return float(x), float(y)

    def local_to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        """
        로컬 미터 좌표 → WGS84 위경도 변환.

        Args:
            x: 로컬 동서 좌표 [m]
            y: 로컬 남북 좌표 [m]

        Returns:
            (lat, lon) WGS84 위경도 [°]
        """
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:32652", "EPSG:4326", always_xy=True)
        utm_x = x + self.origin_utm_x
        utm_y = y + self.origin_utm_y
        lon, lat = transformer.transform(utm_x, utm_y)
        return float(lat), float(lon)

    # ──────────────────────────────────────────────────────────────
    # SyntheticTerrain 호환 지형 쿼리 메서드
    # ──────────────────────────────────────────────────────────────

    def get_elevation(self, x: float, y: float) -> float:
        """
        월드 좌표 (x, y)에서 고도값 반환 (쌍선형 보간).

        Args:
            x: 동서 좌표 [m] (로컬 좌표계)
            y: 남북 좌표 [m] (로컬 좌표계)

        Returns:
            고도값 [m]
        """
        col = x / self.resolution
        row = y / self.resolution

        # 경계 클램프
        col = float(np.clip(col, 0, self.size_x - 1))
        row = float(np.clip(row, 0, self.size_y - 1))

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

        return float(
            h00 * (1 - dc) * (1 - dr)
            + h10 * dc * (1 - dr)
            + h01 * (1 - dc) * dr
            + h11 * dc * dr
        )

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
            거칠기값 [0~1]
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
        cx, cy = self._world_to_grid(x, y)
        cx, cy = int(cx), int(cy)

        r0 = cy - half
        r1 = cy + half
        c0 = cx - half
        c1 = cx + half

        padded = np.pad(
            self.heightmap,
            pad_width=half,
            mode="edge",
        )
        patch = padded[
            r0 + half: r1 + half,
            c0 + half: c1 + half,
        ]

        if patch.shape != (patch_size, patch_size):
            patch = np.zeros((patch_size, patch_size), dtype=np.float32)

        return patch.astype(np.float32)

    # ──────────────────────────────────────────────────────────────
    # 속성 (SyntheticTerrain 호환)
    # ──────────────────────────────────────────────────────────────

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

    @property
    def crs_info(self) -> Dict:
        """좌표 참조 시스템 정보."""
        return {
            "dst_crs": "EPSG:32652 (UTM52N)",
            "src_crs": str(self._src_crs) if self._src_crs else "unknown",
            "origin_utm_x": self.origin_utm_x,
            "origin_utm_y": self.origin_utm_y,
        }

    def __repr__(self) -> str:
        return (
            f"DEMTerrain(size=({self.size_x},{self.size_y}), "
            f"res={self.resolution:.1f}m, "
            f"elev=[{self.heightmap.min():.0f}, {self.heightmap.max():.0f}]m, "
            f"src={os.path.basename(self._source_file)})"
        )

    # ──────────────────────────────────────────────────────────────
    # 내부 헬퍼 메서드
    # ──────────────────────────────────────────────────────────────

    def _world_to_grid(self, x: float, y: float) -> Tuple[float, float]:
        """월드 좌표 → 격자 인덱스 변환."""
        col = x / self.resolution
        row = y / self.resolution
        return col, row

    def _clamp_index(self, col: float, row: float) -> Tuple[int, int]:
        """인덱스를 heightmap 범위로 클램프."""
        col_int = int(np.clip(col, 0, self.size_x - 1))
        row_int = int(np.clip(row, 0, self.size_y - 1))
        return col_int, row_int

    def _build_interpolator(self) -> None:
        """RegularGridInterpolator 빌드 (고속 배치 쿼리용)."""
        # 격자 좌표 생성 (row 방향 = Y, col 방향 = X)
        rows = np.arange(self.size_y, dtype=np.float32) * self.resolution
        cols = np.arange(self.size_x, dtype=np.float32) * self.resolution
        self._interp = RegularGridInterpolator(
            (rows, cols),
            self.heightmap.astype(np.float32),
            method="linear",
            bounds_error=False,
            fill_value=None,  # 경계 외부는 가장 가까운 값으로 extrapolate
        )

    @staticmethod
    def _fill_nodata(
        data: np.ndarray,
        nodata_val: float = -9999.0,
        threshold: float = 100.0,
    ) -> np.ndarray:
        """
        nodata 영역을 주변 유효값으로 보간 처리.

        Args:
            data: 고도 데이터 배열
            nodata_val: nodata 마커 값
            threshold: nodata 판단 기준 (nodata_val과의 차이가 threshold 미만이면 nodata)

        Returns:
            nodata가 보간된 고도 배열
        """
        mask = np.abs(data - nodata_val) < threshold

        if not mask.any():
            return data

        logger.debug(f"nodata 픽셀 수: {mask.sum()} / {mask.size}")

        # 주변 유효값으로 간단한 nearest-neighbor 채우기
        from scipy.ndimage import distance_transform_edt
        _, indices = distance_transform_edt(mask, return_indices=True)
        filled = data.copy()
        filled[mask] = data[indices[0][mask], indices[1][mask]]

        # 여전히 nodata인 경우 전체 평균으로 채우기
        still_nodata = np.abs(filled - nodata_val) < threshold
        if still_nodata.any():
            valid_mean = float(data[~mask].mean()) if (~mask).any() else 200.0
            filled[still_nodata] = valid_mean

        return filled

    @staticmethod
    def _crop_utm(
        data: np.ndarray,
        utm_x_origin: float,
        utm_y_origin: float,
        resolution: float,
        crop_size_km: float,
        crop_center_lat: Optional[float],
        crop_center_lon: Optional[float],
        dst_crs: object,
    ) -> Tuple[np.ndarray, float, float]:
        """
        UTM 좌표계에서 크롭 수행.

        Args:
            data: 전체 고도 데이터 shape=(H, W)
            utm_x_origin: UTM X 원점 (왼쪽 경계)
            utm_y_origin: UTM Y 원점 (아래쪽 경계)
            resolution: 해상도 [m]
            crop_size_km: 크롭 크기 [km]
            crop_center_lat: 크롭 중심 위도
            crop_center_lon: 크롭 중심 경도
            dst_crs: 목적지 CRS 객체

        Returns:
            (cropped_data, new_utm_x_origin, new_utm_y_origin)
        """
        h, w = data.shape
        half_m = (crop_size_km * 1000) / 2.0

        if crop_center_lat is not None and crop_center_lon is not None:
            # 위경도 → UTM 변환
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:32652", always_xy=True)
            center_utm_x, center_utm_y = transformer.transform(crop_center_lon, crop_center_lat)
        else:
            # 파일 중심 사용
            center_utm_x = utm_x_origin + (w * resolution) / 2
            center_utm_y = utm_y_origin + (h * resolution) / 2

        # 로컬 좌표로 변환
        local_cx = (center_utm_x - utm_x_origin) / resolution
        local_cy = (center_utm_y - utm_y_origin) / resolution

        # 크롭 픽셀 범위
        half_px = int(half_m / resolution)
        col_start = max(0, int(local_cx) - half_px)
        col_end   = min(w, int(local_cx) + half_px)
        row_start = max(0, int(local_cy) - half_px)
        row_end   = min(h, int(local_cy) + half_px)

        if col_end <= col_start or row_end <= row_start:
            logger.warning("크롭 범위가 데이터 범위 밖입니다. 전체 데이터를 사용합니다.")
            return data, utm_x_origin, utm_y_origin

        cropped = data[row_start:row_end, col_start:col_end]
        new_utm_x = utm_x_origin + col_start * resolution
        new_utm_y = utm_y_origin + row_start * resolution

        logger.debug(
            f"크롭: ({col_start},{row_start}) ~ ({col_end},{row_end}), "
            f"결과 크기: {cropped.shape}"
        )
        return cropped, new_utm_x, new_utm_y

    @classmethod
    def _find_utm_tile(
        cls,
        resource_dir: Path,
        center_lat: float,
        center_lon: float,
        size_km: float,
    ) -> List[Path]:
        """
        UTM 타일 중 관심 지역을 포함하는 파일 검색.

        Args:
            resource_dir: 리소스 디렉토리
            center_lat: 중심 위도
            center_lon: 중심 경도
            size_km: 관심 영역 크기 [km]

        Returns:
            매칭 파일 경로 리스트
        """
        try:
            import rasterio
            from pyproj import Transformer
        except ImportError:
            return []

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32652", always_xy=True)
        center_utm_x, center_utm_y = transformer.transform(center_lon, center_lat)
        half_m = (size_km * 1000) / 2

        candidates = []
        for tif in resource_dir.glob("*.tif"):
            try:
                with rasterio.open(str(tif)) as ds:
                    if ds.crs.to_epsg() != 32652:
                        continue
                    b = ds.bounds
                    # 중심점이 타일 범위 내에 있는지 확인
                    if (b.left <= center_utm_x <= b.right and
                            b.bottom <= center_utm_y <= b.top):
                        candidates.append(tif)
            except Exception:
                continue

        return candidates

    @classmethod
    def _find_srtm_tiles(
        cls,
        resource_dir: Path,
        center_lat: float,
        center_lon: float,
        size_km: float,
    ) -> List[Path]:
        """
        SRTM 타일 중 관심 지역을 커버하는 타일 파일 검색.

        Args:
            resource_dir: 리소스 디렉토리
            center_lat: 중심 위도
            center_lon: 중심 경도
            size_km: 관심 영역 크기 [km]

        Returns:
            매칭 SRTM 파일 경로 리스트
        """
        # 관심 영역의 위경도 범위 계산
        lat_deg_per_km = 1.0 / 111.0
        lon_deg_per_km = 1.0 / (111.0 * math.cos(math.radians(center_lat)))
        half_km = size_km / 2.0

        lat_min = center_lat - half_km * lat_deg_per_km
        lat_max = center_lat + half_km * lat_deg_per_km
        lon_min = center_lon - half_km * lon_deg_per_km
        lon_max = center_lon + half_km * lon_deg_per_km

        # 필요한 타일 목록 계산
        needed_tiles = set()
        for lat in range(int(math.floor(lat_min)), int(math.floor(lat_max)) + 1):
            for lon in range(int(math.floor(lon_min)), int(math.floor(lon_max)) + 1):
                needed_tiles.add((lat, lon))

        # 실제 파일 검색
        found = []
        for lat, lon in needed_tiles:
            fname = f"n{lat:02d}_e{lon:03d}_1arc_v3.tif"
            fpath = resource_dir / fname
            if fpath.exists():
                found.append(fpath)
            else:
                logger.debug(f"SRTM 타일 없음: {fname}")

        return found

    @classmethod
    def _mosaic_srtm(cls, tif_files: List[Path], resource_dir: Path) -> str:
        """
        복수 SRTM 타일을 모자이크하여 임시 파일로 저장.

        Args:
            tif_files: 모자이크할 파일 리스트
            resource_dir: 임시 파일 저장 경로

        Returns:
            모자이크 파일 경로 문자열
        """
        try:
            import rasterio
            from rasterio.merge import merge
        except ImportError:
            raise ImportError("rasterio가 필요합니다: pip install rasterio")

        import tempfile

        datasets = [rasterio.open(str(f)) for f in tif_files]
        try:
            mosaic_arr, mosaic_transform = merge(datasets)
            profile = datasets[0].profile.copy()
            profile.update({
                "height": mosaic_arr.shape[1],
                "width": mosaic_arr.shape[2],
                "transform": mosaic_transform,
            })

            # 임시 파일에 저장
            tmp_path = str(resource_dir / "_mosaic_tmp.tif")
            with rasterio.open(tmp_path, "w", **profile) as dst:
                dst.write(mosaic_arr)

            logger.info(f"SRTM 모자이크 완료: {len(tif_files)}개 타일 → {tmp_path}")
            return tmp_path
        finally:
            for ds in datasets:
                ds.close()
