"""
지형 특징 분석 모듈
- slope, roughness, local_relief, TPI, risk_map, landing_suitability 계산
- numpy/scipy 기반 최적화된 연산
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter, generic_filter
from typing import Dict, Optional


def compute_slope(
    heightmap: np.ndarray,
    res: float = 30.0,
) -> np.ndarray:
    """
    경사도(slope) 맵 계산.

    slope = atan(sqrt((dz/dx)^2 + (dz/dy)^2))

    Args:
        heightmap: 고도 맵 배열 shape (H, W) [m]
        res: 격자 해상도 [m]

    Returns:
        slope 맵 [rad] shape (H, W)
    """
    # Sobel 커널을 이용한 경사도 계산
    dzdx = np.zeros_like(heightmap, dtype=np.float32)
    dzdy = np.zeros_like(heightmap, dtype=np.float32)

    # 내부 영역: 중앙 차분
    dzdx[:, 1:-1] = (heightmap[:, 2:] - heightmap[:, :-2]) / (2.0 * res)
    dzdy[1:-1, :] = (heightmap[2:, :] - heightmap[:-2, :]) / (2.0 * res)

    # 경계 영역: 전방/후방 차분
    dzdx[:, 0] = (heightmap[:, 1] - heightmap[:, 0]) / res
    dzdx[:, -1] = (heightmap[:, -1] - heightmap[:, -2]) / res
    dzdy[0, :] = (heightmap[1, :] - heightmap[0, :]) / res
    dzdy[-1, :] = (heightmap[-1, :] - heightmap[-2, :]) / res

    slope = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2)).astype(np.float32)
    return slope


def compute_roughness(
    heightmap: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """
    거칠기(roughness) 맵 계산.

    roughness = 지역 창(window) 내 고도 표준편차

    Args:
        heightmap: 고도 맵 배열 shape (H, W) [m]
        window: 분석 창 크기 (홀수 권장)

    Returns:
        roughness 맵 shape (H, W), 0~1 정규화
    """
    def _std(values: np.ndarray) -> float:
        return float(np.std(values))

    roughness = generic_filter(
        heightmap.astype(np.float64),
        _std,
        size=window,
    ).astype(np.float32)

    # 0~1 정규화
    r_max = roughness.max()
    if r_max > 1e-6:
        roughness = roughness / r_max

    return roughness


def compute_local_relief(
    heightmap: np.ndarray,
    window: int = 7,
) -> np.ndarray:
    """
    국지 지형 기복(local relief) 맵 계산.

    local_relief = window 내 최대 고도 - 최소 고도

    Args:
        heightmap: 고도 맵 배열 shape (H, W) [m]
        window: 분석 창 크기

    Returns:
        relief 맵 [m] shape (H, W), 0~1 정규화
    """
    def _max(v: np.ndarray) -> float:
        return float(v.max())

    def _min(v: np.ndarray) -> float:
        return float(v.min())

    hm = heightmap.astype(np.float64)
    local_max = generic_filter(hm, _max, size=window).astype(np.float32)
    local_min = generic_filter(hm, _min, size=window).astype(np.float32)

    relief = local_max - local_min

    # 0~1 정규화
    r_max = relief.max()
    if r_max > 1e-6:
        relief = relief / r_max

    return relief


def compute_tpi(
    heightmap: np.ndarray,
    window: int = 7,
) -> np.ndarray:
    """
    지형 위치 지수(Topographic Position Index, TPI) 맵 계산.

    TPI = 현재 고도 - 주변 창 내 평균 고도
    - TPI > 0: 능선 / 돌출 지형
    - TPI < 0: 계곡 / 함몰 지형

    Args:
        heightmap: 고도 맵 배열 shape (H, W) [m]
        window: 분석 창 크기

    Returns:
        TPI 맵 shape (H, W), -1~1 정규화
    """
    hm = heightmap.astype(np.float32)
    mean_h = uniform_filter(hm, size=window)
    tpi = (hm - mean_h).astype(np.float32)

    # -1~1 정규화
    tpi_max = np.abs(tpi).max()
    if tpi_max > 1e-6:
        tpi = tpi / tpi_max

    return tpi


def compute_risk_map(
    slope: np.ndarray,
    roughness: np.ndarray,
    relief: np.ndarray,
    tpi: np.ndarray,
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    정적 위험도 맵 계산 (Static Risk Map).

    각 레이어의 가중 합산으로 구성. 모든 입력은 0~1 범위.

    R_static = w_slope * slope_norm
             + w_roughness * roughness
             + w_relief * relief
             + w_ridge * ridge_proxy
             + w_valley * valley_proxy

    Args:
        slope: 경사도 맵 (0~1 정규화 또는 [rad])
        roughness: 거칠기 맵 (0~1)
        relief: 국지 기복 맵 (0~1)
        tpi: TPI 맵 (-1~1)
        weights: 각 요소별 가중치 dict. 기본값 사용 시 None.

    Returns:
        risk 맵 [0~1] shape 동일
    """
    default_weights: Dict[str, float] = {
        "slope": 0.35,
        "roughness": 0.25,
        "relief": 0.20,
        "ridge": 0.10,
        "valley": 0.10,
    }
    w = {**default_weights, **(weights or {})}

    # slope 정규화 (rad → 0~1, 최대 90도)
    slope_max = np.pi / 2.0
    slope_norm = np.clip(slope / slope_max, 0.0, 1.0) if slope.max() > 1.0 else np.clip(slope, 0.0, 1.0)

    # 능선 프록시: TPI가 높고 양수 (능선 노출 지형)
    ridge_proxy = np.clip(tpi, 0.0, 1.0)

    # 계곡 프록시: TPI가 낮고 음수 (계곡, 함몰)
    valley_proxy = np.clip(-tpi, 0.0, 1.0)

    risk = (
        w["slope"] * slope_norm
        + w["roughness"] * roughness
        + w["relief"] * relief
        + w["ridge"] * ridge_proxy
        + w["valley"] * valley_proxy
    )

    # 0~1 클램프
    risk = np.clip(risk, 0.0, 1.0).astype(np.float32)
    return risk


def compute_landing_suitability(
    slope: np.ndarray,
    roughness: np.ndarray,
    relief: np.ndarray,
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    비상착륙 적합도 맵 계산.

    경사가 낮고, 거칠기가 낮고, 기복이 적은 곳이 착륙에 적합.

    suitability = 1 - (w_slope * slope_norm + w_rough * roughness + w_relief * relief)

    Args:
        slope: 경사도 맵 [rad] 또는 0~1 정규화
        roughness: 거칠기 맵 (0~1)
        relief: 국지 기복 맵 (0~1)
        weights: 가중치 dict

    Returns:
        착륙 적합도 맵 [0~1]
    """
    default_weights: Dict[str, float] = {
        "slope": 0.5,
        "roughness": 0.3,
        "relief": 0.2,
    }
    w = {**default_weights, **(weights or {})}

    # slope 정규화
    slope_max = np.pi / 2.0
    slope_norm = np.clip(slope / slope_max, 0.0, 1.0) if slope.max() > 1.0 else np.clip(slope, 0.0, 1.0)

    # 위험 합산 (값이 높을수록 착륙 부적합)
    risk = (
        w["slope"] * slope_norm
        + w["roughness"] * roughness
        + w["relief"] * relief
    )
    risk = np.clip(risk, 0.0, 1.0)

    # 적합도 = 1 - 위험
    suitability = (1.0 - risk).astype(np.float32)
    return suitability


def compute_all_features(
    heightmap: np.ndarray,
    res: float = 30.0,
    slope_window: int = 1,  # slope는 차분으로 계산하므로 미사용
    roughness_window: int = 5,
    relief_window: int = 7,
    tpi_window: int = 7,
    risk_weights: Optional[Dict[str, float]] = None,
    landing_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """
    모든 지형 특징을 한 번에 계산하여 dict로 반환.

    Args:
        heightmap: 고도 맵 [m]
        res: 해상도 [m]
        roughness_window: 거칠기 계산 창 크기
        relief_window: 기복 계산 창 크기
        tpi_window: TPI 계산 창 크기
        risk_weights: 위험도 가중치
        landing_weights: 착륙 적합도 가중치

    Returns:
        {
            "elevation": heightmap,
            "slope": slope_map [rad],
            "roughness": roughness_map [0~1],
            "relief": relief_map [0~1],
            "tpi": tpi_map [-1~1],
            "risk": risk_map [0~1],
            "landing": landing_suitability_map [0~1],
        }
    """
    slope = compute_slope(heightmap, res)
    roughness = compute_roughness(heightmap, roughness_window)
    relief = compute_local_relief(heightmap, relief_window)
    tpi = compute_tpi(heightmap, tpi_window)
    risk = compute_risk_map(slope, roughness, relief, tpi, risk_weights)
    landing = compute_landing_suitability(slope, roughness, relief, landing_weights)

    return {
        "elevation": heightmap.astype(np.float32),
        "slope": slope,
        "roughness": roughness,
        "relief": relief,
        "tpi": tpi,
        "risk": risk,
        "landing": landing,
    }
