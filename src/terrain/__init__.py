"""
terrain 패키지
- 합성 지형(synthetic terrain) 생성 및 지형 특징 분석 모듈
- DEM 없이도 동작하는 synthetic terrain 지원
- GeoTIFF DEM 파일 로더 (DEMTerrain) 지원
"""

from src.terrain.synthetic import SyntheticTerrain
from src.terrain.dem_loader import DEMTerrain, list_available_dems
from src.terrain.features import (
    compute_slope,
    compute_roughness,
    compute_local_relief,
    compute_tpi,
    compute_risk_map,
    compute_landing_suitability,
)

__all__ = [
    "SyntheticTerrain",
    "DEMTerrain",
    "list_available_dems",
    "compute_slope",
    "compute_roughness",
    "compute_local_relief",
    "compute_tpi",
    "compute_risk_map",
    "compute_landing_suitability",
]
