"""
항공기 상태(State) 정의 모듈
- AircraftState dataclass
- to_dict / from_dict 직렬화
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class AircraftState:
    """
    회전익기 상태 dataclass.

    월드 프레임(x, y, z)과 운동 상태(v, psi, vz),
    hold 모드 정보를 포함한다.

    Attributes:
        x (float): 동서 위치 [m]
        y (float): 남북 위치 [m]
        z (float): 고도 AMSL [m]
        v (float): 수평 속도 [m/s]
        psi (float): 기수각(heading) [rad] (동쪽=0, 반시계=양수)
        vz (float): 수직 속도 [m/s] (상승=양수)
        hold_mode (bool): hold(호버링/저속 체공) 모드 활성화 여부
        hold_timer (float): hold 모드 지속 시간 [s]
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 300.0
    v: float = 40.0
    psi: float = 0.0
    vz: float = 0.0
    hold_mode: bool = False
    hold_timer: float = 0.0

    # ------------------------------------------------------------------
    # 복사 생성
    # ------------------------------------------------------------------

    def copy(self) -> "AircraftState":
        """현재 상태의 복사본 반환."""
        return AircraftState(
            x=self.x,
            y=self.y,
            z=self.z,
            v=self.v,
            psi=self.psi,
            vz=self.vz,
            hold_mode=self.hold_mode,
            hold_timer=self.hold_timer,
        )

    # ------------------------------------------------------------------
    # 직렬화
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        상태를 dict로 변환.

        Returns:
            {"x": ..., "y": ..., "z": ..., "v": ..., "psi": ...,
             "vz": ..., "hold_mode": ..., "hold_timer": ...}
        """
        return {
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "v": float(self.v),
            "psi": float(self.psi),
            "vz": float(self.vz),
            "hold_mode": bool(self.hold_mode),
            "hold_timer": float(self.hold_timer),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AircraftState":
        """
        dict에서 상태 복원.

        Args:
            d: to_dict()로 생성된 dict

        Returns:
            AircraftState 인스턴스
        """
        return cls(
            x=float(d.get("x", 0.0)),
            y=float(d.get("y", 0.0)),
            z=float(d.get("z", 300.0)),
            v=float(d.get("v", 40.0)),
            psi=float(d.get("psi", 0.0)),
            vz=float(d.get("vz", 0.0)),
            hold_mode=bool(d.get("hold_mode", False)),
            hold_timer=float(d.get("hold_timer", 0.0)),
        )

    # ------------------------------------------------------------------
    # 편의 메서드
    # ------------------------------------------------------------------

    def position_2d(self) -> np.ndarray:
        """2D 위치 벡터 (x, y) 반환."""
        return np.array([self.x, self.y], dtype=np.float32)

    def position_3d(self) -> np.ndarray:
        """3D 위치 벡터 (x, y, z) 반환."""
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def velocity_vector(self) -> np.ndarray:
        """수평 속도 벡터 (vx, vy) 반환."""
        vx = self.v * np.cos(self.psi)
        vy = self.v * np.sin(self.psi)
        return np.array([vx, vy], dtype=np.float32)

    def is_valid(self) -> bool:
        """상태 수치 유효성 검사 (NaN/Inf 없음)."""
        vals = [self.x, self.y, self.z, self.v, self.psi, self.vz, self.hold_timer]
        return all(np.isfinite(v) for v in vals)

    def __repr__(self) -> str:
        return (
            f"AircraftState("
            f"pos=({self.x:.1f},{self.y:.1f},{self.z:.1f}), "
            f"v={self.v:.1f}m/s, psi={np.degrees(self.psi):.1f}deg, "
            f"vz={self.vz:.1f}m/s, hold={self.hold_mode})"
        )
