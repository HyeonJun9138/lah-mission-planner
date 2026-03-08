"""
Frenet 좌표계 변환 모듈
- 항공기 위치를 ref path에 투영하여 (s, d, heading_error) 계산
- z_ref_error 계산
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

# RefPath import는 타입 힌트용 (순환 참조 방지)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.route.ref_path import RefPath


def project_to_path(
    x: float,
    y: float,
    ref_path: "RefPath",
) -> Tuple[float, float, float]:
    """
    월드 좌표 (x, y)를 ref path에 투영하여 Frenet 좌표 반환.

    투영 방법:
    1) ref path의 각 선분에 대해 점과의 수직 거리 계산
    2) 가장 가까운 선분 위의 투영점 찾기
    3) s (along-track), d (cross-track), heading_error 계산

    Args:
        x: 항공기 동서 좌표 [m]
        y: 항공기 남북 좌표 [m]
        ref_path: 참조 경로 객체

    Returns:
        (s, d, heading_error)
        - s: 경로 따른 진행 거리 [m]
        - d: 경로에서 좌우 이탈 거리 [m] (좌=양수, 우=음수)
        - heading_error: 현재 heading에서 ref tangent까지의 각도 차 [rad]
                         (호출자가 heading을 전달하지 않으므로 0 반환)
    """
    pts_xy = ref_path.points[:, :2]  # (N, 2)
    pos = np.array([x, y], dtype=np.float32)

    N = len(pts_xy)

    best_s = 0.0
    best_d = np.inf
    best_seg_idx = 0
    best_proj_frac = 0.0

    for i in range(N - 1):
        p0 = pts_xy[i]
        p1 = pts_xy[i + 1]
        seg = p1 - p0
        seg_len = np.linalg.norm(seg)

        if seg_len < 1e-8:
            continue

        seg_dir = seg / seg_len
        dp = pos - p0
        t = float(np.dot(dp, seg_dir))
        t_clamped = float(np.clip(t, 0.0, seg_len))

        proj = p0 + t_clamped * seg_dir
        dist = float(np.linalg.norm(pos - proj))

        if dist < best_d:
            best_d = dist
            best_seg_idx = i
            best_proj_frac = t_clamped / seg_len
            best_s = (
                float(ref_path.cumulative_s[i])
                + t_clamped
            )

    # d의 부호 결정 (법선 방향 = 접선의 좌측이 양수)
    seg_tangent = ref_path.tangents[best_seg_idx]
    seg_normal = ref_path.normals[best_seg_idx]

    proj_pt = ref_path.get_point_at_s(best_s)[:2]
    offset = pos - proj_pt
    d_signed = float(np.dot(offset, seg_normal))

    # s를 경로 총 길이로 클램프
    s = float(np.clip(best_s, 0.0, ref_path.total_length))

    # heading_error는 별도 함수로 계산 (여기선 0 반환)
    heading_error = 0.0

    return s, d_signed, heading_error


def compute_heading_error(
    heading: float,
    s: float,
    ref_path: "RefPath",
) -> float:
    """
    현재 기수각(heading)과 ref path 접선 방향의 차이 계산.

    Args:
        heading: 현재 기수각 [rad] (NED 기준, 동쪽=0, 반시계=양수)
        s: 현재 경로 진행 거리 [m]
        ref_path: 참조 경로

    Returns:
        heading_error [rad] (-pi ~ pi)
    """
    tangent = ref_path.get_tangent_at_s(s)
    ref_heading = float(np.arctan2(tangent[1], tangent[0]))

    # 각도 차이 [-pi, pi]로 정규화
    err = heading - ref_heading
    err = (err + np.pi) % (2 * np.pi) - np.pi
    return float(err)


def get_z_ref_error(
    z: float,
    s: float,
    ref_path: "RefPath",
) -> float:
    """
    현재 고도와 참조 경로의 권장 고도 차이 계산.

    z_ref_error = z - z_ref
    - 양수: 참조 고도보다 위
    - 음수: 참조 고도보다 아래

    Args:
        z: 현재 고도 [m]
        s: 현재 경로 진행 거리 [m]
        ref_path: 참조 경로

    Returns:
        z_ref_error [m]
    """
    z_ref = ref_path.get_reference_altitude(s)
    return float(z - z_ref)


def frenet_to_world(
    s: float,
    d: float,
    ref_path: "RefPath",
) -> Tuple[float, float]:
    """
    Frenet 좌표 (s, d)를 월드 좌표 (x, y)로 변환.

    Args:
        s: 경로 진행 거리 [m]
        d: 횡방향 이탈 거리 [m] (법선 방향)
        ref_path: 참조 경로

    Returns:
        (x, y) 월드 좌표 [m]
    """
    pt = ref_path.get_point_at_s(s)
    s_idx = np.searchsorted(ref_path.cumulative_s, s)
    s_idx = int(np.clip(s_idx, 0, ref_path.n_points - 1))
    normal = ref_path.normals[s_idx]

    x = float(pt[0] + d * normal[0])
    y = float(pt[1] + d * normal[1])
    return x, y
