"""
LAH 회전익기 안전 경로계획 강화학습 환경 (Gymnasium)
- 설계서 11절(Observation), 10절(Action), 14절(학습 환경 설계) 구현
- DEM 없이 synthetic terrain으로 즉시 동작
- Dict observation: state_vec, lookahead_ref, local_patch
- Box(4,) action: yaw_rate, vz, accel, hold
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, List, Tuple

from src.terrain.synthetic import SyntheticTerrain
from src.terrain.features import compute_slope, compute_roughness, compute_local_relief, compute_tpi
from src.route.ref_path import RefPath
from src.route.frenet import project_to_path, compute_heading_error, get_z_ref_error
from src.route.corridor import Corridor
from src.sim.state import AircraftState
from src.sim.dynamics import kinematic_step
from src.sim.reward import compute_reward, compute_reward_breakdown
from src.sim.termination import check_termination, get_terminal_reward
from src.sim.metrics import EpisodeMetrics


# ------------------------------------------------------------------
# 기본 환경 파라미터
# ------------------------------------------------------------------

DEFAULT_ENV_CONFIG: Dict[str, Any] = {
    # 지형 파라미터
    "terrain_size_x": 500,
    "terrain_size_y": 500,
    "terrain_resolution": 30.0,
    "terrain_base_elev": 200.0,
    "terrain_max_elev": 1200.0,
    "terrain_seed": 42,

    # 경로 파라미터
    "route_resample_m": 50.0,
    "soft_corridor_m": 400.0,
    "hard_corridor_m": 1000.0,

    # 운동학 파라미터
    "dt": 1.0,
    "v_min": 0.0,
    "v_max": 70.0,
    "vz_min": -6.0,
    "vz_max": 6.0,
    "max_turn_deg_s": 12.0,
    "v_init": 40.0,
    "agl_safe_min": 120.0,
    "agl_pref": 180.0,

    # 에피소드 파라미터
    "max_steps": 1000,
    "goal_radius": 200.0,

    # Observation 파라미터
    "patch_size": 32,        # 로컬 패치 픽셀 (초기: 32x32)
    "patch_channels": 3,     # 로컬 패치 채널 (elevation, slope, risk)
    "lookahead_points": 16,  # lookahead 포인트 수
    "lookahead_features": 4, # 포인트당 feature 수
    "route_profile_steps": 20,
    "state_vec_dim": 13,

    # 리셋 랜덤화
    "start_pos_noise_m": 50.0,    # 시작 위치 노이즈 [m]
    "start_heading_noise_deg": 15.0,  # 시작 기수각 노이즈 [deg]
    "start_speed_noise_ms": 5.0,  # 시작 속도 노이즈 [m/s]
    "start_alt_noise_m": 30.0,    # 시작 고도 노이즈 [m]
}


class LAHLocalPathEnv(gym.Env):
    """
    LAH 회전익기 안전 경로계획 강화학습 환경.

    운용자가 입력한 ref path를 기준으로, RL이 지형과 위험을 고려하며
    로컬 3D 경로를 계획하는 Gymnasium 환경.

    Observation Space (Dict):
        - state_vec (13,): 기본 상태 벡터
        - lookahead_ref (lookahead_points, lookahead_features): 앞쪽 참조 경로
        - local_patch (patch_channels, patch_size, patch_size): 로컬 지형 패치

    Action Space:
        Box(4,) ∈ [-1, 1]
        - [0] yaw_rate_cmd
        - [1] vz_cmd
        - [2] accel_cmd
        - [3] hold_cmd
    """

    metadata = {"render_modes": ["human", "ansi", "rgb_array"]}

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        ref_path: Optional[RefPath] = None,
        terrain: Optional[SyntheticTerrain] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """
        Args:
            config: 환경 파라미터 dict (DEFAULT_ENV_CONFIG에 오버라이드)
            ref_path: 참조 경로 (None이면 기본 직선 경로 생성)
            terrain: 지형 객체 (None이면 SyntheticTerrain 생성)
            render_mode: 렌더링 모드 ("human", "ansi")
        """
        super().__init__()

        # 설정 병합
        self.config: Dict[str, Any] = {**DEFAULT_ENV_CONFIG, **(config or {})}
        self.render_mode = render_mode

        # 지형 초기화
        if terrain is not None:
            self.terrain = terrain
        else:
            self.terrain = SyntheticTerrain(
                size_x=self.config["terrain_size_x"],
                size_y=self.config["terrain_size_y"],
                resolution=self.config["terrain_resolution"],
                base_elevation=self.config["terrain_base_elev"],
                max_elevation=self.config["terrain_max_elev"],
                seed=self.config["terrain_seed"],
            )

        # 참조 경로 초기화
        if ref_path is not None:
            self.ref_path = ref_path
        else:
            self.ref_path = self._make_default_ref_path()

        # Corridor 초기화
        self.corridor = Corridor(
            soft_width=self.config["soft_corridor_m"],
            hard_width=self.config["hard_corridor_m"],
        )

        # 지형 특징 맵 사전 계산
        self._precompute_terrain_features()

        # Observation/Action Space 정의
        self._define_spaces()

        # 에피소드 상태 초기화
        self.state: AircraftState = AircraftState()
        self.prev_state: AircraftState = AircraftState()
        self.prev_action: np.ndarray = np.zeros(4, dtype=np.float32)
        self.step_count: int = 0
        self.episode_metrics: EpisodeMetrics = EpisodeMetrics()
        self._episode_count: int = 0

    # ------------------------------------------------------------------
    # 공간 정의
    # ------------------------------------------------------------------

    def _define_spaces(self) -> None:
        """Observation/Action Space 정의."""
        cfg = self.config
        n_state = cfg["state_vec_dim"]
        n_la = cfg["lookahead_points"]
        n_laf = cfg["lookahead_features"]
        patch_c = cfg["patch_channels"]
        patch_sz = cfg["patch_size"]

        # Observation Space
        self.observation_space = spaces.Dict({
            "state_vec": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n_state,),
                dtype=np.float32,
            ),
            "lookahead_ref": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n_la, n_laf),
                dtype=np.float32,
            ),
            "local_patch": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(patch_c, patch_sz, patch_sz),
                dtype=np.float32,
            ),
        })

        # Action Space: Box(4,) ∈ [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gymnasium 기본 메서드
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        에피소드 리셋.

        Args:
            seed: 난수 시드
            options: 추가 옵션 (ref_path 교체 등)

        Returns:
            (observation, info)
        """
        super().reset(seed=seed)

        # ref_path 교체 옵션
        if options and "ref_path" in options:
            self.ref_path = options["ref_path"]
            # terrain도 교체 가능
        if options and "terrain" in options:
            self.terrain = options["terrain"]
            self._precompute_terrain_features()

        # 시작 상태 랜덤화
        self.state = self._random_initial_state()
        self.prev_state = self.state.copy()
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.step_count = 0

        # 에피소드 지표 초기화
        self.episode_metrics.reset()
        self.episode_metrics._ref_path_length = self.ref_path.total_length
        self._episode_count += 1

        obs = self._get_observation()
        info = {
            "episode": self._episode_count,
            "terrain": str(self.terrain),
            "ref_path": str(self.ref_path),
        }
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        환경 한 스텝 진행.

        Args:
            action: 4차원 액션 벡터 ∈ [-1, 1]

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        action = np.clip(
            np.asarray(action, dtype=np.float32), -1.0, 1.0
        )

        # 이전 상태 저장
        self.prev_state = self.state.copy()

        # 운동학 업데이트
        dyn_config = {
            "v_min": self.config["v_min"],
            "v_max": self.config["v_max"],
            "vz_min": self.config["vz_min"],
            "vz_max": self.config["vz_max"],
            "max_turn_rad": np.radians(self.config["max_turn_deg_s"]),
            "dt": self.config["dt"],
        }
        self.state = kinematic_step(self.state, action, self.config["dt"], dyn_config)
        self.step_count += 1

        # 종료 조건 확인
        terminated, truncated, term_info = check_termination(
            self.state,
            self.terrain,
            self.ref_path,
            self.corridor,
            self.step_count,
            self.config["max_steps"],
            {"goal_radius": self.config["goal_radius"]},
        )

        # 보상 계산
        reward_breakdown = compute_reward_breakdown(
            self.state,
            self.prev_state,
            action,
            self.prev_action,
            self.terrain,
            self.ref_path,
            self.corridor,
            self.config,
        )
        reward = reward_breakdown["total"]

        # terminal reward 추가
        if terminated or truncated:
            terminal_r = get_terminal_reward(term_info)
            reward += terminal_r
            reward_breakdown["terminal"] = terminal_r
            reward_breakdown["total"] = reward

        # 지표 기록
        terrain_h = self.terrain.get_elevation(self.state.x, self.state.y)
        agl = self.state.z - terrain_h
        s = reward_breakdown.get("_s", 0.0)
        d = reward_breakdown.get("_d", 0.0)

        self.episode_metrics.record_step(
            step=self.step_count,
            state_dict=self.state.to_dict(),
            action=action,
            reward=reward,
            reward_breakdown=reward_breakdown,
            agl=float(agl),
            s=float(s),
            d=float(d),
            local_risk=float(reward_breakdown.get("_local_risk", 0.0)),
            terrain_h=float(terrain_h),
        )

        if terminated or truncated:
            self.episode_metrics.set_result(
                success=term_info.get("success", False),
                truncated=truncated,
                reason=term_info.get("reason", "unknown"),
                ref_path_length=self.ref_path.total_length,
            )

        # 이전 액션 업데이트
        self.prev_action = action.copy()

        obs = self._get_observation()
        info = {
            **term_info,
            "reward_breakdown": reward_breakdown,
            "step": self.step_count,
            "agl": float(agl),
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> Optional[str]:
        """
        환경 상태 렌더링.

        Returns:
            render_mode="ansi": 상태 문자열
            render_mode="human": 콘솔 출력 (None 반환)
        """
        s, d, _ = project_to_path(self.state.x, self.state.y, self.ref_path)
        terrain_h = self.terrain.get_elevation(self.state.x, self.state.y)
        agl = self.state.z - terrain_h
        goal_dist = float(np.linalg.norm(
            np.array([self.state.x, self.state.y]) - self.ref_path.end_point[:2]
        ))

        status = (
            f"[스텝 {self.step_count:4d}] "
            f"pos=({self.state.x:.0f},{self.state.y:.0f},{self.state.z:.0f}) "
            f"v={self.state.v:.1f}m/s "
            f"AGL={agl:.0f}m "
            f"s={s:.0f}m "
            f"d={d:.0f}m "
            f"goal={goal_dist:.0f}m "
            f"hold={'ON' if self.state.hold_mode else 'off'}"
        )

        if self.render_mode == "human":
            print(status)
            return None
        elif self.render_mode == "ansi":
            return status
        return None

    def close(self) -> None:
        """환경 리소스 해제."""
        pass

    # ------------------------------------------------------------------
    # Observation 생성
    # ------------------------------------------------------------------

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        현재 상태에서 observation dict 생성.

        Returns:
            {
                "state_vec": (13,),
                "lookahead_ref": (N, 4),
                "local_patch": (C, H, W),
            }
        """
        state_vec = self._build_state_vec()
        lookahead = self._build_lookahead_ref()
        patch = self._build_local_patch()

        return {
            "state_vec": state_vec,
            "lookahead_ref": lookahead,
            "local_patch": patch,
        }

    def _build_state_vec(self) -> np.ndarray:
        """
        상태 벡터 생성 (13차원).

        설계서 11.1절 state_vec:
        [v_norm, vz_norm, psi_sin, psi_cos, agl_norm, s_norm, d_norm,
         heading_error_sin, heading_error_cos, z_ref_err_norm,
         dist_goal_norm, local_risk, hold_mode, hold_timer_norm]

        Returns:
            (13,) float32 배열
        """
        s, d, _ = project_to_path(self.state.x, self.state.y, self.ref_path)
        heading_err = compute_heading_error(self.state.psi, s, self.ref_path)
        z_ref_err = get_z_ref_error(self.state.z, s, self.ref_path)
        terrain_h = self.terrain.get_elevation(self.state.x, self.state.y)
        agl = self.state.z - terrain_h
        dist_goal = float(np.linalg.norm(
            np.array([self.state.x, self.state.y]) - self.ref_path.end_point[:2]
        ))
        slope = self.terrain.get_slope(self.state.x, self.state.y)
        rough = self.terrain.get_roughness(self.state.x, self.state.y)
        local_risk = float(np.clip(0.5 * (slope / (np.pi / 4)) + 0.5 * rough, 0.0, 1.0))

        v_max = self.config["v_max"]
        vz_max = self.config["vz_max"]
        agl_pref = self.config["agl_pref"]
        hard_w = self.corridor.hard_width

        state_vec = np.array([
            self.state.v / max(v_max, 1.0),                              # 0: 속도 정규화
            self.state.vz / max(vz_max, 1.0),                            # 1: 수직속도 정규화
            np.sin(self.state.psi),                                       # 2: 기수각 sin
            np.cos(self.state.psi),                                       # 3: 기수각 cos
            float(np.clip(agl / max(agl_pref * 2, 1.0), -0.5, 2.0)),    # 4: AGL 정규화
            s / max(self.ref_path.total_length, 1.0),                    # 5: 진행 거리 비율
            float(np.clip(d / max(hard_w, 1.0), -1.5, 1.5)),            # 6: cross-track 정규화
            np.sin(heading_err),                                          # 7: heading 오차 sin
            np.cos(heading_err),                                          # 8: heading 오차 cos
            float(np.clip(z_ref_err / max(agl_pref, 1.0), -2.0, 2.0)), # 9: 고도 오차 정규화
            float(np.clip(dist_goal / max(self.ref_path.total_length, 1.0), 0.0, 2.0)),  # 10: 목표 거리
            local_risk,                                                   # 11: 현재 위험도
            float(self.state.hold_mode),                                  # 12: hold 상태
        ], dtype=np.float32)

        return state_vec

    def _build_lookahead_ref(self) -> np.ndarray:
        """
        앞쪽 N개 참조 포인트를 항공기 기준 로컬 좌표로 변환.

        설계서 11.1절 B. guide_vec:
        각 포인트: (dx_local, dy_local, z_ref_delta, mean_risk)

        Returns:
            (lookahead_points, 4) float32 배열
        """
        n_la = self.config["lookahead_points"]
        spacing = self.config["route_resample_m"] if "route_resample_m" in self.config else 50.0
        spacing = self.ref_path.total_length / max(self.ref_path.n_points - 1, 1)
        spacing = max(spacing, 30.0)

        s_curr, _, _ = project_to_path(self.state.x, self.state.y, self.ref_path)
        la_pts = self.ref_path.get_lookahead_points(s_curr, n_la, spacing)

        # 로컬 좌표 변환 (항공기 기준)
        cos_psi = np.cos(self.state.psi)
        sin_psi = np.sin(self.state.psi)

        lookahead = np.zeros((n_la, 4), dtype=np.float32)
        for i, pt in enumerate(la_pts):
            dx_world = pt[0] - self.state.x
            dy_world = pt[1] - self.state.y
            # 항공기 기준 로컬 변환
            dx_local = cos_psi * dx_world + sin_psi * dy_world
            dy_local = -sin_psi * dx_world + cos_psi * dy_world
            dz = pt[2] - self.state.z

            # 해당 포인트 주변 위험도
            slope = self.terrain.get_slope(float(pt[0]), float(pt[1]))
            rough = self.terrain.get_roughness(float(pt[0]), float(pt[1]))
            risk = float(np.clip(0.5 * (slope / (np.pi / 4)) + 0.5 * rough, 0.0, 1.0))

            # 정규화
            scale = max(self.ref_path.total_length, 1.0)
            lookahead[i, 0] = float(dx_local / scale)
            lookahead[i, 1] = float(dy_local / scale)
            lookahead[i, 2] = float(np.clip(dz / max(self.config["agl_pref"], 1.0), -2.0, 2.0))
            lookahead[i, 3] = risk

        return lookahead

    def _build_local_patch(self) -> np.ndarray:
        """
        항공기 주변 로컬 지형 패치 생성.

        채널:
        0: 상대 고도 (relative elevation)
        1: 경사도 (slope, 정규화)
        2: 위험도 (static risk)

        Returns:
            (patch_channels, patch_size, patch_size) float32 배열
        """
        patch_sz = self.config["patch_size"]
        patch_c = self.config["patch_channels"]

        # 고도 패치 추출
        elev_patch = self.terrain.get_local_patch(
            self.state.x, self.state.y,
            patch_size=patch_sz,
            patch_resolution=self.terrain.resolution,
        )

        # 채널 구성
        channels = []

        # 채널 0: 상대 고도 (항공기 고도 기준)
        rel_elev = (elev_patch - self.state.z) / max(self.config["agl_pref"] * 2, 1.0)
        channels.append(rel_elev.astype(np.float32))

        if patch_c >= 2:
            # 채널 1: 경사도 (패치 내 slope 근사)
            dzdx = np.zeros_like(elev_patch)
            dzdy = np.zeros_like(elev_patch)
            dzdx[:, 1:-1] = (elev_patch[:, 2:] - elev_patch[:, :-2]) / (2 * self.terrain.resolution)
            dzdy[1:-1, :] = (elev_patch[2:, :] - elev_patch[:-2, :]) / (2 * self.terrain.resolution)
            slope_patch = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2)) / (np.pi / 2.0)
            channels.append(slope_patch.astype(np.float32))

        if patch_c >= 3:
            # 채널 2: 간이 위험도
            from scipy.ndimage import uniform_filter
            rough_patch = np.abs(elev_patch - uniform_filter(elev_patch, size=5))
            rough_norm = rough_patch / max(rough_patch.max(), 1.0)
            slope_for_risk = channels[1] if len(channels) > 1 else np.zeros_like(elev_patch)
            risk_patch = np.clip(0.5 * slope_for_risk + 0.5 * rough_norm, 0.0, 1.0)
            channels.append(risk_patch.astype(np.float32))

        # 부족한 채널은 0으로 패딩
        while len(channels) < patch_c:
            channels.append(np.zeros((patch_sz, patch_sz), dtype=np.float32))

        # (C, H, W) 형태로 스택
        patch = np.stack(channels[:patch_c], axis=0)
        return patch.astype(np.float32)

    # ------------------------------------------------------------------
    # 초기화 헬퍼
    # ------------------------------------------------------------------

    def _make_default_ref_path(self) -> RefPath:
        """
        기본 직선 참조 경로 생성 (테스트/Stage0 용도).

        지형 중앙을 가로지르는 직선 경로.
        """
        x_min, x_max, y_min, y_max = self.terrain.world_extent
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        # 지형 평균 고도 + 권장 AGL
        mean_elev = float(self.terrain.heightmap.mean())
        z_ref = mean_elev + self.config.get("agl_pref", 180.0)

        start = (x_min + 500.0, cy, z_ref)
        end = (x_max - 500.0, cy, z_ref)

        return RefPath.from_waypoints(
            [start, end],
            step_m=self.config["route_resample_m"],
        )

    def _random_initial_state(self) -> AircraftState:
        """
        에피소드 시작 상태 랜덤화.

        ref path 시작점 근처에서 약간의 노이즈를 추가.
        """
        rng = self.np_random

        start = self.ref_path.start_point  # (x, y, z)

        # 위치 노이즈
        pos_noise = self.config["start_pos_noise_m"]
        dx = float(rng.uniform(-pos_noise, pos_noise))
        dy = float(rng.uniform(-pos_noise, pos_noise))

        # 고도 노이즈
        alt_noise = self.config["start_alt_noise_m"]
        dz = float(rng.uniform(-alt_noise, alt_noise))

        # 지형 고도 확인 (지하로 시작하지 않도록)
        x0, y0 = start[0] + dx, start[1] + dy
        terrain_h = self.terrain.get_elevation(x0, y0)
        z0 = max(float(start[2]) + dz, terrain_h + self.config["agl_safe_min"])

        # ref path 첫 접선 방향 + 노이즈
        tangent = self.ref_path.get_tangent_at_s(0.0)
        base_heading = float(np.arctan2(tangent[1], tangent[0]))
        heading_noise = np.radians(self.config["start_heading_noise_deg"])
        psi0 = base_heading + float(rng.uniform(-heading_noise, heading_noise))

        # 속도 노이즈
        v0 = float(np.clip(
            self.config["v_init"] + rng.uniform(
                -self.config["start_speed_noise_ms"],
                self.config["start_speed_noise_ms"]
            ),
            self.config["v_min"], self.config["v_max"]
        ))

        return AircraftState(
            x=float(x0),
            y=float(y0),
            z=float(z0),
            v=v0,
            psi=float(psi0),
            vz=0.0,
            hold_mode=False,
            hold_timer=0.0,
        )

    def _precompute_terrain_features(self) -> None:
        """지형 특징 맵 사전 계산 (학습 속도 향상)."""
        # slope와 roughness는 SyntheticTerrain의 property로 지연 계산됨
        # 여기서 명시적으로 트리거
        _ = self.terrain.slope_map
        _ = self.terrain.roughness_map

    # ------------------------------------------------------------------
    # 속성
    # ------------------------------------------------------------------

    def get_episode_metrics(self) -> Dict[str, Any]:
        """현재 에피소드 지표 반환."""
        return self.episode_metrics.compute()

    @property
    def unwrapped(self) -> "LAHLocalPathEnv":
        return self

    def set_ref_path(self, ref_path: RefPath) -> None:
        """
        참조 경로 교체.

        Args:
            ref_path: 새 RefPath 인스턴스
        """
        self.ref_path = ref_path

    def set_terrain(self, terrain: SyntheticTerrain) -> None:
        """
        지형 교체.

        Args:
            terrain: 새 SyntheticTerrain 인스턴스
        """
        self.terrain = terrain
        self._precompute_terrain_features()

    def __repr__(self) -> str:
        return (
            f"LAHLocalPathEnv("
            f"steps={self.step_count}/{self.config['max_steps']}, "
            f"terrain={self.terrain}, "
            f"ref_path={self.ref_path})"
        )
