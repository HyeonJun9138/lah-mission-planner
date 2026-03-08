# LAH 임무 경로계획 AI 학습 프레임워크

> 회전익기(LAH) 안전 경로계획을 위한 강화학습 시스템 v3.0

![LAH Mission Planner](assets/cover.png)

## 개요

운용자가 사전 입력한 **Ref Path(기준 경로)**를 기본으로 따라가되, DEM 기반 지형 분석과 경로 주변 위험 분석을 통해 **필요할 때만 국소적으로 우회/Holding**하고 다시 복귀하는 **회전익기용 3D 강화학습 로컬 플래너** 프레임워크입니다.

### 핵심 컨셉
- **전역경로(Ref Path)**: 운용자가 사전 입력한 거의 직선 위주의 Polyline
- **RL의 역할**: Ref Path를 기준으로 Corridor 안에서 더 안전한 3D 경로를 찾는 로컬 플래너
- **Route Follower + Local Deviation Planner + Safe Recovery Planner**

## 주요 기능

| 기능 | 설명 |
|------|------|
| 🌍 **지형 분석** | Synthetic/DEM 기반 3D 지형 분석, Slope/Roughness/TPI/Risk Map 산출 |
| 🛤️ **경로 계획** | Ref Path + Frenet 좌표계 + Soft/Hard Corridor + Band Risk Profile |
| 🧠 **AI 학습** | PPO/SAC 기반 강화학습, Optuna 자동 튜닝, 이어학습 지원 |
| 🚁 **비행 시뮬레이션** | 실시간 에피소드 재생, 2D 탑다운 뷰, 계기판, 시계열 차트 |
| 🌐 **3D 시각화** | Three.js 기반 3차원 지형/경로/LAH 헬기 모델 시각화 |
| ✅ **환경 검증** | Rule 기반 Obs/Action/Reward/Dynamics/Termination 합치성 평가 |
| 📊 **모델 비교** | 다중 모델 성능 비교, 레이더 차트, 경로 오버레이 |

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Frontend (HTML/JS)                     │
│  표지 │ 환경설정 │ 학습 │ 시뮬레이션 │ 3D뷰 │ 검증 │ 비교  │
├─────────────────────────────────────────────────────────────┤
│                   FastAPI Backend (Python)                    │
├────────┬────────┬────────┬────────┬────────┬───────────────┤
│Terrain │ Route  │  Sim   │  Envs  │ Train  │   Validate    │
│Analysis│Planning│ Engine │Gym Env │PPO/SAC │  Rule Check   │
└────────┴────────┴────────┴────────┴────────┴───────────────┘
```

## 프로젝트 구조

```
lah-mission-planner/
├── README.md
├── requirements.txt
├── configs/
│   └── base.yaml              # 기본 환경/학습 설정
├── assets/
│   └── cover.png              # 표지 이미지
├── src/
│   ├── terrain/               # 지형 분석 모듈
│   │   ├── synthetic.py       # 합성 지형 생성 (Perlin-like noise)
│   │   └── features.py        # Slope, Roughness, TPI, Risk Map
│   ├── route/                 # 경로 관리 모듈
│   │   ├── ref_path.py        # Ref Path 로딩/리샘플링
│   │   ├── frenet.py          # Frenet 좌표 변환 (s, d, heading_error)
│   │   └── corridor.py        # Soft/Hard Corridor, Band Risk Profile
│   ├── sim/                   # 시뮬레이션 코어
│   │   ├── state.py           # AircraftState 데이터 구조
│   │   ├── dynamics.py        # 운동학 모델 (6DOF 단순화)
│   │   ├── reward.py          # 보상 함수 (11개 항목)
│   │   ├── termination.py     # 종료 조건 판정
│   │   └── metrics.py         # 에피소드 성능 지표
│   ├── envs/
│   │   └── lah_env.py         # Gymnasium 환경 (Dict obs, Box(4) action)
│   ├── train/
│   │   ├── trainer.py         # PPO/SAC 학습 + 이어학습
│   │   └── tuner.py           # Optuna 하이퍼파라미터 튜닝
│   ├── heuristics/
│   │   └── baseline.py        # BaselineTracker, CorridorShift, HoldPolicy
│   ├── validate/
│   │   └── rules.py           # 환경 검증 Rule (6개 카테고리)
│   └── webapp/
│       ├── api.py             # FastAPI 엔드포인트 (18개 API)
│       ├── runner.py          # 서버 실행 스크립트
│       └── static/            # 웹 프론트엔드
│           ├── index.html
│           ├── css/style.css
│           └── js/
│               ├── app.js             # 앱 코어, 탭 관리
│               ├── cover.js           # 표지 히어로
│               ├── environment.js     # 환경 설정 + 2D 미리보기
│               ├── training.js        # 학습 제어 + 실시간 차트
│               ├── simulation.js      # 에피소드 재생 + 계기판
│               ├── visualization3d.js # Three.js 3D 시각화
│               ├── validation.js      # 환경 검증 UI
│               └── utils.js           # 공통 유틸리티
└── DESIGN_SPEC.txt            # 상세 설계서 v3
```

## 빠른 시작

### 1. 환경 설치

```bash
git clone https://github.com/HyeonJun9138/lah-mission-planner.git
cd lah-mission-planner
pip install -r requirements.txt
```

### 2. 서버 실행

```bash
python -m src.webapp.runner
```

서버가 `http://localhost:8000`에서 시작됩니다.

### 3. 웹 UI 접속

브라우저에서 `http://localhost:8000` 접속

### 4. CLI로 직접 학습

```python
from src.envs.lah_env import LAHLocalPathEnv
from src.train.trainer import LAHTrainer

# 환경 생성
env = LAHLocalPathEnv()

# PPO 학습
trainer = LAHTrainer(algo='PPO')
model = trainer.train(total_timesteps=100000)

# 평가
metrics = trainer.evaluate(model, n_episodes=10)
print(f"성공률: {metrics['success_rate']:.1%}")

# 모델 저장
trainer.save_model("outputs/models/ppo_v1")

# 이어학습
trainer.resume_training("outputs/models/ppo_v1", additional_timesteps=50000)
```

### 5. Optuna 자동 튜닝

```python
from src.train.tuner import HyperparamTuner

tuner = HyperparamTuner(algo='PPO')
best_params = tuner.tune(n_trials=20)
print(f"최적 파라미터: {best_params}")
```

### 6. 환경 검증

```python
from src.validate.rules import EnvironmentValidator
from src.envs.lah_env import LAHLocalPathEnv

env = LAHLocalPathEnv()
validator = EnvironmentValidator()
report = validator.run_full_validation(env)
print(f"전체 통과율: {report['overall_pass_rate']:.1%}")
```

## Observation Space

| 키 | Shape | 설명 |
|-----|-------|------|
| `state_vec` | (13,) | v, vz, psi, agl, s_progress, d_cross_track, heading_error, z_ref_error, dist_to_goal, risk, clearance, hold_mode, hold_timer |
| `lookahead_ref` | (16, 4) | Ref Path 전방 16개 포인트 (dx, dy, z_delta, risk) |
| `local_patch` | (3, 32, 32) | 항공기 주변 지형 패치 (elevation, slope, risk) |

## Action Space

| 차원 | 범위 | 설명 |
|------|------|------|
| a0 | [-1, 1] | Yaw rate → [-12, +12] deg/s |
| a1 | [-1, 1] | 수직 속도 → [-6, +6] m/s |
| a2 | [-1, 1] | 가속도 → speed delta |
| a3 | [-1, 1] | Hold 명령 (>0.5: hold on, <-0.5: hold off) |

## Reward 구성

```
r_t = + w_prog    · Δs_progress
      + w_goal    · Δgoal_distance_reduction
      - w_d       · cross_track_error
      - w_zref    · altitude_ref_error
      - w_risk    · current_risk
      - w_clear   · clearance_deficit
      - w_smooth  · control_jerk
      - w_hold    · hold_time_penalty
      + terminal_bonus_or_penalty
```

## 기술 스택

| 구분 | 기술 |
|------|------|
| **RL Framework** | Gymnasium + Stable-Baselines3 |
| **알고리즘** | PPO, SAC (RecurrentPPO, TQC 확장 가능) |
| **튜닝** | Optuna |
| **백엔드** | FastAPI + Uvicorn |
| **프론트엔드** | HTML5 + CSS3 + Vanilla JS |
| **3D 시각화** | Three.js |
| **차트** | Plotly.js |
| **지형 분석** | NumPy + SciPy |

## 설계서

상세 설계 문서는 `DESIGN_SPEC.txt`를 참조하세요. 주요 내용:

- 시스템 목표 및 Ref Path 설계
- DEM 전처리 및 지형 특징 추출
- Route-Centric 지형 분석 (Corridor Band Profile)
- 위험도(Risk) 설계 (Hard/Soft 분리)
- 시뮬레이션 상태 및 운동학 모델
- Observation/Action/Reward 상세 설계
- Curriculum 학습 단계 (Stage 0~5)
- 네트워크 구조 및 하이퍼파라미터 권장치
- Heuristic Baseline 정책
- 평가 지표 및 검증 방법

## 라이선스

MIT License
