/**
 * app.js — LAH Mission Planner 글로벌 앱 관리자
 * 탭 전환, API 통신, 글로벌 상태 관리
 */

'use strict';

window.LAHApp = (() => {
  const U = window.LAHUtils;

  // ── Global State ──
  const state = {
    currentTab: 0,
    serverConnected: false,
    demoMode: true,
    envConfig: getDefaultEnvConfig(),
    trainingState: {
      running: false,
      paused: false,
      algorithm: 'PPO',
      timestep: 0,
      fps: 0,
      avgReward: 0,
      successRate: 0,
      episodeCount: 0
    },
    simState: {
      running: false,
      paused: false,
      speed: 1,
      currentFrame: 0,
      frames: [],
      selectedModel: 'heuristic_tracker'
    },
    models: [],
    currentEpisode: null,
    terrain: null,
    refPath: null
  };

  // ── Tab Config ──
  const tabs = [
    { id: 0, name: '표지', breadcrumb: '표지' },
    { id: 1, name: '환경 설정', breadcrumb: '환경 설정' },
    { id: 2, name: '학습', breadcrumb: '학습' },
    { id: 3, name: '시뮬레이션', breadcrumb: '시뮬레이션' },
    { id: 4, name: '3D 시각화', breadcrumb: '3D 시각화' },
    { id: 5, name: '환경 검증', breadcrumb: '환경 검증' },
    { id: 6, name: '모델 비교', breadcrumb: '모델 비교' },
  ];

  // ── Initialization ──
  function init() {
    setupTabNavigation();
    setupSidebar();
    startClock();
    checkServerConnection();
    setInterval(checkServerConnection, 10000);

    // Initialize first visible tab
    switchTab(0);
  }

  // ── Tab Navigation ──
  function setupTabNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
      item.addEventListener('click', () => {
        const tabId = parseInt(item.dataset.tab);
        switchTab(tabId);
      });
    });
  }

  function switchTab(tabId) {
    if (tabId === state.currentTab && tabId !== 0) return;

    // Deactivate current
    const currentPanel = document.getElementById(`tab-${state.currentTab}`);
    const currentNav = document.querySelector(`.nav-item[data-tab="${state.currentTab}"]`);
    if (currentPanel) currentPanel.classList.remove('active');
    if (currentNav) currentNav.classList.remove('active');

    // Notify outgoing module
    onTabDeactivate(state.currentTab);

    state.currentTab = tabId;

    // Activate new
    const newPanel = document.getElementById(`tab-${tabId}`);
    const newNav = document.querySelector(`.nav-item[data-tab="${tabId}"]`);
    if (newPanel) newPanel.classList.add('active');
    if (newNav) newNav.classList.add('active');

    // Update breadcrumb
    const tab = tabs.find(t => t.id === tabId);
    if (tab) U.setText('headerBreadcrumb', tab.breadcrumb);

    // Notify incoming module
    onTabActivate(tabId);
  }

  function onTabActivate(tabId) {
    switch (tabId) {
      case 0: if (window.LAHCover) window.LAHCover.activate(); break;
      case 1: if (window.LAHEnvironment) window.LAHEnvironment.activate(); break;
      case 2: if (window.LAHTraining) window.LAHTraining.activate(); break;
      case 3: if (window.LAHSimulation) window.LAHSimulation.activate(); break;
      case 4: if (window.LAHViz3D) window.LAHViz3D.activate(); break;
      case 5: if (window.LAHValidation) window.LAHValidation.activate(); break;
      case 6: initComparison(); break;
    }
  }

  function onTabDeactivate(tabId) {
    switch (tabId) {
      case 4: if (window.LAHViz3D) window.LAHViz3D.deactivate(); break;
    }
  }

  // ── Sidebar Toggle ──
  function setupSidebar() {
    const toggle = document.getElementById('sidebarToggle');
    const sidebar = document.getElementById('sidebar');
    if (toggle && sidebar) {
      toggle.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
      });
    }
  }

  // ── Clock ──
  function startClock() {
    const el = document.getElementById('headerTime');
    if (!el) return;
    const tick = () => {
      el.textContent = new Date().toLocaleTimeString('ko-KR', { hour12: false });
    };
    tick();
    setInterval(tick, 1000);
  }

  // ── Server Connection ──
  async function checkServerConnection() {
    const { data, error } = await U.apiCall('/health');
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    const modeEl = document.getElementById('statusMode');

    if (data && !error) {
      state.serverConnected = true;
      state.demoMode = false;
      if (dot) { dot.className = 'status-dot connected'; }
      if (text) text.textContent = '서버 연결됨';
      if (modeEl) modeEl.textContent = '모드: 온라인';
    } else {
      state.serverConnected = false;
      state.demoMode = true;
      if (dot) { dot.className = 'status-dot disconnected'; }
      if (text) text.textContent = '오프라인';
      if (modeEl) modeEl.textContent = '모드: 데모';
    }
  }

  // ── Default Config ──
  function getDefaultEnvConfig() {
    return {
      terrainType: 'synthetic',
      mapSize: 20000,
      resolution: 30,
      complexity: 1.0,
      startPoint: { x: 1000, y: 1000, z: 400 },
      endPoint:   { x: 19000, y: 19000, z: 400 },
      waypoints: [],
      resampleInterval: 50,
      softCorridor: 400,
      hardCorridor: 1000,
      vMin: 0, vMax: 70,
      vzMin: -6, vzMax: 6,
      maxTurnRate: 12,
      aglSafeMin: 120,
      aglPreferred: 180,
      rewards: {
        w_prog: 2.0, w_goal: 0.5, w_d: 0.8, w_zref: 0.2,
        w_risk: 1.0, w_risk_a: 0.7, w_clear: 3.0,
        w_smooth: 0.05, w_energy: 0.05, w_hold: 0.03
      }
    };
  }

  // ── Status Updates ──
  function updateStatusBar() {
    const envEl = document.getElementById('statusEnv');
    const modelEl = document.getElementById('statusModel');
    if (envEl) envEl.textContent = `환경: ${state.terrain ? '초기화됨' : '미초기화'}`;
    if (modelEl) modelEl.textContent = `모델: ${state.models.length > 0 ? `${state.models.length}개` : '없음'}`;
    U.setText('envStatus', state.terrain ? '초기화됨' : '미초기화');
    U.setText('modelStatus', state.models.length > 0 ? `${state.models.length}개` : '없음');
    U.setText('timestepStatus', U.fmtInt(state.trainingState.timestep));
  }

  // ── Model Comparison Tab (Tab 6) ──
  function initComparison() {
    const container = document.getElementById('comparison-container');
    if (!container || container.dataset.initialized) return;
    container.dataset.initialized = 'true';

    const mockModels = [
      { name: 'PPO_v1', algo: 'PPO', steps: 1000000, successRate: 0.71, avgReturn: 142.3, collisionRate: 0.12 },
      { name: 'PPO_v2', algo: 'PPO', steps: 2000000, successRate: 0.82, avgReturn: 178.5, collisionRate: 0.07 },
      { name: 'SAC_v1', algo: 'SAC', steps: 500000, successRate: 0.68, avgReturn: 135.1, collisionRate: 0.14 },
      { name: 'Heuristic', algo: 'Tracker', steps: 0, successRate: 0.55, avgReturn: 98.2, collisionRate: 0.22 },
    ];

    container.innerHTML = `
      <div class="cmp-layout">
        <div class="section-header">
          <h2 class="section-title">모델 비교</h2>
          <div class="flex gap-2">
            <button class="btn btn-secondary btn-sm">에피소드 실행</button>
            <button class="btn btn-primary btn-sm">경로 오버레이</button>
          </div>
        </div>

        <div class="grid-2" style="gap:var(--space-4)">
          <!-- Model Selection -->
          <div class="card">
            <div class="card-header"><span class="card-title">모델 선택</span></div>
            <div style="display:flex;flex-direction:column;gap:var(--space-2)">
              ${mockModels.map((m, i) => `
                <label class="checkbox-label">
                  <input type="checkbox" ${i < 3 ? 'checked' : ''} data-model="${m.name}">
                  <span>${m.name}</span>
                  <span class="badge badge-cyan" style="margin-left:auto">${m.algo}</span>
                </label>
              `).join('')}
            </div>
          </div>

          <!-- Radar Chart -->
          <div class="card">
            <div class="card-header"><span class="card-title">성능 레이더 차트</span></div>
            <div id="radar-chart" style="height:220px"></div>
          </div>
        </div>

        <!-- Performance Table -->
        <div class="card">
          <div class="card-header"><span class="card-title">성능 지표 비교</span></div>
          <table class="data-table">
            <thead>
              <tr>
                <th>모델명</th><th>알고리즘</th><th>학습 스텝</th>
                <th>성공률</th><th>평균 리턴</th><th>충돌률</th><th>비고</th>
              </tr>
            </thead>
            <tbody>
              ${mockModels.map((m, i) => `
                <tr>
                  <td class="mono" style="color:var(--cyan)">${m.name}</td>
                  <td><span class="badge badge-muted">${m.algo}</span></td>
                  <td class="mono">${U.fmtInt(m.steps)}</td>
                  <td>
                    <div class="flex items-center gap-2">
                      <div class="progress-bar" style="width:60px">
                        <div class="progress-fill" style="width:${m.successRate*100}%"></div>
                      </div>
                      <span class="mono">${U.fmtPct(m.successRate)}</span>
                    </div>
                  </td>
                  <td class="mono">${U.fmt(m.avgReturn, 1)}</td>
                  <td class="mono" style="color:${m.collisionRate > 0.15 ? 'var(--orange)' : 'var(--green)'}">${U.fmtPct(m.collisionRate)}</td>
                  <td>${i === 1 ? '<span class="badge badge-green">Best</span>' : ''}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>

        <!-- Path Overlay Placeholder -->
        <div class="card">
          <div class="card-header">
            <span class="card-title">경로 오버레이 비교</span>
            <span class="text-xs text-muted">동일 지형에서 여러 모델의 경로</span>
          </div>
          <div id="path-overlay-canvas" style="height:280px;background:#050810;border-radius:var(--radius-md);position:relative">
            <canvas id="overlay-canvas" style="width:100%;height:100%;display:block"></canvas>
          </div>
        </div>
      </div>
    `;

    // Render radar chart
    const categories = ['성공률', '경로 추종', '안전도', '효율성', '부드러움'];
    const traces = [
      { name: 'PPO_v2', values: [0.82, 0.78, 0.85, 0.76, 0.72], color: '#00d4ff' },
      { name: 'PPO_v1', values: [0.71, 0.65, 0.77, 0.69, 0.63], color: '#4ade80' },
      { name: 'SAC_v1', values: [0.68, 0.72, 0.71, 0.74, 0.80], color: '#f59e0b' },
    ];

    const radarData = traces.map(t => ({
      type: 'scatterpolar',
      r: [...t.values, t.values[0]],
      theta: [...categories, categories[0]],
      name: t.name,
      fill: 'toself',
      fillcolor: t.color + '20',
      line: { color: t.color, width: 1.5 }
    }));

    Plotly.newPlot('radar-chart', radarData, {
      ...U.plotlyLayout(),
      polar: {
        bgcolor: '#050810',
        radialaxis: { visible: true, range: [0, 1], color: '#475569', gridcolor: '#1e2a3d' },
        angularaxis: { color: '#475569', gridcolor: '#1e2a3d' }
      },
      showlegend: true,
      legend: { font: { size: 9, color: '#94a3b8' }, bgcolor: 'transparent' },
      margin: { l: 30, r: 30, t: 20, b: 20 }
    }, { responsive: true, displayModeBar: false });

    // Draw overlay canvas
    setTimeout(() => {
      const canvas = document.getElementById('overlay-canvas');
      if (!canvas) return;
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
      const ctx = canvas.getContext('2d');
      const w = canvas.width, h = canvas.height;

      // Background
      ctx.fillStyle = '#050810';
      ctx.fillRect(0, 0, w, h);
      U.drawGrid(ctx, w, h, 40);

      // Draw mock paths
      const pathColors = ['#00d4ff', '#4ade80', '#f59e0b', '#94a3b8'];
      const pathNames = ['PPO_v2', 'PPO_v1', 'SAC_v1', 'Heuristic'];
      const numPts = 60;

      pathNames.forEach((name, idx) => {
        ctx.beginPath();
        for (let i = 0; i < numPts; i++) {
          const t = i / (numPts - 1);
          const x = 40 + t * (w - 80);
          const wobble = Math.sin(t * Math.PI * 3 + idx * 1.5) * (20 + idx * 8);
          const y = h / 2 + wobble;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = pathColors[idx];
        ctx.lineWidth = idx === 0 ? 2.5 : 1.5;
        ctx.globalAlpha = idx === 0 ? 1 : 0.7;
        ctx.stroke();
      });

      // Ref path
      ctx.beginPath();
      ctx.moveTo(40, h/2); ctx.lineTo(w-40, h/2);
      ctx.strokeStyle = '#475569';
      ctx.lineWidth = 1;
      ctx.setLineDash([6, 4]);
      ctx.globalAlpha = 1;
      ctx.stroke();
      ctx.setLineDash([]);

      // Legend
      pathNames.forEach((name, idx) => {
        ctx.fillStyle = pathColors[idx];
        ctx.fillRect(10, 10 + idx * 16, 20, 2);
        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px monospace';
        ctx.fillText(name, 36, 15 + idx * 16);
      });
    }, 100);
  }

  // ── Public API ──
  return {
    init,
    switchTab,
    getState: () => state,
    updateStatusBar,
    checkServerConnection
  };
})();
