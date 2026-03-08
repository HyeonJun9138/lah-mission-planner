/**
 * training.js — LAH Mission Planner 학습 탭 (탭 2)
 * PPO/SAC 하이퍼파라미터, 학습 제어, 실시간 모니터링
 */

'use strict';

window.LAHTraining = (() => {
  const U = window.LAHUtils;
  let initialized = false;
  let logger = null;
  let mockInterval = null;
  let rewardHistory = [], lengthHistory = [], successHistory = [];
  let timestep = 0, isRunning = false, isPaused = false;
  let selectedAlgo = 'PPO';
  let optunaRunning = false;
  let optunaTrials = [];

  function activate() {
    if (!initialized) {
      initialized = true;
      render();
    }
  }

  function render() {
    const container = document.getElementById('training-container');
    if (!container) return;

    container.innerHTML = `
      <div class="training-layout" style="height:100%">

        <!-- Controls Section -->
        <div class="training-controls">
          <div class="section-header" style="margin-bottom:var(--space-3)">
            <h2 class="section-title">학습 설정 및 제어</h2>
            <div id="training-status-badge" class="badge badge-muted">대기 중</div>
          </div>

          <!-- Algorithm Selection + Hyperparams (side by side) -->
          <div class="grid-2" style="gap:var(--space-4)">

            <!-- Algorithm Tabs -->
            <div>
              <div class="form-label" style="margin-bottom:var(--space-2)">알고리즘 선택</div>
              <div class="flex gap-2 mb-3" id="algoTabs">
                ${['PPO', 'SAC', 'RecurrentPPO', 'TQC'].map((a, i) => `
                  <button class="inner-tab ${i === 0 ? 'active' : ''} ${i >= 2 ? 'soon' : ''}"
                    data-algo="${a}" ${i >= 2 ? 'title="준비 중"' : ''}>
                    ${a}${i >= 2 ? '<sup style="font-size:8px;color:var(--text-muted)"> 준비중</sup>' : ''}
                  </button>
                `).join('')}
              </div>

              <!-- PPO Hyperparams -->
              <div id="params-PPO" class="algo-params">
                <div class="grid-2" style="gap:var(--space-2)">
                  ${[
                    { id: 'lr', label: '학습률', type: 'text', val: '3e-4' },
                    { id: 'total_steps', label: '총 타임스텝', type: 'number', val: '1000000' },
                  ].map(f => `
                    <div class="form-group" style="margin-bottom:0">
                      <label class="form-label">${f.label}</label>
                      <input type="${f.type}" class="input input-sm mono" id="hp_${f.id}" value="${f.val}">
                    </div>
                  `).join('')}
                </div>
                <div style="margin-top:var(--space-2);display:flex;flex-direction:column;gap:var(--space-2)">
                  ${[
                    { id: 'n_steps', label: 'n_steps', opts: [512, 1024, 2048], default: 1024 },
                    { id: 'batch_size', label: 'batch_size', opts: [1024, 2048, 4096], default: 2048 },
                  ].map(f => `
                    <div class="form-group" style="margin-bottom:0">
                      <label class="form-label">${f.label}</label>
                      <select class="input input-sm mono" id="hp_${f.id}">
                        ${f.opts.map(o => `<option value="${o}" ${o === f.default ? 'selected' : ''}>${o}</option>`).join('')}
                      </select>
                    </div>
                  `).join('')}
                  ${[
                    { id: 'gamma', label: 'gamma', min: 900, max: 999, val: 995, scale: 0.001 },
                    { id: 'gae_lambda', label: 'gae_lambda', min: 900, max: 990, val: 970, scale: 0.001 },
                    { id: 'ent_coef', label: 'ent_coef', min: 1, max: 50, val: 5, scale: 0.001 },
                    { id: 'clip_range', label: 'clip_range', min: 10, max: 30, val: 20, scale: 0.01 },
                  ].map(s => `
                    <div class="slider-row" style="gap:var(--space-2)">
                      <span class="slider-label mono" style="min-width:80px">${s.label}</span>
                      <input type="range" id="hp_${s.id}" min="${s.min}" max="${s.max}" value="${s.val}" data-display="hpv_${s.id}" data-scale="${s.scale}">
                      <span class="slider-value" id="hpv_${s.id}">${(s.val * s.scale).toFixed(3)}</span>
                    </div>
                  `).join('')}
                </div>
              </div>

              <!-- SAC Hyperparams -->
              <div id="params-SAC" class="algo-params" style="display:none">
                <div class="grid-2" style="gap:var(--space-2)">
                  ${[
                    { id: 'sac_lr', label: '학습률', type: 'text', val: '3e-4' },
                    { id: 'buffer_size', label: 'buffer_size', type: 'number', val: '1000000' },
                    { id: 'learning_starts', label: 'learning_starts', type: 'number', val: '20000' },
                    { id: 'sac_batch', label: 'batch_size', type: 'number', val: '512' },
                  ].map(f => `
                    <div class="form-group" style="margin-bottom:0">
                      <label class="form-label">${f.label}</label>
                      <input type="${f.type}" class="input input-sm mono" id="hp_${f.id}" value="${f.val}">
                    </div>
                  `).join('')}
                </div>
                <div style="margin-top:var(--space-2)">
                  <div class="slider-row" style="gap:var(--space-2)">
                    <span class="slider-label mono" style="min-width:80px">gamma</span>
                    <input type="range" min="900" max="999" value="995" data-display="hpv_sac_gamma" data-scale="0.001">
                    <span class="slider-value" id="hpv_sac_gamma">0.995</span>
                  </div>
                  <div class="slider-row" style="gap:var(--space-2);margin-top:var(--space-2)">
                    <span class="slider-label mono" style="min-width:80px">tau</span>
                    <input type="range" min="1" max="20" value="5" data-display="hpv_sac_tau" data-scale="0.001">
                    <span class="slider-value" id="hpv_sac_tau">0.005</span>
                  </div>
                </div>
              </div>
            </div>

            <!-- Training Controls -->
            <div>
              <div class="form-label" style="margin-bottom:var(--space-2)">학습 제어</div>
              <div class="flex gap-2 mb-3" style="flex-wrap:wrap">
                <button class="btn btn-primary" id="trainStartBtn">▶️ 학습 시작</button>
                <button class="btn btn-secondary" id="trainPauseBtn" disabled>⏸️ 일시정지</button>
                <button class="btn btn-danger btn-sm" id="trainStopBtn" disabled>⏹️ 중단</button>
              </div>
              <div class="flex gap-2 mb-3">
                <button class="btn btn-secondary btn-sm" id="saveModelBtn" disabled>💾 모델 저장</button>
                <button class="btn btn-secondary btn-sm" id="loadModelBtn">📂 모델 불러오기</button>
              </div>

              <!-- Current State Card -->
              <div class="card card-sm">
                <div class="kpi-grid" style="grid-template-columns:repeat(2,1fr);gap:var(--space-2)">
                  ${[
                    { id: 'kpi_timestep', label: 'TIMESTEP', val: '0', color: 'var(--cyan)' },
                    { id: 'kpi_fps', label: 'FPS', val: '0', color: 'var(--green)' },
                    { id: 'kpi_reward', label: 'AVG REWARD', val: '—', color: 'var(--yellow)' },
                    { id: 'kpi_success', label: 'SUCCESS', val: '—', color: 'var(--orange)' },
                  ].map(k => `
                    <div class="kpi-card" style="--accent:${k.color};padding:var(--space-2)">
                      <div class="kpi-label">${k.label}</div>
                      <div class="kpi-value" id="${k.id}" style="font-size:var(--text-xl)">${k.val}</div>
                    </div>
                  `).join('')}
                </div>
              </div>

              <!-- Progress -->
              <div style="margin-top:var(--space-3)">
                <div class="flex justify-between mb-2" style="font-size:var(--text-xs);color:var(--text-muted)">
                  <span>학습 진행률</span>
                  <span class="mono" id="progressPct">0%</span>
                </div>
                <div class="progress-bar"><div class="progress-fill" id="trainingProgress" style="width:0%"></div></div>
              </div>
            </div>
          </div>
        </div>

        <!-- Monitor Section -->
        <div class="training-monitor">
          <!-- Reward Chart -->
          <div class="card">
            <div class="card-header">
              <span class="card-title">에피소드 Reward</span>
              <span class="badge badge-cyan" id="rewardEma">EMA: —</span>
            </div>
            <div id="rewardChart" style="height:160px"></div>
          </div>

          <!-- Success Rate Chart -->
          <div class="card">
            <div class="card-header">
              <span class="card-title">성공률 / 에피소드 길이</span>
            </div>
            <div id="successChart" style="height:160px"></div>
          </div>

          <!-- Learning Curves -->
          <div class="card" style="grid-column:1/3">
            <div class="card-header">
              <span class="card-title">학습 커브</span>
              <div class="flex gap-2">
                <span class="badge badge-muted">PPO Loss</span>
                <span class="badge badge-muted">Value Loss</span>
                <span class="badge badge-muted">Entropy</span>
              </div>
            </div>
            <div id="lossChart" style="height:140px"></div>
          </div>

          <!-- Log Console -->
          <div class="card" style="grid-column:1/3">
            <div class="card-header">
              <span class="card-title">학습 로그</span>
              <button class="btn btn-ghost btn-sm" onclick="document.getElementById('logConsole').innerHTML=''">지우기</button>
            </div>
            <div class="log-console" id="logConsole"></div>
          </div>

          <!-- Optuna Tuning -->
          <div class="card" style="grid-column:1/3">
            <div class="collapsible-header" data-default-open="false">
              <div class="collapsible-title">
                🔬 자동 하이퍼파라미터 튜닝 (Optuna)
              </div>
              <svg class="collapsible-arrow" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
            </div>
            <div class="collapsible-body" id="optunaBody">
              <div class="flex gap-3 mb-3 items-center">
                <button class="btn btn-primary btn-sm" id="optunaStartBtn">Optuna 튜닝 시작</button>
                <button class="btn btn-danger btn-sm" id="optunaStopBtn" disabled>중단</button>
                <div class="form-group" style="margin-bottom:0;flex:1">
                  <div class="flex items-center gap-2">
                    <span class="form-label">시도 횟수</span>
                    <input type="number" class="input input-sm mono" id="optunaN" value="20" style="width:70px">
                  </div>
                </div>
              </div>
              <div id="optunaTable" style="overflow-x:auto">
                <table class="data-table">
                  <thead><tr><th>#</th><th>학습률</th><th>n_steps</th><th>gamma</th><th>최종 reward</th><th>상태</th></tr></thead>
                  <tbody id="optunaBody2"><tr><td colspan="6" style="text-align:center;color:var(--text-muted);padding:var(--space-4)">튜닝 대기 중...</td></tr></tbody>
                </table>
              </div>
              <div style="margin-top:var(--space-3);display:flex;gap:var(--space-2)">
                <button class="btn btn-secondary btn-sm" id="applyBestBtn" disabled>최적 파라미터 적용</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;

    // Bind sliders
    container.querySelectorAll('input[type="range"][data-display]').forEach(slider => {
      const displayId = slider.dataset.display;
      const display = document.getElementById(displayId);
      if (!display) return;
      const scale = parseFloat(slider.dataset.scale) || 1;
      const update = () => { display.textContent = (parseFloat(slider.value) * scale).toFixed(3); };
      slider.addEventListener('input', update);
      update();
    });

    // Collapsibles
    U.initCollapsibles(container);

    // Logger
    const logEl = document.getElementById('logConsole');
    if (logEl) logger = U.createLogger(logEl);

    // Algorithm tabs
    setupAlgoTabs();

    // Train controls
    setupTrainControls();

    // Optuna
    setupOptuna();

    // Initial charts
    initCharts();
    logger?.info('시스템 초기화 완료. 학습 시작 버튼을 눌러 학습을 시작하세요.');
  }

  function setupAlgoTabs() {
    const tabs = document.querySelectorAll('#algoTabs .inner-tab');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        if (tab.disabled) return;
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        selectedAlgo = tab.dataset.algo;

        // Show/hide param panels
        document.querySelectorAll('.algo-params').forEach(p => p.style.display = 'none');
        const target = document.getElementById(`params-${selectedAlgo}`);
        if (target) target.style.display = '';
        else {
          const ppoPan = document.getElementById('params-PPO');
          if (ppoPan) ppoPan.style.display = '';
        }
      });
    });
  }

  function setupTrainControls() {
    const startBtn = document.getElementById('trainStartBtn');
    const pauseBtn = document.getElementById('trainPauseBtn');
    const stopBtn  = document.getElementById('trainStopBtn');
    const saveBtn  = document.getElementById('saveModelBtn');
    const loadBtn  = document.getElementById('loadModelBtn');

    startBtn?.addEventListener('click', startTraining);
    pauseBtn?.addEventListener('click', togglePause);
    stopBtn?.addEventListener('click', stopTraining);
    saveBtn?.addEventListener('click', saveModel);
    loadBtn?.addEventListener('click', loadModel);
  }

  function startTraining() {
    if (isRunning) return;
    isRunning = true; isPaused = false;
    rewardHistory = []; lengthHistory = []; successHistory = [];
    timestep = 0;

    const startBtn = document.getElementById('trainStartBtn');
    const pauseBtn = document.getElementById('trainPauseBtn');
    const stopBtn  = document.getElementById('trainStopBtn');
    const saveBtn  = document.getElementById('saveModelBtn');
    const badge    = document.getElementById('training-status-badge');

    if (startBtn) startBtn.disabled = true;
    if (pauseBtn) pauseBtn.disabled = false;
    if (stopBtn)  stopBtn.disabled = false;
    if (saveBtn)  saveBtn.disabled = false;
    if (badge)    { badge.className = 'badge badge-green'; badge.textContent = '학습 중'; }

    logger?.ok(`${selectedAlgo} 학습 시작 — 타임스텝: ${document.getElementById('hp_total_steps')?.value || 1000000}`);
    logger?.info(`학습률: ${document.getElementById('hp_lr')?.value || '3e-4'}`);

    // Mock training loop
    mockInterval = setInterval(tickTraining, 200);
  }

  function tickTraining() {
    if (isPaused) return;

    const totalSteps = parseInt(document.getElementById('hp_total_steps')?.value || 1000000);
    timestep += Math.floor(U.randRange(800, 1200));
    if (timestep > totalSteps) { stopTraining(); return; }

    const progress = timestep / totalSteps;
    const reward = -50 + progress * 180 + U.randRange(-20, 20) * (1 - progress * 0.5);
    const success = Math.min(0.95, 0.1 + progress * 0.7 + U.randRange(-0.05, 0.05));
    const epLen = Math.round(600 - progress * 200 + U.randRange(-50, 50));
    const fps = Math.round(U.randRange(1800, 2400));

    rewardHistory.push(reward);
    successHistory.push(success * 100);
    lengthHistory.push(epLen);
    if (rewardHistory.length > 500) rewardHistory.shift();
    if (successHistory.length > 500) successHistory.shift();

    // Update KPIs
    U.setText('kpi_timestep', U.fmtInt(timestep));
    U.setText('kpi_fps', fps.toLocaleString());
    U.setText('kpi_reward', U.fmt(reward, 1));
    U.setText('kpi_success', U.fmtPct(success));

    // Progress
    const pct = Math.round(progress * 100);
    const progEl = document.getElementById('trainingProgress');
    if (progEl) progEl.style.width = pct + '%';
    U.setText('progressPct', pct + '%');

    // Update global state
    const appState = LAHApp.getState();
    appState.trainingState.timestep = timestep;
    appState.trainingState.fps = fps;
    appState.trainingState.avgReward = reward;
    appState.trainingState.successRate = success;
    LAHApp.updateStatusBar();

    // EMA
    const ema = rewardHistory.slice(-20).reduce((a, b) => a + b, 0) / Math.min(20, rewardHistory.length);
    U.setText('rewardEma', `EMA: ${U.fmt(ema, 1)}`);

    // Log periodically
    if (rewardHistory.length % 50 === 0) {
      logger?.info(`Step ${U.fmtInt(timestep)} | reward: ${U.fmt(reward, 1)} | success: ${U.fmtPct(success)} | fps: ${fps}`);
    }

    // Update charts every 5 ticks
    if (rewardHistory.length % 5 === 0) updateCharts();
  }

  function togglePause() {
    isPaused = !isPaused;
    const btn = document.getElementById('trainPauseBtn');
    const badge = document.getElementById('training-status-badge');
    if (btn) btn.textContent = isPaused ? '▶️ 재개' : '⏸️ 일시정지';
    if (badge) { badge.className = isPaused ? 'badge badge-yellow' : 'badge badge-green'; badge.textContent = isPaused ? '일시정지' : '학습 중'; }
    logger?.[isPaused ? 'warn' : 'ok'](isPaused ? '학습 일시정지됨' : '학습 재개됨');
  }

  function stopTraining() {
    isRunning = false; isPaused = false;
    clearInterval(mockInterval);

    const startBtn = document.getElementById('trainStartBtn');
    const pauseBtn = document.getElementById('trainPauseBtn');
    const stopBtn  = document.getElementById('trainStopBtn');
    const badge    = document.getElementById('training-status-badge');

    if (startBtn) startBtn.disabled = false;
    if (pauseBtn) { pauseBtn.disabled = true; pauseBtn.textContent = '⏸️ 일시정지'; }
    if (stopBtn)  stopBtn.disabled = true;
    if (badge)    { badge.className = 'badge badge-muted'; badge.textContent = '중단됨'; }

    logger?.warn('학습이 중단되었습니다.');
    updateCharts();
  }

  function saveModel() {
    U.toast('데모 모드: 모델 저장 시뮬레이션됨', 'ok');
    logger?.ok(`모델 저장됨: ${selectedAlgo}_step${U.fmtInt(timestep)}.zip`);
    const appState = LAHApp.getState();
    appState.models.push(`${selectedAlgo}_step${U.fmtInt(timestep)}`);
    LAHApp.updateStatusBar();
  }

  function loadModel() {
    const input = document.createElement('input');
    input.type = 'file'; input.accept = '.zip,.pt';
    input.onchange = () => { U.toast('모델을 불러왔습니다.', 'ok'); logger?.ok('모델 불러오기 완료'); };
    input.click();
  }

  // ── Charts ──
  function initCharts() {
    const dummyX = Array.from({ length: 10 }, (_, i) => i * 1000);
    const dummyY = dummyX.map(() => 0);

    // Reward Chart
    Plotly.newPlot('rewardChart', [
      { x: dummyX, y: dummyY, type: 'scatter', mode: 'lines', name: 'Reward',
        line: { color: '#00d4ff', width: 1.5 }, fill: 'tozeroy', fillcolor: 'rgba(0,212,255,0.06)' }
    ], {
      ...U.plotlyLayout({ margin: { l: 35, r: 8, t: 8, b: 25 } }),
      xaxis: { gridcolor: '#1e2a3d', color: '#475569', title: { text: 'Timestep', font: { size: 9 } }, tickfont: { size: 8 } },
      yaxis: { gridcolor: '#1e2a3d', color: '#475569', title: { text: 'Reward', font: { size: 9 } }, tickfont: { size: 8 } },
    }, { responsive: true, displayModeBar: false });

    // Success/Length Chart
    Plotly.newPlot('successChart', [
      { x: dummyX, y: dummyY, type: 'scatter', mode: 'lines', name: '성공률 (%)',
        line: { color: '#4ade80', width: 1.5 }, yaxis: 'y' },
      { x: dummyX, y: dummyY, type: 'scatter', mode: 'lines', name: '에피소드 길이',
        line: { color: '#f59e0b', width: 1, dash: 'dot' }, yaxis: 'y2' },
    ], {
      ...U.plotlyLayout({ margin: { l: 35, r: 35, t: 8, b: 25 } }),
      showlegend: true,
      legend: { orientation: 'h', y: -0.3, font: { size: 8, color: '#94a3b8' }, bgcolor: 'transparent' },
      yaxis: { gridcolor: '#1e2a3d', color: '#4ade80', tickfont: { size: 8 }, title: { text: '%', font: { size: 9 } } },
      yaxis2: { overlaying: 'y', side: 'right', color: '#f59e0b', tickfont: { size: 8 }, showgrid: false },
    }, { responsive: true, displayModeBar: false });

    // Loss Chart
    Plotly.newPlot('lossChart', [
      { x: dummyX, y: dummyY, type: 'scatter', mode: 'lines', name: 'Policy Loss',
        line: { color: '#a78bfa', width: 1.5 } },
      { x: dummyX, y: dummyY, type: 'scatter', mode: 'lines', name: 'Value Loss',
        line: { color: '#ff6b35', width: 1.5 } },
      { x: dummyX, y: dummyY, type: 'scatter', mode: 'lines', name: 'Entropy',
        line: { color: '#00d4ff', width: 1, dash: 'dash' } },
    ], {
      ...U.plotlyLayout({ margin: { l: 35, r: 8, t: 8, b: 25 } }),
      showlegend: true,
      legend: { orientation: 'h', y: -0.35, font: { size: 8, color: '#94a3b8' }, bgcolor: 'transparent' },
    }, { responsive: true, displayModeBar: false });
  }

  function updateCharts() {
    const n = rewardHistory.length;
    const steps = Array.from({ length: n }, (_, i) => (i + 1) * 1000);

    Plotly.update('rewardChart', { x: [steps], y: [rewardHistory] }, {}, [0]);

    Plotly.update('successChart',
      { x: [steps, Array.from({ length: lengthHistory.length }, (_, i) => (i + 1) * 1000)],
        y: [successHistory, lengthHistory] }, {}, [0, 1]);

    // Mock loss
    const pLoss = rewardHistory.map((_, i) => 0.3 - i / (n * 3) + (Math.random() - 0.5) * 0.05);
    const vLoss = rewardHistory.map((_, i) => 0.5 - i / (n * 2.5) + (Math.random() - 0.5) * 0.1);
    const entropy = rewardHistory.map((_, i) => 0.01 + Math.sin(i * 0.1) * 0.003);
    Plotly.update('lossChart', { x: [steps, steps, steps], y: [pLoss, vLoss, entropy] }, {}, [0, 1, 2]);
  }

  // ── Optuna ──
  function setupOptuna() {
    document.getElementById('optunaStartBtn')?.addEventListener('click', startOptuna);
    document.getElementById('optunaStopBtn')?.addEventListener('click', stopOptuna);
    document.getElementById('applyBestBtn')?.addEventListener('click', applyBest);
  }

  function startOptuna() {
    optunaRunning = true;
    optunaTrials = [];
    const startBtn = document.getElementById('optunaStartBtn');
    const stopBtn  = document.getElementById('optunaStopBtn');
    if (startBtn) startBtn.disabled = true;
    if (stopBtn)  stopBtn.disabled = false;
    logger?.info('Optuna 튜닝 시작...');

    const nTrials = parseInt(document.getElementById('optunaN')?.value || 20);
    let trial = 0;

    const runTrial = () => {
      if (!optunaRunning || trial >= nTrials) { stopOptuna(); return; }
      trial++;
      const lr = [1e-4, 3e-4, 5e-4, 1e-3][U.randInt(0, 3)];
      const nSteps = [512, 1024, 2048][U.randInt(0, 2)];
      const gamma = (0.990 + U.randRange(0, 0.009)).toFixed(4);
      const reward = 80 + trial * 3 + U.randRange(-15, 15);
      const t = { trial, lr, n_steps: nSteps, gamma, reward: U.fmt(reward, 1), status: '완료' };
      optunaTrials.push(t);
      renderOptunaTable();
      logger?.info(`Trial ${trial}/${nTrials}: lr=${lr}, reward=${U.fmt(reward, 1)}`);
      setTimeout(runTrial, 600);
    };
    setTimeout(runTrial, 300);
  }

  function stopOptuna() {
    optunaRunning = false;
    const startBtn = document.getElementById('optunaStartBtn');
    const stopBtn  = document.getElementById('optunaStopBtn');
    const applyBtn = document.getElementById('applyBestBtn');
    if (startBtn) startBtn.disabled = false;
    if (stopBtn)  stopBtn.disabled = true;
    if (applyBtn) applyBtn.disabled = false;
    logger?.ok('Optuna 튜닝 완료');
  }

  function renderOptunaTable() {
    const tbody = document.getElementById('optunaBody2');
    if (!tbody) return;
    const best = Math.max(...optunaTrials.map(t => parseFloat(t.reward)));
    tbody.innerHTML = optunaTrials.slice(-10).reverse().map(t => `
      <tr>
        <td class="mono">${t.trial}</td>
        <td class="mono">${t.lr}</td>
        <td class="mono">${t.n_steps}</td>
        <td class="mono">${t.gamma}</td>
        <td class="mono" style="color:${parseFloat(t.reward) >= best ? 'var(--green)' : 'var(--text-primary)'}">${t.reward}</td>
        <td><span class="badge badge-green">${t.status}</span></td>
      </tr>
    `).join('');
  }

  function applyBest() {
    if (!optunaTrials.length) return;
    const best = optunaTrials.reduce((a, b) => parseFloat(a.reward) > parseFloat(b.reward) ? a : b);
    const lrEl = document.getElementById('hp_lr');
    if (lrEl) lrEl.value = best.lr;
    logger?.ok(`최적 파라미터 적용: lr=${best.lr}, n_steps=${best.n_steps}, gamma=${best.gamma}`);
    U.toast('최적 파라미터가 적용되었습니다.', 'ok');
  }

  return { activate };
})();
