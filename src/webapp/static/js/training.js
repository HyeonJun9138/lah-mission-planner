/**
 * training.js
 * Real backend training monitor for LAH Mission Planner.
 */

'use strict';

window.LAHTraining = (() => {
  const U = window.LAHUtils;

  let initialized = false;
  let logger = null;
  let pollTimer = null;
  let selectedAlgo = 'PPO';
  let lastLogCount = 0;
  let lastRequestedTotal = 0;
  let lastComputeError = null;

  function activate() {
    if (!initialized) {
      initialized = true;
      render();
    }
    startPolling();
    refreshAll();
  }

  function render() {
    const container = document.getElementById('training-container');
    if (!container) return;

    container.innerHTML = `
      <div class="training-layout" style="height:100%">
        <div class="training-controls">
          <div class="section-header" style="margin-bottom:var(--space-3)">
            <h2 class="section-title">Training Control</h2>
            <div id="training-status-badge" class="badge badge-muted">IDLE</div>
          </div>

          <div class="grid-2" style="gap:var(--space-4)">
            <div class="card card-sm">
              <div class="form-label" style="margin-bottom:var(--space-2)">Algorithm</div>
              <div class="flex gap-2 mb-3" id="algoTabs">
                ${['PPO', 'SAC'].map((algo, index) => `
                  <button class="inner-tab ${index === 0 ? 'active' : ''}" data-algo="${algo}">${algo}</button>
                `).join('')}
              </div>

              <div class="grid-2" style="gap:var(--space-2)">
                <div class="form-group" style="margin-bottom:0">
                  <label class="form-label">learning_rate</label>
                  <input type="text" class="input input-sm mono" id="hp_learning_rate" value="3e-4">
                </div>
                <div class="form-group" style="margin-bottom:0">
                  <label class="form-label">total_timesteps</label>
                  <input type="number" class="input input-sm mono" id="hp_total_steps" value="500000">
                </div>
                <div class="form-group" style="margin-bottom:0">
                  <label class="form-label">device</label>
                  <select class="input input-sm mono" id="hp_device">
                    <option value="auto" selected>auto</option>
                    <option value="cuda">cuda</option>
                    <option value="cpu">cpu</option>
                  </select>
                </div>
                <div class="form-group" style="margin-bottom:0">
                  <label class="form-label">batch_size</label>
                  <input type="number" class="input input-sm mono" id="hp_batch_size" value="2048">
                </div>
              </div>

              <div class="grid-2" style="gap:var(--space-2);margin-top:var(--space-2)">
                <div class="form-group" style="margin-bottom:0">
                  <label class="form-label">n_steps / buffer_size</label>
                  <input type="number" class="input input-sm mono" id="hp_steps_like" value="1024">
                </div>
                <div class="form-group" style="margin-bottom:0">
                  <label class="form-label">gamma</label>
                  <input type="text" class="input input-sm mono" id="hp_gamma" value="0.995">
                </div>
                <div class="form-group" style="margin-bottom:0">
                  <label class="form-label">gae_lambda / tau</label>
                  <input type="text" class="input input-sm mono" id="hp_second_coeff" value="0.97">
                </div>
                <div class="form-group" style="margin-bottom:0">
                  <label class="form-label">ent_coef / learning_starts</label>
                  <input type="text" class="input input-sm mono" id="hp_third_coeff" value="0.005">
                </div>
              </div>
            </div>

            <div class="card card-sm">
              <div class="form-label" style="margin-bottom:var(--space-2)">Actions</div>
              <div class="flex gap-2 mb-3" style="flex-wrap:wrap">
                <button class="btn btn-primary" id="trainStartBtn">Start Training</button>
                <button class="btn btn-danger" id="trainStopBtn" disabled>Stop</button>
                <button class="btn btn-secondary" id="trainCheckConnectionBtn">Connection Check</button>
                <button class="btn btn-secondary" id="trainRefreshComputeBtn">Refresh GPU</button>
              </div>

              <div class="kpi-grid" style="grid-template-columns:repeat(2,1fr);gap:var(--space-2)">
                <div class="kpi-card" style="--accent:var(--cyan);padding:var(--space-2)">
                  <div class="kpi-label">TIMESTEP</div>
                  <div class="kpi-value" id="kpi_timestep" style="font-size:var(--text-xl)">0</div>
                </div>
                <div class="kpi-card" style="--accent:var(--yellow);padding:var(--space-2)">
                  <div class="kpi-label">AVG REWARD</div>
                  <div class="kpi-value" id="kpi_reward" style="font-size:var(--text-xl)">-</div>
                </div>
                <div class="kpi-card" style="--accent:var(--green);padding:var(--space-2)">
                  <div class="kpi-label">MODEL DEVICE</div>
                  <div class="kpi-value" id="kpi_device" style="font-size:var(--text-xl)">-</div>
                </div>
                <div class="kpi-card" style="--accent:var(--orange);padding:var(--space-2)">
                  <div class="kpi-label">GPU MEMORY</div>
                  <div class="kpi-value" id="kpi_gpu_mem" style="font-size:var(--text-xl)">-</div>
                </div>
              </div>

              <div style="margin-top:var(--space-3)">
                <div class="flex justify-between mb-2" style="font-size:var(--text-xs);color:var(--text-muted)">
                  <span>Progress</span>
                  <span class="mono" id="progressPct">0%</span>
                </div>
                <div class="progress-bar"><div class="progress-fill" id="trainingProgress" style="width:0%"></div></div>
              </div>
            </div>
          </div>
        </div>

        <div class="training-monitor">
          <div class="card">
            <div class="card-header">
              <span class="card-title">Reward Curve</span>
              <span class="badge badge-cyan" id="rewardEma">EMA: -</span>
            </div>
            <div id="rewardChart" style="height:220px"></div>
          </div>

          <div class="card">
            <div class="card-header">
              <span class="card-title">CUDA / GPU Monitor</span>
              <span class="badge badge-muted" id="computeRuntimeBadge">UNKNOWN</span>
            </div>
            <div class="kpi-grid" style="grid-template-columns:repeat(2,1fr);gap:var(--space-2)">
              <div class="kpi-card" style="--accent:var(--cyan);padding:var(--space-2)">
                <div class="kpi-label">TORCH</div>
                <div class="kpi-value" id="computeTorch">-</div>
              </div>
              <div class="kpi-card" style="--accent:var(--green);padding:var(--space-2)">
                <div class="kpi-label">CUDA</div>
                <div class="kpi-value" id="computeCuda">-</div>
              </div>
              <div class="kpi-card" style="--accent:var(--yellow);padding:var(--space-2)">
                <div class="kpi-label">REQUESTED</div>
                <div class="kpi-value" id="computeRequested">-</div>
              </div>
              <div class="kpi-card" style="--accent:var(--orange);padding:var(--space-2)">
                <div class="kpi-label">RESOLVED</div>
                <div class="kpi-value" id="computeResolved">-</div>
              </div>
            </div>
            <div style="margin-top:var(--space-3);font-size:var(--text-xs);color:var(--text-muted);display:grid;gap:6px">
              <div><span class="mono">GPU:</span> <span id="computeGpuName">-</span></div>
              <div><span class="mono">UTIL:</span> <span id="computeGpuUtil">-</span></div>
              <div><span class="mono">MEM:</span> <span id="computeGpuMemory">-</span></div>
              <div><span class="mono">DETAIL:</span> <span id="computeDetail">-</span></div>
            </div>
            <div id="computeError" class="mono" style="margin-top:var(--space-3);font-size:11px;color:#fca5a5;white-space:pre-wrap"></div>
          </div>

          <div class="card" style="grid-column:1/3">
            <div class="card-header">
              <span class="card-title">Training Log</span>
              <button class="btn btn-ghost btn-sm" id="clearTrainingLogBtn">Clear</button>
            </div>
            <div class="log-console" id="logConsole"></div>
          </div>
        </div>
      </div>
    `;

    logger = U.createLogger(document.getElementById('logConsole'));
    initChart();
    bindEvents();
    logger.info('Backend training monitor is ready.');
  }

  function bindEvents() {
    document.querySelectorAll('#algoTabs .inner-tab').forEach((button) => {
      button.addEventListener('click', () => {
        document.querySelectorAll('#algoTabs .inner-tab').forEach((node) => node.classList.remove('active'));
        button.classList.add('active');
        selectedAlgo = button.dataset.algo || 'PPO';
        syncAlgoFields();
      });
    });

    document.getElementById('trainStartBtn')?.addEventListener('click', startTraining);
    document.getElementById('trainStopBtn')?.addEventListener('click', stopTraining);
    document.getElementById('trainCheckConnectionBtn')?.addEventListener('click', checkConnection);
    document.getElementById('trainRefreshComputeBtn')?.addEventListener('click', refreshComputeStatus);
    document.getElementById('clearTrainingLogBtn')?.addEventListener('click', () => logger?.clear());
    syncAlgoFields();
  }

  function syncAlgoFields() {
    if (selectedAlgo === 'PPO') {
      document.getElementById('hp_batch_size').value = '2048';
      document.getElementById('hp_steps_like').value = '1024';
      document.getElementById('hp_second_coeff').value = '0.97';
      document.getElementById('hp_third_coeff').value = '0.005';
    } else {
      document.getElementById('hp_batch_size').value = '512';
      document.getElementById('hp_steps_like').value = '500000';
      document.getElementById('hp_second_coeff').value = '0.005';
      document.getElementById('hp_third_coeff').value = '5000';
    }
  }

  function getNumber(id, fallback) {
    const raw = document.getElementById(id)?.value;
    const value = Number(raw);
    return Number.isFinite(value) ? value : fallback;
  }

  function buildTrainPayload() {
    const payload = {
      algo: selectedAlgo,
      total_timesteps: getNumber('hp_total_steps', 500000),
      env_config: window.LAHApp?.getState?.().envConfig || {},
      train_config: {
        learning_rate: getNumber('hp_learning_rate', 3e-4),
        batch_size: getNumber('hp_batch_size', selectedAlgo === 'PPO' ? 2048 : 512),
        gamma: getNumber('hp_gamma', 0.995),
        device: document.getElementById('hp_device')?.value || 'auto',
      },
    };

    if (selectedAlgo === 'PPO') {
      payload.train_config.n_steps = getNumber('hp_steps_like', 1024);
      payload.train_config.gae_lambda = getNumber('hp_second_coeff', 0.97);
      payload.train_config.ent_coef = getNumber('hp_third_coeff', 0.005);
    } else {
      payload.train_config.buffer_size = getNumber('hp_steps_like', 500000);
      payload.train_config.tau = getNumber('hp_second_coeff', 0.005);
      payload.train_config.learning_starts = getNumber('hp_third_coeff', 5000);
    }

    return payload;
  }

  async function checkConnection() {
    const { data, error } = await U.apiCall('/health');
    if (error) {
      U.toast(`Connection failed: ${error}`, 'error');
      logger?.error(`Connection check failed: ${error}`);
      return;
    }
    U.toast(`Server connected (${data.time || 'ok'})`, 'ok');
    logger?.ok(`Connection OK. version=${data.version || '-'}`);
  }

  async function startTraining() {
    const payload = buildTrainPayload();
    lastRequestedTotal = payload.total_timesteps;

    const { error } = await U.apiCall('/train/start', {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    if (error) {
      U.toast(`Training start failed: ${error}`, 'error', 5000);
      logger?.error(`Training start failed: ${error}`);
      return;
    }

    logger?.ok(`Training started. algo=${payload.algo}, device=${payload.train_config.device}, total_steps=${U.fmtInt(payload.total_timesteps)}`);
    setRunState(true);
    startPolling();
    refreshAll();
  }

  async function stopTraining() {
    const { error } = await U.apiCall('/train/stop', { method: 'POST' });
    if (error) {
      U.toast(`Stop failed: ${error}`, 'error', 5000);
      logger?.error(`Stop failed: ${error}`);
      return;
    }

    logger?.warn('Stop requested. Backend stops after the current step boundary.');
    refreshAll();
  }

  function setRunState(isRunning) {
    const startBtn = document.getElementById('trainStartBtn');
    const stopBtn = document.getElementById('trainStopBtn');
    const badge = document.getElementById('training-status-badge');

    if (startBtn) startBtn.disabled = isRunning;
    if (stopBtn) stopBtn.disabled = !isRunning;
    if (badge) {
      badge.className = isRunning ? 'badge badge-green' : 'badge badge-muted';
      badge.textContent = isRunning ? 'RUNNING' : 'IDLE';
    }
  }

  function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(refreshAll, 3000);
  }

  async function refreshAll() {
    await Promise.all([
      refreshTrainingStatus(),
      refreshComputeStatus(),
    ]);
  }

  async function refreshTrainingStatus() {
    const [statusResult, logResult] = await Promise.all([
      U.apiCall('/train/status'),
      U.apiCall('/train/log'),
    ]);

    if (statusResult.error) {
      logger?.error(`Training status failed: ${statusResult.error}`);
      return;
    }

    const status = statusResult.data || {};
    const entries = Array.isArray(logResult.data?.log) ? logResult.data.log : [];
    const latest = status.latest_log || entries[entries.length - 1] || null;
    const running = Boolean(status.is_training);

    if (entries.length < lastLogCount) {
      lastLogCount = 0;
      logger?.clear();
    }

    entries.slice(lastLogCount).forEach((entry) => {
      logger?.info(`[${entry.time || U.fmtNow()}] step=${U.fmtInt(entry.timestep || 0)} reward=${U.fmt(entry.avg_reward || 0, 3)}`);
    });
    lastLogCount = entries.length;

    updateRewardChart(entries);
    updateTrainingCards(status, latest, running);
  }

  function updateTrainingCards(status, latest, running) {
    const timestep = latest?.timestep || 0;
    const reward = latest?.avg_reward;

    U.setText('kpi_timestep', U.fmtInt(timestep));
    U.setText('kpi_reward', latest ? U.fmt(reward, 2) : '-');

    const pct = lastRequestedTotal > 0 ? Math.min(100, Math.round((timestep / lastRequestedTotal) * 100)) : 0;
    const progress = document.getElementById('trainingProgress');
    if (progress) progress.style.width = `${pct}%`;
    U.setText('progressPct', `${pct}%`);

    const appState = window.LAHApp?.getState?.();
    if (appState) {
      appState.trainingState.running = running;
      appState.trainingState.algorithm = status.algo || selectedAlgo;
      appState.trainingState.timestep = timestep;
      appState.trainingState.avgReward = reward || 0;
      window.LAHApp?.updateStatusBar?.();
    }

    setRunState(running);
  }

  async function refreshComputeStatus() {
    const { data, error } = await U.apiCall('/system/compute');
    if (error) {
      document.getElementById('computeError').textContent = error;
      logger?.error(`Compute monitor failed: ${error}`);
      return;
    }

    const trainer = data?.trainer || {};
    const gpu = data?.nvidia_smi?.gpus?.[0] || data?.devices?.[0] || null;
    const torchOk = Boolean(data?.torch_available) && !data?.torch_import_error;
    const cudaOk = Boolean(data?.cuda_available);
    const requested = trainer.device_request || document.getElementById('hp_device')?.value || 'auto';
    const resolved = trainer.model_device || '-';
    const runtimeBadge = document.getElementById('computeRuntimeBadge');
    const computeError = data?.torch_import_error || data?.nvidia_smi?.error || trainer.error || '';

    U.setText('computeTorch', torchOk ? `OK (${data.torch_version || '-'})` : 'ERROR');
    U.setText('computeCuda', cudaOk ? `YES (${data.cuda_version || '-'})` : 'NO');
    U.setText('computeRequested', requested);
    U.setText('computeResolved', resolved);
    U.setText('computeGpuName', gpu?.name || '-');
    U.setText('computeGpuUtil', gpu?.utilization_gpu !== undefined ? `${gpu.utilization_gpu}%` : '-');
    U.setText(
      'computeGpuMemory',
      gpu?.memory_used_mb !== undefined && gpu?.memory_total_mb !== undefined
        ? `${gpu.memory_used_mb} / ${gpu.memory_total_mb} MB`
        : (gpu?.total_memory_mb !== undefined ? `0 / ${gpu.total_memory_mb} MB` : '-')
    );
    U.setText(
      'computeDetail',
      trainer.algo
        ? `${trainer.algo} | training=${trainer.is_training ? 'yes' : 'no'} | logs=${trainer.log_entries ?? 0}`
        : 'trainer not created yet'
    );

    U.setText('kpi_device', resolved);
    U.setText(
      'kpi_gpu_mem',
      gpu?.memory_used_mb !== undefined && gpu?.memory_total_mb !== undefined
        ? `${gpu.memory_used_mb}MB`
        : '-'
    );

    if (runtimeBadge) {
      runtimeBadge.className = torchOk && cudaOk ? 'badge badge-green' : (computeError ? 'badge badge-red' : 'badge badge-yellow');
      runtimeBadge.textContent = torchOk && cudaOk ? 'CUDA READY' : (computeError ? 'CUDA ERROR' : 'CPU / UNKNOWN');
    }

    document.getElementById('computeError').textContent = computeError;
    if (computeError && computeError !== lastComputeError) {
      logger?.error(`Torch/CUDA issue detected: ${computeError}`);
    }
    lastComputeError = computeError || null;
  }

  function initChart() {
    Plotly.newPlot('rewardChart', [
      {
        x: [0],
        y: [0],
        type: 'scatter',
        mode: 'lines',
        name: 'avg_reward',
        line: { color: '#00d4ff', width: 1.8 },
        fill: 'tozeroy',
        fillcolor: 'rgba(0,212,255,0.08)',
      },
    ], {
      ...U.plotlyLayout({ margin: { l: 40, r: 12, t: 10, b: 30 } }),
      xaxis: { gridcolor: '#1e2a3d', color: '#475569', title: { text: 'Timestep', font: { size: 10 } } },
      yaxis: { gridcolor: '#1e2a3d', color: '#475569', title: { text: 'Avg Reward', font: { size: 10 } } },
    }, { responsive: true, displayModeBar: false });
  }

  function updateRewardChart(entries) {
    const steps = entries.map((entry) => entry.timestep || 0);
    const rewards = entries.map((entry) => entry.avg_reward || 0);
    Plotly.update('rewardChart', { x: [steps.length ? steps : [0]], y: [rewards.length ? rewards : [0]] }, {}, [0]);

    if (rewards.length) {
      const recent = rewards.slice(-Math.min(10, rewards.length));
      const ema = recent.reduce((sum, value) => sum + value, 0) / recent.length;
      U.setText('rewardEma', `EMA: ${U.fmt(ema, 2)}`);
    } else {
      U.setText('rewardEma', 'EMA: -');
    }
  }

  return { activate };
})();