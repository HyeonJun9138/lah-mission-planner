/**
 * simulation.js — LAH Mission Planner 시뮬레이션 탭 (탭 3)
 * 에피소드 재생, 2D 뷰, 계기판
 */

'use strict';

window.LAHSimulation = (() => {
  const U = window.LAHUtils;
  let initialized = false;
  let canvas, ctx;
  let frames = [], currentFrame = 0;
  let isRunning = false, isPaused = false, playSpeed = 1;
  let animFrame = null, playInterval = null;
  let terrainData = null;
  let showRiskOverlay = false;
  let rotor1Angle = 0;

  function activate() {
    if (!initialized) {
      initialized = true;
      render();
      loadDemoEpisode();
    }
  }

  function render() {
    const container = document.getElementById('simulation-container');
    if (!container) return;

    const models = ['heuristic_tracker', 'heuristic_corridor', 'heuristic_hold', 'PPO_v1', 'PPO_v2'];

    container.innerHTML = `
      <div class="sim-layout" style="height:100%">

        <!-- Control Bar -->
        <div class="sim-controls-bar">
          <!-- Model Select -->
          <div style="display:flex;align-items:center;gap:var(--space-2)">
            <label class="form-label" style="margin:0;white-space:nowrap">모델:</label>
            <select class="input input-sm mono" id="simModelSelect" style="min-width:160px">
              ${models.map(m => `<option value="${m}">${m}</option>`).join('')}
            </select>
          </div>

          <div class="status-divider">|</div>

          <!-- Playback Controls -->
          <div style="display:flex;align-items:center;gap:var(--space-2)">
            <button class="btn btn-primary btn-sm" id="simRunBtn">▶ 실행</button>
            <button class="btn btn-secondary btn-sm" id="simPauseBtn" disabled>⏸ 일시정지</button>
            <button class="btn btn-secondary btn-sm" id="simStopBtn" disabled>⏹ 정지</button>
            <button class="btn btn-secondary btn-sm" id="simFastBtn">⏩ 빠르게</button>
          </div>

          <!-- Speed -->
          <div style="display:flex;align-items:center;gap:var(--space-2)">
            <span class="form-label" style="margin:0">속도:</span>
            <div class="flex gap-1">
              ${[1, 2, 5, 10].map(s => `
                <button class="btn btn-ghost btn-sm speed-btn ${s === 1 ? 'active' : ''}" data-speed="${s}" style="min-width:36px">x${s}</button>
              `).join('')}
            </div>
          </div>

          <!-- Step Counter -->
          <div class="mono" style="margin-left:auto;font-size:var(--text-sm);color:var(--text-secondary)">
            스텝: <span id="stepCounter" style="color:var(--cyan)">0</span> / <span id="totalSteps">0</span>
          </div>

          <!-- Risk Overlay Toggle -->
          <button class="btn btn-ghost btn-sm" id="toggleRiskOverlay">위험도 오버레이</button>
        </div>

        <!-- Main View -->
        <div class="sim-main">
          <!-- 2D Canvas -->
          <div class="sim-canvas-area" style="position:relative">
            <canvas id="simCanvas" style="width:100%;height:100%;display:block"></canvas>

            <!-- Overlay Info -->
            <div style="position:absolute;top:8px;left:8px;display:flex;flex-direction:column;gap:4px;pointer-events:none">
              <div style="background:rgba(10,14,23,0.8);border:1px solid var(--border-default);border-radius:4px;padding:4px 8px;font-size:10px;color:var(--text-secondary)">
                탑다운 뷰 · 지형 hillshade
              </div>
              <div id="coordOverlay" style="background:rgba(10,14,23,0.8);border:1px solid var(--border-default);border-radius:4px;padding:4px 8px;font-family:monospace;font-size:10px;color:var(--cyan)">
                X: — Y: — Z: —
              </div>
            </div>

            <!-- Scale Bar -->
            <div style="position:absolute;bottom:10px;left:10px;pointer-events:none">
              <div style="width:80px;height:2px;background:rgba(255,255,255,0.6)"></div>
              <div style="font-size:9px;color:rgba(255,255,255,0.6);font-family:monospace;margin-top:2px">1 km</div>
            </div>

            <!-- HUD corners -->
            <div class="hud-corner hud-corner-tl"></div>
            <div class="hud-corner hud-corner-tr"></div>
            <div class="hud-corner hud-corner-bl"></div>
            <div class="hud-corner hud-corner-br"></div>
          </div>

          <!-- Instruments Panel -->
          <div class="sim-instruments">

            <div class="instrument-card">
              <div class="instrument-label">AGL</div>
              <div class="instrument-value" id="inst_agl">—</div>
              <div class="instrument-unit">m</div>
              <div class="progress-bar" id="aglBar" style="width:100%;margin-top:4px">
                <div class="progress-fill" id="aglFill" style="width:60%"></div>
              </div>
            </div>

            <div class="instrument-card">
              <div class="instrument-label">속도</div>
              <div class="instrument-value" id="inst_speed">—</div>
              <div class="instrument-unit">m/s</div>
              <div class="progress-bar" style="width:100%;margin-top:4px">
                <div class="progress-fill" id="speedFill" style="width:50%"></div>
              </div>
            </div>

            <!-- Compass -->
            <div class="instrument-card">
              <div class="instrument-label">방위각</div>
              <canvas id="compassCanvas" width="80" height="80" style="display:block;margin:0 auto"></canvas>
              <div class="instrument-value" id="inst_heading" style="font-size:var(--text-md)">—°</div>
            </div>

            <div class="instrument-card">
              <div class="instrument-label">위험도</div>
              <div class="instrument-value" id="inst_risk" style="color:var(--green)">—</div>
              <div class="progress-bar danger" style="width:100%;margin-top:4px">
                <div class="progress-fill" id="riskFill" style="width:20%"></div>
              </div>
            </div>

            <div class="instrument-card">
              <div class="instrument-label">X-TRACK ERR</div>
              <div class="instrument-value" id="inst_xtrack" style="font-size:var(--text-md)">—</div>
              <div class="instrument-unit">m</div>
            </div>

            <div class="instrument-card">
              <div class="instrument-label">HOLD</div>
              <div id="holdStatus" style="font-size:var(--text-lg);font-weight:700;color:var(--text-muted)">OFF</div>
            </div>

            <div class="instrument-card">
              <div class="instrument-label">진행률</div>
              <div class="instrument-value" id="inst_progress" style="font-size:var(--text-md)">0%</div>
              <div class="progress-bar" style="width:100%;margin-top:4px">
                <div class="progress-fill" id="progressFill2" style="width:0%"></div>
              </div>
            </div>

            <div class="instrument-card">
              <div class="instrument-label">누적 REWARD</div>
              <div class="instrument-value" id="inst_reward" style="font-size:var(--text-md)">—</div>
            </div>
          </div>
        </div>

        <!-- Bottom Charts -->
        <div class="sim-charts">
          <div id="simTimeSeries" style="height:100%"></div>
        </div>
      </div>
    `;

    setupEventListeners();
    initCanvas();
    initTimeSeries();
  }

  // ── Event Listeners ──
  function setupEventListeners() {
    document.getElementById('simRunBtn')?.addEventListener('click', runEpisode);
    document.getElementById('simPauseBtn')?.addEventListener('click', togglePause);
    document.getElementById('simStopBtn')?.addEventListener('click', stopEpisode);
    document.getElementById('simFastBtn')?.addEventListener('click', () => setSpeed(Math.min(playSpeed * 2, 10)));
    document.getElementById('toggleRiskOverlay')?.addEventListener('click', () => {
      showRiskOverlay = !showRiskOverlay;
      drawFrame(currentFrame);
    });

    // Speed buttons
    document.querySelectorAll('.speed-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        setSpeed(parseInt(btn.dataset.speed));
      });
    });
  }

  // ── Canvas Init ──
  function initCanvas() {
    canvas = document.getElementById('simCanvas');
    if (!canvas) return;
    ctx = canvas.getContext('2d');

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width || 600;
      canvas.height = rect.height || 400;
      drawFrame(currentFrame);
    };
    resize();
    window.addEventListener('resize', U.debounce(resize, 200));

    canvas.addEventListener('mousemove', e => {
      if (!frames.length) return;
      const f = frames[currentFrame];
      const rect = canvas.getBoundingClientRect();
      const px = (e.clientX - rect.left) / rect.width;
      const py = (e.clientY - rect.top) / rect.height;
      const mapSize = 20000;
      const coord = document.getElementById('coordOverlay');
      if (coord) coord.textContent = `X: ${Math.round(px * mapSize)} Y: ${Math.round(py * mapSize)} Z: ${Math.round(f.z)}m`;
    });
  }

  // ── Load Demo Episode ──
  function loadDemoEpisode() {
    frames = U.generateMockEpisode(400);
    terrainData = U.generateSyntheticTerrain(200, 200, 1.0);
    U.setText('totalSteps', frames.length);
    drawFrame(0);
    updateTimeSeries(0);
  }

  // ── Episode Control ──
  async function runEpisode() {
    const model = document.getElementById('simModelSelect')?.value;

    // Try API first
    const { data, error } = await U.apiCall('/episode/run_heuristic', {
      method: 'POST',
      body: JSON.stringify({ model: model, n_steps: 400 })
    });

    if (data?.frames) {
      frames = data.frames;
    } else {
      // Demo mode
      frames = U.generateMockEpisode(U.randInt(300, 500));
    }

    terrainData = U.generateSyntheticTerrain(200, 200, U.randRange(0.7, 1.3));
    currentFrame = 0;
    U.setText('totalSteps', frames.length);

    isRunning = true; isPaused = false;
    document.getElementById('simRunBtn').disabled = true;
    document.getElementById('simPauseBtn').disabled = false;
    document.getElementById('simStopBtn').disabled = false;

    startPlayback();
  }

  function startPlayback() {
    if (playInterval) clearInterval(playInterval);
    const fps = 20 * playSpeed;
    const ms = 1000 / fps;

    playInterval = setInterval(() => {
      if (isPaused || !isRunning) return;
      currentFrame++;
      if (currentFrame >= frames.length) { stopEpisode(); return; }
      drawFrame(currentFrame);
      updateInstruments(frames[currentFrame]);
      updateTimeSeries(currentFrame);
    }, ms);
  }

  function togglePause() {
    isPaused = !isPaused;
    const btn = document.getElementById('simPauseBtn');
    if (btn) btn.textContent = isPaused ? '▶ 재개' : '⏸ 일시정지';
  }

  function stopEpisode() {
    isRunning = false; isPaused = false;
    clearInterval(playInterval);
    document.getElementById('simRunBtn').disabled = false;
    document.getElementById('simPauseBtn').disabled = true;
    document.getElementById('simStopBtn').disabled = true;
    const pauseBtn = document.getElementById('simPauseBtn');
    if (pauseBtn) pauseBtn.textContent = '⏸ 일시정지';
  }

  function setSpeed(s) {
    playSpeed = s;
    if (isRunning && !isPaused) { startPlayback(); }
  }

  // ── 2D Drawing ──
  function drawFrame(frameIdx) {
    if (!canvas || !ctx) return;
    const cw = canvas.width, ch = canvas.height;
    ctx.clearRect(0, 0, cw, ch);

    // Draw terrain
    if (terrainData) {
      const tw = 200, th = 200;
      let minH = Infinity, maxH = -Infinity;
      for (let i = 0; i < terrainData.length; i++) {
        if (terrainData[i] < minH) minH = terrainData[i];
        if (terrainData[i] > maxH) maxH = terrainData[i];
      }

      const cellW = cw / tw, cellH = ch / th;
      for (let y = 0; y < th; y++) {
        for (let x = 0; x < tw; x++) {
          const h = terrainData[y * tw + x];
          // Hillshade effect
          let lightness = (h - minH) / (maxH - minH);
          if (x > 0 && y > 0) {
            const dx = terrainData[y * tw + x] - terrainData[y * tw + (x - 1)];
            const dy = terrainData[y * tw + x] - terrainData[(y - 1) * tw + x];
            const shade = Math.max(0.2, 1 - (dx + dy) / 200);
            lightness *= shade;
          }
          const g = Math.round(30 + lightness * 80);
          const r = Math.round(20 + lightness * 50);
          const b = Math.round(10 + lightness * 30);
          ctx.fillStyle = `rgb(${r},${g},${b})`;
          ctx.fillRect(x * cellW, y * cellH, Math.ceil(cellW), Math.ceil(cellH));
        }
      }
    } else {
      ctx.fillStyle = '#0a1420';
      ctx.fillRect(0, 0, cw, ch);
    }

    // Risk overlay
    if (showRiskOverlay && frames.length > 0) {
      const img = ctx.getImageData(0, 0, cw, ch);
      // Apply simple risk tint based on terrain slope
      ctx.globalAlpha = 0.3;
      // Simplified: just show as overlay pattern
      ctx.globalAlpha = 1;
    }

    // Grid
    U.drawGrid(ctx, cw, ch, cw / 10);

    // Corridor
    const sw = parseInt(400 / 20000 * cw * 10);
    const hw = parseInt(1000 / 20000 * cw * 10);
    drawCorridor(ctx, cw, ch, sw, hw);

    // Ref path
    drawRefPath(ctx, cw, ch);

    // Actual path
    if (frames.length > 0 && frameIdx >= 0) {
      drawActualPath(ctx, cw, ch, frameIdx);
    }

    // Helicopter marker
    if (frames.length > 0 && frameIdx < frames.length) {
      const f = frames[frameIdx];
      const px = (f.x / 20000) * cw;
      const py = (f.y / 20000) * ch;
      U.drawHelicopterMarker(ctx, px, py, f.psi, 14, '#00d4ff');
    }

    // Step counter
    U.setText('stepCounter', frameIdx);
  }

  function drawCorridor(ctx, cw, ch, soft, hard) {
    // Draw simple center corridor line along diagonal
    const start = { x: cw * 0.05, y: ch * 0.05 };
    const end   = { x: cw * 0.95, y: ch * 0.95 };
    const dx = end.x - start.x, dy = end.y - start.y;
    const len = Math.sqrt(dx*dx + dy*dy);
    const nx = -dy / len, ny = dx / len;

    [hard, soft].forEach((w, i) => {
      const color = i === 0 ? 'rgba(255,107,53,0.12)' : 'rgba(0,212,255,0.10)';
      const stroke = i === 0 ? 'rgba(255,107,53,0.3)' : 'rgba(0,212,255,0.25)';
      ctx.beginPath();
      ctx.moveTo(start.x + nx * w, start.y + ny * w);
      ctx.lineTo(end.x + nx * w, end.y + ny * w);
      ctx.lineTo(end.x - nx * w, end.y - ny * w);
      ctx.lineTo(start.x - nx * w, start.y - ny * w);
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    });
  }

  function drawRefPath(ctx, cw, ch) {
    const pts = [
      { x: cw * 0.05, y: ch * 0.05 },
      { x: cw * 0.3,  y: ch * 0.2 },
      { x: cw * 0.5,  y: ch * 0.5 },
      { x: cw * 0.7,  y: ch * 0.75 },
      { x: cw * 0.95, y: ch * 0.95 },
    ];
    ctx.beginPath();
    pts.forEach((p, i) => { if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y); });
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([8, 4]);
    ctx.globalAlpha = 0.7;
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    // Start/end markers
    ctx.beginPath(); ctx.arc(pts[0].x, pts[0].y, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#4ade80'; ctx.fill();
    ctx.beginPath(); ctx.arc(pts[pts.length-1].x, pts[pts.length-1].y, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#ff6b35'; ctx.fill();
  }

  function drawActualPath(ctx, cw, ch, upTo) {
    if (upTo < 1) return;
    const n = Math.min(upTo, frames.length - 1);

    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const f = frames[i];
      const px = (f.x / 20000) * cw;
      const py = (f.y / 20000) * ch;

      if (i === 0) { ctx.moveTo(px, py); continue; }

      // Color by risk
      const risk = f.risk || 0.2;
      ctx.strokeStyle = U.riskColor(risk);
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = 0.85;
      ctx.lineTo(px, py);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(px, py);
    }
    ctx.globalAlpha = 1;
  }

  // ── Instruments ──
  function updateInstruments(f) {
    if (!f) return;

    U.setText('inst_agl', Math.round(f.agl));
    U.setText('inst_speed', U.fmt(f.v, 1));
    U.setText('inst_heading', Math.round(f.psi * 180 / Math.PI) + '°');
    U.setText('inst_risk', U.fmt(f.risk, 3));
    U.setText('inst_xtrack', Math.round(f.d));
    U.setText('inst_progress', Math.round((f.t / frames.length) * 100) + '%');
    U.setText('inst_reward', U.fmt(f.reward_cum, 1));

    // AGL bar
    const aglPct = Math.min(100, (f.agl / 400) * 100);
    const aglFill = document.getElementById('aglFill');
    if (aglFill) {
      aglFill.style.width = aglPct + '%';
      const aglBar = document.getElementById('aglBar');
      if (aglBar) aglBar.className = `progress-bar ${f.agl < 120 ? 'critical' : f.agl < 150 ? 'danger' : ''}`;
    }

    // Speed bar
    const spdFill = document.getElementById('speedFill');
    if (spdFill) spdFill.style.width = ((f.v / 70) * 100) + '%';

    // Risk
    const riskColor = U.riskColor(f.risk);
    const riskVal = document.getElementById('inst_risk');
    if (riskVal) riskVal.style.color = riskColor;
    const riskFill = document.getElementById('riskFill');
    if (riskFill) riskFill.style.width = (f.risk * 100) + '%';

    // Hold status
    const holdEl = document.getElementById('holdStatus');
    if (holdEl) {
      holdEl.textContent = f.hold_mode ? 'HOLD' : 'FLY';
      holdEl.style.color = f.hold_mode ? 'var(--yellow)' : 'var(--green)';
    }

    // Progress
    const progFill = document.getElementById('progressFill2');
    if (progFill) progFill.style.width = ((f.t / frames.length) * 100) + '%';

    // Compass
    drawCompass(f.psi);
  }

  function drawCompass(heading) {
    const c = document.getElementById('compassCanvas');
    if (!c) return;
    const ctx2 = c.getContext('2d');
    const w = c.width, h = c.height, cx = w / 2, cy = h / 2, r = w / 2 - 4;

    ctx2.clearRect(0, 0, w, h);

    // Background
    ctx2.beginPath(); ctx2.arc(cx, cy, r, 0, Math.PI * 2);
    ctx2.fillStyle = '#050810'; ctx2.fill();
    ctx2.strokeStyle = '#253448'; ctx2.lineWidth = 1; ctx2.stroke();

    // Cardinal marks
    const cardinals = ['N', 'E', 'S', 'W'];
    cardinals.forEach((c2, i) => {
      const a = (i * Math.PI / 2) - Math.PI / 2;
      const x = cx + Math.cos(a) * (r - 6);
      const y = cy + Math.sin(a) * (r - 6);
      ctx2.font = 'bold 9px monospace';
      ctx2.fillStyle = c2 === 'N' ? '#ff6b35' : '#475569';
      ctx2.textAlign = 'center';
      ctx2.textBaseline = 'middle';
      ctx2.fillText(c2, x, y);
    });

    // Heading needle
    ctx2.save();
    ctx2.translate(cx, cy);
    ctx2.rotate(heading);
    ctx2.beginPath();
    ctx2.moveTo(0, -r + 10);
    ctx2.lineTo(-4, 6); ctx2.lineTo(4, 6);
    ctx2.closePath();
    ctx2.fillStyle = '#00d4ff';
    ctx2.fill();
    ctx2.restore();

    // Center dot
    ctx2.beginPath(); ctx2.arc(cx, cy, 3, 0, Math.PI * 2);
    ctx2.fillStyle = '#94a3b8'; ctx2.fill();
  }

  // ── Time Series Charts ──
  function initTimeSeries() {
    const container = document.getElementById('simTimeSeries');
    if (!container) return;

    Plotly.newPlot('simTimeSeries', [
      { x: [], y: [], name: 'AGL (m)', line: { color: '#00d4ff', width: 1.5 }, yaxis: 'y' },
      { x: [], y: [], name: 'AGL 최소', line: { color: '#f59e0b', width: 1, dash: 'dot' }, yaxis: 'y' },
      { x: [], y: [], name: '속도', line: { color: '#4ade80', width: 1.5 }, yaxis: 'y2' },
      { x: [], y: [], name: '위험도', line: { color: '#ff6b35', width: 1.5 }, yaxis: 'y3' },
    ], {
      ...U.plotlyLayout({ margin: { l: 30, r: 30, t: 5, b: 20 } }),
      showlegend: true,
      legend: { orientation: 'h', x: 0, y: 1.15, font: { size: 8, color: '#94a3b8' }, bgcolor: 'transparent' },
      grid: { rows: 1, columns: 1 },
      yaxis:  { title: { text: 'AGL', font: { size: 8 } }, gridcolor: '#1e2a3d', tickfont: { size: 7 }, domain: [0.65, 1] },
      yaxis2: { title: { text: 'm/s', font: { size: 8 } }, gridcolor: '#1e2a3d', tickfont: { size: 7 }, domain: [0.33, 0.62], anchor: 'x', overlaying: false },
      yaxis3: { title: { text: 'risk', font: { size: 8 } }, gridcolor: '#1e2a3d', tickfont: { size: 7 }, domain: [0, 0.30], anchor: 'x', overlaying: false },
      xaxis: { gridcolor: '#1e2a3d', tickfont: { size: 7 } },
    }, { responsive: true, displayModeBar: false });
  }

  let tsUpdateThrottle = 0;
  function updateTimeSeries(frameIdx) {
    const now = Date.now();
    if (now - tsUpdateThrottle < 100) return;
    tsUpdateThrottle = now;

    const slice = frames.slice(0, frameIdx + 1);
    const x = slice.map(f => f.t);
    const agl = slice.map(f => f.agl);
    const aglMin = slice.map(() => 120);
    const spd = slice.map(f => f.v);
    const risk = slice.map(f => f.risk);

    Plotly.update('simTimeSeries', {
      x: [x, x, x, x],
      y: [agl, aglMin, spd, risk]
    }, {}, [0, 1, 2, 3]);
  }

  return { activate };
})();
