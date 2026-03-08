/**
 * environment.js — LAH Mission Planner 환경 설정 탭 (탭 1)
 * 지형 설정, Ref Path, Corridor, 비행체, 보상 가중치
 */

'use strict';

window.LAHEnvironment = (() => {
  const U = window.LAHUtils;
  let initialized = false;
  let canvas2D = null;
  let ctx2D = null;
  let terrainData = null;
  let refPath = [];
  let animFrame = null;

  // ── Activate ──
  function activate() {
    if (!initialized) {
      initialized = true;
      render();
    }
    startPreviewLoop();
  }

  // ── Render Layout ──
  function render() {
    const container = document.getElementById('environment-container');
    if (!container) return;

    container.innerHTML = `
      <div class="env-layout" style="height:100%">
        <!-- Left: Settings Panel -->
        <div class="env-settings">
          <div class="section-header">
            <h2 class="section-title">환경 설정</h2>
            <div class="flex gap-2">
              <button class="btn btn-secondary btn-sm" id="loadConfigBtn">📂 불러오기</button>
              <button class="btn btn-primary btn-sm" id="saveConfigBtn">💾 저장</button>
            </div>
          </div>

          <!-- Terrain Settings -->
          <div class="form-section">
            <div class="form-section-title">지형 설정</div>

            <div class="form-group">
              <label class="form-label">지형 유형</label>
              <div class="radio-group">
                <label class="radio-btn active" id="terrainSynthBtn">
                  <input type="radio" name="terrainType" value="synthetic" checked> 합성 지형
                </label>
                <label class="radio-btn" id="terrainDEMBtn">
                  <input type="radio" name="terrainType" value="dem"> DEM
                </label>
              </div>
            </div>

            <div class="form-group">
              <label class="form-label">맵 크기 (m)</label>
              <div class="slider-row">
                <input type="range" id="mapSizeSlider" min="5000" max="50000" step="1000" value="20000" data-display="mapSizeVal">
                <span class="slider-value" id="mapSizeVal">20000</span>
              </div>
            </div>

            <div class="form-group">
              <label class="form-label">해상도 (m/px)</label>
              <div class="slider-row">
                <input type="range" id="resolutionSlider" min="10" max="100" step="5" value="30" data-display="resolutionVal">
                <span class="slider-value" id="resolutionVal">30</span>
              </div>
            </div>

            <div class="form-group">
              <label class="form-label">지형 복잡도</label>
              <div class="slider-row">
                <input type="range" id="complexitySlider" min="10" max="200" step="5" value="100" data-display="complexityVal">
                <span class="slider-value" id="complexityVal" style="min-width:40px">1.00</span>
              </div>
            </div>
          </div>

          <!-- Ref Path Settings -->
          <div class="form-section">
            <div class="form-section-title">Ref Path 설정</div>

            <div class="form-group">
              <label class="form-label">시작점 (x, y, z)</label>
              <div class="grid-3" style="gap:var(--space-2)">
                <input type="number" class="input input-sm mono" id="startX" value="1000" placeholder="X">
                <input type="number" class="input input-sm mono" id="startY" value="1000" placeholder="Y">
                <input type="number" class="input input-sm mono" id="startZ" value="400" placeholder="Z">
              </div>
            </div>

            <div class="form-group">
              <label class="form-label">끝점 (x, y, z)</label>
              <div class="grid-3" style="gap:var(--space-2)">
                <input type="number" class="input input-sm mono" id="endX" value="19000" placeholder="X">
                <input type="number" class="input input-sm mono" id="endY" value="19000" placeholder="Y">
                <input type="number" class="input input-sm mono" id="endZ" value="400" placeholder="Z">
              </div>
            </div>

            <div class="form-group">
              <label class="form-label">중간 웨이포인트</label>
              <div id="waypointList" style="display:flex;flex-direction:column;gap:var(--space-2);margin-bottom:var(--space-2)"></div>
              <button class="btn btn-secondary btn-sm" id="addWaypointBtn" style="width:100%">+ 웨이포인트 추가</button>
            </div>

            <div class="form-group">
              <label class="form-label">리샘플링 간격</label>
              <div class="radio-group">
                <label class="radio-btn active"><input type="radio" name="resample" value="50" checked> 50m</label>
                <label class="radio-btn"><input type="radio" name="resample" value="100"> 100m</label>
              </div>
            </div>
          </div>

          <!-- Corridor Settings -->
          <div class="form-section">
            <div class="form-section-title">Corridor 설정</div>

            <div class="form-group">
              <label class="form-label">Soft Corridor (m)</label>
              <div class="slider-row">
                <input type="range" id="softCorridorSlider" min="100" max="1000" step="50" value="400" data-display="softCorridorVal">
                <span class="slider-value" id="softCorridorVal">400</span>
              </div>
            </div>

            <div class="form-group">
              <label class="form-label">Hard Corridor (m)</label>
              <div class="slider-row">
                <input type="range" id="hardCorridorSlider" min="500" max="3000" step="100" value="1000" data-display="hardCorridorVal">
                <span class="slider-value" id="hardCorridorVal">1000</span>
              </div>
            </div>
          </div>

          <!-- Aircraft Settings -->
          <div class="form-section">
            <div class="form-section-title">비행체 설정</div>
            <div class="grid-2" style="gap:var(--space-2)">
              ${[
                { id: 'vMin', label: 'v_min (m/s)', val: 0 },
                { id: 'vMax', label: 'v_max (m/s)', val: 70 },
                { id: 'vzMin', label: 'vz_min (m/s)', val: -6 },
                { id: 'vzMax', label: 'vz_max (m/s)', val: 6 },
                { id: 'maxTurnRate', label: '최대 선회율 (°/s)', val: 12 },
                { id: 'aglSafeMin', label: 'AGL 최소 안전고도 (m)', val: 120 },
                { id: 'aglPreferred', label: 'AGL 선호 고도 (m)', val: 180 },
              ].map(f => `
                <div class="form-group" style="margin-bottom:0">
                  <label class="form-label">${f.label}</label>
                  <input type="number" class="input input-sm mono" id="${f.id}" value="${f.val}">
                </div>
              `).join('')}
            </div>
          </div>

          <!-- Reward Weights -->
          <div class="form-section">
            <div class="form-section-title">보상 가중치</div>
            <div id="rewardSliders">
              ${[
                { id: 'w_prog',   label: 'w_prog',    val: 2.0,  hint: '진행 보상' },
                { id: 'w_goal',   label: 'w_goal',    val: 0.5,  hint: '목표 도달' },
                { id: 'w_d',      label: 'w_d',       val: 0.8,  hint: 'Cross-track 패널티' },
                { id: 'w_zref',   label: 'w_zref',    val: 0.2,  hint: '고도 기준 패널티' },
                { id: 'w_risk',   label: 'w_risk',    val: 1.0,  hint: '위험 패널티' },
                { id: 'w_risk_a', label: 'w_risk_a',  val: 0.7,  hint: '전방 위험 패널티' },
                { id: 'w_clear',  label: 'w_clear',   val: 3.0,  hint: 'AGL 여유 패널티' },
                { id: 'w_smooth', label: 'w_smooth',  val: 0.05, hint: '스무딩 패널티' },
              ].map(r => `
                <div class="form-group" data-tooltip="${r.hint}">
                  <div class="slider-row">
                    <span class="slider-label mono" style="min-width:64px">${r.label}</span>
                    <input type="range" id="rw_${r.id}" min="0" max="500" step="1" value="${Math.round(r.val * 100)}"
                      data-display="rv_${r.id}" data-scale="0.01">
                    <span class="slider-value" id="rv_${r.id}">${r.val.toFixed(2)}</span>
                  </div>
                </div>
              `).join('')}
            </div>
          </div>

          <!-- Actions -->
          <div style="display:flex;gap:var(--space-2);padding-bottom:var(--space-4)">
            <button class="btn btn-primary" id="initEnvBtn" style="flex:1">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 11-2.12-9.36L23 10"/></svg>
              환경 초기화
            </button>
            <button class="btn btn-danger btn-sm" id="resetConfigBtn">초기화</button>
          </div>
        </div>

        <!-- Right: Preview Panel -->
        <div class="env-preview">
          <div class="section-header">
            <h2 class="section-title">2D 지형 미리보기</h2>
            <div class="flex gap-2">
              <button class="btn btn-ghost btn-sm" id="toggleRiskBtn">위험도 표시</button>
              <button class="btn btn-ghost btn-sm" id="toggleCorridorBtn">Corridor 표시</button>
            </div>
          </div>

          <!-- 2D Canvas -->
          <div class="card" style="padding:0;overflow:hidden;flex-shrink:0">
            <canvas id="terrain2d" style="width:100%;height:360px;display:block;cursor:crosshair"></canvas>
          </div>

          <!-- Cross-section -->
          <div class="card">
            <div class="card-header">
              <span class="card-title">지형 단면도 (Ref Path 따라)</span>
            </div>
            <div id="crossSectionChart" style="height:140px"></div>
          </div>

          <!-- Config Summary -->
          <div class="card">
            <div class="card-header"><span class="card-title">현재 설정 요약</span></div>
            <div id="configSummary" class="grid-2" style="gap:var(--space-2)">
              ${[
                ['지형 유형', '합성 지형'], ['맵 크기', '20,000 m'],
                ['해상도', '30 m/px'], ['Soft Corridor', '±400 m'],
                ['Hard Corridor', '±1,000 m'], ['AGL 최소', '120 m'],
                ['v_max', '70 m/s'], ['학습 알고리즘', 'PPO'],
              ].map(([k, v]) => `
                <div style="display:flex;justify-content:space-between;font-size:var(--text-xs)">
                  <span style="color:var(--text-muted)">${k}</span>
                  <span class="mono" style="color:var(--text-primary)">${v}</span>
                </div>
              `).join('')}
            </div>
          </div>
        </div>
      </div>
    `;

    setupEventListeners();
    bindSliders();
    initCanvas();
    generateTerrain();
    renderCrossSection();
  }

  // ── Event Listeners ──
  function setupEventListeners() {
    // Terrain type toggle
    document.querySelectorAll('input[name="terrainType"]').forEach(r => {
      r.addEventListener('change', () => {
        document.querySelectorAll('[id$="Btn"].radio-btn').forEach(b => b.classList.remove('active'));
        const parent = r.closest('.radio-btn');
        if (parent) parent.classList.add('active');
        generateTerrain();
      });
    });

    // Regenerate on change
    ['mapSizeSlider', 'complexitySlider', 'resolutionSlider'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('input', U.debounce(generateTerrain, 300));
    });

    ['softCorridorSlider', 'hardCorridorSlider'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('input', () => drawPreview());
    });

    // Add waypoint
    const addBtn = document.getElementById('addWaypointBtn');
    if (addBtn) addBtn.addEventListener('click', addWaypoint);

    // Init env
    const initBtn = document.getElementById('initEnvBtn');
    if (initBtn) initBtn.addEventListener('click', initEnv);

    // Save/Load
    document.getElementById('saveConfigBtn')?.addEventListener('click', saveConfig);
    document.getElementById('loadConfigBtn')?.addEventListener('click', loadConfig);

    // Reset
    document.getElementById('resetConfigBtn')?.addEventListener('click', () => {
      U.toast('설정이 기본값으로 초기화되었습니다.', 'info');
      render();
    });

    // Toggles
    let showRisk = true, showCorridor = true;
    document.getElementById('toggleRiskBtn')?.addEventListener('click', () => {
      showRisk = !showRisk;
      drawPreview(showRisk, showCorridor);
    });
    document.getElementById('toggleCorridorBtn')?.addEventListener('click', () => {
      showCorridor = !showCorridor;
      drawPreview(showRisk, showCorridor);
    });
  }

  // ── Bind Sliders ──
  function bindSliders() {
    // Standard sliders
    document.querySelectorAll('input[type="range"][data-display]').forEach(slider => {
      const displayId = slider.dataset.display;
      const display = document.getElementById(displayId);
      if (!display) return;
      const scale = parseFloat(slider.dataset.scale) || 1;
      const update = () => {
        const val = parseFloat(slider.value) * scale;
        display.textContent = scale < 1 ? val.toFixed(2) : val;
      };
      slider.addEventListener('input', update);
      update();
    });
  }

  // ── Canvas Init ──
  function initCanvas() {
    canvas2D = document.getElementById('terrain2d');
    if (!canvas2D) return;
    ctx2D = canvas2D.getContext('2d');

    const resize = () => {
      const rect = canvas2D.getBoundingClientRect();
      canvas2D.width = rect.width || 600;
      canvas2D.height = 360;
      drawPreview();
    };
    resize();
    window.addEventListener('resize', U.debounce(resize, 200));

    // Mouse coordinate display
    canvas2D.addEventListener('mousemove', e => {
      if (!terrainData) return;
      const rect = canvas2D.getBoundingClientRect();
      const px = (e.clientX - rect.left) / rect.width;
      const py = (e.clientY - rect.top) / rect.height;
      const mapSize = parseInt(document.getElementById('mapSizeSlider')?.value || 20000);
      const wx = Math.round(px * mapSize);
      const wy = Math.round(py * mapSize);
      canvas2D.title = `X: ${wx}m, Y: ${wy}m`;
    });
  }

  // ── Terrain Generation ──
  function generateTerrain() {
    const complexity = (parseFloat(document.getElementById('complexitySlider')?.value || 100)) / 100;
    const w = 200, h = 200;
    terrainData = U.generateSyntheticTerrain(w, h, complexity);
    drawPreview();
    renderCrossSection();
  }

  // ── Draw 2D Preview ──
  function drawPreview(showRisk = true, showCorridor = true) {
    if (!canvas2D || !ctx2D || !terrainData) return;

    const cw = canvas2D.width, ch = canvas2D.height;
    const w = 200, h = 200;
    ctx2D.clearRect(0, 0, cw, ch);

    // Find min/max elevation
    let minH = Infinity, maxH = -Infinity;
    for (let i = 0; i < terrainData.length; i++) {
      if (terrainData[i] < minH) minH = terrainData[i];
      if (terrainData[i] > maxH) maxH = terrainData[i];
    }

    const cellW = cw / w, cellH = ch / h;

    // Draw terrain
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const elev = terrainData[y * w + x];
        const color = U.heightColor(elev, minH, maxH);
        ctx2D.fillStyle = color;
        ctx2D.fillRect(x * cellW, y * cellH, Math.ceil(cellW), Math.ceil(cellH));
      }
    }

    // Risk overlay
    if (showRisk) {
      const imgData = ctx2D.getImageData(0, 0, cw, ch);
      // Subtle risk tint on high slopes
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const i = y * w + x;
          if (x > 0 && y > 0) {
            const slope = Math.abs(terrainData[i] - terrainData[(y-1)*w + x]) +
                          Math.abs(terrainData[i] - terrainData[y*w + (x-1)]);
            if (slope > 15) {
              const px = Math.round(y * cellH) * cw * 4 + Math.round(x * cellW) * 4;
              if (px + 3 < imgData.data.length) {
                imgData.data[px] = Math.min(255, imgData.data[px] + 60);
                imgData.data[px+1] = Math.max(0, imgData.data[px+1] - 20);
                imgData.data[px+3] = 200;
              }
            }
          }
        }
      }
      ctx2D.putImageData(imgData, 0, 0);
    }

    // Grid overlay
    U.drawGrid(ctx2D, cw, ch, cw / 10);

    // Ref Path
    const softC = parseInt(document.getElementById('softCorridorSlider')?.value || 400);
    const hardC = parseInt(document.getElementById('hardCorridorSlider')?.value || 1000);
    const mapSize = parseInt(document.getElementById('mapSizeSlider')?.value || 20000);

    const toScreen = pt => ({
      x: (pt.x / mapSize) * cw,
      y: (pt.y / mapSize) * ch
    });

    const startX = parseFloat(document.getElementById('startX')?.value || 1000);
    const startY = parseFloat(document.getElementById('startY')?.value || 1000);
    const endX = parseFloat(document.getElementById('endX')?.value || 19000);
    const endY = parseFloat(document.getElementById('endY')?.value || 19000);

    refPath = U.generateRefPath(startX, startY, 400, endX, endY, 400, 5);
    const pathPts = refPath.map(toScreen);

    // Corridor
    if (showCorridor) {
      // Hard corridor
      drawCorridorBand(ctx2D, pathPts, (hardC / mapSize) * cw, 'rgba(255,107,53,0.10)', 'rgba(255,107,53,0.4)');
      // Soft corridor
      drawCorridorBand(ctx2D, pathPts, (softC / mapSize) * cw, 'rgba(0,212,255,0.08)', 'rgba(0,212,255,0.3)');
    }

    // Path line
    if (pathPts.length > 1) {
      ctx2D.beginPath();
      ctx2D.moveTo(pathPts[0].x, pathPts[0].y);
      pathPts.forEach(p => ctx2D.lineTo(p.x, p.y));
      ctx2D.strokeStyle = '#00d4ff';
      ctx2D.lineWidth = 2;
      ctx2D.setLineDash([8, 4]);
      ctx2D.stroke();
      ctx2D.setLineDash([]);
    }

    // Waypoints
    pathPts.forEach((p, i) => {
      ctx2D.beginPath();
      ctx2D.arc(p.x, p.y, i === 0 || i === pathPts.length - 1 ? 6 : 4, 0, Math.PI * 2);
      ctx2D.fillStyle = i === 0 ? '#4ade80' : i === pathPts.length - 1 ? '#ff6b35' : '#00d4ff';
      ctx2D.fill();
    });

    // Labels
    ctx2D.font = '11px monospace';
    ctx2D.fillStyle = '#4ade80';
    ctx2D.fillText('시작', pathPts[0].x + 8, pathPts[0].y + 4);
    ctx2D.fillStyle = '#ff6b35';
    ctx2D.fillText('목표', pathPts[pathPts.length-1].x + 8, pathPts[pathPts.length-1].y + 4);

    // Scale bar
    const barWidth = 60;
    const scale = mapSize / cw;
    ctx2D.fillStyle = '#64748b';
    ctx2D.fillRect(10, ch - 20, barWidth, 2);
    ctx2D.font = '9px monospace';
    ctx2D.fillStyle = '#64748b';
    ctx2D.fillText(`${Math.round(barWidth * scale)}m`, 10, ch - 6);
  }

  function drawCorridorBand(ctx, points, width, fillColor, strokeColor) {
    if (points.length < 2) return;
    ctx.save();
    ctx.beginPath();

    // Left side
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      let nx = 0, ny = 1;
      if (i < points.length - 1) {
        const dx = points[i+1].x - p.x, dy = points[i+1].y - p.y;
        const len = Math.sqrt(dx*dx + dy*dy) || 1;
        nx = -dy / len; ny = dx / len;
      }
      const lx = p.x + nx * width, ly = p.y + ny * width;
      if (i === 0) ctx.moveTo(lx, ly); else ctx.lineTo(lx, ly);
    }

    // Right side (reverse)
    for (let i = points.length - 1; i >= 0; i--) {
      const p = points[i];
      let nx = 0, ny = 1;
      if (i < points.length - 1) {
        const dx = points[i+1].x - p.x, dy = points[i+1].y - p.y;
        const len = Math.sqrt(dx*dx + dy*dy) || 1;
        nx = -dy / len; ny = dx / len;
      }
      const rx = p.x - nx * width, ry = p.y - ny * width;
      ctx.lineTo(rx, ry);
    }

    ctx.closePath();
    ctx.fillStyle = fillColor;
    ctx.fill();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 0.5;
    ctx.stroke();
    ctx.restore();
  }

  // ── Cross-Section Chart ──
  function renderCrossSection() {
    const el = document.getElementById('crossSectionChart');
    if (!el || !terrainData) return;

    const n = 100;
    const x_vals = Array.from({ length: n }, (_, i) => i * 200);
    const terrain_vals = [], refAlt = [], safeMin = [];
    const aglMin = 120, aglPref = 180;

    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      const tx = Math.floor(t * 199), ty = Math.floor(t * 199);
      const h = terrainData[ty * 200 + tx];
      terrain_vals.push(h);
      refAlt.push(h + aglPref + Math.sin(i * 0.1) * 20);
      safeMin.push(h + aglMin);
    }

    const traces = [
      { x: x_vals, y: terrain_vals, fill: 'tozeroy', fillcolor: 'rgba(74,100,80,0.4)', line: { color: '#4a7a55', width: 1 }, name: '지형', mode: 'lines' },
      { x: x_vals, y: safeMin, line: { color: '#f59e0b', width: 1, dash: 'dot' }, name: 'AGL 최소', mode: 'lines' },
      { x: x_vals, y: refAlt, line: { color: '#00d4ff', width: 1.5 }, name: 'Ref 고도', mode: 'lines' },
    ];

    Plotly.newPlot(el, traces, {
      ...U.plotlyLayout({
        margin: { l: 35, r: 5, t: 5, b: 25 },
        showlegend: true,
        legend: { orientation: 'h', y: -0.3, font: { size: 9, color: '#94a3b8' }, bgcolor: 'transparent' },
        xaxis: { title: { text: '경로 거리 (m)', font: { size: 9 } }, gridcolor: '#1e2a3d', color: '#475569', tickfont: { size: 8 } },
        yaxis: { title: { text: '고도 (m)', font: { size: 9 } }, gridcolor: '#1e2a3d', color: '#475569', tickfont: { size: 8 } },
      })
    }, { responsive: true, displayModeBar: false });
  }

  // ── Waypoint Management ──
  let waypointCount = 0;
  function addWaypoint() {
    const list = document.getElementById('waypointList');
    if (!list) return;
    waypointCount++;
    const div = U.create('div', { style: 'display:flex;gap:6px;align-items:center' });
    div.innerHTML = `
      <span style="color:var(--text-muted);font-size:11px;min-width:16px">WP${waypointCount}</span>
      <input type="number" class="input input-sm mono" placeholder="X" style="flex:1">
      <input type="number" class="input input-sm mono" placeholder="Y" style="flex:1">
      <input type="number" class="input input-sm mono" placeholder="Z" style="flex:1;max-width:64px">
      <button class="btn btn-ghost btn-icon" style="color:var(--red);padding:4px" onclick="this.parentElement.remove()">✕</button>
    `;
    list.appendChild(div);
  }

  // ── Init Environment ──
  async function initEnv() {
    const btn = document.getElementById('initEnvBtn');
    if (btn) { btn.disabled = true; btn.textContent = '초기화 중...'; }

    const config = readConfig();
    const { data, error } = await U.apiCall('/env/init', {
      method: 'POST',
      body: JSON.stringify(config)
    });

    if (!error && data) {
      U.toast('환경이 초기화되었습니다.', 'ok');
      const state = LAHApp.getState();
      state.terrain = config;
      LAHApp.updateStatusBar();
    } else {
      // Demo mode — simulate success
      U.toast('데모 모드: 환경 초기화 시뮬레이션됨', 'info');
      const appState = LAHApp.getState();
      appState.terrain = config;
      LAHApp.updateStatusBar();
    }

    if (btn) { btn.disabled = false; btn.textContent = '환경 초기화'; }
    generateTerrain();
  }

  // ── Config R/W ──
  function readConfig() {
    return {
      terrainType: document.querySelector('input[name="terrainType"]:checked')?.value || 'synthetic',
      mapSize: parseInt(document.getElementById('mapSizeSlider')?.value || 20000),
      resolution: parseInt(document.getElementById('resolutionSlider')?.value || 30),
      complexity: (parseInt(document.getElementById('complexitySlider')?.value || 100)) / 100,
      startPoint: {
        x: parseFloat(document.getElementById('startX')?.value || 1000),
        y: parseFloat(document.getElementById('startY')?.value || 1000),
        z: parseFloat(document.getElementById('startZ')?.value || 400),
      },
      endPoint: {
        x: parseFloat(document.getElementById('endX')?.value || 19000),
        y: parseFloat(document.getElementById('endY')?.value || 19000),
        z: parseFloat(document.getElementById('endZ')?.value || 400),
      },
      softCorridor: parseInt(document.getElementById('softCorridorSlider')?.value || 400),
      hardCorridor: parseInt(document.getElementById('hardCorridorSlider')?.value || 1000),
      aglSafeMin: parseFloat(document.getElementById('aglSafeMin')?.value || 120),
      aglPreferred: parseFloat(document.getElementById('aglPreferred')?.value || 180),
    };
  }

  function saveConfig() {
    const config = readConfig();
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `lah_env_config_${Date.now()}.json`;
    a.click();
    U.toast('설정이 저장되었습니다.', 'ok');
  }

  function loadConfig() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = e => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = evt => {
        try {
          // JSON.parse(evt.target.result);
          U.toast('설정을 불러왔습니다.', 'ok');
        } catch {
          U.toast('잘못된 설정 파일입니다.', 'error');
        }
      };
      reader.readAsText(file);
    };
    input.click();
  }

  // ── Preview Loop ──
  function startPreviewLoop() {
    stopPreviewLoop();
    // Only redraw on demand - no continuous loop needed
  }

  function stopPreviewLoop() {
    if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
  }

  return { activate };
})();
