/**
 * environment.js — LAH Mission Planner 환경 설정 탭 (탭 1)
 * 지형 설정(합성/DEM), Ref Path, Corridor, 비행체, 보상 가중치
 */

'use strict';

window.LAHEnvironment = (() => {
  const U = window.LAHUtils;
  let initialized = false;
  let canvas2D = null;
  let ctx2D = null;
  let terrainData = null;          // 합성 terrain float array
  let terrainMeta = null;          // DEM terrain metadata from API
  let refPath = [];
  let animFrame = null;
  let availableDems = [];          // DEM 파일 목록
  let coordMode = 'local';         // 'local' | 'latlon'
  let showRiskGlobal = true;
  let showCorridorGlobal = true;

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

            <!-- 합성 지형 설정 -->
            <div id="synthTerrainSettings">
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

            <!-- DEM 지형 설정 -->
            <div id="demTerrainSettings" style="display:none">
              <div class="form-group">
                <label class="form-label">DEM 프리셋</label>
                <div style="display:flex;gap:4px;flex-wrap:wrap">
                  <button class="btn btn-secondary btn-sm" id="demPresetHongik">홍익</button>
                  <button class="btn btn-secondary btn-sm" id="demPresetInje">인제</button>
                  <button class="btn btn-secondary btn-sm" id="demPresetJipo">지포</button>
                </div>
              </div>

              <div class="form-group">
                <label class="form-label">DEM 파일</label>
                <select class="input input-sm" id="demFileSelect" style="width:100%">
                  <option value="">-- 로딩 중... --</option>
                </select>
              </div>

              <div class="form-group">
                <label class="form-label">영역 중심 (lat, lon)</label>
                <div style="display:flex;gap:4px">
                  <input type="number" class="input input-sm mono" id="demCenterLat" placeholder="위도" step="0.01" style="flex:1">
                  <input type="number" class="input input-sm mono" id="demCenterLon" placeholder="경도" step="0.01" style="flex:1">
                </div>
              </div>

              <div class="form-group">
                <label class="form-label">DEM 크기 (km)</label>
                <div class="slider-row">
                  <input type="range" id="demSizeSlider" min="5" max="100" step="5" value="20" data-display="demSizeVal">
                  <span class="slider-value" id="demSizeVal">20</span>
                </div>
              </div>

              <div id="demMetaInfo" style="font-size:11px;color:var(--text-muted);margin-top:4px"></div>
            </div>

            <!-- 좌표 입력 모드 토글 -->
            <div class="form-group" style="margin-top:8px">
              <label class="form-label">웨이포인트 좌표 모드</label>
              <div class="radio-group" id="coordModeGroup">
                <label class="radio-btn active" id="coordModeLocalBtn">
                  <input type="radio" name="coordMode" value="local" checked> 로컈 (m)
                </label>
                <label class="radio-btn" id="coordModeLatLonBtn">
                  <input type="radio" name="coordMode" value="latlon"> WGS84 (lat/lon)
                </label>
              </div>
            </div>
          </div>

          <!-- Ref Path Settings -->
          <div class="form-section">
            <div class="form-section-title">Ref Path 설정</div>

            <!-- 로컈 모드 (m) -->
            <div id="waypointLocalMode">
              <div class="form-group">
                <label class="form-label">시작점 (x, y, z) [m]</label>
                <div class="grid-3" style="gap:var(--space-2)">
                  <input type="number" class="input input-sm mono" id="startX" value="1000" placeholder="X">
                  <input type="number" class="input input-sm mono" id="startY" value="1000" placeholder="Y">
                  <input type="number" class="input input-sm mono" id="startZ" value="400" placeholder="Z">
                </div>
              </div>

              <div class="form-group">
                <label class="form-label">끝점 (x, y, z) [m]</label>
                <div class="grid-3" style="gap:var(--space-2)">
                  <input type="number" class="input input-sm mono" id="endX" value="19000" placeholder="X">
                  <input type="number" class="input input-sm mono" id="endY" value="19000" placeholder="Y">
                  <input type="number" class="input input-sm mono" id="endZ" value="400" placeholder="Z">
                </div>
              </div>
            </div>

            <!-- WGS84 모드 (lat/lon) -->
            <div id="waypointLatLonMode" style="display:none">
              <div class="form-group">
                <label class="form-label">시작점 (lat, lon, alt[m])</label>
                <div class="grid-3" style="gap:var(--space-2)">
                  <input type="number" class="input input-sm mono" id="startLat" placeholder="위도" step="0.0001">
                  <input type="number" class="input input-sm mono" id="startLon" placeholder="경도" step="0.0001">
                  <input type="number" class="input input-sm mono" id="startAlt" placeholder="고도" value="400">
                </div>
              </div>

              <div class="form-group">
                <label class="form-label">끝점 (lat, lon, alt[m])</label>
                <div class="grid-3" style="gap:var(--space-2)">
                  <input type="number" class="input input-sm mono" id="endLat" placeholder="위도" step="0.0001">
                  <input type="number" class="input input-sm mono" id="endLon" placeholder="경도" step="0.0001">
                  <input type="number" class="input input-sm mono" id="endAlt" placeholder="고도" value="400">
                </div>
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
        document.querySelectorAll('.radio-btn').forEach(b => b.classList.remove('active'));
        const parent = r.closest('.radio-btn');
        if (parent) parent.classList.add('active');
        onTerrainTypeChange(r.value);
      });
    });

    // Regenerate on change
    ['mapSizeSlider', 'complexitySlider', 'resolutionSlider'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('input', U.debounce(generateTerrain, 300));
    });

    ['softCorridorSlider', 'hardCorridorSlider'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('input', () => drawPreview(showRiskGlobal, showCorridorGlobal));
    });

    // DEM 파일 선택시 메타 표시
    document.getElementById('demFileSelect')?.addEventListener('change', onDemFileSelect);

    // DEM 크기 슬라이더
    document.getElementById('demSizeSlider')?.addEventListener('input', U.debounce(() => {
      drawPreview(showRiskGlobal, showCorridorGlobal);
    }, 300));

    // DEM 프리셋 버튼
    document.getElementById('demPresetHongik')?.addEventListener('click', () => applyDemPreset('Hongik_48km'));
    document.getElementById('demPresetInje')?.addEventListener('click', () => applyDemPreset('Inje_48km'));
    document.getElementById('demPresetJipo')?.addEventListener('click', () => applyDemPreset('Jipo_48km'));

    // 좌표 모드 토글
    document.querySelectorAll('input[name="coordMode"]').forEach(r => {
      r.addEventListener('change', () => {
        document.querySelectorAll('#coordModeGroup .radio-btn').forEach(b => b.classList.remove('active'));
        const parent = r.closest('.radio-btn');
        if (parent) parent.classList.add('active');
        coordMode = r.value;
        onCoordModeChange(r.value);
      });
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
      terrainMeta = null;
      render();
    });

    // Toggles
    document.getElementById('toggleRiskBtn')?.addEventListener('click', () => {
      showRiskGlobal = !showRiskGlobal;
      drawPreview(showRiskGlobal, showCorridorGlobal);
    });
    document.getElementById('toggleCorridorBtn')?.addEventListener('click', () => {
      showCorridorGlobal = !showCorridorGlobal;
      drawPreview(showRiskGlobal, showCorridorGlobal);
    });

    // DEM 목록 로드
    loadDemList();
  }

  // ── Terrain Type Change ──
  function onTerrainTypeChange(type) {
    const synthDiv = document.getElementById('synthTerrainSettings');
    const demDiv   = document.getElementById('demTerrainSettings');
    if (synthDiv) synthDiv.style.display = type === 'synthetic' ? '' : 'none';
    if (demDiv)   demDiv.style.display   = type === 'dem'       ? '' : 'none';
    if (type === 'dem') {
      loadDemList();
    } else {
      generateTerrain();
    }
  }

  // ── Coord Mode Change ──
  function onCoordModeChange(mode) {
    const localDiv  = document.getElementById('waypointLocalMode');
    const latLonDiv = document.getElementById('waypointLatLonMode');
    if (localDiv)  localDiv.style.display  = mode === 'local'  ? '' : 'none';
    if (latLonDiv) latLonDiv.style.display = mode === 'latlon' ? '' : 'none';
  }

  // ── DEM List Loader ──
  async function loadDemList() {
    const { data, error } = await U.apiCall('/terrain/dem/list');
    if (error || !data) return;

    availableDems = data.dems || [];
    const sel = document.getElementById('demFileSelect');
    if (!sel) return;

    sel.innerHTML = '<option value="">개요 전용</option>';
    availableDems.forEach(d => {
      const opt = document.createElement('option');
      opt.value = d.path;
      const typeLabel = d.type === 'srtm_1arc' ? 'SRTM' : 'UTM48km';
      opt.textContent = `${d.name} [${typeLabel}, ${d.resolution_m}m, 엘브: ${d.elev_min}~${d.elev_max}m]`;
      sel.appendChild(opt);
    });
  }

  // ── DEM File Select Handler ──
  function onDemFileSelect() {
    const sel = document.getElementById('demFileSelect');
    if (!sel || !sel.value) return;
    const info = availableDems.find(d => d.path === sel.value);
    const metaEl = document.getElementById('demMetaInfo');
    if (!metaEl) return;
    if (!info) { metaEl.textContent = ''; return; }
    metaEl.innerHTML = [
      `분해능: ${info.resolution_m}m`,
      `크기: ${info.shape[0]}x${info.shape[1]} px`,
      `고도: ${info.elev_min}~${info.elev_max}m`,
      `CRS: ${info.crs}`,
    ].join(' &nbsp;|  ');
  }

  // ── DEM Preset ──
  function applyDemPreset(name) {
    const sel = document.getElementById('demFileSelect');
    if (!sel) return;
    const match = availableDems.find(d => d.name === name);
    if (match) {
      sel.value = match.path;
      onDemFileSelect();
      U.toast(`${name} 선택됨`, 'info');
    } else {
      // 사용 가능한 DEM에 없으면 이름으로 비주얼 매칭 시도
      for (let opt of sel.options) {
        if (opt.text.includes(name)) {
          sel.value = opt.value;
          onDemFileSelect();
          U.toast(`${name} 선택됨`, 'info');
          return;
        }
      }
      U.toast(`${name} 파일을 찾을 수 없습니다.`, 'warn');
    }
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
      const rect = canvas2D.getBoundingClientRect();
      const px = (e.clientX - rect.left) / rect.width;
      const py = (e.clientY - rect.top) / rect.height;

      if (terrainMeta && terrainMeta.world_extent) {
        // DEM 모드: 로컈 좌표 + lat/lon 병렴 표시
        const ext = terrainMeta.world_extent; // [xmin, xmax, ymin, ymax]
        const wx = Math.round(ext[0] + px * (ext[1] - ext[0]));
        const wy = Math.round(ext[2] + (1 - py) * (ext[3] - ext[2]));
        if (terrainMeta.crs && terrainMeta.crs.includes('32652') && terrainMeta.origin_utm) {
          const utmX = terrainMeta.origin_utm[0] + wx;
          const utmY = terrainMeta.origin_utm[1] + wy;
          const ll = U.coordConvert(0, 0, false, utmX, utmY);
          canvas2D.title = `X: ${wx}m, Y: ${wy}m | lat: ${ll.lat.toFixed(5)}, lon: ${ll.lon.toFixed(5)}`;
        } else {
          canvas2D.title = `X: ${wx}m, Y: ${wy}m`;
        }
      } else {
        const mapSize = parseInt(document.getElementById('mapSizeSlider')?.value || 20000);
        const wx = Math.round(px * mapSize);
        const wy = Math.round(py * mapSize);
        canvas2D.title = `X: ${wx}m, Y: ${wy}m`;
      }
    });
  }

  // ── Terrain Generation ──
  function generateTerrain() {
    const type = document.querySelector('input[name="terrainType"]:checked')?.value || 'synthetic';
    if (type === 'dem') {
      // DEM 모드에서는 서버 에서 데이터를 받아와 표시하므로 클라이언트 합성 괴늘리합니다.
      // 서버에서 받은 한초 데이터를 사용 (있으면)
      if (!terrainData && !terrainMeta) {
        // DEM 데이터 없음: 늘리 'DEM 초기화' 단추 표시
        drawPlaceholder();
        return;
      }
      drawPreview(showRiskGlobal, showCorridorGlobal);
    } else {
      const complexity = (parseFloat(document.getElementById('complexitySlider')?.value || 100)) / 100;
      const w = 200, h = 200;
      terrainData = U.generateSyntheticTerrain(w, h, complexity);
      terrainMeta = null;
      drawPreview(showRiskGlobal, showCorridorGlobal);
      renderCrossSection();
    }
  }

  // ── DEM 프리뷷 컜네버스 그리기 ──
  function drawPlaceholder() {
    if (!canvas2D || !ctx2D) return;
    const cw = canvas2D.width, ch = canvas2D.height;
    ctx2D.clearRect(0, 0, cw, ch);
    ctx2D.fillStyle = '#0a0f1a';
    ctx2D.fillRect(0, 0, cw, ch);
    ctx2D.fillStyle = '#475569';
    ctx2D.font = '13px monospace';
    ctx2D.textAlign = 'center';
    ctx2D.fillText('DEM 모드: 아래 "환경 초기화" 버튼을 눌러 DEM을 로드하세요.', cw / 2, ch / 2);
    ctx2D.textAlign = 'start';
  }

  // ── Draw 2D Preview ──
  function drawPreview(showRisk = true, showCorridor = true) {
    if (!canvas2D || !ctx2D) return;

    // DEM 모드: terrainMeta의 heightmap_preview 사용
    let renderData = terrainData;
    let renderW = 200, renderH = 200;
    if (terrainMeta && terrainMeta._heightmapPreview) {
      renderData = terrainMeta._heightmapPreview;
      renderW = terrainMeta._previewW;
      renderH = terrainMeta._previewH;
    }

    if (!renderData) {
      drawPlaceholder();
      return;
    }

    const cw = canvas2D.width, ch = canvas2D.height;
    const w = renderW, h = renderH;
    ctx2D.clearRect(0, 0, cw, ch);

    // Find min/max elevation
    let minH = Infinity, maxH = -Infinity;
    for (let row = 0; row < h; row++) {
      for (let col = 0; col < w; col++) {
        const v = Array.isArray(renderData[0]) ? renderData[row][col] : renderData[row * w + col];
        if (v < minH) minH = v;
        if (v > maxH) maxH = v;
      }
    }

    // Helper: 로우/콜 인덱스로 값 가져오기
    const getVal = (row, col) => Array.isArray(renderData[0]) ? renderData[row][col] : renderData[row * w + col];

    const cellW = cw / w, cellH = ch / h;

    // Draw terrain
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const elev = getVal(y, x);
        const color = U.heightColor(elev, minH, maxH);
        ctx2D.fillStyle = color;
        ctx2D.fillRect(x * cellW, y * cellH, Math.ceil(cellW), Math.ceil(cellH));
      }
    }

    // Risk overlay
    if (showRisk) {
      const imgData = ctx2D.getImageData(0, 0, cw, ch);
      // Subtle risk tint on high slopes
      for (let y = 1; y < h; y++) {
        for (let x = 1; x < w; x++) {
          const slope = Math.abs(getVal(y, x) - getVal(y - 1, x)) +
                        Math.abs(getVal(y, x) - getVal(y, x - 1));
          if (slope > 15) {
            const px = Math.round(y * cellH) * cw * 4 + Math.round(x * cellW) * 4;
            if (px + 3 < imgData.data.length) {
              imgData.data[px] = Math.min(255, imgData.data[px] + 60);
              imgData.data[px + 1] = Math.max(0, imgData.data[px + 1] - 20);
              imgData.data[px + 3] = 200;
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
    if (btn) { btn.disabled = true; btn.innerHTML = '<span style="opacity:.7">환경 초기화 중...</span>'; }

    const config = readConfig();
    const { data, error } = await U.apiCall('/env/init', {
      method: 'POST',
      body: JSON.stringify(config)
    });

    if (!error && data && data.status === 'ok') {
      U.toast('환경이 초기화되었습니다.', 'ok');

      // DEM 클라이언트 상태 업데이트
      if (data.terrain_meta) {
        terrainMeta = data.terrain_meta;

        // heightmap_preview 가콓 데이터 저장
        if (data.heightmap_preview) {
          const raw = data.heightmap_preview;
          terrainMeta._heightmapPreview = raw;
          terrainMeta._previewH = raw.length;
          terrainMeta._previewW = raw[0] ? raw[0].length : 0;
          terrainData = null;  // 합성 terrain 데이터 무효화
        }
      }

      // 앱 글로벌 상태 업데이트
      if (window.LAHApp) {
        const state = LAHApp.getState();
        state.terrain = { ...config, meta: data.terrain_meta };
        LAHApp.updateStatusBar();
      }

      // 캔버스 업데이트
      drawPreview(showRiskGlobal, showCorridorGlobal);

      // 단면도 업데이트 (DEM ref path 사용 시)
      if (data.ref_path_pts && data.ref_path_pts.length > 0) {
        renderCrossSectionFromPath(data.ref_path_pts, data.terrain_meta);
      } else {
        renderCrossSection();
      }

    } else if (error || !data) {
      // 에러 시: 데모 모드로 줄라가지 않고 메시지 표시
      U.toast(`환경 초기화 실패: ${error || '알 수 없는 오류'}`, 'error');
    }

    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 11-2.12-9.36L23 10"/></svg>
        환경 초기화
      `;
    }
  }

  // ── Config R/W ──
  function readConfig() {
    const terrainType = document.querySelector('input[name="terrainType"]:checked')?.value || 'synthetic';
    const cm = document.querySelector('input[name="coordMode"]:checked')?.value || 'local';

    const cfg = {
      terrain_type: terrainType,
      coord_mode: cm,
      soft_corridor_m: parseInt(document.getElementById('softCorridorSlider')?.value || 400),
      hard_corridor_m: parseInt(document.getElementById('hardCorridorSlider')?.value || 1000),
      agl_safe_min: parseFloat(document.getElementById('aglSafeMin')?.value || 120),
      agl_pref: parseFloat(document.getElementById('aglPreferred')?.value || 180),
    };

    if (terrainType === 'synthetic') {
      const mapSize = parseInt(document.getElementById('mapSizeSlider')?.value || 20000);
      const res = parseInt(document.getElementById('resolutionSlider')?.value || 30);
      cfg.terrain_size_x = Math.floor(mapSize / res);
      cfg.terrain_size_y = Math.floor(mapSize / res);
      cfg.terrain_resolution = res;
      cfg.terrain_seed = 42;
    } else {
      // DEM 설정
      const demFile = document.getElementById('demFileSelect')?.value || '';
      if (demFile) cfg.dem_file = demFile;
      const lat = parseFloat(document.getElementById('demCenterLat')?.value);
      const lon = parseFloat(document.getElementById('demCenterLon')?.value);
      if (!isNaN(lat)) cfg.dem_center_lat = lat;
      if (!isNaN(lon)) cfg.dem_center_lon = lon;
      cfg.dem_size_km = parseInt(document.getElementById('demSizeSlider')?.value || 20);
    }

    // 웨이포인트
    const waypoints = [];
    if (cm === 'latlon') {
      const sLat = parseFloat(document.getElementById('startLat')?.value);
      const sLon = parseFloat(document.getElementById('startLon')?.value);
      const sAlt = parseFloat(document.getElementById('startAlt')?.value || 400);
      const eLat = parseFloat(document.getElementById('endLat')?.value);
      const eLon = parseFloat(document.getElementById('endLon')?.value);
      const eAlt = parseFloat(document.getElementById('endAlt')?.value || 400);
      if (!isNaN(sLat) && !isNaN(sLon)) waypoints.push({ lat: sLat, lon: sLon, alt: sAlt });
      if (!isNaN(eLat) && !isNaN(eLon)) waypoints.push({ lat: eLat, lon: eLon, alt: eAlt });
    } else {
      const sx = parseFloat(document.getElementById('startX')?.value || 1000);
      const sy = parseFloat(document.getElementById('startY')?.value || 1000);
      const sz = parseFloat(document.getElementById('startZ')?.value || 400);
      const ex = parseFloat(document.getElementById('endX')?.value || 19000);
      const ey = parseFloat(document.getElementById('endY')?.value || 19000);
      const ez = parseFloat(document.getElementById('endZ')?.value || 400);
      waypoints.push({ x: sx, y: sy, z: sz });
      waypoints.push({ x: ex, y: ey, z: ez });
    }

    // 중간 웨이포인트 파싱
    const wpList = document.getElementById('waypointList');
    if (wpList) {
      wpList.querySelectorAll('div').forEach((row, i) => {
        const inputs = row.querySelectorAll('input[type="number"]');
        if (inputs.length >= 2) {
          if (cm === 'latlon') {
            const lat2 = parseFloat(inputs[0].value);
            const lon2 = parseFloat(inputs[1].value);
            const alt2 = inputs[2] ? parseFloat(inputs[2].value || 400) : 400;
            if (!isNaN(lat2) && !isNaN(lon2))
              waypoints.splice(waypoints.length - 1, 0, { lat: lat2, lon: lon2, alt: alt2 });
          } else {
            const x2 = parseFloat(inputs[0].value);
            const y2 = parseFloat(inputs[1].value);
            const z2 = inputs[2] ? parseFloat(inputs[2].value || 400) : 400;
            if (!isNaN(x2) && !isNaN(y2))
              waypoints.splice(waypoints.length - 1, 0, { x: x2, y: y2, z: z2 });
          }
        }
      });
    }
    if (waypoints.length >= 2) cfg.waypoints = waypoints;

    return cfg;
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

  // ── Cross-section from DEM ref path ──
  function renderCrossSectionFromPath(refPts, meta) {
    const el = document.getElementById('crossSectionChart');
    if (!el || !refPts || refPts.length < 2 || !meta) return;

    // 계산: 각 포인트의 누적 거리
    const x_vals = [0];
    for (let i = 1; i < refPts.length; i++) {
      const dx = refPts[i].x - refPts[i-1].x;
      const dy = refPts[i].y - refPts[i-1].y;
      x_vals.push(x_vals[x_vals.length-1] + Math.sqrt(dx*dx + dy*dy));
    }

    const z_vals = refPts.map(p => p.z || meta.mean_elev + 180);
    const aglMin = 120;
    const aglPref = meta.mean_elev ? meta.mean_elev + 180 : 380;

    // 지형 높이 가샘: 이상적으로 ref_path_pts는 z에 지형 고도와 AGL을 포함하면 좋았지만
    // 여기서는 mean_elev 기준 모의 지형 프로파일로 표시
    const terrain_vals = z_vals.map(z => Math.max(meta.min_elev, z - (meta.agl_pref || 180)));
    const safe_vals    = terrain_vals.map(h => h + aglMin);

    const traces = [
      {
        x: x_vals, y: terrain_vals,
        fill: 'tozeroy', fillcolor: 'rgba(74,100,80,0.4)',
        line: { color: '#4a7a55', width: 1 }, name: '지형', mode: 'lines',
      },
      {
        x: x_vals, y: safe_vals,
        line: { color: '#f59e0b', width: 1, dash: 'dot' }, name: 'AGL 최소', mode: 'lines',
      },
      {
        x: x_vals, y: z_vals,
        line: { color: '#00d4ff', width: 1.5 }, name: 'Ref 고도', mode: 'lines',
      },
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
