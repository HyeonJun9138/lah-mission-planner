/**
 * visualization3d.js — LAH Mission Planner 3D 시각화 탭 (탭 4)
 * Three.js 기반 3D 씬, LAH 헬기 모델, 지형 메쉬
 */

'use strict';

window.LAHViz3D = (() => {
  const U = window.LAHUtils;
  let initialized = false;
  let renderer, scene, camera, controls;
  let animId = null;
  let isActive = false;

  // Scene objects
  let terrainMesh, refPathLine, flightPathLine;
  let helicopterGroup, mainRotor, tailRotor;
  let corridorTube;
  let clock;

  // Episode data
  let frames = [], currentFrame = 0;
  let isPlaying = false;
  let playInterval = null;

  // Camera modes
  const CAM_FREE = 'free', CAM_FOLLOW = 'follow', CAM_TOP = 'top', CAM_COCKPIT = 'cockpit';
  let camMode = CAM_FREE;

  function activate() {
    if (!initialized) {
      initialized = true;
      render();
      setTimeout(initThree, 100);
    }
    isActive = true;
    startRenderLoop();
  }

  function deactivate() {
    isActive = false;
    stopRenderLoop();
  }

  // ── UI Render ──
  function render() {
    const container = document.getElementById('viz3d-container');
    if (!container) return;

    container.innerHTML = `
      <div class="viz3d-layout" style="height:100%">

        <!-- 3D Scene -->
        <div class="viz3d-scene" id="viz3dScene">
          <canvas id="three-canvas"></canvas>

          <!-- Camera controls bar -->
          <div style="position:absolute;top:8px;left:8px;display:flex;gap:6px;z-index:10">
            <button class="btn btn-secondary btn-sm cam-btn active" data-cam="free">자유 시점</button>
            <button class="btn btn-secondary btn-sm cam-btn" data-cam="follow">추적</button>
            <button class="btn btn-secondary btn-sm cam-btn" data-cam="top">탑다운</button>
            <button class="btn btn-secondary btn-sm cam-btn" data-cam="cockpit">1인칭</button>
          </div>

          <!-- Playback bar -->
          <div style="position:absolute;bottom:8px;left:8px;right:220px;display:flex;align-items:center;gap:8px;z-index:10">
            <button class="btn btn-primary btn-sm" id="viz3dPlayBtn">▶</button>
            <button class="btn btn-secondary btn-sm" id="viz3dPauseBtn" disabled>⏸</button>
            <button class="btn btn-secondary btn-sm" id="viz3dStopBtn" disabled>⏹</button>
            <input type="range" id="viz3dTimeline" min="0" max="100" value="0" style="flex:1">
            <span class="mono" style="font-size:11px;color:var(--cyan);min-width:60px">
              <span id="viz3dStep">0</span> / <span id="viz3dTotal">0</span>
            </span>
          </div>

          <!-- HUD corners -->
          <div class="hud-corner hud-corner-tl"></div>
          <div class="hud-corner hud-corner-tr"></div>
          <div class="hud-corner hud-corner-bl"></div>
          <div class="hud-corner hud-corner-br"></div>

          <!-- Telemetry overlay -->
          <div style="position:absolute;top:8px;right:228px;background:rgba(10,14,23,0.8);border:1px solid var(--border-default);border-radius:6px;padding:8px 12px;font-family:monospace;font-size:11px;z-index:10">
            <div style="color:var(--text-muted);margin-bottom:4px">TELEMETRY</div>
            <div>ALT: <span id="t3d_alt" style="color:var(--cyan)">—</span> m</div>
            <div>SPD: <span id="t3d_spd" style="color:var(--green)">—</span> m/s</div>
            <div>HDG: <span id="t3d_hdg" style="color:var(--yellow)">—</span>°</div>
            <div>RSK: <span id="t3d_risk" style="color:var(--orange)">—</span></div>
          </div>
        </div>

        <!-- Right Controls Panel -->
        <div class="viz3d-controls">
          <div style="font-size:var(--text-xs);font-weight:600;color:var(--cyan);letter-spacing:0.08em;text-transform:uppercase;margin-bottom:var(--space-3)">
            3D 시각화 제어
          </div>

          <!-- Overlay Toggles -->
          <div class="form-section">
            <div class="form-section-title" style="font-size:10px">오버레이</div>
            <div style="display:flex;flex-direction:column;gap:8px">
              ${[
                { id: 'tog_terrain', label: '지형 메쉬', checked: true },
                { id: 'tog_refpath', label: 'Ref Path', checked: true },
                { id: 'tog_flight',  label: '비행 경로', checked: true },
                { id: 'tog_corridor', label: 'Corridor', checked: true },
                { id: 'tog_waypts',  label: '웨이포인트', checked: true },
                { id: 'tog_fog',     label: '안개 효과', checked: true },
                { id: 'tog_grid',    label: '고도 그리드', checked: false },
              ].map(t => `
                <label class="checkbox-label" style="font-size:11px">
                  <input type="checkbox" id="${t.id}" ${t.checked ? 'checked' : ''}>
                  ${t.label}
                </label>
              `).join('')}
            </div>
          </div>

          <!-- Episode Load -->
          <div class="form-section">
            <div class="form-section-title" style="font-size:10px">에피소드</div>
            <button class="btn btn-primary btn-sm" style="width:100%;margin-bottom:6px" id="viz3dLoadDemo">
              데모 에피소드 로드
            </button>
            <button class="btn btn-secondary btn-sm" style="width:100%" id="viz3dLoadFile">
              파일에서 불러오기
            </button>
          </div>

          <!-- Terrain Config -->
          <div class="form-section">
            <div class="form-section-title" style="font-size:10px">지형 설정</div>
            <div class="form-group" style="margin-bottom:8px">
              <label class="form-label" style="font-size:10px">세그먼트 수</label>
              <input type="range" id="terrainSegments" min="30" max="150" value="80" style="width:100%">
            </div>
            <div class="form-group" style="margin-bottom:0">
              <label class="form-label" style="font-size:10px">고도 스케일</label>
              <input type="range" id="terrainScale" min="1" max="10" value="3" style="width:100%">
            </div>
          </div>

          <!-- Render Stats -->
          <div class="card card-sm" style="margin-top:auto">
            <div style="font-size:10px;color:var(--text-muted)">렌더 통계</div>
            <div style="font-family:monospace;font-size:10px;margin-top:4px">
              <div>FPS: <span id="renderFps" style="color:var(--cyan)">—</span></div>
              <div>삼각형: <span id="renderTris" style="color:var(--green)">—</span></div>
            </div>
          </div>
        </div>
      </div>
    `;

    setupVizControls();
  }

  // ── Controls ──
  function setupVizControls() {
    // Camera mode buttons
    document.querySelectorAll('.cam-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.cam-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        setCameraMode(btn.dataset.cam);
      });
    });

    // Overlay toggles
    ['tog_refpath', 'tog_flight', 'tog_corridor', 'tog_fog', 'tog_terrain', 'tog_waypts'].forEach(id => {
      document.getElementById(id)?.addEventListener('change', e => {
        updateVisibility(id, e.target.checked);
      });
    });

    // Playback
    document.getElementById('viz3dPlayBtn')?.addEventListener('click', startPlayback);
    document.getElementById('viz3dPauseBtn')?.addEventListener('click', togglePlayback);
    document.getElementById('viz3dStopBtn')?.addEventListener('click', stopPlayback);

    // Timeline slider
    document.getElementById('viz3dTimeline')?.addEventListener('input', e => {
      const idx = Math.round(parseFloat(e.target.value));
      currentFrame = Math.min(idx, frames.length - 1);
      updateHelicopterPosition(currentFrame);
      U.setText('viz3dStep', currentFrame);
    });

    // Load demo
    document.getElementById('viz3dLoadDemo')?.addEventListener('click', loadDemoEpisode);
    document.getElementById('viz3dLoadFile')?.addEventListener('click', loadEpisodeFile);

    // Terrain controls
    document.getElementById('terrainSegments')?.addEventListener('input', U.debounce(rebuildTerrain, 400));
    document.getElementById('terrainScale')?.addEventListener('input', U.debounce(rebuildTerrain, 400));
  }

  function updateVisibility(id, visible) {
    if (!scene) return;
    const map = {
      tog_refpath: 'refpath',
      tog_flight: 'flightpath',
      tog_corridor: 'corridor',
      tog_terrain: 'terrain',
    };
    const name = map[id];
    if (name) {
      scene.traverse(obj => { if (obj.userData.role === name) obj.visible = visible; });
    }
    if (id === 'tog_fog') {
      if (scene) scene.fog = visible ? new THREE.FogExp2(0x0a1422, 0.00018) : null;
    }
  }

  // ── Three.js Init ──
  function initThree() {
    const canvas = document.getElementById('three-canvas');
    const parent = document.getElementById('viz3dScene');
    if (!canvas || !parent || typeof THREE === 'undefined') {
      console.warn('Three.js or canvas not ready');
      return;
    }

    // Renderer
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.setClearColor(0x0a0e17, 1);

    const w = parent.clientWidth - 220;
    const h = parent.clientHeight;
    renderer.setSize(w, h);

    // Scene
    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x0a1422, 0.00018);
    scene.background = new THREE.Color(0x0a0e17);

    // Camera
    camera = new THREE.PerspectiveCamera(50, w / h, 1, 80000);
    camera.position.set(5000, 8000, 5000);
    camera.lookAt(10000, 0, 10000);

    // Clock
    clock = new THREE.Clock();

    // Controls
    if (typeof THREE.OrbitControls !== 'undefined') {
      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.minDistance = 200;
      controls.maxDistance = 30000;
      controls.maxPolarAngle = Math.PI / 2.1;
    }

    // Lights
    setupLights();

    // Build scene
    buildTerrain();
    buildRefPath();
    buildHelicopter();
    buildFlightPath([]);
    buildCorridor();
    buildStarfield();

    // Resize
    window.addEventListener('resize', onResize);

    loadDemoEpisode();
  }

  // ── Lights ──
  function setupLights() {
    const ambient = new THREE.AmbientLight(0x1a2a3a, 1.5);
    scene.add(ambient);

    const sun = new THREE.DirectionalLight(0x6699aa, 2.5);
    sun.position.set(-5000, 8000, 3000);
    sun.castShadow = true;
    scene.add(sun);

    const fill = new THREE.DirectionalLight(0x334455, 0.8);
    fill.position.set(5000, 2000, -3000);
    scene.add(fill);

    // Cyan rim light for HUD aesthetic
    const rim = new THREE.PointLight(0x00d4ff, 0.5, 5000);
    rim.position.set(10000, 3000, 10000);
    scene.add(rim);
  }

  // ── Terrain ──
  function buildTerrain() {
    if (terrainMesh) { scene.remove(terrainMesh); terrainMesh.geometry.dispose(); }

    const segments = parseInt(document.getElementById('terrainSegments')?.value || 80);
    const hScale = parseInt(document.getElementById('terrainScale')?.value || 3) * 10;
    const mapSize = 20000;

    const geo = new THREE.PlaneGeometry(mapSize, mapSize, segments, segments);
    geo.rotateX(-Math.PI / 2);

    const positions = geo.attributes.position;
    const data = U.generateSyntheticTerrain(segments + 1, segments + 1, 1.2);
    let minH = Infinity, maxH = -Infinity;
    for (let i = 0; i < data.length; i++) { if (data[i] < minH) minH = data[i]; if (data[i] > maxH) maxH = data[i]; }

    for (let i = 0; i < positions.count; i++) {
      const h = data[i] || 0;
      positions.setY(i, (h - minH) * hScale);
    }
    positions.needsUpdate = true;
    geo.computeVertexNormals();

    // Vertex colors
    const colors = new Float32Array(positions.count * 3);
    for (let i = 0; i < positions.count; i++) {
      const h = positions.getY(i);
      const norm = h / (400 * hScale);
      let r, g, b;
      if (norm < 0.25) { r = 0.1; g = 0.25 + norm * 0.4; b = 0.35; }
      else if (norm < 0.5) { r = 0.15 + norm * 0.3; g = 0.35; b = 0.25 - norm * 0.1; }
      else if (norm < 0.75) { r = 0.35 + norm * 0.2; g = 0.25 - norm * 0.1; b = 0.18; }
      else { r = 0.7 + norm * 0.3; g = 0.65 + norm * 0.2; b = 0.6 + norm * 0.3; }
      colors[i * 3] = r; colors[i * 3 + 1] = g; colors[i * 3 + 2] = b;
    }
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const mat = new THREE.MeshLambertMaterial({ vertexColors: true, wireframe: false });
    terrainMesh = new THREE.Mesh(geo, mat);
    terrainMesh.receiveShadow = true;
    terrainMesh.userData.role = 'terrain';
    terrainMesh.position.set(mapSize / 2, 0, mapSize / 2);
    scene.add(terrainMesh);

    // Grid helper
    const grid = new THREE.GridHelper(mapSize, 20, 0x1e2a3d, 0x1e2a3d);
    grid.userData.role = 'grid';
    grid.position.set(mapSize / 2, 0, mapSize / 2);
    scene.add(grid);
  }

  function rebuildTerrain() { buildTerrain(); }

  // ── Ref Path ──
  function buildRefPath() {
    if (refPathLine) scene.remove(refPathLine);
    const pts = [
      new THREE.Vector3(1000, 600, 1000),
      new THREE.Vector3(5000, 700, 4000),
      new THREE.Vector3(10000, 750, 10000),
      new THREE.Vector3(15000, 720, 16000),
      new THREE.Vector3(19000, 680, 19000),
    ];
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const mat = new THREE.LineBasicMaterial({ color: 0x00d4ff, linewidth: 2 });
    refPathLine = new THREE.Line(geo, mat);
    refPathLine.userData.role = 'refpath';
    scene.add(refPathLine);

    // Waypoint markers
    pts.forEach((p, i) => {
      const g = new THREE.SphereGeometry(i === 0 || i === pts.length - 1 ? 60 : 40, 8, 8);
      const m = new THREE.MeshBasicMaterial({ color: i === 0 ? 0x4ade80 : i === pts.length - 1 ? 0xff6b35 : 0x00d4ff });
      const sphere = new THREE.Mesh(g, m);
      sphere.position.copy(p);
      sphere.userData.role = 'refpath';
      scene.add(sphere);
    });
  }

  // ── Flight Path ──
  function buildFlightPath(flightFrames) {
    if (flightPathLine) scene.remove(flightPathLine);
    if (!flightFrames || flightFrames.length < 2) return;

    const pts = flightFrames.map(f => new THREE.Vector3(f.x, Math.max(f.z, 300), f.y));
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(pts.length * 3);
    const colors = new Float32Array(pts.length * 3);

    pts.forEach((p, i) => {
      positions[i * 3] = p.x; positions[i * 3 + 1] = p.y; positions[i * 3 + 2] = p.z;
      const risk = flightFrames[i]?.risk || 0.2;
      const t = risk;
      colors[i * 3] = t; colors[i * 3 + 1] = 1 - t * 0.8; colors[i * 3 + 2] = 0.2;
    });

    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const mat = new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 });
    flightPathLine = new THREE.Line(geo, mat);
    flightPathLine.userData.role = 'flightpath';
    scene.add(flightPathLine);
  }

  // ── Corridor Tube ──
  function buildCorridor() {
    if (corridorTube) scene.remove(corridorTube);
    const pts = [
      new THREE.Vector3(1000, 600, 1000),
      new THREE.Vector3(5000, 700, 4000),
      new THREE.Vector3(10000, 750, 10000),
      new THREE.Vector3(15000, 720, 16000),
      new THREE.Vector3(19000, 680, 19000),
    ];
    const curve = new THREE.CatmullRomCurve3(pts);
    const geo = new THREE.TubeGeometry(curve, 50, 500, 8, false);
    const mat = new THREE.MeshBasicMaterial({ color: 0x00d4ff, transparent: true, opacity: 0.06, side: THREE.DoubleSide });
    corridorTube = new THREE.Mesh(geo, mat);
    corridorTube.userData.role = 'corridor';
    scene.add(corridorTube);
  }

  // ── LAH Helicopter Model ──
  function buildHelicopter() {
    if (helicopterGroup) scene.remove(helicopterGroup);
    helicopterGroup = new THREE.Group();

    const oliveDark  = new THREE.MeshLambertMaterial({ color: 0x3d4a2e });
    const glassBlue  = new THREE.MeshLambertMaterial({ color: 0x002244, transparent: true, opacity: 0.5 });
    const darkGrey   = new THREE.MeshLambertMaterial({ color: 0x1a1f14 });
    const lightGrey  = new THREE.MeshLambertMaterial({ color: 0x5c6452 });
    const rotorMat   = new THREE.MeshLambertMaterial({ color: 0x1a1f14 });

    // Fuselage — elongated (using CylinderGeometry for r128 compatibility)
    const fuseBodyGeo = new THREE.CylinderGeometry(55, 50, 280, 12);
    const fuseBody = new THREE.Mesh(fuseBodyGeo, oliveDark);
    fuseBody.rotation.z = Math.PI / 2;
    helicopterGroup.add(fuseBody);
    // Front cap
    const fuseFrontGeo = new THREE.SphereGeometry(55, 12, 8, 0, Math.PI * 2, 0, Math.PI / 2);
    const fuseFront = new THREE.Mesh(fuseFrontGeo, oliveDark);
    fuseFront.rotation.z = Math.PI / 2;
    fuseFront.position.set(140, 0, 0);
    helicopterGroup.add(fuseFront);
    // Back cap
    const fuseBackGeo = new THREE.SphereGeometry(50, 12, 8, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2);
    const fuseBack = new THREE.Mesh(fuseBackGeo, oliveDark);
    fuseBack.rotation.z = Math.PI / 2;
    fuseBack.position.set(-140, 0, 0);
    helicopterGroup.add(fuseBack);

    // Nose bubble (canopy)
    const noseGeo = new THREE.SphereGeometry(65, 12, 8, 0, Math.PI);
    const nose = new THREE.Mesh(noseGeo, glassBlue);
    nose.rotation.z = Math.PI / 2;
    nose.position.set(170, 10, 0);
    helicopterGroup.add(nose);

    // Tail boom
    const tailGeo = new THREE.CylinderGeometry(15, 25, 350, 8);
    const tail = new THREE.Mesh(tailGeo, darkGrey);
    tail.rotation.z = Math.PI / 2;
    tail.position.set(-250, 0, 0);
    helicopterGroup.add(tail);

    // Tail fin
    const finGeo = new THREE.BoxGeometry(60, 120, 8);
    const fin = new THREE.Mesh(finGeo, darkGrey);
    fin.position.set(-400, 40, 0);
    helicopterGroup.add(fin);

    // Main rotor hub
    const hubGeo = new THREE.CylinderGeometry(20, 20, 40, 8);
    const hub = new THREE.Mesh(hubGeo, darkGrey);
    hub.position.set(0, 80, 0);
    helicopterGroup.add(hub);

    // Main rotor blades (4 blades)
    mainRotor = new THREE.Group();
    mainRotor.position.set(0, 95, 0);
    for (let i = 0; i < 4; i++) {
      const bladeGeo = new THREE.BoxGeometry(600, 8, 50);
      const blade = new THREE.Mesh(bladeGeo, rotorMat);
      blade.rotation.y = (i / 4) * Math.PI * 2;
      blade.position.x = 300 * Math.cos((i / 4) * Math.PI * 2);
      blade.position.z = 300 * Math.sin((i / 4) * Math.PI * 2);
      mainRotor.add(blade);
    }
    helicopterGroup.add(mainRotor);

    // Tail rotor
    tailRotor = new THREE.Group();
    tailRotor.position.set(-430, 20, 20);
    for (let i = 0; i < 3; i++) {
      const bladeGeo = new THREE.BoxGeometry(80, 6, 20);
      const blade = new THREE.Mesh(bladeGeo, darkGrey);
      blade.rotation.x = (i / 3) * Math.PI * 2;
      tailRotor.add(blade);
    }
    helicopterGroup.add(tailRotor);

    // Landing skids
    const skidMat = darkGrey;
    [[-30, 60], [30, 60]].forEach(([ox, oz]) => {
      const skidGeo = new THREE.CylinderGeometry(8, 8, 320, 6);
      const skid = new THREE.Mesh(skidGeo, skidMat);
      skid.rotation.z = Math.PI / 2;
      skid.position.set(0, -80, oz);
      helicopterGroup.add(skid);

      // Struts
      [-100, 100].forEach(sx => {
        const strutGeo = new THREE.CylinderGeometry(5, 5, 90, 6);
        const strut = new THREE.Mesh(strutGeo, skidMat);
        strut.rotation.x = 0.3;
        strut.position.set(sx, -45, oz * 0.5);
        helicopterGroup.add(strut);
      });
    });

    // Scale up for visibility
    helicopterGroup.scale.set(1.5, 1.5, 1.5);
    helicopterGroup.position.set(1000, 650, 1000);
    helicopterGroup.castShadow = true;
    scene.add(helicopterGroup);
  }

  // ── Starfield ──
  function buildStarfield() {
    const geo = new THREE.BufferGeometry();
    const count = 2000;
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      pos[i * 3]     = (Math.random() - 0.5) * 60000;
      pos[i * 3 + 1] = Math.random() * 20000 + 5000;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 60000;
    }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({ color: 0x334466, size: 15, sizeAttenuation: true });
    scene.add(new THREE.Points(geo, mat));
  }

  // ── Episode Data ──
  function loadDemoEpisode() {
    frames = U.generateMockEpisode(300);
    buildFlightPath(frames);
    U.setText('viz3dTotal', frames.length);
    const slider = document.getElementById('viz3dTimeline');
    if (slider) slider.max = frames.length - 1;
    currentFrame = 0;
    updateHelicopterPosition(0);
  }

  function loadEpisodeFile() {
    const input = document.createElement('input');
    input.type = 'file'; input.accept = '.json,.csv';
    input.onchange = () => { U.toast('파일 로드 (데모 모드 전용)', 'info'); };
    input.click();
  }

  function updateHelicopterPosition(idx) {
    if (!helicopterGroup || !frames.length || idx >= frames.length) return;
    const f = frames[idx];
    helicopterGroup.position.set(f.x, Math.max(f.z, 350), f.y);
    helicopterGroup.rotation.y = -f.psi;

    // Telemetry
    U.setText('t3d_alt', Math.round(f.z));
    U.setText('t3d_spd', U.fmt(f.v, 1));
    U.setText('t3d_hdg', Math.round(f.psi * 180 / Math.PI));
    U.setText('t3d_risk', U.fmt(f.risk, 3));
    U.setText('viz3dStep', idx);

    // Follow camera
    if (camMode === CAM_FOLLOW && camera) {
      const offset = new THREE.Vector3(-1500, 600, 0).applyQuaternion(helicopterGroup.quaternion);
      camera.position.lerp(helicopterGroup.position.clone().add(offset), 0.1);
      camera.lookAt(helicopterGroup.position);
    } else if (camMode === CAM_COCKPIT && camera) {
      camera.position.copy(helicopterGroup.position.clone().add(new THREE.Vector3(0, 100, 0)));
      camera.lookAt(helicopterGroup.position.clone().add(new THREE.Vector3(1000, 0, 0)));
    }

    // Update timeline
    const slider = document.getElementById('viz3dTimeline');
    if (slider) slider.value = idx;

    // Update partial flight path
    if (flightPathLine) {
      const pts = frames.slice(0, idx + 1).map(fr => new THREE.Vector3(fr.x, Math.max(fr.z, 300), fr.y));
      if (pts.length >= 2) {
        const pos = flightPathLine.geometry.attributes.position;
        pts.forEach((p, i) => { pos.setXYZ(i, p.x, p.y, p.z); });
        pos.needsUpdate = true;
      }
    }
  }

  // ── Playback ──
  function startPlayback() {
    if (!frames.length) { loadDemoEpisode(); return; }
    isPlaying = true;
    document.getElementById('viz3dPlayBtn').disabled = true;
    document.getElementById('viz3dPauseBtn').disabled = false;
    document.getElementById('viz3dStopBtn').disabled = false;
    if (currentFrame >= frames.length - 1) currentFrame = 0;
  }

  function togglePlayback() {
    isPlaying = !isPlaying;
    const btn = document.getElementById('viz3dPauseBtn');
    if (btn) btn.textContent = isPlaying ? '⏸' : '▶';
  }

  function stopPlayback() {
    isPlaying = false;
    currentFrame = 0;
    document.getElementById('viz3dPlayBtn').disabled = false;
    document.getElementById('viz3dPauseBtn').disabled = true;
    document.getElementById('viz3dStopBtn').disabled = true;
    const pauseBtn = document.getElementById('viz3dPauseBtn');
    if (pauseBtn) pauseBtn.textContent = '⏸';
    updateHelicopterPosition(0);
  }

  // ── Camera Modes ──
  function setCameraMode(mode) {
    camMode = mode;
    if (!camera) return;
    if (mode === CAM_TOP) {
      camera.position.set(10000, 15000, 10000);
      camera.lookAt(10000, 0, 10000);
      if (controls) controls.enabled = false;
    } else {
      if (controls) controls.enabled = true;
    }
  }

  // ── Render Loop ──
  let lastFpsTime = 0, fpsCount = 0;

  function startRenderLoop() {
    stopRenderLoop();
    const loop = () => {
      if (!isActive) return;
      animId = requestAnimationFrame(loop);
      tick();
    };
    animId = requestAnimationFrame(loop);
  }

  function stopRenderLoop() {
    if (animId) { cancelAnimationFrame(animId); animId = null; }
  }

  function tick() {
    if (!renderer || !scene || !camera) return;

    const delta = clock ? clock.getDelta() : 0.016;

    // Rotate rotors
    if (mainRotor) mainRotor.rotation.y += delta * 12;
    if (tailRotor) tailRotor.rotation.x += delta * 18;

    // Episode playback
    if (isPlaying && frames.length > 0) {
      currentFrame = (currentFrame + 1) % frames.length;
      updateHelicopterPosition(currentFrame);
    }

    // OrbitControls
    if (controls) controls.update();

    // FPS
    fpsCount++;
    const now = performance.now();
    if (now - lastFpsTime > 1000) {
      U.setText('renderFps', Math.round(fpsCount * 1000 / (now - lastFpsTime)));
      fpsCount = 0;
      lastFpsTime = now;
      if (renderer.info) {
        U.setText('renderTris', U.fmtInt(renderer.info.render.triangles));
      }
    }

    renderer.render(scene, camera);
  }

  // ── Resize ──
  function onResize() {
    if (!renderer || !camera) return;
    const parent = document.getElementById('viz3dScene');
    if (!parent) return;
    const w = parent.clientWidth - 220;
    const h = parent.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }

  return { activate, deactivate };
})();
