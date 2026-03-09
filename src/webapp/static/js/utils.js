/**
 * utils.js — LAH Mission Planner 공통 유틸리티
 */

'use strict';

window.LAHUtils = (() => {

  // ── Number Formatting ──
  function fmt(val, dec = 1) {
    if (val === null || val === undefined || isNaN(val)) return '—';
    return Number(val).toFixed(dec);
  }

  function fmtInt(val) {
    if (val === null || val === undefined) return '—';
    return Math.round(val).toLocaleString('ko-KR');
  }

  function fmtPct(val, dec = 1) {
    return fmt(val * 100, dec) + '%';
  }

  function fmtTime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return [h, m, s].map(v => String(v).padStart(2, '0')).join(':');
  }

  function fmtNow() {
    return new Date().toLocaleTimeString('ko-KR', { hour12: false });
  }

  // ── Color Interpolation ──
  function colorScale(val, min = 0, max = 1) {
    // Blue -> Green -> Yellow -> Red
    const t = Math.max(0, Math.min(1, (val - min) / (max - min)));
    if (t < 0.33) {
      const f = t / 0.33;
      return `rgb(${Math.round(0 + f * 74)}, ${Math.round(100 + f * 174)}, ${Math.round(255 - f * 55)})`;
    } else if (t < 0.67) {
      const f = (t - 0.33) / 0.34;
      return `rgb(${Math.round(74 + f * 181)}, ${Math.round(274 - f * 34)}, ${Math.round(200 - f * 200)})`;
    } else {
      const f = (t - 0.67) / 0.33;
      return `rgb(${Math.round(255)}, ${Math.round(240 - f * 240)}, 0)`;
    }
  }

  function riskColor(risk) {
    if (risk < 0.2)  return '#4ade80';
    if (risk < 0.4)  return '#a3e635';
    if (risk < 0.6)  return '#f59e0b';
    if (risk < 0.8)  return '#fb923c';
    return '#ef4444';
  }

  function heightColor(h, minH, maxH) {
    const t = (h - minH) / (maxH - minH + 0.001);
    if (t < 0.25) { const f = t/0.25; return `rgb(30,${Math.round(60+f*80)},${Math.round(140+f*60)})`; }
    if (t < 0.5)  { const f = (t-0.25)/0.25; return `rgb(${Math.round(30+f*60)},${Math.round(140+f*60)},${Math.round(200-f*120)})`; }
    if (t < 0.75) { const f = (t-0.5)/0.25; return `rgb(${Math.round(90+f*100)},${Math.round(200-f*80)},${Math.round(80-f*40)})`; }
    { const f = (t-0.75)/0.25; return `rgb(${Math.round(190+f*65)},${Math.round(120+f*100)},${Math.round(40+f*60)})`; }
  }

  // ── DOM Helpers ──
  function el(id) { return document.getElementById(id); }

  function create(tag, attrs = {}, text = '') {
    const e = document.createElement(tag);
    Object.entries(attrs).forEach(([k, v]) => {
      if (k === 'class') e.className = v;
      else if (k === 'style') e.style.cssText = v;
      else e.setAttribute(k, v);
    });
    if (text) e.textContent = text;
    return e;
  }

  function clearEl(element) {
    while (element.firstChild) element.removeChild(element.firstChild);
  }

  function setHTML(id, html) {
    const e = el(id);
    if (e) e.innerHTML = html;
  }

  function setText(id, text) {
    const e = el(id);
    if (e) e.textContent = text;
  }

  // ── Slider Binding ──
  function bindSlider(sliderId, displayId, transform) {
    const slider = el(sliderId);
    const display = el(displayId);
    if (!slider || !display) return;
    const update = () => {
      const val = parseFloat(slider.value);
      display.textContent = transform ? transform(val) : val;
    };
    slider.addEventListener('input', update);
    update();
  }

  function bindAllSliders(container) {
    const sliders = container.querySelectorAll('input[type="range"][data-display]');
    sliders.forEach(slider => {
      const displayId = slider.dataset.display;
      const display = document.getElementById(displayId);
      if (display) {
        const update = () => { display.textContent = slider.value; };
        slider.addEventListener('input', update);
        update();
      }
    });
  }

  // ── Collapsible ──
  function initCollapsibles(container) {
    const headers = container.querySelectorAll('.collapsible-header');
    headers.forEach(header => {
      const body = header.nextElementSibling;
      if (!body || !body.classList.contains('collapsible-body')) return;
      header.addEventListener('click', () => {
        header.classList.toggle('open');
        body.classList.toggle('open');
      });
      // Open first one by default
      if (header.dataset.defaultOpen === 'true') {
        header.classList.add('open');
        body.classList.add('open');
      }
    });
  }

  // ── Log Console ──
  function createLogger(consoleEl) {
    return {
      log(msg, type = 'info') {
        const entry = create('div', { class: `log-entry ${type}` });
        const prefix = create('span', { class: 'log-prefix' });
        const now = new Date();
        prefix.textContent = `[${now.toTimeString().slice(0,8)}]`;
        const msgEl = create('span', { class: 'log-msg' }, ' ' + msg);
        entry.appendChild(prefix);
        entry.appendChild(msgEl);
        consoleEl.appendChild(entry);
        consoleEl.scrollTop = consoleEl.scrollHeight;
        // Keep last 200 entries
        while (consoleEl.children.length > 200) {
          consoleEl.removeChild(consoleEl.firstChild);
        }
      },
      info:  function(m) { this.log(m, 'info'); },
      warn:  function(m) { this.log(m, 'warn'); },
      error: function(m) { this.log(m, 'error'); },
      ok:    function(m) { this.log(m, 'ok'); },
      clear() { clearEl(consoleEl); }
    };
  }

  // ── Synthetic Terrain Generation ──
  function generateSyntheticTerrain(width, height, complexity = 1.0) {
    const data = new Float32Array(width * height);
    const scales = [
      { scale: 0.003, amp: 300 },
      { scale: 0.01, amp: 120 },
      { scale: 0.03, amp: 50 },
      { scale: 0.08, amp: 20 },
    ];

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let h = 200; // base elevation
        scales.forEach(({ scale, amp }) => {
          h += amp * complexity * noise2d(x * scale, y * scale);
        });
        data[y * width + x] = Math.max(0, h);
      }
    }
    return data;
  }

  // Simple 2D Perlin-like noise
  function noise2d(x, y) {
    const ix = Math.floor(x), iy = Math.floor(y);
    const fx = x - ix, fy = y - iy;
    const ux = fx * fx * (3 - 2 * fx);
    const uy = fy * fy * (3 - 2 * fy);
    const a = pseudoRand(ix, iy);
    const b = pseudoRand(ix + 1, iy);
    const c = pseudoRand(ix, iy + 1);
    const d = pseudoRand(ix + 1, iy + 1);
    return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
  }

  function pseudoRand(x, y) {
    const n = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
    return n - Math.floor(n);
  }

  // ── Path Utilities ──
  function generateRefPath(startX, startY, startZ, endX, endY, endZ, numWaypoints = 3) {
    const path = [{ x: startX, y: startY, z: startZ }];
    for (let i = 1; i < numWaypoints - 1; i++) {
      const t = i / (numWaypoints - 1);
      path.push({
        x: startX + (endX - startX) * t + (Math.random() - 0.5) * 200,
        y: startY + (endY - startY) * t + (Math.random() - 0.5) * 200,
        z: startZ + (endZ - startZ) * t
      });
    }
    path.push({ x: endX, y: endY, z: endZ });
    return path;
  }

  function interpolatePath(path, numPoints = 100) {
    if (path.length < 2) return path;
    const result = [];
    const segLen = numPoints / (path.length - 1);
    for (let i = 0; i < path.length - 1; i++) {
      const a = path[i], b = path[i + 1];
      for (let j = 0; j < segLen; j++) {
        const t = j / segLen;
        result.push({
          x: a.x + (b.x - a.x) * t,
          y: a.y + (b.y - a.y) * t,
          z: a.z + (b.z - a.z) * t
        });
      }
    }
    result.push(path[path.length - 1]);
    return result;
  }

  // ── Mock Episode Data ──
  function generateMockEpisode(steps = 300) {
    const frames = [];
    let x = 1000, y = 1000, z = 400;
    let v = 40, psi = Math.PI / 4, vz = 0;
    let s = 0, d = 0, risk = 0.15, agl = 250;
    let reward_cum = 0;

    for (let t = 0; t < steps; t++) {
      const progress = t / steps;
      psi += (Math.random() - 0.5) * 0.05;
      v = Math.max(30, Math.min(60, v + (Math.random() - 0.5) * 2));
      vz = (Math.random() - 0.5) * 1;
      x += v * Math.cos(psi);
      y += v * Math.sin(psi);
      z = Math.max(300, z + vz);
      agl = Math.max(100, 200 + Math.sin(t * 0.08) * 60 + (Math.random() - 0.5) * 20);
      s = progress * 30000;
      d = Math.sin(t * 0.05) * 80 + (Math.random() - 0.5) * 20;
      risk = 0.1 + Math.abs(Math.sin(t * 0.06)) * 0.4 + (Math.random() - 0.5) * 0.05;
      risk = Math.max(0, Math.min(1, risk));
      const reward = 0.5 + (1 - risk) * 0.3 - Math.abs(d) / 1000;
      reward_cum += reward;

      frames.push({ t, x, y, z, v, psi, vz, agl, s, d, risk, reward, reward_cum,
        hold_mode: risk > 0.75 ? 1 : 0 });
    }
    return frames;
  }

  // ── API Helpers ──
  async function apiCall(endpoint, options = {}) {
    try {
      const response = await fetch(`/api${endpoint}`, {
        headers: { 'Content-Type': 'application/json', ...options.headers },
        ...options
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return { data: await response.json(), error: null };
    } catch (err) {
      return { data: null, error: err.message };
    }
  }

  // ── Plotly Default Layout ──
  function plotlyLayout(overrides = {}) {
    return {
      paper_bgcolor: 'transparent',
      plot_bgcolor: '#050810',
      font: { family: '"JetBrains Mono", monospace', size: 10, color: '#94a3b8' },
      margin: { l: 40, r: 10, t: 24, b: 30 },
      xaxis: {
        gridcolor: '#1e2a3d', zerolinecolor: '#1e2a3d',
        color: '#475569', tickfont: { size: 9 }
      },
      yaxis: {
        gridcolor: '#1e2a3d', zerolinecolor: '#1e2a3d',
        color: '#475569', tickfont: { size: 9 }
      },
      showlegend: false,
      ...overrides
    };
  }

  // ── Canvas 2D Helpers ──
  function drawGrid(ctx, w, h, step = 50, color = 'rgba(30,42,61,0.5)') {
    ctx.strokeStyle = color;
    ctx.lineWidth = 0.5;
    for (let x = 0; x < w; x += step) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y < h; y += step) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
  }

  function drawHelicopterMarker(ctx, x, y, heading, size = 12, color = '#00d4ff') {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(heading);

    // Body
    ctx.beginPath();
    ctx.ellipse(0, 0, size * 0.5, size * 0.8, 0, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.9;
    ctx.fill();

    // Main rotor
    ctx.beginPath();
    ctx.moveTo(-size, 0); ctx.lineTo(size, 0);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 1;
    ctx.stroke();

    // Tail
    ctx.beginPath();
    ctx.moveTo(0, size * 0.8); ctx.lineTo(0, size * 1.6);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Direction arrow
    ctx.beginPath();
    ctx.moveTo(0, -size * 0.8);
    ctx.lineTo(-size * 0.3, -size * 0.4);
    ctx.lineTo(size * 0.3, -size * 0.4);
    ctx.closePath();
    ctx.fillStyle = '#ff6b35';
    ctx.fill();

    ctx.restore();
  }

  // ── Debounce / Throttle ──
  function debounce(fn, delay) {
    let timer;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), delay);
    };
  }

  function throttle(fn, limit) {
    let last = 0;
    return (...args) => {
      const now = Date.now();
      if (now - last >= limit) { last = now; fn(...args); }
    };
  }

  // ── Random ──
  function randRange(min, max) { return min + Math.random() * (max - min); }
  function randInt(min, max)   { return Math.floor(randRange(min, max + 1)); }
  function randChoice(arr)     { return arr[Math.floor(Math.random() * arr.length)]; }

  // ── Toast Notification ──
  function toast(message, type = 'info', duration = 3000) {
    let container = document.getElementById('toast-container');
    if (!container) {
      container = create('div', {
        id: 'toast-container',
        style: 'position:fixed;bottom:40px;right:20px;z-index:9999;display:flex;flex-direction:column;gap:8px;'
      });
      document.body.appendChild(container);
    }
    const colors = { info: '#00d4ff', warn: '#f59e0b', error: '#ef4444', ok: '#4ade80' };
    const toast = create('div', {
      style: `background:#0f1623;border:1px solid ${colors[type]||colors.info};border-left:3px solid ${colors[type]||colors.info};
              border-radius:6px;padding:8px 14px;font-size:12px;color:#e2e8f0;
              animation:fadeIn 0.3s ease;max-width:280px;word-break:break-word;`
    }, message);
    container.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; toast.style.transition = 'opacity 0.3s'; setTimeout(() => toast.remove(), 300); }, duration);
  }

  // ── Coordinate Conversion (Client-side Approximate) ──
  /**
   * WGS84 위경도 ↔ UTM52N 근사 변환 (클라이언트 측 표시용).
   * 정밀 변환은 서버 측(/api/terrain/dem/elevation) 사용 권장.
   *
   * @param {number} lat   위도 [°]
   * @param {number} lon   경도 [°]
   * @param {boolean} toUTM  true: lat/lon→UTM, false: UTM→lat/lon
   * @param {number} [utmX]  UTM X (toUTM=false 시)
   * @param {number} [utmY]  UTM Y (toUTM=false 시)
   * @returns {{ x: number, y: number } | { lat: number, lon: number }}
   */
  function coordConvert(lat, lon, toUTM = true, utmX = 0, utmY = 0) {
    // UTM Zone 52N (중앙 경선 129°E) 기준 Mercator 근사
    const DEG_TO_RAD = Math.PI / 180;
    const a = 6378137.0;       // WGS84 장반경 [m]
    const k0 = 0.9996;         // 축척 계수
    const lon0 = 129.0;        // Zone 52N 중앙 경선 [°]
    const E0 = 500000.0;       // 동방향 가산수 [m]
    const N0 = 0.0;            // 북방향 가산수 (북반구) [m]

    if (toUTM) {
      // WGS84 → UTM52N (근사 Transverse Mercator)
      const latR = lat * DEG_TO_RAD;
      const lonR = lon * DEG_TO_RAD;
      const lon0R = lon0 * DEG_TO_RAD;
      const N = a / Math.sqrt(1 - 0.00669438 * Math.sin(latR) ** 2);
      const T = Math.tan(latR) ** 2;
      const C = 0.006739496742 * Math.cos(latR) ** 2;
      const A = Math.cos(latR) * (lonR - lon0R);
      const M = a * (
        (1 - 0.00669438 / 4 - 3 * 0.00669438 ** 2 / 64) * latR
        - (3 * 0.00669438 / 8 + 3 * 0.00669438 ** 2 / 32) * Math.sin(2 * latR)
        + (15 * 0.00669438 ** 2 / 256) * Math.sin(4 * latR)
      );
      const x = k0 * N * (A + (1 - T + C) * A ** 3 / 6) + E0;
      const y = k0 * (M + N * Math.tan(latR) * (A ** 2 / 2 + (5 - T + 9 * C) * A ** 4 / 24)) + N0;
      return { x: Math.round(x), y: Math.round(y) };
    } else {
      // UTM52N → WGS84 (역 근사)
      const x1 = utmX - E0;
      const y1 = utmY - N0;
      const lon0R = lon0 * DEG_TO_RAD;
      const M = y1 / k0;
      const mu = M / (a * (1 - 0.00669438 / 4 - 3 * 0.00669438 ** 2 / 64));
      const e1 = (1 - Math.sqrt(1 - 0.00669438)) / (1 + Math.sqrt(1 - 0.00669438));
      const phi1 = mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * Math.sin(2 * mu)
        + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * Math.sin(4 * mu);
      const N1 = a / Math.sqrt(1 - 0.00669438 * Math.sin(phi1) ** 2);
      const T1 = Math.tan(phi1) ** 2;
      const C1 = 0.006739496742 * Math.cos(phi1) ** 2;
      const R1 = a * (1 - 0.00669438) / (1 - 0.00669438 * Math.sin(phi1) ** 2) ** 1.5;
      const D = x1 / (N1 * k0);
      const latOut = phi1 - (N1 * Math.tan(phi1) / R1) * (
        D ** 2 / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2) * D ** 4 / 24
      );
      const lonOut = lon0R + (D - (1 + 2 * T1 + C1) * D ** 3 / 6) / Math.cos(phi1);
      return {
        lat: +(latOut / (Math.PI / 180)).toFixed(7),
        lon: +(lonOut / (Math.PI / 180)).toFixed(7),
      };
    }
  }

  return {
    fmt, fmtInt, fmtPct, fmtTime, fmtNow,
    colorScale, riskColor, heightColor,
    el, create, clearEl, setHTML, setText,
    bindSlider, bindAllSliders, initCollapsibles,
    createLogger, generateSyntheticTerrain,
    generateRefPath, interpolatePath, generateMockEpisode,
    apiCall, plotlyLayout,
    drawGrid, drawHelicopterMarker,
    debounce, throttle,
    randRange, randInt, randChoice,
    toast,
    coordConvert,
  };
})();
