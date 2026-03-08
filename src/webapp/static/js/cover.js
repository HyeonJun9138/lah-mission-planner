/**
 * cover.js — LAH Mission Planner 표지 탭 (탭 0)
 */

'use strict';

window.LAHCover = (() => {
  const U = window.LAHUtils;
  let initialized = false;

  function activate() {
    if (!initialized) {
      initialized = true;
      render();
    }
  }

  function render() {
    const container = document.getElementById('cover-container');
    if (!container) return;

    container.innerHTML = `
      <div class="cover-hero">
        <div class="cover-bg"></div>
        <div class="cover-bg-overlay"></div>

        <div class="cover-content">
          <!-- HUD Corners -->
          <div style="position:fixed;top:50px;left:var(--sidebar-width);right:0;bottom:28px;pointer-events:none;z-index:3">
            <div class="hud-corner hud-corner-tl"></div>
            <div class="hud-corner hud-corner-tr"></div>
            <div class="hud-corner hud-corner-bl"></div>
            <div class="hud-corner hud-corner-br"></div>
          </div>

          <!-- Animated scan line -->
          <div id="cover-scanline" style="
            position:fixed;top:0;left:0;right:0;height:2px;
            background:linear-gradient(90deg,transparent,var(--cyan),transparent);
            animation:scanLine 4s ease-in-out infinite;
            z-index:3;pointer-events:none;opacity:0.4;
          "></div>

          <!-- Top Badge -->
          <div class="cover-badge animate-fadein">
            <span style="width:6px;height:6px;background:var(--green);border-radius:50%;animation:pulse 2s ease infinite"></span>
            SYSTEM ACTIVE — DEMO MODE
          </div>

          <!-- Main Title -->
          <h1 class="cover-title animate-fadein" style="animation-delay:0.1s">
            <span class="cover-title-accent">LAH</span> 임무 경로계획<br>
            <span style="color:var(--text-primary)">AI 학습 프레임워크</span>
          </h1>

          <!-- Subtitle -->
          <p class="cover-subtitle animate-fadein" style="animation-delay:0.2s">
            회전익기 안전 경로계획을 위한 강화학습 시스템 <span style="color:var(--cyan);font-family:var(--font-mono)">v3.0</span><br>
            <span style="font-size:var(--text-sm);color:var(--text-muted)">
              DEM 기반 지형분석 · Ref Path 기준 로컬 플래너 · PPO/SAC 강화학습
            </span>
          </p>

          <!-- CTA Buttons -->
          <div class="cover-cta animate-fadein" style="animation-delay:0.3s">
            <button class="btn btn-primary btn-lg" onclick="LAHApp.switchTab(1)">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 010 14.14M4.93 19.07a10 10 0 010-14.14"/></svg>
              시작하기
            </button>
            <button class="btn btn-secondary btn-lg" onclick="LAHApp.switchTab(3)">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
              시뮬레이션 보기
            </button>
          </div>

          <!-- Feature Cards -->
          <div class="cover-features animate-fadein" style="animation-delay:0.4s">
            ${[
              { icon: '🌍', title: '지형 분석', desc: 'DEM 기반 3D 지형 분석 및 위험도 산출', tab: 1 },
              { icon: '🛤️', title: '경로 계획', desc: 'Ref Path 기반 Corridor 로컬 경로계획', tab: 1 },
              { icon: '🧠', title: 'AI 학습', desc: 'PPO/SAC 기반 강화학습 및 자동 튜닝', tab: 2 },
              { icon: '🚁', title: '비행 시뮬레이션', desc: '실시간 에피소드 재생 및 분석', tab: 3 },
              { icon: '📊', title: '3D 시각화', desc: 'Three.js 기반 3차원 비행 시각화', tab: 4 },
              { icon: '✅', title: '환경 검증', desc: 'Rule 기반 환경 합치성 종합 평가', tab: 5 },
            ].map(f => `
              <div class="feature-card" onclick="LAHApp.switchTab(${f.tab})">
                <span class="feature-icon">${f.icon}</span>
                <div class="feature-title">${f.title}</div>
                <div class="feature-desc">${f.desc}</div>
              </div>
            `).join('')}
          </div>

          <!-- System Stats -->
          <div class="animate-fadein" style="animation-delay:0.5s;width:100%;max-width:900px;margin-bottom:var(--space-6)">
            <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:var(--space-3)">
              ${[
                { label: '알고리즘', value: 'PPO / SAC', sub: 'SB3 기반', color: 'var(--cyan)' },
                { label: '관측 공간', value: 'Dict', sub: 'state+patch+path', color: 'var(--green)' },
                { label: '액션 공간', value: 'Box(4,)', sub: 'Continuous', color: 'var(--yellow)' },
                { label: 'dt', value: '1.0s', sub: '시뮬레이션 step', color: 'var(--purple)' },
                { label: '지형 해상도', value: '30m', sub: 'DEM 기반', color: 'var(--orange)' },
              ].map(s => `
                <div class="card card-sm" style="text-align:center;--accent:${s.color}">
                  <div style="font-size:var(--text-xs);color:var(--text-muted);margin-bottom:4px;letter-spacing:0.06em;text-transform:uppercase">${s.label}</div>
                  <div style="font-family:var(--font-mono);font-size:var(--text-lg);font-weight:700;color:${s.color}">${s.value}</div>
                  <div style="font-size:10px;color:var(--text-muted);margin-top:2px">${s.sub}</div>
                </div>
              `).join('')}
            </div>
          </div>

          <!-- Tech Stack Badges -->
          <div class="cover-tech-stack animate-fadein" style="animation-delay:0.6s">
            ${[
              'Python 3.12', 'Gymnasium', 'stable-baselines3', 'FastAPI',
              'Three.js r163', 'Plotly.js', 'numpy', 'scipy', 'optuna'
            ].map(t => `<span class="tech-tag">${t}</span>`).join('')}
          </div>
        </div>
      </div>
    `;

    // Inject scan line animation
    const style = document.createElement('style');
    style.textContent = `
      @keyframes scanLine {
        0%   { top: 48px; opacity: 0; }
        5%   { opacity: 0.4; }
        95%  { opacity: 0.4; }
        100% { top: calc(100vh - 28px); opacity: 0; }
      }
    `;
    document.head.appendChild(style);

    // Animated counter for the title
    animateTitle();
  }

  function animateTitle() {
    // Typewriter on badge text — subtle
    const badge = document.querySelector('.cover-badge');
    if (badge) {
      const original = badge.textContent.trim();
      badge.textContent = '';
      let i = 0;
      const timer = setInterval(() => {
        badge.textContent = original.slice(0, i + 1);
        i++;
        if (i >= original.length) clearInterval(timer);
      }, 30);
    }
  }

  return { activate };
})();
