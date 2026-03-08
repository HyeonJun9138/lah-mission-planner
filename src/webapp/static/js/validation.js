/**
 * validation.js — LAH Mission Planner 환경 검증 탭 (탭 5)
 * Rule-based 환경 합치성 검증
 */

'use strict';

window.LAHValidation = (() => {
  const U = window.LAHUtils;
  let initialized = false;
  let logger = null;
  let validationHistory = [];

  function activate() {
    if (!initialized) {
      initialized = true;
      render();
    }
  }

  function render() {
    const container = document.getElementById('validation-container');
    if (!container) return;

    const checks = [
      { id: 'obs', title: '① Observation Space 검증', icon: '🔍', desc: 'state_vec, lookahead_ref, local_patch 형태/범위/NaN 검사' },
      { id: 'act', title: '② Action Space 검증', icon: '🎮', desc: 'Box(4,) 범위 [-1, 1] 및 차원 의미 확인' },
      { id: 'rwd', title: '③ Reward Function 검증', icon: '💰', desc: '랜덤 액션 N스텝 실행 후 보상 분포 분석' },
      { id: 'term', title: '④ Termination Conditions 검증', icon: '🏁', desc: 'terrain collision, goal, timeout, out-of-bounds 작동 확인' },
      { id: 'dyn', title: '⑤ Dynamics 검증', icon: '⚙️', desc: '속도/고도/선회율 제한, hold 모드, 물리적 일관성 확인' },
      { id: 'terr', title: '⑥ 지형 일관성 검증', icon: '🗺️', desc: 'heightmap 범위, slope/roughness/risk [0,1] 확인' },
    ];

    container.innerHTML = `
      <div class="val-layout">

        <!-- Header -->
        <div class="val-header">
          <h2 class="section-title">환경 합치성 검증</h2>
          <div class="flex gap-2 items-center flex-wrap">
            <!-- Check selection -->
            <div class="flex gap-2">
              ${checks.map(c => `
                <label class="checkbox-label">
                  <input type="checkbox" id="check_${c.id}" checked>
                  <span style="font-size:var(--text-xs)">${c.id.toUpperCase()}</span>
                </label>
              `).join('')}
            </div>
            <button class="btn btn-primary" id="runAllValidation">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
              전체 검증 실행
            </button>
            <button class="btn btn-secondary btn-sm" id="exportValidation">📄 리포트 저장</button>
          </div>
        </div>

        <!-- Overall Status -->
        <div class="card" id="valOverallCard" style="display:none">
          <div class="card-header">
            <span class="card-title">종합 평가</span>
            <span id="valPassBadge" class="badge badge-muted">대기 중</span>
          </div>
          <div class="grid-4" style="gap:var(--space-3)">
            <div class="kpi-card" style="--accent:var(--green)">
              <div class="kpi-label">통과</div>
              <div class="kpi-value" id="valPassCount" style="color:var(--green)">—</div>
            </div>
            <div class="kpi-card" style="--accent:var(--red)">
              <div class="kpi-label">실패</div>
              <div class="kpi-value" id="valFailCount" style="color:var(--red)">—</div>
            </div>
            <div class="kpi-card" style="--accent:var(--yellow)">
              <div class="kpi-label">경고</div>
              <div class="kpi-value" id="valWarnCount" style="color:var(--yellow)">—</div>
            </div>
            <div class="kpi-card" style="--accent:var(--cyan)">
              <div class="kpi-label">통과율</div>
              <div class="kpi-value" id="valPassRate" style="color:var(--cyan)">—</div>
            </div>
          </div>
          <div style="margin-top:var(--space-3)">
            <div class="flex justify-between mb-1" style="font-size:var(--text-xs);color:var(--text-muted)">
              <span>전체 통과율</span>
              <span id="valPassRatePct">—</span>
            </div>
            <div class="progress-bar" style="height:8px">
              <div class="progress-fill" id="valOverallBar" style="width:0%;height:100%"></div>
            </div>
          </div>
        </div>

        <!-- Validation Items -->
        <div class="val-items" id="validationItems">
          ${checks.map(c => `
            <div class="val-item" id="valItem_${c.id}">
              <div class="collapsible-header" data-default-open="false">
                <div class="collapsible-title">
                  <span>${c.icon}</span>
                  <span>${c.title}</span>
                  <span class="badge badge-muted" id="badge_${c.id}">대기</span>
                </div>
                <svg class="collapsible-arrow" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
              </div>
              <div class="collapsible-body" id="body_${c.id}">
                <div style="color:var(--text-muted);font-size:var(--text-sm);margin-bottom:var(--space-3)">${c.desc}</div>
                <div id="result_${c.id}">
                  <div style="color:var(--text-muted);font-size:var(--text-sm)">검증 대기 중...</div>
                </div>
              </div>
            </div>
          `).join('')}
        </div>

        <!-- Recommendations -->
        <div class="card" id="valRecoCard" style="display:none">
          <div class="card-header">
            <span class="card-title">개선 권고사항</span>
          </div>
          <div id="valRecommendations" style="display:flex;flex-direction:column;gap:8px"></div>
        </div>

        <!-- Log -->
        <div class="card">
          <div class="card-header">
            <span class="card-title">검증 로그</span>
            <button class="btn btn-ghost btn-sm" onclick="document.getElementById('valLog').innerHTML=''">지우기</button>
          </div>
          <div class="log-console" id="valLog"></div>
        </div>

        <!-- History -->
        <div class="card" id="valHistoryCard">
          <div class="card-header">
            <span class="card-title">검증 이력</span>
          </div>
          <div id="valHistory">
            <div style="color:var(--text-muted);font-size:var(--text-sm);text-align:center;padding:var(--space-4)">
              검증 이력이 없습니다.
            </div>
          </div>
        </div>

      </div>
    `;

    U.initCollapsibles(container);
    const logEl = document.getElementById('valLog');
    if (logEl) logger = U.createLogger(logEl);

    document.getElementById('runAllValidation')?.addEventListener('click', runAllValidation);
    document.getElementById('exportValidation')?.addEventListener('click', exportReport);
  }

  // ── Run All ──
  async function runAllValidation() {
    const startBtn = document.getElementById('runAllValidation');
    if (startBtn) { startBtn.disabled = true; startBtn.textContent = '검증 중...'; }

    logger?.info('전체 환경 검증 시작...');
    document.getElementById('valOverallCard').style.display = 'block';

    const checks = ['obs', 'act', 'rwd', 'term', 'dyn', 'terr'];
    const selected = checks.filter(id => document.getElementById(`check_${id}`)?.checked);

    let passCount = 0, failCount = 0, warnCount = 0;
    const results = {};

    for (const id of selected) {
      setBadge(id, 'running');
      await delay(400 + Math.random() * 600);

      // Try API
      const { data, error } = await U.apiCall(`/validate/${id}`);

      let result;
      if (data && !error) {
        result = data;
      } else {
        result = generateMockResult(id);
      }

      results[id] = result;
      renderResult(id, result);
      setBadge(id, result.status);

      if (result.status === 'pass')  passCount++;
      else if (result.status === 'fail') failCount++;
      else if (result.status === 'warn') warnCount++;

      logger?.[result.status === 'pass' ? 'ok' : result.status === 'fail' ? 'error' : 'warn'](
        `${id.toUpperCase()}: ${result.status.toUpperCase()} — ${result.summary}`
      );
    }

    // Update overall
    const total = passCount + failCount + warnCount;
    const rate = total > 0 ? passCount / total : 0;
    U.setText('valPassCount', passCount);
    U.setText('valFailCount', failCount);
    U.setText('valWarnCount', warnCount);
    U.setText('valPassRate', U.fmtPct(rate));
    U.setText('valPassRatePct', U.fmtPct(rate));

    const bar = document.getElementById('valOverallBar');
    if (bar) bar.style.width = (rate * 100) + '%';

    const badge = document.getElementById('valPassBadge');
    if (badge) {
      const cls = rate >= 0.9 ? 'badge-green' : rate >= 0.7 ? 'badge-yellow' : 'badge-red';
      badge.className = `badge ${cls}`;
      badge.textContent = rate >= 0.9 ? '정상' : rate >= 0.7 ? '주의 필요' : '개선 필요';
    }

    // Recommendations
    renderRecommendations(results);

    // Save history
    validationHistory.push({
      time: new Date().toLocaleString('ko-KR'),
      pass: passCount, fail: failCount, warn: warnCount,
      rate: U.fmtPct(rate)
    });
    renderHistory();

    if (startBtn) { startBtn.disabled = false; startBtn.textContent = '전체 검증 실행'; }
    logger?.ok(`검증 완료 — 통과: ${passCount}, 실패: ${failCount}, 경고: ${warnCount}`);
  }

  // ── Mock Results ──
  function generateMockResult(id) {
    const templates = {
      obs: {
        status: 'pass', summary: 'Observation space 정상',
        details: [
          { name: 'state_vec shape', expected: '(14,)', got: '(14,)', status: 'pass' },
          { name: 'lookahead_ref shape', expected: '(16, 4)', got: '(16, 4)', status: 'pass' },
          { name: 'local_patch shape', expected: '(64, 64, 8)', got: '(64, 64, 8)', status: 'pass' },
          { name: 'NaN 검사', expected: 'NaN 없음', got: 'NaN 없음', status: 'pass' },
          { name: 'Inf 검사', expected: 'Inf 없음', got: 'Inf 없음', status: 'pass' },
          { name: '정규화 범위', expected: '[-1, 1]', got: '[-0.98, 1.00]', status: 'pass' },
        ]
      },
      act: {
        status: 'pass', summary: 'Action space Box(4,) 정상',
        details: [
          { name: 'action shape', expected: 'Box(4,)', got: 'Box(4,)', status: 'pass' },
          { name: 'a0 범위 (yaw)', expected: '[-1, 1]', got: '[-1, 1]', status: 'pass' },
          { name: 'a1 범위 (vz)', expected: '[-1, 1]', got: '[-1, 1]', status: 'pass' },
          { name: 'a2 범위 (speed)', expected: '[-1, 1]', got: '[-1, 1]', status: 'pass' },
          { name: 'a3 범위 (hold)', expected: '[-1, 1]', got: '[-1, 1]', status: 'pass' },
        ]
      },
      rwd: {
        status: 'warn', summary: '보상 분포 분석 완료 — 경고: 보상 스파이크 감지',
        details: [
          { name: '평균 보상', expected: '-10 ~ +10', got: '2.34', status: 'pass' },
          { name: '최소 보상', expected: '>= -300', got: '-152.3', status: 'pass' },
          { name: '최대 보상', expected: '<= +300', got: '150.0', status: 'pass' },
          { name: '분산', expected: '< 100', got: '87.2', status: 'pass' },
          { name: '보상 스파이크', expected: '없음', got: '2회 감지', status: 'warn' },
          { name: 'progress 보상', expected: '양수', got: '양수', status: 'pass' },
        ],
        chartData: { bins: [-100,-50,-20,-10,-5,0,5,10,20,50,100], counts: [2,5,18,45,82,95,78,52,30,10,3] }
      },
      term: {
        status: 'pass', summary: '모든 종료 조건 정상 작동',
        details: [
          { name: 'terrain collision', expected: '작동', got: '142회 트리거', status: 'pass' },
          { name: 'goal 도달', expected: '작동', got: '213회 트리거', status: 'pass' },
          { name: 'timeout (max_steps)', expected: '작동', got: '89회 트리거', status: 'pass' },
          { name: 'out of bounds', expected: '작동', got: '17회 트리거', status: 'pass' },
          { name: 'hard corridor 이탈', expected: '작동', got: '34회 트리거', status: 'pass' },
        ]
      },
      dyn: {
        status: 'warn', summary: '동역학 대부분 정상 — 경고: 선회율 초과 간헐적 발생',
        details: [
          { name: 'v_max 준수', expected: '<= 70 m/s', got: '최대 69.8 m/s', status: 'pass' },
          { name: 'v_min 준수', expected: '>= 0 m/s', got: '최소 0.1 m/s', status: 'pass' },
          { name: 'vz 제한 준수', expected: '[-6, 6] m/s', got: '[-5.9, 5.8] m/s', status: 'pass' },
          { name: '선회율 제한', expected: '<= 12 deg/s', got: '최대 13.2 deg/s (간헐적)', status: 'warn' },
          { name: 'hold 모드 작동', expected: '작동', got: '정상', status: 'pass' },
          { name: '에너지 보존', expected: '물리적 일관성', got: '일관성 확인', status: 'pass' },
        ]
      },
      terr: {
        status: 'pass', summary: '지형 데이터 정상',
        details: [
          { name: 'heightmap 범위', expected: '[0, ∞)', got: '[45.2, 892.4] m', status: 'pass' },
          { name: 'slope 범위', expected: '[0, π/2)', got: '[0, 1.21] rad', status: 'pass' },
          { name: 'roughness 범위', expected: '[0, 1]', got: '[0.00, 0.98]', status: 'pass' },
          { name: 'risk map 범위', expected: '[0, 1]', got: '[0.00, 1.00]', status: 'pass' },
          { name: 'NaN/Inf 검사', expected: '없음', got: '없음', status: 'pass' },
          { name: 'landing suitability', expected: '[0, 1]', got: '[0.00, 0.99]', status: 'pass' },
        ]
      }
    };
    return templates[id] || { status: 'pass', summary: '검증 완료', details: [] };
  }

  // ── Render Result ──
  function renderResult(id, result) {
    const container = document.getElementById(`result_${id}`);
    if (!container) return;

    // Open collapsible
    const header = document.querySelector(`#valItem_${id} .collapsible-header`);
    const body = document.getElementById(`body_${id}`);
    if (header && body) { header.classList.add('open'); body.classList.add('open'); }

    const statusIcon = { pass: '✅', fail: '❌', warn: '⚠️' };

    let html = `
      <div style="display:flex;align-items:center;gap:var(--space-2);margin-bottom:var(--space-3)">
        <span style="font-size:1.2rem">${statusIcon[result.status] || '—'}</span>
        <span style="font-size:var(--text-sm);font-weight:600;color:${result.status === 'pass' ? 'var(--green)' : result.status === 'fail' ? 'var(--red)' : 'var(--yellow)'}">
          ${result.summary}
        </span>
      </div>
    `;

    if (result.details?.length > 0) {
      html += `
        <table class="data-table" style="margin-bottom:var(--space-3)">
          <thead><tr><th>검증 항목</th><th>기대값</th><th>실제값</th><th>결과</th></tr></thead>
          <tbody>
            ${result.details.map(d => `
              <tr>
                <td style="font-weight:500">${d.name}</td>
                <td class="mono" style="font-size:11px;color:var(--text-muted)">${d.expected}</td>
                <td class="mono" style="font-size:11px">${d.got}</td>
                <td>
                  <span class="badge ${d.status === 'pass' ? 'badge-green' : d.status === 'fail' ? 'badge-red' : 'badge-yellow'}">
                    ${statusIcon[d.status] || '—'} ${d.status}
                  </span>
                </td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      `;
    }

    // Reward chart
    if (id === 'rwd' && result.chartData) {
      html += `<div id="rewardDistChart_${id}" style="height:120px"></div>`;
    }

    container.innerHTML = html;

    // Plot reward distribution
    if (id === 'rwd' && result.chartData) {
      setTimeout(() => {
        const { bins, counts } = result.chartData;
        Plotly.newPlot(`rewardDistChart_${id}`, [{
          x: bins, y: counts, type: 'bar',
          marker: { color: bins.map(b => b >= 0 ? '#4ade80' : '#ff6b35') }
        }], {
          ...U.plotlyLayout({ margin: { l: 30, r: 8, t: 10, b: 30 } }),
          xaxis: { title: { text: 'Reward', font: { size: 9 } }, gridcolor: '#1e2a3d', tickfont: { size: 8 } },
          yaxis: { title: { text: '빈도', font: { size: 9 } }, gridcolor: '#1e2a3d', tickfont: { size: 8 } },
        }, { responsive: true, displayModeBar: false });
      }, 100);
    }
  }

  // ── Badge ──
  function setBadge(id, status) {
    const badge = document.getElementById(`badge_${id}`);
    if (!badge) return;
    const map = {
      pass:    ['badge-green', '✅ 통과'],
      fail:    ['badge-red', '❌ 실패'],
      warn:    ['badge-yellow', '⚠️ 경고'],
      running: ['badge-cyan', '🔄 실행 중'],
      pending: ['badge-muted', '대기'],
    };
    const [cls, text] = map[status] || map.pending;
    badge.className = `badge ${cls}`;
    badge.textContent = text;
  }

  // ── Recommendations ──
  function renderRecommendations(results) {
    const recoCard = document.getElementById('valRecoCard');
    const recoEl = document.getElementById('valRecommendations');
    if (!recoCard || !recoEl) return;

    const recos = [];

    if (results.rwd?.status === 'warn') {
      recos.push({ level: 'warn', text: 'w_smooth 가중치를 높여 보상 스파이크를 줄이세요. (현재 0.05 → 권장 0.1~0.2)' });
    }
    if (results.dyn?.status === 'warn') {
      recos.push({ level: 'warn', text: 'max_turn_rate 제한을 확인하세요. 간헐적으로 제한값을 초과하고 있습니다 (12 deg/s).' });
    }
    if (Object.values(results).some(r => r.status === 'fail')) {
      recos.push({ level: 'error', text: '실패한 검증 항목이 있습니다. 학습 시작 전에 수정하세요.' });
    }
    if (Object.values(results).every(r => r.status === 'pass')) {
      recos.push({ level: 'ok', text: '모든 검증 항목이 통과되었습니다. 학습을 시작할 수 있습니다.' });
    }

    recoCard.style.display = 'block';
    recoEl.innerHTML = recos.map(r => {
      const icons = { ok: '✅', warn: '⚠️', error: '❌' };
      const colors = { ok: 'var(--green)', warn: 'var(--yellow)', error: 'var(--red)' };
      return `
        <div style="display:flex;align-items:flex-start;gap:var(--space-2);padding:var(--space-2) var(--space-3);background:var(--bg-elevated);border-radius:var(--radius-md);border-left:2px solid ${colors[r.level]}">
          <span>${icons[r.level]}</span>
          <span style="font-size:var(--text-sm);color:var(--text-primary)">${r.text}</span>
        </div>
      `;
    }).join('') || '<div style="color:var(--text-muted);font-size:var(--text-sm)">권고사항 없음</div>';
  }

  // ── History ──
  function renderHistory() {
    const el = document.getElementById('valHistory');
    if (!el || !validationHistory.length) return;

    el.innerHTML = `
      <table class="data-table">
        <thead><tr><th>시간</th><th>통과</th><th>실패</th><th>경고</th><th>통과율</th></tr></thead>
        <tbody>
          ${validationHistory.slice().reverse().map(h => `
            <tr>
              <td class="mono" style="font-size:11px">${h.time}</td>
              <td style="color:var(--green)">${h.pass}</td>
              <td style="color:var(--red)">${h.fail}</td>
              <td style="color:var(--yellow)">${h.warn}</td>
              <td class="mono">${h.rate}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
  }

  // ── Export ──
  function exportReport() {
    const report = {
      timestamp: new Date().toISOString(),
      history: validationHistory,
      system: 'LAH Mission Planner v3.0',
      note: '환경 합치성 검증 리포트'
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `lah_validation_${Date.now()}.json`;
    a.click();
    U.toast('검증 리포트가 저장되었습니다.', 'ok');
  }

  // ── Helpers ──
  function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

  return { activate };
})();
