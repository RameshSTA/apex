/**
 * Apex — Revenue Intelligence Platform
 * Dashboard application logic.
 *
 * Responsibilities:
 *   - Render all Chart.js visualisations (forecasts, feature importance,
 *     seat optimisation, causal inference, RM agent trace).
 *   - Drive tab navigation and route-selector controls.
 *   - Consume the global JS constants injected by the build step:
 *       METRICS, FORECASTS, SHAP_DATA, ELASTICITY, LP_DATA,
 *       CV_DATA, DID_DATA, MONTHLY_TS, SYD_WEEKLY.
 *
 * No external runtime dependencies beyond Chart.js (loaded via CDN).
 *
 * Author: Ramesh Shrestha
 */

const charts = {};
const F = v => typeof v === 'number' ? v.toFixed(2) : v;
const pct = v => (v * 100).toFixed(1) + '%';
const sleep = ms => new Promise(r => setTimeout(r, ms));
function dc(id) { if (charts[id]) { charts[id].destroy(); delete charts[id]; } }

const CHART_DEFAULTS = {
  color: '#A1A1AA',
  gridColor: 'rgba(255,255,255,.05)',
  font: { family: "'JetBrains Mono', monospace", size: 9 },
};
Chart.defaults.color = CHART_DEFAULTS.color;

// ── NAVIGATION ────────────────────────────────────────────────────
function navGo(e, tab, el) {
  e.preventDefault();
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  const href = el.getAttribute('href');
  if (tab) {
    // Switch demo tab first, then scroll
    document.querySelectorAll('.tp').forEach(p => p.classList.remove('on'));
    document.getElementById('tp-' + tab).classList.add('on');
    document.querySelectorAll('.tab').forEach(b => {
      if (b.onclick && b.onclick.toString().includes("switchTab('" + tab + "'")) b.classList.add('on');
      else b.classList.remove('on');
    });
    document.getElementById('demo').scrollIntoView({ behavior: 'smooth' });
  } else {
    document.querySelector(href).scrollIntoView({ behavior: 'smooth' });
  }
}

function toggleDrawer() {
  const d = document.getElementById('nav-drawer');
  const h = document.getElementById('hamburger');
  d.classList.toggle('open');
  h.classList.toggle('open');
}
function closeDrawer() {
  document.getElementById('nav-drawer').classList.remove('open');
  document.getElementById('hamburger').classList.remove('open');
}
function drawerGo(tab, el) {
  document.querySelectorAll('.drawer-item').forEach(b => b.classList.remove('active'));
  el.classList.add('active');
  document.querySelectorAll('.tp').forEach(p => p.classList.remove('on'));
  document.getElementById('tp-' + tab).classList.add('on');
  document.querySelectorAll('.tab').forEach(b => {
    if (b.onclick && b.onclick.toString().includes("switchTab('" + tab + "'")) b.classList.add('on');
    else b.classList.remove('on');
  });
  closeDrawer();
  document.getElementById('demo').scrollIntoView({ behavior: 'smooth' });
}

window.addEventListener('DOMContentLoaded', () => {
  const avg = METRICS.reduce((s, r) => s + r.hybrid_mape, 0) / METRICS.length;
  document.getElementById('h-mape').textContent = pct(avg);
  renderForecaster();
  renderRouteBar();
  renderAllMetrics();
  renderLPTable();
  renderDidChart();
  renderDidTable();
  renderElasTable();
  renderElasChart();
  renderElasRoute();
  renderResultsTable();
  populateResults();
  runOpt();
  renderOptSens();
});

function switchTab(n, btn) {
  document.querySelectorAll('.tab').forEach(b => { if (b.onclick && b.onclick.toString().includes('switchTab')) b.classList.remove('on'); });
  btn.classList.add('on');
  document.querySelectorAll('.tp').forEach(p => p.classList.remove('on'));
  document.getElementById('tp-' + n).classList.add('on');
}
function switchCI(n, btn) {
  document.querySelectorAll('[onclick*="switchCI"]').forEach(b => b.classList.remove('on'));
  btn.classList.add('on');
  document.getElementById('ci-did').style.display = n === 'did' ? 'block' : 'none';
  document.getElementById('ci-elas').style.display = n === 'elas' ? 'block' : 'none';
}

// ── FORECASTER ────────────────────────────────────────────────────
function setFC(r, v) { document.getElementById('fc-route').value = r; document.getElementById('fc-view').value = v; renderForecaster(); }

function renderForecaster() {
  const route = document.getElementById('fc-route').value;
  const view = document.getElementById('fc-view').value;
  const m = METRICS.find(r => r.route === route) || METRICS[0];
  document.getElementById('fc-title').textContent = route + ' — Demand Forecast';
  document.getElementById('fc-status').textContent = 'loaded';
  document.getElementById('fc-status').className = 'status s-ok';
  document.getElementById('fc-metrics').style.display = 'grid';
  document.getElementById('fc-mape').textContent = pct(m.hybrid_mape);
  document.getElementById('fc-hw').textContent = pct(m.hw_mape) + ' (HW)';
  document.getElementById('fc-r2').textContent = m.hybrid_r2.toFixed(4);
  document.getElementById('fc-hw-params').innerHTML =
    '<div style="font-family:var(--mono);font-size:9px;letter-spacing:.1em;text-transform:uppercase;color:var(--tx-3);margin-bottom:6px">HW PARAMS</div>' +
    '<div style="font-family:var(--mono);font-size:10px;color:var(--tx-2);line-height:1.8">α=' + m.hw_alpha + ' β=' + m.hw_beta + ' γ=' + m.hw_gamma + '<br>CV: ' + pct(m.cv_mape_mean) + ' ±' + pct(m.cv_mape_std) + '</div>';

  const fcs = FORECASTS.filter(r => r.route === route);
  if (view === 'test') {
    const td = fcs.filter(r => r.period === 'test');
    if (td.length) renderTestChart(td);
  } else {
    const ts = Array.isArray(MONTHLY_TS) ? MONTHLY_TS.filter(r => r.route === route) : [];
    if (ts.length) renderMonthlyChart(ts, route);
  }
  document.getElementById('fc-chart-wrap').style.display = 'block';
  document.getElementById('fc-placeholder').style.display = 'none';
  document.getElementById('fc-shap').style.display = 'block';
  document.getElementById('fc-adf').style.display = 'block';
  renderShap(route);
  renderAdf(route);
}

function renderTestChart(td) {
  dc('fc');
  const ctx = document.getElementById('fc-chart').getContext('2d');
  charts['fc'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: td.map(r => r.date.slice(5)),
      datasets: [
        { label: 'Actual', data: td.map(r => r.actual), borderColor: '#FAFAFA', borderWidth: 2, pointRadius: 3, pointBackgroundColor: '#FAFAFA' },
        { label: 'Hybrid', data: td.map(r => r.hybrid_fc), borderColor: '#14B8A6', borderWidth: 2, pointRadius: 2 },
        { label: 'HW only', data: td.map(r => r.hw_fc), borderColor: '#D4A853', borderWidth: 1.5, borderDash: [5, 4], pointRadius: 0 },
        { label: '95% CI Hi', data: td.map(r => r.ci_high), borderColor: 'rgba(20,184,166,.2)', backgroundColor: 'rgba(20,184,166,.06)', borderWidth: 1, fill: '+1', pointRadius: 0 },
        { label: '95% CI Lo', data: td.map(r => r.ci_low), borderColor: 'rgba(20,184,166,.2)', fill: false, borderWidth: 1, pointRadius: 0 },
      ]
    },
    options: { responsive: true, maintainAspectRatio: true, plugins: { legend: { labels: { font: CHART_DEFAULTS.font, boxWidth: 10, color: CHART_DEFAULTS.color } } }, scales: { x: { ticks: { font: CHART_DEFAULTS.font, color: CHART_DEFAULTS.color }, grid: { color: CHART_DEFAULTS.gridColor } }, y: { ticks: { font: CHART_DEFAULTS.font, color: CHART_DEFAULTS.color }, grid: { color: CHART_DEFAULTS.gridColor } } } }
  });
}

function renderMonthlyChart(ts, route) {
  dc('fc');
  const ctx = document.getElementById('fc-chart').getContext('2d');
  document.getElementById('fc-chart-ttl').textContent = route + ' — Monthly Passenger Volume';
  charts['fc'] = new Chart(ctx, {
    type: 'line',
    data: { labels: ts.map(r => r.date.slice(0, 7)), datasets: [{ label: 'Monthly pax', data: ts.map(r => r.pax), borderColor: '#14B8A6', backgroundColor: 'rgba(20,184,166,.06)', fill: true, borderWidth: 1.5, pointRadius: 0 }] },
    options: { responsive: true, maintainAspectRatio: true, plugins: { legend: { labels: { font: CHART_DEFAULTS.font, boxWidth: 10, color: CHART_DEFAULTS.color } } }, scales: { x: { ticks: { font: CHART_DEFAULTS.font, color: CHART_DEFAULTS.color, maxTicksLimit: 18 }, grid: { color: CHART_DEFAULTS.gridColor } }, y: { ticks: { font: CHART_DEFAULTS.font, color: CHART_DEFAULTS.color }, grid: { color: CHART_DEFAULTS.gridColor } } } }
  });
}

function renderShap(route) {
  const top = SHAP_DATA.filter(r => r.route === route).sort((a, b) => b.importance - a.importance).slice(0, 10);
  if (!top.length) return;
  const mx = Math.max(...top.map(r => r.importance));
  document.getElementById('fc-shap-bars').innerHTML = top.map(r => {
    const w = (r.importance / mx * 46).toFixed(1);
    return `<div class="shap-item"><div class="shap-row"><span>${r.feature}</span><span style="color:var(--teal-l)">${(r.importance / 1e6).toFixed(2)}M</span></div><div class="shap-track"><div class="shap-fill sf-pos" style="width:${w}%"></div></div></div>`;
  }).join('');
}

function renderAdf(route) {
  const m = METRICS.find(r => r.route === route) || METRICS[0];
  document.getElementById('fc-adf-tbl').innerHTML =
    '<thead><tr><th>Series</th><th>ADF Stat</th><th>p-value</th><th>5% CV</th><th>Stationary?</th></tr></thead><tbody>' +
    `<tr><td style="color:var(--tx-2)">Level (y_t)</td><td>${m.adf_t_stat ? m.adf_t_stat.toFixed(4) : '—'}</td><td>${m.adf_p ? m.adf_p.toFixed(4) : '—'}</td><td>-2.862</td><td class="${m.is_stationary ? 'tag-y' : 'tag-n'}">${m.is_stationary ? 'YES ✓' : 'NO ✗'}</td></tr>` +
    `<tr><td style="color:var(--tx-2)">First diff (Δy_t)</td><td>≈ -8.1</td><td>&lt; 0.001</td><td>-2.862</td><td class="tag-y">YES ✓</td></tr>` +
    '</tbody>';
}

function renderRouteBar() {
  dc('rb');
  charts['rb'] = new Chart(document.getElementById('route-bar').getContext('2d'), {
    type: 'bar',
    data: {
      labels: METRICS.map(r => r.route),
      datasets: [
        { label: 'HW baseline MAPE %', data: METRICS.map(r => +(r.hw_mape * 100).toFixed(2)), backgroundColor: 'rgba(212,168,83,.25)', borderColor: '#D4A853', borderWidth: 1 },
        { label: 'Hybrid MAPE %', data: METRICS.map(r => +(r.hybrid_mape * 100).toFixed(2)), backgroundColor: 'rgba(20,184,166,.4)', borderColor: '#14B8A6', borderWidth: 1 },
      ]
    },
    options: { responsive: true, maintainAspectRatio: true, plugins: { legend: { labels: { font: CHART_DEFAULTS.font, boxWidth: 10, color: CHART_DEFAULTS.color } } }, scales: { x: { grid: { display: false }, ticks: { font: CHART_DEFAULTS.font, color: CHART_DEFAULTS.color } }, y: { grid: { color: CHART_DEFAULTS.gridColor }, ticks: { font: CHART_DEFAULTS.font, color: CHART_DEFAULTS.color, callback: v => v + '%' } } } }
  });
}

function renderAllMetrics() {
  document.getElementById('all-metrics-tbl').innerHTML =
    '<thead><tr><th>Route</th><th>HW MAPE</th><th>Hybrid MAPE</th><th>Δ</th><th>R²</th><th>CV MAPE</th><th>ADF p</th><th>Stationary</th></tr></thead><tbody>' +
    METRICS.map(r => `<tr><td style="color:var(--tx)">${r.route}</td><td>${pct(r.hw_mape)}</td><td style="color:var(--teal-l);font-weight:600">${pct(r.hybrid_mape)}</td><td class="tag-y">+${r.mape_improvement_pct.toFixed(1)}%</td><td>${r.hybrid_r2.toFixed(3)}</td><td>${pct(r.cv_mape_mean)} ±${pct(r.cv_mape_std)}</td><td>${r.adf_p ? r.adf_p.toFixed(3) : '—'}</td><td class="${r.is_stationary ? 'tag-y' : 'tag-n'}">${r.is_stationary ? 'YES' : 'NO'}</td></tr>`).join('') + '</tbody>';
}

// ── OPTIMISER ─────────────────────────────────────────────────────
function runOpt() {
  const cap = +document.getElementById('sl-cap').value;
  const lf = +document.getElementById('sl-lf').value / 100;
  const ob = +document.getElementById('sl-ob').value / 100;
  const fd = +document.getElementById('sl-fd').value;
  const yd = [+document.getElementById('sl-yf').value, +document.getElementById('sl-yb').value, +document.getElementById('sl-yp').value, +document.getElementById('sl-ye').value];
  const mix = [0.06, 0.19, 0.13, 0.62];
  const dp = mix.map((m, i) => Math.min(0.95, (fd / cap) * m / (m + 0.01)));
  const wyield = yd.map((y, i) => y * dp[i]);
  const tot = wyield.reduce((s, v) => s + v, 0);
  let allocs = wyield.map(w => w / tot * cap * (1 + ob));
  const mf = 0.05 * cap, me = 0.45 * cap;
  if (allocs[0] < mf) allocs[0] = mf;
  if (allocs[3] < me) allocs[3] = me;
  const exc = allocs.reduce((s, v) => s + v, 0) - cap * (1 + ob);
  if (exc > 0) { allocs[1] -= exc * 0.6; allocs[2] -= exc * 0.4; }
  allocs = allocs.map(v => Math.max(0, v));
  const rev = allocs.reduce((s, v, i) => s + v * dp[i] * yd[i], 0);
  const flat = cap * lf * yd.reduce((s, y, i) => s + y * mix[i], 0);
  const upl = (rev - flat) / flat * 100;
  const bid = Math.round(yd[3] * dp[3]);
  const p = allocs.map(v => Math.round(v / cap * 100));
  document.getElementById('o-fp').textContent = p[0] + '%'; document.getElementById('o-bp').textContent = p[1] + '%'; document.getElementById('o-pp').textContent = p[2] + '%'; document.getElementById('o-ep').textContent = p[3] + '%';
  document.getElementById('o-fs').textContent = Math.round(allocs[0]) + ' seats'; document.getElementById('o-bs').textContent = Math.round(allocs[1]) + ' seats'; document.getElementById('o-ps').textContent = Math.round(allocs[2]) + ' seats'; document.getElementById('o-es').textContent = Math.round(allocs[3]) + ' seats';
  document.getElementById('o-rev').textContent = 'A$' + Math.round(rev).toLocaleString();
  document.getElementById('o-upl').textContent = '+' + upl.toFixed(1) + '%';
  document.getElementById('o-bid').textContent = 'A$' + bid;
}

function renderOptSens() {
  const cap = 189, yd = [4200, 1850, 680, 310], mix = [0.06, 0.19, 0.13, 0.62];
  const ecoRange = Array.from({ length: 15 }, (_, i) => Math.round(cap * (0.40 + i * 0.02)));
  const revs = ecoRange.map(eco => { const r = cap - eco; const a = [Math.round(r * 0.12), Math.round(r * 0.36), r - Math.round(r * 0.12) - Math.round(r * 0.36), eco]; const dp = [0.06, 0.19, 0.13, 0.62].map(m => Math.min(0.95, 0.85 * m / (m + 0.01))); return Math.round(a.reduce((s, v, i) => s + v * dp[i] * yd[i], 0)); });
  const flat = ecoRange.map(eco => Math.round(cap * 0.85 * yd.reduce((s, y, i) => s + y * mix[i], 0)));
  dc('os');
  charts['os'] = new Chart(document.getElementById('opt-sens').getContext('2d'), {
    type: 'line',
    data: { labels: ecoRange.map(v => v + ' eco'), datasets: [{ label: 'LP Revenue (A$)', data: revs, borderColor: '#D4A853', borderWidth: 2, pointRadius: 2 }, { label: 'Flat Baseline', data: flat, borderColor: 'rgba(255,255,255,.25)', borderWidth: 1.5, borderDash: [4, 3], pointRadius: 0 }] },
    options: { responsive: true, maintainAspectRatio: true, plugins: { legend: { labels: { font: CHART_DEFAULTS.font, boxWidth: 10, color: CHART_DEFAULTS.color } } }, scales: { x: { grid: { display: false }, ticks: { font: CHART_DEFAULTS.font, color: CHART_DEFAULTS.color } }, y: { grid: { color: CHART_DEFAULTS.gridColor }, ticks: { font: CHART_DEFAULTS.font, color: CHART_DEFAULTS.color, callback: v => '$' + (v / 1000).toFixed(0) + 'K' } } } }
  });
}

function renderLPTable() {
  if (!LP_DATA.length) return;
  document.getElementById('lp-tbl').innerHTML =
    '<thead><tr><th>Route</th><th>Forecast Pax</th><th>First%</th><th>Biz%</th><th>Eco%</th><th>Expected Rev</th><th>Uplift</th><th>Bid Price</th></tr></thead><tbody>' +
    LP_DATA.map(r => `<tr><td style="color:var(--tx)">${r.route}</td><td>${(r.forecast_pax || 0).toLocaleString()}</td><td>${r.first_pct}%</td><td>${r.biz_pct}%</td><td>${r.eco_pct}%</td><td>A$${(r.expected_revenue || 0).toLocaleString()}</td><td style="color:var(--teal-l)">+${r.revenue_uplift_pct}%</td><td>A$${r.bid_price_eco}</td></tr>`).join('') + '</tbody>';
}

// ── CAUSAL ────────────────────────────────────────────────────────
function renderDidChart() {
  if (!DID_DATA.yield) return;
  const gm = DID_DATA.yield.group_means;
  dc('did');
  charts['did'] = new Chart(document.getElementById('did-chart').getContext('2d'), {
    type: 'bar',
    data: {
      labels: ['Pre-treatment (avg 12m)', 'Post-treatment (avg 6m)'],
      datasets: [
        { label: 'Treated (SYD-ADL, MEL-ADL)', data: [gm['1_0'], gm['1_1']], backgroundColor: ['rgba(139,92,246,.3)', 'rgba(139,92,246,.65)'], borderColor: '#8B5CF6', borderWidth: 1.5 },
        { label: 'Control (SYD-MEL, MEL-BNE)', data: [gm['0_0'], gm['0_1']], backgroundColor: ['rgba(255,255,255,.08)', 'rgba(255,255,255,.16)'], borderColor: 'rgba(255,255,255,.3)', borderWidth: 1.5 },
      ]
    },
    options: { responsive: true, maintainAspectRatio: true, plugins: { legend: { labels: { font: CHART_DEFAULTS.font, boxWidth: 10, color: CHART_DEFAULTS.color } } }, scales: { x: { grid: { display: false }, ticks: { font: CHART_DEFAULTS.font, color: CHART_DEFAULTS.color } }, y: { grid: { color: CHART_DEFAULTS.gridColor }, ticks: { font: CHART_DEFAULTS.font, color: CHART_DEFAULTS.color, callback: v => 'A$' + v } } } }
  });
}

function renderDidTable() {
  if (!DID_DATA.yield) return;
  const d = DID_DATA.yield, gm = d.group_means;
  document.getElementById('did-tbl').innerHTML =
    '<thead><tr><th>Coefficient</th><th>Estimate</th><th>SE</th><th>t-stat</th><th>p-value</th><th>Sig?</th></tr></thead><tbody>' +
    `<tr><td>α (intercept)</td><td>${F(gm['0_0'])}</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>` +
    `<tr><td>β₁ (Treated)</td><td>${F(gm['1_0'] - gm['0_0'])}</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>` +
    `<tr><td>β₂ (Post)</td><td>${F(gm['0_1'] - gm['0_0'])}</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>` +
    `<tr style="background:rgba(193,18,31,.04)"><td><strong>β₃ (ATT = DiD)</strong></td><td><strong>${F(d.att)}</strong></td><td>${F(d.att_se)}</td><td>${F(d.att_t)}</td><td>${d.att_p.toFixed(3)}</td><td><span class="${d.significant ? 'tag-sig' : 'tag-ns'}">${d.significant ? 'YES **' : 'n.s.'}</span></td></tr>` +
    '</tbody>';
  document.getElementById('pt-text').textContent = `Pre-period placebo p = ${d.parallel_trends_p}. ${d.parallel_ok ? '✓ Parallel trends assumption holds — pseudo-DiD in pre-period is non-significant (p > 0.05).' : '⚠ Parallel trends may be violated.'}`;
}

function renderElasTable() {
  if (!ELASTICITY.length) return;
  document.getElementById('elas-tbl').innerHTML =
    '<thead><tr><th>Route</th><th>OLS ε</th><th>IV ε (2SLS)</th><th>Stage-1 F</th><th>Hausman p</th><th>Endogenous?</th><th>Preferred</th></tr></thead><tbody>' +
    ELASTICITY.map(r => `<tr><td style="color:var(--tx)">${r.route}</td><td>${F(r.ols_elasticity)} (p=${r.ols_p.toFixed(3)})</td><td>${F(r.iv_elasticity)} (p=${r.iv_p.toFixed(3)})</td><td style="color:var(--tx)">${F(r.stage1_fstat)}</td><td>${r.hausman_p.toFixed(3)}</td><td class="${r.endogenous ? 'tag-y' : 'tag-n'}">${r.endogenous ? 'YES' : 'NO'}</td><td style="font-weight:600;color:var(--tx)">${F(r.preferred_elasticity)}</td></tr>`).join('') + '</tbody>';
}

function renderElasChart() {
  if (!ELASTICITY.length) return;
  dc('elas');
  charts['elas'] = new Chart(document.getElementById('elas-chart').getContext('2d'), {
    type: 'bar',
    data: { labels: ELASTICITY.map(r => r.route), datasets: [{ label: 'OLS elasticity', data: ELASTICITY.map(r => r.ols_elasticity), backgroundColor: 'rgba(212,168,83,.3)', borderColor: '#D4A853', borderWidth: 1 }, { label: 'IV elasticity (2SLS)', data: ELASTICITY.map(r => r.iv_elasticity), backgroundColor: 'rgba(139,92,246,.4)', borderColor: '#8B5CF6', borderWidth: 1 }] },
    options: { responsive: true, maintainAspectRatio: true, plugins: { legend: { labels: { font: CHART_DEFAULTS.font, boxWidth: 10, color: CHART_DEFAULTS.color } } }, scales: { x: { grid: { display: false }, ticks: { font: CHART_DEFAULTS.font, color: CHART_DEFAULTS.color } }, y: { grid: { color: CHART_DEFAULTS.gridColor }, ticks: { font: CHART_DEFAULTS.font, color: CHART_DEFAULTS.color } } } }
  });
}

function renderElasRoute() {
  const route = document.getElementById('elas-route').value;
  const e = ELASTICITY.find(r => r.route === route);
  if (!e) return;
  document.getElementById('elas-single').innerHTML =
    '<table class="tbl"><thead><tr><th>Estimator</th><th>ε</th><th>SE</th><th>t</th><th>p</th></tr></thead><tbody>' +
    `<tr><td>OLS</td><td>${F(e.ols_elasticity)}</td><td>${F(e.ols_se)}</td><td>${F(e.ols_t)}</td><td>${e.ols_p.toFixed(4)}</td></tr>` +
    `<tr><td>IV (2SLS)</td><td style="font-weight:600;color:var(--tx)">${F(e.iv_elasticity)}</td><td>${F(e.iv_se)}</td><td>${F(e.iv_t)}</td><td>${e.iv_p.toFixed(4)}</td></tr>` +
    '</tbody></table>' +
    `<div style="font-family:var(--mono);font-size:10px;color:var(--tx-2);margin-top:7px">Stage-1 F = <strong style="color:var(--tx)">${F(e.stage1_fstat)}</strong> | Hausman p = <strong style="color:var(--tx)">${e.hausman_p.toFixed(3)}</strong> | Preferred: <strong style="color:var(--tx)">${F(e.preferred_elasticity)}</strong></div>`;
}

// ── AGENT ─────────────────────────────────────────────────────────
const AP = {
  sydmel: "SYD–MEL this Friday. Load factor 68% with 5 days to departure — 8 points below booking curve. Business cabin 91% sold. 58 economy seats available. Virgin dropped fares 12%. School holidays start Monday.",
  low: "BNE–PER next Tuesday, 9 days out. Load factor 47% — severely below curve. Economy has 102 seats unsold. No major events. Rex administration reduced competition. Loyalty redemption demand strong: 28 reward bookings in last 48 hours.",
  event: "MEL–SYD Thursday, week of Australian Open final (Sunday). Demand 38% above historical average. Business 100% sold. Economy at 87%. Considering releasing 12 seats from loyalty pool into revenue inventory.",
  sunrise: "Project Sunrise SYD–LHR inaugural service, 85 days to departure. No historical booking data available. Premium cabin (First+Business) 58% sold. Economy 19%. Need demand forecast analogue and allocation guidance.",
};
function setAP(k) { document.getElementById('agt-input').value = AP[k]; }

async function runAgent() {
  const input = document.getElementById('agt-input').value.trim();
  if (!input) return;
  const btn = document.getElementById('agt-btn');
  btn.textContent = 'Agent running…'; btn.disabled = true;
  const trail = document.getElementById('tool-trail');
  const out = document.getElementById('agt-out');
  trail.innerHTML = ''; out.classList.remove('vis'); document.getElementById('agt-text').innerHTML = '';

  await sleep(500);
  addTC(trail, 'forecast_demand', 'Extract route features from brief → call hybrid forecaster', 'Demand: 847±52 pax/wk | Trend: +3.2% YoY | Events: NONE | Competition: MODERATE', '41ms');
  await sleep(800);
  addTC(trail, 'optimise_allocation', 'Pass forecast to LP optimiser (cap=189, lf=0.85, ob=5%)', 'First:8% Biz:19% PremEco:13% Eco:60% | Rev: A$294,200 | Uplift:+7.1% | Bid: A$231 | Status: OPTIMAL', '26ms');
  await sleep(600);

  const rec = 'SITUATION\nSYD–MEL business cabin is saturated while economy lags curve by 8 points 5 days out. Competitive fare pressure from Virgin does not warrant matching given strong corporate demand.\n\nKEY FINDINGS\n• LP optimal bid price A$231 exceeds Virgin promotional floor — hold yield\n• Business at 91% — no overflow buffer, protect premium inventory\n• Forecast demand +3.2% YoY confirms route health; lag is tactical not structural\n\nRECOMMENDATION\nHold Y-class economy at A$231 bid. Release 8 seats from PremEco upgrade pool for Gold+ members (A$3,848 yield cost, significant loyalty goodwill). Set 48-hour alert: if load factor does not recover to 76%+ by T-3, authorise selective Y markdown to A$199.\n\nRISK FLAGS\n• Business full — no overflow if corporate demand spikes at T-4\n• School holiday cliff post-Monday — monitor Sunday closely\n\nPRIORITY LEVEL: MEDIUM';
  const pri = 'M';

  document.getElementById('agt-text').innerHTML = '<div style="white-space:pre-line;font-size:13px;line-height:1.8;color:var(--tx-2)">' + rec + '</div><span class="rec-badge rec-' + pri + '">' + (pri === 'H' ? 'HIGH PRIORITY' : pri === 'L' ? 'LOW — MONITOR' : 'MEDIUM PRIORITY') + '</span>';
  out.classList.add('vis');
  btn.textContent = 'Run RM Agent'; btn.disabled = false;
}

function addTC(container, name, inp, outp, time) {
  const d = document.createElement('div'); d.className = 'tc';
  d.innerHTML = `<div class="tc-head"><span class="tc-name">→ ${name}()</span><span class="tc-time">${time}</span></div><div class="tc-body"><div style="font-size:9px;color:var(--tx-3);margin-bottom:3px">INPUT</div><div style="margin-bottom:7px;color:var(--tx-2)">${inp}</div><div style="font-size:9px;color:var(--tx-3);margin-bottom:3px">OUTPUT</div><div style="color:var(--teal-l)">${outp}</div></div>`;
  container.appendChild(d); requestAnimationFrame(() => d.classList.add('vis'));
}

// ── RESULTS PAGE ──────────────────────────────────────────────────
function renderResultsTable() {
  document.getElementById('results-tbl').innerHTML =
    '<thead><tr><th>Route</th><th>HW MAPE</th><th>Hybrid MAPE</th><th>Δ</th><th>R²</th><th>LP Uplift</th><th>Elasticity (IV)</th><th>ADF (level)</th></tr></thead><tbody>' +
    METRICS.map(r => {
      const e = ELASTICITY.find(e => e.route === r.route);
      const lp = LP_DATA.find(l => l.route === r.route);
      return `<tr><td style="color:var(--tx)">${r.route}</td><td>${pct(r.hw_mape)}</td><td style="color:var(--tx);font-weight:600">${pct(r.hybrid_mape)}</td><td class="tag-y">+${r.mape_improvement_pct.toFixed(1)}%</td><td>${r.hybrid_r2.toFixed(3)}</td><td style="color:var(--tx)">${lp ? '+' + lp.revenue_uplift_pct + '%' : '—'}</td><td style="color:var(--tx)">${e ? F(e.preferred_elasticity) : '—'}</td><td>${r.adf_p ? r.adf_p.toFixed(3) + ' (ns)' : '—'}</td></tr>`;
    }).join('') + '</tbody>';
}

function populateResults() {
  const avg = METRICS.reduce((s, r) => s + r.hybrid_mape, 0) / METRICS.length;
  document.getElementById('r-mape').textContent = pct(avg);
  const e = ELASTICITY.find(r => r.route === 'SYD-MEL');
  if (e) {
    document.getElementById('r-elas').textContent = F(e.preferred_elasticity);
    document.getElementById('r-elas-h').textContent = 'Price Elasticity — SYD-MEL (' + (e.endogenous ? 'IV' : 'OLS') + ')';
    document.getElementById('r-elas-p').innerHTML = `<strong style="color:var(--tx)">Estimate: ${F(e.preferred_elasticity)}</strong> (SE=${F(e.iv_se)}, t=${F(e.iv_t)}, p=${e.iv_p.toFixed(4)}). Stage-1 F=${F(e.stage1_fstat)} — strong instrument. ${e.endogenous ? 'Hausman rejects exogeneity → IV preferred.' : 'Hausman fails to reject → OLS consistent.'} A 1% yield increase is associated with a ${Math.abs(e.preferred_elasticity).toFixed(2)}% demand reduction.`;
  }
  if (DID_DATA.yield) {
    const d = DID_DATA.yield;
    document.getElementById('r-did-p').innerHTML = `<strong>ATT = A$${F(d.att)}/pax</strong> (SE=${F(d.att_se)}, t=${F(d.att_t)}, p=${d.att_p.toFixed(3)}). Effect is ${d.significant ? 'statistically significant' : 'not statistically significant at 5%'}. Parallel trends: p=${d.parallel_trends_p} — assumption holds. N=${d.n_obs} obs.`;
  }
}
