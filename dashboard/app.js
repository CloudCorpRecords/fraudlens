// FraudLens dashboard - renders only numbers produced by the real model run
// (window.FRAUDLENS, generated into data.js by run.py). No hard-coded metrics.
const D = window.FRAUDLENS;
const m = D.test_metrics_model, r = D.test_metrics_rules_baseline, ds = D.dataset;
const pct = x => (x * 100).toFixed(1) + "%";

const cards = [
  [ds.total_transactions.toLocaleString(), "transactions (synthetic)"],
  [pct(m.precision), "precision (test)"],
  [pct(m.recall), "recall (test)"],
  [pct(m.f1), "F1 score (test)"],
  [m.roc_auc.toFixed(3), "ROC-AUC (test)"],
  [D.flagged_test_transactions.length, "flagged for review"],
];
document.getElementById("cards").innerHTML =
  cards.map(([v, k]) => `<div class="card"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");

function metricTable(a, b) {
  const row = (k, x, y, fmt) => `<tr><td>${k}</td><td><strong>${fmt(x)}</strong></td><td>${fmt(y)}</td></tr>`;
  return `<table><thead><tr><th>Metric (test set, n=${ds.test_size})</th><th>FraudLens model</th><th>Rules baseline</th></tr></thead><tbody>
  ${row("Precision", a.precision, b.precision, pct)}
  ${row("Recall", a.recall, b.recall, pct)}
  ${row("F1", a.f1, b.f1, pct)}
  ${row("ROC-AUC", a.roc_auc, b.roc_auc, x => x.toFixed(3))}
  <tr><td>Confusion (TP / FP / TN / FN)</td><td><strong>${a.tp} / ${a.fp} / ${a.tn} / ${a.fn}</strong></td><td>${b.tp} / ${b.fp} / ${b.tn} / ${b.fn}</td></tr>
  </tbody></table>`;
}
document.getElementById("compare").innerHTML = metricTable(m, r);
document.getElementById("threshold-note").textContent =
  `Model threshold ${D.model.threshold} was selected on the training split only (max F1), then frozen before test evaluation. Rules baseline: ${r.rule}.`;

// Feature drivers as diverging bars (coefficient scale normalised to max).
const maxAbs = Math.max(...D.feature_drivers.map(d => Math.abs(d.coefficient)));
document.getElementById("drivers").innerHTML = D.feature_drivers.map(d => {
  const w = Math.abs(d.coefficient) / maxAbs * 50;
  const cls = d.coefficient >= 0 ? "pos" : "neg";
  const left = d.coefficient >= 0 ? 50 : 50 - w;
  return `<div class="bar-row"><div class="bar-label">${d.label}</div>
    <div class="bar-track"><div class="bar-fill ${cls}" style="left:${left}%;width:${w}%"></div></div>
    <div class="bar-val">coef ${d.coefficient} · OR ${d.odds_ratio_per_sd}×/SD</div></div>`;
}).join("");

// Flagged transactions table (top 40).
document.querySelector("#flagged tbody").innerHTML =
  D.flagged_test_transactions.slice(0, 40).map(t => `<tr>
    <td>${t.txn_id}</td><td>$${t.amount_usd.toFixed(2)}</td><td>${t.merchant_category}</td>
    <td>${String(t.hour_of_day).padStart(2, "0")}:00</td><td>${t.country_mismatch ? "Yes" : "No"}</td>
    <td>${t.device_known ? "Yes" : "No"}</td><td>${t.velocity_1h}</td>
    <td><strong>${t.score.toFixed(3)}</strong></td>
    <td class="${t.actual_fraud ? "yes" : "no"}">${t.actual_fraud ? "FRAUD" : "legit (false alert)"}</td></tr>`).join("");

// Interactive threshold explorer over the real test scores.
const scores = D.test_scores_for_dashboard;
const thr = document.getElementById("thr"), thrVal = document.getElementById("thr-val"),
      live = document.getElementById("live-metrics");
thr.value = Math.round(D.model.threshold * 100);
function update() {
  const t = thr.value / 100; thrVal.textContent = t.toFixed(2);
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (const s of scores) {
    const pred = s.score >= t ? 1 : 0;
    if (pred && s.fraud) tp++; else if (pred) fp++;
    else if (s.fraud) fn++; else tn++;
  }
  const P = tp + fp ? tp / (tp + fp) : 0, R = tp + fn ? tp / (tp + fn) : 0;
  const F1 = P + R ? 2 * P * R / (P + R) : 0;
  live.innerHTML = `<span>Precision <strong>${pct(P)}</strong></span><span>Recall <strong>${pct(R)}</strong></span><span>F1 <strong>${pct(F1)}</strong></span><span>Alerts <strong>${tp + fp}</strong> (${tp} true fraud, ${fp} false alerts, ${fn} missed)</span>`;
}
thr.addEventListener("input", update); update();
