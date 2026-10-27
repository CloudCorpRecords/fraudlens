# FraudLens — Interpretable Fraud Detection for Payment Transactions

**DATANOVA 2026 GLOBAL DATA-THON · Track: Cybersecurity & Fraud Detection**

> **Problem statement note:** The DataNova problem statements on
> datanovathon.com load dynamically and could not be retrieved as text in our
> environment, so this project defines its problem directly from the official
> Cybersecurity & Fraud Detection track description on the Devpost page:
> build a data-driven solution that detects fraudulent payment activity and
> communicates its reasoning. Problem defined below.

## Problem

Payment fraud is rare (~2% of transactions), fast, and expensive — and the
analysts who review alerts are drowning. Most small and mid-size processors
still rely on hand-written rules ("flag if amount > $500 AND foreign AND new
device…") that miss evolving patterns, while black-box ML scores can't be
explained to a customer, an auditor, or a regulator asking *"why was this
transaction blocked?"*

**FraudLens** is a fraud-detection pipeline and analyst dashboard that
answers two questions for every transaction — *how risky is this?* and
*why?* — and proves, on a held-out test set, that an interpretable model can
beat the rules baseline it would replace.

## Solution & methodology

1. **Synthetic, reproducible data.** `src/generate_data.py` generates 8,000
   transactions (seed 42) with a 2.14% fraud rate. Fraud is drawn from a
   separate behavioural process modelled on stolen-card usage — foreign
   country, unrecognised device, burst velocity, night hours, high-risk
   merchant categories, bimodal "card-testing probe / large cash-out"
   amounts. Using synthetic data means zero privacy risk and full
   reproducibility: anyone can regenerate the exact dataset.
2. **Interpretable model.** Logistic regression implemented from scratch in
   the Python standard library (gradient descent, class weighting, L2
   regularisation) over 10 human-readable features: log amount, night flag,
   merchant risk, country mismatch, device unknown, velocity 1h/24h, log
   account age, log distance from home, days since previous transaction.
   Coefficients *are* the explanation — the dashboard turns them into
   reason codes for every alert.
3. **Honest validation.** Stratified 70/30 split; the decision threshold
   (0.97) is chosen on the training split only and frozen before test
   evaluation. A transparent rules-only baseline is scored on the identical
   test rows for a fair comparison. Every metric is recomputed from
   `data/test_predictions.csv` — see `RESULTS.md`.
4. **Analyst dashboard.** A static web dashboard (no build step, no
   dependencies) showing headline metrics, model-vs-rules comparison, the
   feature-driver chart, the flagged-transaction queue with reason data,
   and an **interactive threshold slider** that recomputes
   precision/recall/confusion live from the real embedded test scores —
   the exact trade-off a fraud team tunes to its review capacity.

## Results (held-out test set, n = 2,401 — full detail in RESULTS.md)

| | FraudLens model | Rules baseline |
|---|---|---|
| Precision | **93.6%** | 91.4% |
| Recall | **84.6%** (44/52 frauds) | 61.5% (32/52) |
| F1 | **0.889** | 0.736 |
| ROC-AUC | **0.997** | 0.981 |
| False alerts | 3 | 3 |
| Fraud value caught | **$14,662.78 / $16,523.91 (88.7%)** | $13,282.29 (80.4%) |

The model catches 12 more frauds than rules at the *same* false-alert count.
Top drivers: burst behaviour (short gap since previous transaction, high
24h velocity), unrecognised device, and night-time activity raise risk;
account age lowers it.

*Limitation, stated plainly:* the data is synthetic with learnable patterns,
so these metrics validate the method and the explainability layer — a
production deployment would retrain and recalibrate on real labelled
transactions. See RESULTS.md → "Honest limitations".

## Tech stack

- **Python 3 standard library only** — data generation, logistic regression,
  evaluation (precision/recall/F1/ROC-AUC via Mann-Whitney). No pip installs.
- **Static HTML/CSS/vanilla JS dashboard** — charts and the threshold
  explorer are hand-rolled; data is injected as `dashboard/data.js` by the
  pipeline.
- **CSV/JSON artifacts** — `data/transactions.csv`,
  `data/test_predictions.csv`, `results.json` for full auditability.

## How to run

```bash
python3 run.py                                          # 1 command: data → model → results → dashboard data
python3 -m http.server 8000 --directory dashboard       # serve the dashboard
# open http://localhost:8000
```

(Opening `dashboard/index.html` directly from disk also works.)

## Repository layout

```
run.py                  one-command pipeline
src/generate_data.py    seeded synthetic dataset generator
src/model.py            features, logistic regression, rules baseline, metrics
data/                   generated dataset + per-row test predictions
dashboard/              index.html · styles.css · app.js · data.js (generated)
results.json            machine-readable results (source of every number)
RESULTS.md              full results, drivers, limitations
SLIDES.md               slide-deck content (PPT source)
SUBMISSION.md           Devpost submission copy
DEMO.md                 demo walkthrough script
LICENSE                 MIT
```

## Data provenance & responsible AI

- 100% synthetic data generated by this repository — no real customer data,
  no personal data, no external datasets, no paid APIs.
- No demographic features are used. Country/distance features are flagged in
  RESULTS.md as needing bias review before any real-world use.
- All reported results are genuine outputs of the included run; the
  competition's integrity rules (no plagiarised or fabricated results) are
  satisfied by construction — the pipeline *is* the evidence.

## License

MIT — see `LICENSE`.

## Build Disclosure

Build Disclosure: AI-built. Built by AI agents (Tree / Muse fleet) directed
by Rene Turcios during the hackathon period; Rene set the direction and
reviewed; agents designed, coded, ran, and documented the analysis.
