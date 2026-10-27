# FraudLens — Interpretable Fraud Detection for Payment Transactions

**Track:** Cybersecurity & Fraud Detection · DATANOVA 2026 GLOBAL DATA-THON
**Deliverables:** Slide deck (`FraudLens-DATANOVA.pptx` — attached to the
Devpost submission; slide content in SLIDES.md) · Working demo:
https://fraudlens-eight.vercel.app · Source code:
https://github.com/CloudCorpRecords/fraudlens · Optional video
(walkthrough script in DEMO.md)

> Problem-statement note: the problem statements on datanovathon.com load
> dynamically and could not be retrieved as text in our build environment,
> so the problem below is defined directly from the official Cybersecurity &
> Fraud Detection track description on this Devpost page.

## Overview

FraudLens is a fraud-detection pipeline and analyst dashboard that scores
every payment transaction for risk **and explains every score**. An
interpretable logistic-regression model — implemented from scratch in the
Python standard library — is validated honestly against the rules-only
baseline it is designed to replace, on a held-out test set, with every
reported figure reproducible from artifacts included in the repository.

## Problem

Payment fraud is rare (~2% of transactions), fast-moving, and expensive.
Small and mid-size processors still depend on hand-written rules that miss
evolving patterns (in our test set, a typical rules baseline misses 20 of
52 frauds), while black-box ML scores cannot answer the question customers,
auditors, and regulators ask first: *why was this transaction flagged?*
Fraud teams need to catch more fraud, keep false alerts low enough for
their review capacity, and explain every single decision.

## Solution / Methodology

1. **Reproducible synthetic data.** A seeded generator (seed 42) creates
   8,000 transactions with a 2.14% fraud rate. Fraud is drawn from a separate
   behavioural process modelled on stolen-card usage: foreign country,
   unrecognised device, burst velocity, night hours, high-risk merchant
   categories, and bimodal probe/cash-out amounts. No real customer data is
   used, and anyone can regenerate the identical dataset.
2. **Interpretable model.** Logistic regression (gradient descent, class
   weighting, L2 regularisation) over 10 human-readable features. The
   coefficients are the explanation layer — the dashboard renders them as
   feature drivers and attaches reason data to every alert.
3. **Honest validation.** Stratified 70/30 split; the decision threshold is
   selected on the training split only and frozen before testing. A
   transparent points-based rules baseline is evaluated on the identical
   test rows. Per-row predictions are published (`test_predictions.csv`) so
   every metric can be recomputed independently.

## Tech Stack

- Python 3 standard library only — data generation, model training,
  evaluation (precision/recall/F1, ROC-AUC via Mann-Whitney). No installs.
- Static HTML/CSS/vanilla JavaScript dashboard — hand-rolled charts, no
  build step, no dependencies.
- CSV/JSON audit artifacts: `transactions.csv`, `test_predictions.csv`,
  `results.json`.

## Implementation / Demo

One command runs the whole pipeline:

```bash
python3 run.py
python3 -m http.server 8000 --directory dashboard   # → http://localhost:8000
```

The dashboard shows headline metric cards, a model-vs-rules comparison
table, an **interactive threshold slider** that recomputes
precision/recall/confusion live from the real embedded test scores, the
feature-driver chart, and the flagged-transaction queue (47 alerts at the
operating threshold) with the data behind each flag.

## Results / Validation

Held-out test set (n = 2,401; 52 fraud / 2,349 legitimate):

| Metric | FraudLens model | Rules baseline |
|---|---|---|
| Precision | **93.6%** | 91.4% |
| Recall | **84.6%** (44/52) | 61.5% (32/52) |
| F1 | **0.889** | 0.736 |
| ROC-AUC | **0.997** | 0.981 |
| Confusion (TP/FP/TN/FN) | 44 / 3 / 2,346 / 8 | 32 / 3 / 2,346 / 20 |
| Fraud value caught | **$14,662.78 of $16,523.91 (88.7%)** | $13,282.29 (80.4%) |

The model catches **12 more frauds than rules at the identical false-alert
count (3)**; missed fraud falls 60% (20 → 8). Strongest drivers: burst
behaviour (short gap since previous transaction; 24h velocity, OR 2.26× per
SD), unrecognised device (1.91×), night activity (1.87×); account age is
protective (0.60×).

*Stated limitation:* the dataset is synthetic with learnable patterns, so
these metrics validate the method and the interpretability layer rather
than predict production performance; deployment would retrain and
recalibrate on real labelled transactions. Full detail, including all
limitations, is in RESULTS.md in the repository.

## Impact

- **More fraud stopped, same team:** +23.1 points of recall at zero extra
  false alerts means more loss prevented with the same review headcount.
- **Explainable decisions:** every alert carries human-readable drivers —
  usable in customer disputes, audits, and regulatory reviews.
- **Accessible to small teams:** zero-infrastructure (stdlib Python + a
  static page) — no ML platform, cloud account, or paid API required.
- **Responsible by construction:** synthetic data only, no demographic
  features, and documented bias-review flags for country/distance features.

## Team Contributions

- **Rene Turcios (solo participant)** — track and problem selection,
  direction, and acceptance review.
- **AI agent fleet (Tree / Muse)** — dataset design and generation, model
  implementation, evaluation, dashboard build, and documentation, working
  under Rene's direction.

## Build Disclosure

Build Disclosure: AI-built. Built by AI agents (Tree / Muse fleet) directed
by Rene Turcios during the hackathon period; Rene set the direction and
reviewed; agents designed, coded, ran, and documented the analysis.

All results in this submission are genuine outputs of the included pipeline
(`python3 run.py` → `results.json`); no results are plagiarised or
fabricated. Code is MIT-licensed.
