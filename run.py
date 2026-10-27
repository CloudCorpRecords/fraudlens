#!/usr/bin/env python3
"""FraudLens - one-command pipeline.

    python3 run.py

Steps (all deterministic, Python stdlib only, no installs, no network):
  1. Generate the synthetic transactions dataset  -> data/transactions.csv
  2. Train interpretable logistic regression + rules baseline,
     evaluate on a held-out stratified test split -> results.json,
     data/test_predictions.csv
  3. Render the dashboard data file                -> dashboard/data.js

Then view the dashboard:
    python3 -m http.server 8000 --directory dashboard
    # open http://localhost:8000
"""

import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "src"))

from generate_data import generate, write_csv          # noqa: E402
from model import run_pipeline                          # noqa: E402


def main():
    rows = generate()
    write_csv(rows, os.path.join(BASE, "data", "transactions.csv"))
    print(f"[1/3] Dataset: {len(rows)} transactions "
          f"({sum(r['is_fraud'] for r in rows)} fraud) -> data/transactions.csv")

    results = run_pipeline(BASE)
    mm = results["test_metrics_model"]
    print(f"[2/3] Model evaluated on held-out test set "
          f"(n={results['dataset']['test_size']}): "
          f"precision={mm['precision']:.3f} recall={mm['recall']:.3f} "
          f"F1={mm['f1']:.3f} ROC-AUC={mm['roc_auc']:.3f} "
          f"(threshold {results['model']['threshold']})")

    with open(os.path.join(BASE, "dashboard", "data.js"), "w") as f:
        f.write("window.FRAUDLENS = ")
        json.dump(results, f)
        f.write(";\n")
    print("[3/3] Dashboard data -> dashboard/data.js")
    print("View:  python3 -m http.server 8000 --directory dashboard"
          "  ->  http://localhost:8000")


if __name__ == "__main__":
    main()
