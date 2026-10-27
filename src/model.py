"""FraudLens model: interpretable logistic regression (Python stdlib only)
plus a transparent rules-only baseline, evaluated honestly on a held-out
stratified test split.

No third-party packages. Logistic regression is trained with full-batch
gradient descent; AUC is computed with the Mann-Whitney rank statistic.
Every number reported in RESULTS.md / results.json comes out of this file.
"""

import json
import math
import os
import random

from generate_data import generate

# ---------------------------------------------------------------- features
# (name, human label, transform) - transforms keep raw units interpretable
# in the dashboard while giving the model well-scaled inputs.
def build_features(row):
    return {
        "log_amount": math.log1p(row["amount_usd"]),
        "is_night": 1.0 if row["hour_of_day"] < 6 else 0.0,
        "merchant_risk": float(row["merchant_risk"]),
        "country_mismatch": float(row["country_mismatch"]),
        "device_unknown": 1.0 - float(row["device_known"]),
        "velocity_1h": float(row["velocity_1h"]),
        "velocity_24h": float(row["velocity_24h"]),
        "log_account_age": math.log1p(row["account_age_days"]),
        "log_distance_km": math.log1p(row["distance_from_home_km"]),
        "days_since_prev_txn": float(row["days_since_prev_txn"]),
    }

FEATURE_LABELS = {
    "log_amount": "Transaction amount (log)",
    "is_night": "Night-time transaction (00:00-05:59)",
    "merchant_risk": "Merchant category risk",
    "country_mismatch": "Country differs from home country",
    "device_unknown": "Unrecognised device",
    "velocity_1h": "Transactions in past hour",
    "velocity_24h": "Transactions in past 24 hours",
    "log_account_age": "Account age (log days)",
    "log_distance_km": "Distance from home (log km)",
    "days_since_prev_txn": "Days since previous transaction",
}
FEATURES = list(FEATURE_LABELS.keys())


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


# ---------------------------------------------------------------- training
def train_logistic(X, y, iters=900, lr=0.15, l2=0.01):
    """X: list of standardised feature vectors. Returns (weights, bias)."""
    d = len(X[0])
    w = [0.0] * d
    b = 0.0
    n = len(X)
    # Class weighting: fraud is rare, so positives count more in the loss.
    pos = sum(y)
    neg = n - pos
    pos_weight = neg / max(pos, 1)
    for _ in range(iters):
        gw = [0.0] * d
        gb = 0.0
        for xi, yi in zip(X, y):
            p = sigmoid(sum(wj * xj for wj, xj in zip(w, xi)) + b)
            err = (p - yi) * (pos_weight if yi == 1 else 1.0)
            for j in range(d):
                gw[j] += err * xi[j]
            gb += err
        for j in range(d):
            w[j] -= lr * (gw[j] / n + l2 * w[j])
        b -= lr * gb / n
    return w, b


def predict_scores(X, w, b):
    return [sigmoid(sum(wj * xj for wj, xj in zip(w, xi)) + b) for xi in X]


# ---------------------------------------------------------------- metrics
def confusion(y, scores, threshold):
    tp = fp = tn = fn = 0
    for yi, s in zip(y, scores):
        pred = 1 if s >= threshold else 0
        if pred == 1 and yi == 1:
            tp += 1
        elif pred == 1 and yi == 0:
            fp += 1
        elif pred == 0 and yi == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def prfa(tp, fp, tn, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    return precision, recall, f1, accuracy


def auc_score(y, scores):
    pos = [s for yi, s in zip(y, scores) if yi == 1]
    neg = [s for yi, s in zip(y, scores) if yi == 0]
    wins = ties = 0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1
            elif p == q:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def best_threshold(y, scores):
    best_t, best_f1 = 0.5, -1.0
    t = 0.02
    while t <= 0.98:
        f1 = prfa(*confusion(y, scores, t))[2]
        if f1 > best_f1:
            best_f1, best_t = f1, t
        t += 0.01
    return round(best_t, 2)


# ---------------------------------------------------------------- rules
def rule_score(row):
    """Transparent analyst-style rules baseline (points system)."""
    s = 0
    if row["amount_usd"] > 500:
        s += 2
    if row["country_mismatch"]:
        s += 2
    if not row["device_known"]:
        s += 2
    if row["velocity_1h"] >= 3:
        s += 2
    if row["hour_of_day"] < 6:
        s += 1
    if row["merchant_risk"] >= 0.7:
        s += 1
    return s


RULE_FLAG_AT = 5  # flag when points >= 5 (normalised to a 0..1 "score" /10)


# ---------------------------------------------------------------- pipeline
def run_pipeline(base_dir):
    rows = generate()
    rng = random.Random(7)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    fraud_idx = [i for i in idx if rows[i]["is_fraud"] == 1]
    legit_idx = [i for i in idx if rows[i]["is_fraud"] == 0]
    cut_f, cut_l = int(0.7 * len(fraud_idx)), int(0.7 * len(legit_idx))
    train_idx = set(fraud_idx[:cut_f] + legit_idx[:cut_l])
    train_rows = [r for i, r in enumerate(rows) if i in train_idx]
    test_rows = [r for i, r in enumerate(rows) if i not in train_idx]

    Xtr_raw = [[build_features(r)[f] for f in FEATURES] for r in train_rows]
    ytr = [r["is_fraud"] for r in train_rows]
    Xte_raw = [[build_features(r)[f] for f in FEATURES] for r in test_rows]
    yte = [r["is_fraud"] for r in test_rows]

    means = [sum(c) / len(c) for c in zip(*Xtr_raw)]
    stds = []
    for j, col in enumerate(zip(*Xtr_raw)):
        var = sum((v - means[j]) ** 2 for v in col) / len(col)
        stds.append(math.sqrt(var) or 1.0)

    def stdz(X):
        return [[(v - means[j]) / stds[j] for j, v in enumerate(row)]
                for row in X]

    Xtr, Xte = stdz(Xtr_raw), stdz(Xte_raw)
    w, b = train_logistic(Xtr, ytr)
    tr_scores = predict_scores(Xtr, w, b)
    te_scores = predict_scores(Xte, w, b)
    threshold = best_threshold(ytr, tr_scores)

    tp, fp, tn, fn = confusion(yte, te_scores, threshold)
    precision, recall, f1, accuracy = prfa(tp, fp, tn, fn)
    auc = auc_score(yte, te_scores)

    # Rules baseline on the same test set.
    r_scores = [rule_score(r) / 10.0 for r in test_rows]
    rtp, rfp, rtn, rfn = confusion(yte, r_scores, RULE_FLAG_AT / 10.0)
    rp, rr, rf1, racc = prfa(rtp, rfp, rtn, rfn)
    rauc = auc_score(yte, r_scores)

    # Feature drivers: standardised coefficients -> odds ratios.
    drivers = sorted(
        ({"feature": f, "label": FEATURE_LABELS[f],
          "coefficient": round(wj, 4),
          "odds_ratio_per_sd": round(math.exp(wj), 3)}
         for f, wj in zip(FEATURES, w)),
        key=lambda d: abs(d["coefficient"]), reverse=True)

    # Flagged test transactions (model-flagged), highest score first.
    flagged = []
    for r, s in zip(test_rows, te_scores):
        if s >= threshold:
            flagged.append({
                "txn_id": r["txn_id"], "amount_usd": r["amount_usd"],
                "merchant_category": r["merchant_category"],
                "hour_of_day": r["hour_of_day"],
                "country_mismatch": r["country_mismatch"],
                "device_known": r["device_known"],
                "velocity_1h": r["velocity_1h"],
                "score": round(s, 4),
                "actual_fraud": r["is_fraud"],
            })
    flagged.sort(key=lambda d: d["score"], reverse=True)

    results = {
        "dataset": {
            "total_transactions": len(rows),
            "fraud_transactions": sum(r["is_fraud"] for r in rows),
            "fraud_rate": round(sum(r["is_fraud"] for r in rows) / len(rows), 4),
            "train_size": len(train_rows), "test_size": len(test_rows),
            "test_fraud": sum(yte), "test_legit": len(yte) - sum(yte),
            "seed_dataset": 42, "seed_split": 7,
            "provenance": "100% synthetic, generated by src/generate_data.py "
                          "(seeded RNG, no real customer data)",
        },
        "model": {
            "type": "Logistic regression (stdlib gradient descent, "
                    "class-weighted, L2=0.01) + rules baseline",
            "threshold": threshold,
            "threshold_chosen_on": "training split (max F1)",
        },
        "test_metrics_model": {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "accuracy": round(accuracy, 4),
            "roc_auc": round(auc, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        },
        "test_metrics_rules_baseline": {
            "precision": round(rp, 4), "recall": round(rr, 4),
            "f1": round(rf1, 4), "accuracy": round(racc, 4),
            "roc_auc": round(rauc, 4),
            "tp": rtp, "fp": rfp, "tn": rtn, "fn": rfn,
            "rule": "points: amount>500 +2, country mismatch +2, unknown "
                    "device +2, velocity_1h>=3 +2, night +1, merchant "
                    "risk>=0.7 +1; flag at >=5 points",
        },
        "feature_drivers": drivers,
        "flagged_test_transactions": flagged,
        "test_scores_for_dashboard": [
            {"score": round(s, 4), "fraud": yi}
            for s, yi in zip(te_scores, yte)
        ],
    }

    with open(os.path.join(base_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Test predictions CSV for auditability.
    import csv as _csv
    pred_path = os.path.join(base_dir, "data", "test_predictions.csv")
    with open(pred_path, "w", newline="") as f:
        wr = _csv.writer(f)
        wr.writerow(["txn_id", "actual_fraud", "model_score",
                     "flagged_at_threshold", "rule_score"])
        for r, s, rs in zip(test_rows, te_scores, r_scores):
            wr.writerow([r["txn_id"], r["is_fraud"], round(s, 4),
                         1 if s >= threshold else 0, int(rs * 10)])
    return results


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..")
    res = run_pipeline(base)
    print(json.dumps(res["test_metrics_model"], indent=2))
    print(json.dumps(res["test_metrics_rules_baseline"], indent=2))
