"""Synthetic payment-transaction generator for FraudLens.

Fully reproducible: a single seeded RNG (seed 42) drives every draw.
The dataset is 100% synthetic - no real customer data is used or needed.

Fraud is generated from a *different behavioural process* than legitimate
traffic (stolen-card style patterns: foreign country, unknown device,
burst velocity, night hours, high-risk merchant categories, amounts that
are either tiny "card-testing" probes or large cash-outs). The model in
model.py never sees these generation rules; it must learn them from the
labelled features alone.
"""

import csv
import math
import random

SEED = 42
N_TRANSACTIONS = 8000
TARGET_FRAUD_RATE = 0.022  # ~2.2% of transactions are fraudulent

MERCHANTS = {
    # category: (share of legit traffic, inherent risk score 0..1)
    "grocery":       (0.22, 0.10),
    "fuel":          (0.13, 0.15),
    "restaurant":    (0.16, 0.20),
    "retail":        (0.17, 0.35),
    "online_retail": (0.14, 0.45),
    "travel":        (0.06, 0.55),
    "electronics":   (0.07, 0.70),
    "gift_cards":    (0.05, 0.85),
}
MERCHANT_NAMES = list(MERCHANTS.keys())
MERCHANT_WEIGHTS = [MERCHANTS[m][0] for m in MERCHANT_NAMES]
MERCHANT_RISK = {m: MERCHANTS[m][1] for m in MERCHANT_NAMES}


def _pick_merchant(rng, fraud):
    if fraud:
        # Fraud skews hard toward cash-out friendly categories.
        weights = []
        for m in MERCHANT_NAMES:
            w = MERCHANTS[m][0] * (0.25 + 2.6 * MERCHANTS[m][1])
            weights.append(w)
        return rng.choices(MERCHANT_NAMES, weights=weights)[0]
    return rng.choices(MERCHANT_NAMES, weights=MERCHANT_WEIGHTS)[0]


def _amount(rng, fraud):
    if fraud:
        # Bimodal: small card-testing probes or large cash-outs.
        if rng.random() < 0.38:
            return round(rng.uniform(1.0, 15.0), 2)
        return round(rng.lognormvariate(math.log(420), 0.65), 2)
    return round(rng.lognormvariate(math.log(52), 0.85), 2)


def generate(n=N_TRANSACTIONS, seed=SEED):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        fraud = 1 if rng.random() < TARGET_FRAUD_RATE else 0
        merchant = _pick_merchant(rng, fraud)
        if fraud:
            hour = rng.choices(
                range(24),
                weights=[3, 4, 5, 5, 4, 3, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3],
            )[0]
            country_mismatch = 1 if rng.random() < 0.58 else 0
            device_known = 1 if rng.random() < 0.22 else 0
            velocity_1h = rng.choices([1, 2, 3, 4, 5, 6],
                                      weights=[18, 20, 22, 18, 12, 10])[0]
            velocity_24h = velocity_1h + rng.randint(0, 9)
            account_age_days = int(rng.expovariate(1 / 260)) + 1
            distance_km = round(rng.lognormvariate(math.log(420), 1.0), 1) \
                if country_mismatch else round(rng.uniform(0, 60), 1)
            days_since_prev = round(rng.uniform(0, 3), 2)
        else:
            hour = rng.choices(
                range(24),
                weights=[1, 1, 1, 1, 1, 2, 4, 6, 7, 7, 7, 7,
                         7, 6, 6, 6, 6, 7, 7, 6, 5, 4, 3, 2],
            )[0]
            country_mismatch = 1 if rng.random() < 0.045 else 0
            device_known = 0 if rng.random() < 0.09 else 1
            velocity_1h = rng.choices([1, 2, 3, 4],
                                      weights=[78, 16, 4.5, 1.5])[0]
            velocity_24h = velocity_1h + rng.randint(0, 5)
            account_age_days = int(rng.expovariate(1 / 640)) + 7
            distance_km = round(rng.lognormvariate(math.log(9), 0.9), 1) \
                if not country_mismatch else round(rng.uniform(300, 4000), 1)
            days_since_prev = round(rng.uniform(0, 21), 2)

        rows.append({
            "txn_id": f"T{i + 1:05d}",
            "amount_usd": _amount(rng, fraud),
            "hour_of_day": hour,
            "merchant_category": merchant,
            "merchant_risk": MERCHANT_RISK[merchant],
            "country_mismatch": country_mismatch,
            "device_known": device_known,
            "velocity_1h": velocity_1h,
            "velocity_24h": velocity_24h,
            "account_age_days": account_age_days,
            "distance_from_home_km": distance_km,
            "days_since_prev_txn": days_since_prev,
            "is_fraud": fraud,
        })
    rng.shuffle(rows)
    # Re-assign ids after shuffle so id order leaks nothing about the label.
    for idx, r in enumerate(rows):
        r["txn_id"] = f"T{idx + 1:05d}"
    return rows


def write_csv(rows, path):
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(__file__), "..", "data", "transactions.csv")
    data = generate()
    write_csv(data, out)
    fraud_n = sum(r["is_fraud"] for r in data)
    print(f"Wrote {len(data)} transactions ({fraud_n} fraud, "
          f"{fraud_n / len(data):.2%}) to {out}")
