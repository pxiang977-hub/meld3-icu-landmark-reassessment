#!/usr/bin/env python3
"""Generate the full 72-hour MIMIC A/B/C coefficient vectors locally.

This helper exists only for the public code release. It avoids redistributing a
fitted MIMIC model artifact while preserving the fixed-coefficient external
validation workflow: authorized users derive the coefficients locally from the
same MIMIC H72 landmark cohort, then eICU scripts apply them unchanged.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home() / "Desktop" / "MIMIC_IV_3_1"
INFILE = ROOT / "work" / "meld3_snapshots_wide.parquet"
OUTFILE = ROOT / "work" / "14b_h72_fixed_coefficients_for_external_validation.csv"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def fit_logistic(X, y, max_iter=100, tol=1e-9):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = sigmoid(X @ beta)
        w = np.clip(p * (1 - p), 1e-9, None)
        grad = X.T @ (y - p)
        info = X.T @ (X * w[:, None]) + np.eye(X.shape[1]) * 1e-10
        step = np.linalg.pinv(info) @ grad
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    p = sigmoid(X @ beta)
    w = np.clip(p * (1 - p), 1e-9, None)
    info = X.T @ (X * w[:, None]) + np.eye(X.shape[1]) * 1e-10
    cov = np.linalg.pinv(info)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return beta, se


if not INFILE.exists():
    raise FileNotFoundError(INFILE)

df = pd.read_parquet(INFILE)
cohort = df[
    (df["eligible_h72"] == 1)
    & df["meld3_primary_baseline"].notna()
    & df["meld3_primary_h72"].notna()
].copy()

y = cohort["hospital_expire_flag"].astype(int).to_numpy()
baseline = cohort["meld3_primary_baseline"].astype(float).to_numpy()
h72 = cohort["meld3_primary_h72"].astype(float).to_numpy()
delta = h72 - baseline

specs = {
    "A_baseline": (np.column_stack([np.ones(len(cohort)), baseline]), ["intercept", "baseline_meld"]),
    "B_baseline_plus_delta72": (
        np.column_stack([np.ones(len(cohort)), baseline, delta]),
        ["intercept", "baseline_meld", "delta_meld_72h"],
    ),
    "C_h72_alone": (np.column_stack([np.ones(len(cohort)), h72]), ["intercept", "h72_meld"]),
}

rows = []
for model, (X, terms) in specs.items():
    beta, se = fit_logistic(X, y)
    for term, b, s in zip(terms, beta, se):
        rows.append({"model": model, "term": term, "beta": float(b), "se": float(s)})

pd.DataFrame(rows).to_csv(OUTFILE, index=False)
print(f"Saved locally generated coefficient file: {OUTFILE}")
