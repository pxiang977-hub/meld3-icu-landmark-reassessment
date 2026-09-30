#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Checkpoint E4 - Final robustness before analysis lock

Purpose
-------
1) eICU 72-h one-stay-per-patient sensitivity
2) MIMIC vs eICU local delta-MELD72 effect comparison
3) Classical between-database heterogeneity test
4) Patient-cluster bootstrap heterogeneity (2000 reps)

This script does NOT redefine:
- the 48-h primary landmark
- the 72-h secondary landmark
- MELD 3.0
- the primary outcome
- the eICU fixed-coefficient external validation

It uses:
MIMIC:
  eligible_h72 == 1
  meld3_primary_baseline nonmissing
  meld3_primary_h72 nonmissing
  expected n=766, deaths=164

eICU:
  formal E3B Hospital72 observed-primary cohort
  expected n=293, unique patients=281, hospitals=80, deaths=61
"""

from __future__ import annotations

from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd


HOME = Path.home()
MIMIC_WORK = HOME / "Desktop" / "MIMIC_IV_3_1" / "work"
EICU_WORK = HOME / "Desktop" / "eICU_CRD_2_0" / "work"

EICU_CANDIDATES = [
    EICU_WORK / "E3B_h72_external_predictions.parquet",
    EICU_WORK / "E3B_h72_external_candidate.parquet",
]

OUT_EFFECTS = EICU_WORK / "E4_h72_local_effects.csv"
OUT_ONE_STAY = EICU_WORK / "E4_eicu_one_stay_sensitivity.csv"
OUT_BOOT = EICU_WORK / "E4_h72_heterogeneity_bootstrap.csv"
OUT_HET = EICU_WORK / "E4_h72_heterogeneity_summary.csv"
OUT_QC = EICU_WORK / "E4_final_robustness_qc.csv"
MIMIC_H72_FIXED_COEF = MIMIC_WORK / "14b_h72_fixed_coefficients_for_external_validation.csv"

N_BOOT = 2000
SEED = 20260921

if not MIMIC_H72_FIXED_COEF.exists():
    raise FileNotFoundError(
        f"Missing {MIMIC_H72_FIXED_COEF}. Run mimic/code/14b_export_h72_fixed_coefficients.py first."
    )

_coef_src = pd.read_csv(MIMIC_H72_FIXED_COEF)

def _coef(model, term, field="beta"):
    z = _coef_src[(_coef_src["model"] == model) & (_coef_src["term"] == term)]
    if len(z) != 1:
        raise RuntimeError(f"Could not uniquely load MIMIC coefficient: {model}/{term}")
    return float(z.iloc[0][field])

LOCKED_MIMIC_DELTA_BETA = _coef("B_baseline_plus_delta72", "delta_meld_72h", "beta")
LOCKED_MIMIC_DELTA_SE = _coef("B_baseline_plus_delta72", "delta_meld_72h", "se")

LOCKED_A = (
    _coef("A_baseline", "intercept"),
    _coef("A_baseline", "baseline_meld"),
)
LOCKED_B = (
    _coef("B_baseline_plus_delta72", "intercept"),
    _coef("B_baseline_plus_delta72", "baseline_meld"),
    _coef("B_baseline_plus_delta72", "delta_meld_72h"),
)
LOCKED_C = (
    _coef("C_h72_alone", "intercept"),
    _coef("C_h72_alone", "h72_meld"),
)


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=float), -35, 35)
    return 1.0 / (1.0 + np.exp(-x))


def fit_logistic(X, y, max_iter=200, tol=1e-10):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    beta = np.zeros(X.shape[1], dtype=float)

    for _ in range(max_iter):
        eta = X @ beta
        p = sigmoid(eta)
        w = np.clip(p * (1.0 - p), 1e-9, None)

        score = X.T @ (y - p)
        info = X.T @ (X * w[:, None])
        info += np.eye(info.shape[0]) * 1e-10

        step = np.linalg.pinv(info) @ score
        beta_new = beta + step

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break

        beta = beta_new

    eta = X @ beta
    p = sigmoid(eta)
    w = np.clip(p * (1.0 - p), 1e-9, None)
    info = X.T @ (X * w[:, None]) + np.eye(X.shape[1]) * 1e-10
    cov = np.linalg.pinv(info)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))

    return {
        "beta": beta,
        "se": se,
        "pred": p,
    }


def fit_delta_model(df, y_col, base_col, h72_col):
    y = pd.to_numeric(df[y_col], errors="coerce").astype(int).to_numpy()
    base = pd.to_numeric(df[base_col], errors="coerce").astype(float).to_numpy()
    h72 = pd.to_numeric(df[h72_col], errors="coerce").astype(float).to_numpy()
    delta = h72 - base

    X = np.column_stack([
        np.ones(len(df)),
        base,
        delta,
    ])

    fit = fit_logistic(X, y)

    return {
        "n": len(df),
        "deaths": int(y.sum()),
        "beta_intercept": float(fit["beta"][0]),
        "beta_baseline": float(fit["beta"][1]),
        "beta_delta": float(fit["beta"][2]),
        "se_delta": float(fit["se"][2]),
        "or_delta": float(np.exp(fit["beta"][2])),
        "or_delta_ci_low": float(np.exp(fit["beta"][2] - 1.96 * fit["se"][2])),
        "or_delta_ci_high": float(np.exp(fit["beta"][2] + 1.96 * fit["se"][2])),
    }


def auc_score(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=float)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan

    ranks = pd.Series(pred).rank(method="average").to_numpy()
    return float(
        (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    )


def brier_score(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean((y - pred) ** 2))


def log_loss(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 1e-15, 1 - 1e-15)
    return float(
        -np.mean(
            y * np.log(pred) + (1 - y) * np.log(1 - pred)
        )
    )


def fixed_model_performance(df):
    y = pd.to_numeric(df["hospital_death"], errors="coerce").astype(int).to_numpy()
    base = pd.to_numeric(
        df["meld3_observed_primary_baseline"],
        errors="coerce"
    ).astype(float).to_numpy()
    h72 = pd.to_numeric(
        df["meld3_observed_primary_h72"],
        errors="coerce"
    ).astype(float).to_numpy()
    delta = h72 - base

    lp_A = LOCKED_A[0] + LOCKED_A[1] * base
    lp_B = LOCKED_B[0] + LOCKED_B[1] * base + LOCKED_B[2] * delta
    lp_C = LOCKED_C[0] + LOCKED_C[1] * h72

    pred_A = sigmoid(lp_A)
    pred_B = sigmoid(lp_B)
    pred_C = sigmoid(lp_C)

    return {
        "n": len(df),
        "unique_patients": int(df["uniquepid"].nunique()),
        "hospitals": int(df["hospitalid"].nunique()),
        "deaths": int(y.sum()),
        "mortality_pct": float(100 * y.mean()),
        "auc_A": auc_score(y, pred_A),
        "auc_B": auc_score(y, pred_B),
        "auc_C": auc_score(y, pred_C),
        "delta_auc_B_vs_A": auc_score(y, pred_B) - auc_score(y, pred_A),
        "brier_A": brier_score(y, pred_A),
        "brier_B": brier_score(y, pred_B),
        "brier_C": brier_score(y, pred_C),
        "delta_brier_B_vs_A": brier_score(y, pred_B) - brier_score(y, pred_A),
        "logloss_A": log_loss(y, pred_A),
        "logloss_B": log_loss(y, pred_B),
        "logloss_C": log_loss(y, pred_C),
    }


def locate_mimic_source():
    """
    Use the canonical Checkpoint 12 wide MELD snapshot dataset.

    Do not auto-select among downstream sensitivity datasets that may
    contain the same h72-eligible rows. In particular,
    13c_ipw_analysis_dataset.parquet is an IPW sensitivity dataset,
    not the canonical source for Checkpoint 14.
    """
    p = MIMIC_WORK / "meld3_snapshots_wide.parquet"

    if not p.exists():
        raise FileNotFoundError(
            f"Canonical MIMIC snapshot source not found: {p}"
        )

    required = {
        "eligible_h72",
        "meld3_primary_baseline",
        "meld3_primary_h72",
        "hospital_expire_flag",
    }

    df = pd.read_parquet(p)

    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(
            "Canonical MIMIC snapshot file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    c = df[
        (pd.to_numeric(df["eligible_h72"], errors="coerce") == 1)
        & df["meld3_primary_baseline"].notna()
        & df["meld3_primary_h72"].notna()
    ].copy()

    deaths = int(
        pd.to_numeric(
            c["hospital_expire_flag"],
            errors="coerce"
        ).fillna(0).sum()
    )

    if len(c) != 766 or deaths != 164:
        raise RuntimeError(
            "Canonical MIMIC h72 source failed locked-cohort guard: "
            f"n={len(c)}, deaths={deaths}; expected n=766, deaths=164."
        )

    return p


def read_mimic():
    p = locate_mimic_source()
    df = pd.read_parquet(p)

    c = df[
        (pd.to_numeric(df["eligible_h72"], errors="coerce") == 1)
        & df["meld3_primary_baseline"].notna()
        & df["meld3_primary_h72"].notna()
    ].copy()

    if len(c) != 766:
        raise RuntimeError(f"Expected MIMIC n=766, found {len(c)}")

    deaths = int(
        pd.to_numeric(
            c["hospital_expire_flag"],
            errors="coerce"
        ).fillna(0).sum()
    )

    if deaths != 164:
        raise RuntimeError(f"Expected MIMIC deaths=164, found {deaths}")

    return p, c


def read_eicu():
    for p in EICU_CANDIDATES:
        if p.exists():
            e = pd.read_parquet(p)

            required = {
                "patientunitstayid",
                "uniquepid",
                "hospitalid",
                "hospital_death",
                "meld3_observed_primary_baseline",
                "meld3_observed_primary_h72",
            }

            if required.issubset(e.columns):
                if (
                    len(e) == 293
                    and int(e["hospital_death"].sum()) == 61
                    and int(e["uniquepid"].nunique()) == 281
                ):
                    return p, e.copy()

    raise RuntimeError(
        "Could not locate formal E3B Hospital72 eICU cohort "
        "with n=293, deaths=61, unique patients=281."
    )


def make_one_stay_per_patient(e):
    d = e.copy()

    sort_cols = []

    for c in [
        "hospitaldischargeyear",
        "patienthealthsystemstayid",
        "patientunitstayid",
    ]:
        if c in d.columns:
            sort_cols.append(c)

    if not sort_cols:
        sort_cols = ["patientunitstayid"]

    d = d.sort_values(sort_cols, kind="mergesort")
    d = d.drop_duplicates("uniquepid", keep="first").copy()

    return d, sort_cols


def normal_two_sided_p(z):
    # 2 * upper-tail standard normal probability
    return float(math.erfc(abs(float(z)) / math.sqrt(2.0)))


def bootstrap_eicu_clusters(df, rng):
    ids = df["uniquepid"].astype(str).to_numpy()
    unique_ids = np.unique(ids)

    idx_map = {
        cid: np.where(ids == cid)[0]
        for cid in unique_ids
    }

    sampled = rng.choice(
        unique_ids,
        size=len(unique_ids),
        replace=True,
    )

    idx = np.concatenate([
        idx_map[cid]
        for cid in sampled
    ])

    return df.iloc[idx].copy()


def bootstrap_mimic_rows(df, rng):
    idx = rng.integers(
        low=0,
        high=len(df),
        size=len(df),
    )
    return df.iloc[idx].copy()


print("=" * 86)
print("CHECKPOINT E4 - FINAL ROBUSTNESS BEFORE ANALYSIS LOCK")
print("=" * 86)

mimic_path, m = read_mimic()
eicu_path, e = read_eicu()

print(f"MIMIC source : {mimic_path}")
print(f"eICU source  : {eicu_path}")
print(f"MIMIC h72    : {len(m)} rows, {int(m['hospital_expire_flag'].sum())} deaths")
print(
    f"eICU h72     : {len(e)} stays, "
    f"{e['uniquepid'].nunique()} patients, "
    f"{e['hospitalid'].nunique()} hospitals, "
    f"{int(e['hospital_death'].sum())} deaths"
)

# ------------------------------------------------------------------
# Local delta-MELD72 effects
# ------------------------------------------------------------------

m_effect = fit_delta_model(
    m,
    "hospital_expire_flag",
    "meld3_primary_baseline",
    "meld3_primary_h72",
)

e_effect = fit_delta_model(
    e,
    "hospital_death",
    "meld3_observed_primary_baseline",
    "meld3_observed_primary_h72",
)

effect_rows = [
    {
        "database": "MIMIC-IV",
        **m_effect,
    },
    {
        "database": "eICU-CRD",
        **e_effect,
    },
]

effects = pd.DataFrame(effect_rows)
effects.to_csv(OUT_EFFECTS, index=False)

# Reproducibility guard: same MIMIC model as Checkpoint 14.
if abs(m_effect["beta_delta"] - LOCKED_MIMIC_DELTA_BETA) > 1e-6:
    raise RuntimeError(
        "MIMIC delta beta does not reproduce Checkpoint 14: "
        f"{m_effect['beta_delta']} vs {LOCKED_MIMIC_DELTA_BETA}"
    )

if abs(m_effect["se_delta"] - LOCKED_MIMIC_DELTA_SE) > 1e-6:
    raise RuntimeError(
        "MIMIC delta SE does not reproduce Checkpoint 14: "
        f"{m_effect['se_delta']} vs {LOCKED_MIMIC_DELTA_SE}"
    )

# ------------------------------------------------------------------
# One-stay-per-patient eICU sensitivity
# ------------------------------------------------------------------

one, one_sort_cols = make_one_stay_per_patient(e)

if len(one) != 281 or one["uniquepid"].nunique() != 281:
    raise RuntimeError(
        f"Expected one-stay sensitivity n=281 unique patients, "
        f"found n={len(one)}, unique={one['uniquepid'].nunique()}"
    )

one_fixed = fixed_model_performance(one)
one_local = fit_delta_model(
    one,
    "hospital_death",
    "meld3_observed_primary_baseline",
    "meld3_observed_primary_h72",
)

one_row = {
    "selection_rule": "one deterministic hospitalization per uniquepid",
    "sort_columns": ";".join(one_sort_cols),
    **one_fixed,
    "local_delta_beta": one_local["beta_delta"],
    "local_delta_se": one_local["se_delta"],
    "local_delta_or": one_local["or_delta"],
    "local_delta_or_ci_low": one_local["or_delta_ci_low"],
    "local_delta_or_ci_high": one_local["or_delta_ci_high"],
}

one_df = pd.DataFrame([one_row])
one_df.to_csv(OUT_ONE_STAY, index=False)

# ------------------------------------------------------------------
# Classical heterogeneity
# ------------------------------------------------------------------

diff = e_effect["beta_delta"] - m_effect["beta_delta"]
se_diff = math.sqrt(
    e_effect["se_delta"] ** 2
    + m_effect["se_delta"] ** 2
)
z = diff / se_diff
p_classical = normal_two_sided_p(z)

# ------------------------------------------------------------------
# Bootstrap heterogeneity
# MIMIC: first-admission design -> row-level resampling
# eICU: resample uniquepid clusters
# ------------------------------------------------------------------

rng = np.random.default_rng(SEED)
boot_rows = []

for b in range(N_BOOT):
    mb = bootstrap_mimic_rows(m, rng)
    eb = bootstrap_eicu_clusters(e, rng)

    try:
        mfit = fit_delta_model(
            mb,
            "hospital_expire_flag",
            "meld3_primary_baseline",
            "meld3_primary_h72",
        )
        efit = fit_delta_model(
            eb,
            "hospital_death",
            "meld3_observed_primary_baseline",
            "meld3_observed_primary_h72",
        )
    except Exception:
        continue

    boot_rows.append({
        "rep": b + 1,
        "mimic_beta_delta": mfit["beta_delta"],
        "eicu_beta_delta": efit["beta_delta"],
        "difference_eicu_minus_mimic": (
            efit["beta_delta"] - mfit["beta_delta"]
        ),
    })

    if (b + 1) % 200 == 0:
        print(f"Heterogeneity bootstrap {b+1}/{N_BOOT}")

boot = pd.DataFrame(boot_rows)
boot.to_csv(OUT_BOOT, index=False)

if len(boot) < 1900:
    raise RuntimeError(
        f"Too few successful bootstrap reps: {len(boot)}"
    )

ci_low = float(
    boot["difference_eicu_minus_mimic"].quantile(0.025)
)
ci_high = float(
    boot["difference_eicu_minus_mimic"].quantile(0.975)
)

prob_eicu_smaller = float(
    (
        boot["difference_eicu_minus_mimic"] < 0
    ).mean()
)

het = pd.DataFrame([{
    "mimic_n": m_effect["n"],
    "mimic_deaths": m_effect["deaths"],
    "mimic_beta_delta": m_effect["beta_delta"],
    "mimic_se_delta": m_effect["se_delta"],
    "mimic_or_delta": m_effect["or_delta"],
    "mimic_or_ci_low": m_effect["or_delta_ci_low"],
    "mimic_or_ci_high": m_effect["or_delta_ci_high"],
    "eicu_n": e_effect["n"],
    "eicu_deaths": e_effect["deaths"],
    "eicu_beta_delta": e_effect["beta_delta"],
    "eicu_se_delta": e_effect["se_delta"],
    "eicu_or_delta": e_effect["or_delta"],
    "eicu_or_ci_low": e_effect["or_delta_ci_low"],
    "eicu_or_ci_high": e_effect["or_delta_ci_high"],
    "difference_eicu_minus_mimic": diff,
    "difference_se_classical": se_diff,
    "z_classical": z,
    "p_classical": p_classical,
    "bootstrap_reps": len(boot),
    "bootstrap_difference_ci_low": ci_low,
    "bootstrap_difference_ci_high": ci_high,
    "prob_eicu_effect_smaller": prob_eicu_smaller,
}])

het.to_csv(OUT_HET, index=False)

# ------------------------------------------------------------------
# QC
# ------------------------------------------------------------------

qc_rows = []

def add_qc(check, value, expected, passed):
    qc_rows.append({
        "check": check,
        "value": value,
        "expected": expected,
        "pass": bool(passed),
    })

add_qc("mimic_n", len(m), 766, len(m) == 766)
add_qc(
    "mimic_deaths",
    int(m["hospital_expire_flag"].sum()),
    164,
    int(m["hospital_expire_flag"].sum()) == 164,
)
add_qc("eicu_n", len(e), 293, len(e) == 293)
add_qc(
    "eicu_unique_patients",
    e["uniquepid"].nunique(),
    281,
    e["uniquepid"].nunique() == 281,
)
add_qc(
    "eicu_hospitals",
    e["hospitalid"].nunique(),
    80,
    e["hospitalid"].nunique() == 80,
)
add_qc(
    "eicu_deaths",
    int(e["hospital_death"].sum()),
    61,
    int(e["hospital_death"].sum()) == 61,
)
add_qc(
    "mimic_delta_beta_reproduced",
    m_effect["beta_delta"],
    LOCKED_MIMIC_DELTA_BETA,
    abs(m_effect["beta_delta"] - LOCKED_MIMIC_DELTA_BETA) <= 1e-6,
)
add_qc(
    "mimic_delta_se_reproduced",
    m_effect["se_delta"],
    LOCKED_MIMIC_DELTA_SE,
    abs(m_effect["se_delta"] - LOCKED_MIMIC_DELTA_SE) <= 1e-6,
)
add_qc(
    "one_stay_n",
    len(one),
    281,
    len(one) == 281,
)
add_qc(
    "bootstrap_successful_reps",
    len(boot),
    f">=1900 of {N_BOOT}",
    len(boot) >= 1900,
)

qc = pd.DataFrame(qc_rows)
qc.to_csv(OUT_QC, index=False)

# ------------------------------------------------------------------
# Console output
# ------------------------------------------------------------------

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)

print()
print("===== LOCAL DELTA-MELD72 EFFECTS =====")
print(effects.round(6).to_string(index=False))

print()
print("===== eICU ONE-STAY-PER-PATIENT SENSITIVITY =====")
print(one_df.round(6).to_string(index=False))

print()
print("===== 72H CROSS-DATABASE HETEROGENEITY =====")
print(het.round(6).to_string(index=False))

print()
print("===== FINAL ROBUSTNESS QC =====")
print(qc.to_string(index=False))

if not bool(qc["pass"].all()):
    print()
    print("WARNING: At least one hard QC check failed.")
    sys.exit(2)

print()
print("=" * 86)
print("CHECKPOINT E4 COMPLETE - ALL HARD QC CHECKS PASS")
print("=" * 86)
print(f"Local effects      : {OUT_EFFECTS}")
print(f"One-stay sensitivity: {OUT_ONE_STAY}")
print(f"Bootstrap          : {OUT_BOOT}")
print(f"Heterogeneity      : {OUT_HET}")
print(f"QC                 : {OUT_QC}")
