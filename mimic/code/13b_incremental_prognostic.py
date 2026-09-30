from pathlib import Path
import math

import numpy as np
import pandas as pd


# ============================================================
# Checkpoint 13B
# Primary incremental prognostic analysis
#
# Outcome:
#   hospital mortality after the 48-h landmark
#
# Primary cohort:
#   n = 904 paired baseline + 48-h MELD 3.0
#
# Models:
#   A: baseline MELD
#   B: baseline MELD + delta MELD at 48 h
#
# Equivalent QC:
#   baseline MELD + updated 48-h MELD
#
# Secondary:
#   48-h MELD alone
#
# Sensitivity:
#   strict explicit-CVVHD definition
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INFILE = ROOT / "work" / "h48_primary_analysis_cohort.parquet"

OUT_MODELS = ROOT / "work" / "13b_model_summary.csv"
OUT_COEF = ROOT / "work" / "13b_model_coefficients.csv"
OUT_BOOT = ROOT / "work" / "13b_bootstrap_internal_validation.csv"
OUT_CAL = ROOT / "work" / "13b_calibration_deciles.csv"
OUT_PRED = ROOT / "work" / "13b_predictions.parquet"

if not INFILE.exists():
    raise FileNotFoundError(INFILE)


# ============================================================
# Utility functions
# ============================================================

def sigmoid(x):
    x = np.clip(x, -35, 35)
    return 1.0 / (1.0 + np.exp(-x))


def fit_logistic(X, y, max_iter=100, tol=1e-9):
    """
    Unpenalized logistic maximum-likelihood fit by Newton-Raphson.
    X must already include intercept.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    beta = np.zeros(X.shape[1], dtype=float)

    for _ in range(max_iter):

        eta = X @ beta
        p = sigmoid(eta)

        w = np.clip(
            p * (1.0 - p),
            1e-9,
            None
        )

        grad = X.T @ (y - p)

        h = X.T @ (
            X * w[:, None]
        )

        # Tiny stabilizer, not substantive penalization.
        h = h + np.eye(
            h.shape[0]
        ) * 1e-10

        step = np.linalg.pinv(h) @ grad

        beta_new = beta + step

        if np.max(
            np.abs(beta_new - beta)
        ) < tol:
            beta = beta_new
            break

        beta = beta_new

    eta = X @ beta
    p = sigmoid(eta)

    ll = np.sum(
        y * np.log(np.clip(p, 1e-15, 1))
        + (1-y) * np.log(
            np.clip(1-p, 1e-15, 1)
        )
    )

    w = np.clip(
        p * (1-p),
        1e-9,
        None
    )

    info = X.T @ (
        X * w[:, None]
    )

    cov = np.linalg.pinv(info)

    se = np.sqrt(
        np.clip(
            np.diag(cov),
            0,
            None
        )
    )

    return {
        "beta": beta,
        "se": se,
        "pred": p,
        "lp": eta,
        "ll": ll,
        "cov": cov,
    }


def auc_score(y, p):

    y = np.asarray(y)
    p = np.asarray(p)

    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())

    if n1 == 0 or n0 == 0:
        return np.nan

    ranks = (
        pd.Series(p)
        .rank(method="average")
        .to_numpy()
    )

    auc = (
        ranks[y == 1].sum()
        - n1 * (n1 + 1) / 2
    ) / (n1 * n0)

    return float(auc)


def brier_score(y, p):

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    return float(
        np.mean((y-p)**2)
    )


def log_loss(y, p):

    y = np.asarray(y, dtype=float)

    p = np.clip(
        np.asarray(p, dtype=float),
        1e-15,
        1-1e-15
    )

    return float(
        -np.mean(
            y*np.log(p)
            + (1-y)*np.log(1-p)
        )
    )


def calibration_on_predictions(y, p):
    """
    Logistic recalibration:
        logit(Y) = intercept + slope * logit(predicted risk)
    """

    p = np.clip(
        np.asarray(p, dtype=float),
        1e-6,
        1-1e-6
    )

    lp = np.log(
        p / (1-p)
    )

    X = np.column_stack(
        [
            np.ones(len(lp)),
            lp
        ]
    )

    fit = fit_logistic(
        X,
        y
    )

    return (
        float(fit["beta"][0]),
        float(fit["beta"][1])
    )


def model_metrics(name, fit, y, k):

    p = fit["pred"]

    cal_int, cal_slope = (
        calibration_on_predictions(
            y,
            p
        )
    )

    return {
        "model": name,

        "n":
            len(y),

        "deaths":
            int(np.sum(y)),

        "auc":
            auc_score(y, p),

        "brier":
            brier_score(y, p),

        "log_loss":
            log_loss(y, p),

        "log_likelihood":
            float(fit["ll"]),

        "aic":
            float(
                -2*fit["ll"]
                + 2*k
            ),

        "calibration_intercept":
            cal_int,

        "calibration_slope":
            cal_slope,
    }


def coefficient_table(
    model,
    fit,
    names
):

    rows = []

    for i, name in enumerate(names):

        beta = float(
            fit["beta"][i]
        )

        se = float(
            fit["se"][i]
        )

        z = (
            beta / se
            if se > 0
            else np.nan
        )

        # Normal approximation.
        p = (
            math.erfc(
                abs(z)
                / math.sqrt(2)
            )
            if np.isfinite(z)
            else np.nan
        )

        lo = beta - 1.96*se
        hi = beta + 1.96*se

        rows.append({
            "model":
                model,

            "term":
                name,

            "beta":
                beta,

            "se":
                se,

            "odds_ratio":
                math.exp(beta),

            "or_95ci_low":
                math.exp(lo),

            "or_95ci_high":
                math.exp(hi),

            "p_value":
                p,
        })

    return rows


# ============================================================
# Load frozen primary cohort
# ============================================================

df = pd.read_parquet(
    INFILE
).copy()

if len(df) != 904:
    raise RuntimeError(
        f"Expected n=904, found {len(df)}"
    )

y = (
    df["hospital_expire_flag"]
    .astype(int)
    .to_numpy()
)

if y.sum() != 205:
    raise RuntimeError(
        f"Expected 205 deaths, found {y.sum()}"
    )


baseline = (
    df["meld3_primary_baseline"]
    .astype(float)
    .to_numpy()
)

updated = (
    df["meld3_primary_h48"]
    .astype(float)
    .to_numpy()
)

delta = (
    updated - baseline
)


strict_baseline = (
    df["meld3_strict_baseline"]
    .astype(float)
    .to_numpy()
)

strict_updated = (
    df["meld3_strict_h48"]
    .astype(float)
    .to_numpy()
)

strict_delta = (
    strict_updated
    - strict_baseline
)


print("=" * 78)
print("CHECKPOINT 13B - INCREMENTAL PROGNOSTIC ANALYSIS")
print("=" * 78)

print(
    f"Primary cohort: {len(df):,}; "
    f"deaths: {int(y.sum()):,}; "
    f"mortality: {100*y.mean():.2f}%"
)


# ============================================================
# Primary model fits
# ============================================================

X_A = np.column_stack(
    [
        np.ones(len(df)),
        baseline
    ]
)

X_B = np.column_stack(
    [
        np.ones(len(df)),
        baseline,
        delta
    ]
)

# Same linear model space as baseline + delta.
X_B_EQ = np.column_stack(
    [
        np.ones(len(df)),
        baseline,
        updated
    ]
)

X_C = np.column_stack(
    [
        np.ones(len(df)),
        updated
    ]
)

# Strict RRT sensitivity.
X_B_STRICT = np.column_stack(
    [
        np.ones(len(df)),
        strict_baseline,
        strict_delta
    ]
)


fit_A = fit_logistic(
    X_A,
    y
)

fit_B = fit_logistic(
    X_B,
    y
)

fit_B_EQ = fit_logistic(
    X_B_EQ,
    y
)

fit_C = fit_logistic(
    X_C,
    y
)

fit_B_STRICT = fit_logistic(
    X_B_STRICT,
    y
)


# Exact equivalence QC.
equiv_ll_diff = abs(
    fit_B["ll"]
    - fit_B_EQ["ll"]
)

equiv_pred_diff = np.max(
    np.abs(
        fit_B["pred"]
        - fit_B_EQ["pred"]
    )
)

if equiv_ll_diff > 1e-6:
    raise RuntimeError(
        "baseline+delta and baseline+updated "
        "are not numerically equivalent."
    )


# ============================================================
# Model performance
# ============================================================

metrics = []

metrics.append(
    model_metrics(
        "A_baseline_meld",
        fit_A,
        y,
        k=2
    )
)

metrics.append(
    model_metrics(
        "B_baseline_plus_delta",
        fit_B,
        y,
        k=3
    )
)

metrics.append(
    model_metrics(
        "B_equivalent_baseline_plus_h48",
        fit_B_EQ,
        y,
        k=3
    )
)

metrics.append(
    model_metrics(
        "C_h48_meld_alone",
        fit_C,
        y,
        k=2
    )
)

metrics.append(
    model_metrics(
        "B_strict_rrt_sensitivity",
        fit_B_STRICT,
        y,
        k=3
    )
)

model_summary = pd.DataFrame(
    metrics
)


# ============================================================
# Likelihood-ratio improvement: B versus A
#
# One extra degree of freedom: delta MELD.
# For chi-square df=1:
# survival function = erfc(sqrt(LR/2))
# ============================================================

lr = max(
    0.0,
    2.0 * (
        fit_B["ll"]
        - fit_A["ll"]
    )
)

lr_p = math.erfc(
    math.sqrt(
        lr / 2.0
    )
)

model_summary[
    "lr_vs_baseline"
] = np.nan

model_summary[
    "lr_p_vs_baseline"
] = np.nan

m = (
    model_summary["model"]
    == "B_baseline_plus_delta"
)

model_summary.loc[
    m,
    "lr_vs_baseline"
] = lr

model_summary.loc[
    m,
    "lr_p_vs_baseline"
] = lr_p


# Incremental performance changes.
auc_A = float(
    model_summary.loc[
        model_summary["model"]
        == "A_baseline_meld",
        "auc"
    ].iloc[0]
)

auc_B = float(
    model_summary.loc[
        model_summary["model"]
        == "B_baseline_plus_delta",
        "auc"
    ].iloc[0]
)

brier_A = float(
    model_summary.loc[
        model_summary["model"]
        == "A_baseline_meld",
        "brier"
    ].iloc[0]
)

brier_B = float(
    model_summary.loc[
        model_summary["model"]
        == "B_baseline_plus_delta",
        "brier"
    ].iloc[0]
)

model_summary[
    "delta_auc_vs_A"
] = np.nan

model_summary[
    "delta_brier_vs_A"
] = np.nan

model_summary.loc[
    m,
    "delta_auc_vs_A"
] = (
    auc_B - auc_A
)

model_summary.loc[
    m,
    "delta_brier_vs_A"
] = (
    brier_B - brier_A
)


model_summary[
    "equivalent_model_ll_abs_diff"
] = np.nan

model_summary[
    "equivalent_model_max_pred_abs_diff"
] = np.nan

model_summary.loc[
    m,
    "equivalent_model_ll_abs_diff"
] = equiv_ll_diff

model_summary.loc[
    m,
    "equivalent_model_max_pred_abs_diff"
] = equiv_pred_diff


# ============================================================
# Coefficients
# ============================================================

coef_rows = []

coef_rows += coefficient_table(
    "A_baseline_meld",
    fit_A,
    [
        "intercept",
        "baseline_meld"
    ]
)

coef_rows += coefficient_table(
    "B_baseline_plus_delta",
    fit_B,
    [
        "intercept",
        "baseline_meld",
        "delta_meld_48h"
    ]
)

coef_rows += coefficient_table(
    "C_h48_meld_alone",
    fit_C,
    [
        "intercept",
        "h48_meld"
    ]
)

coef_rows += coefficient_table(
    "B_strict_rrt_sensitivity",
    fit_B_STRICT,
    [
        "intercept",
        "baseline_meld",
        "delta_meld_48h"
    ]
)

coef = pd.DataFrame(
    coef_rows
)


# ============================================================
# Calibration deciles
# ============================================================

prediction_sets = {
    "A_baseline_meld":
        fit_A["pred"],

    "B_baseline_plus_delta":
        fit_B["pred"],

    "C_h48_meld_alone":
        fit_C["pred"],
}

cal_rows = []

for name, pred in prediction_sets.items():

    tmp = pd.DataFrame({
        "y": y,
        "pred": pred
    })

    # qcut may drop duplicate edges.
    tmp["group"] = pd.qcut(
        tmp["pred"],
        q=10,
        duplicates="drop"
    )

    z = (
        tmp.groupby(
            "group",
            observed=True
        )
        .agg(
            n=("y", "size"),
            deaths=("y", "sum"),
            mean_predicted=("pred", "mean"),
            observed_mortality=("y", "mean"),
        )
        .reset_index(drop=True)
    )

    z["model"] = name
    z["decile"] = (
        np.arange(len(z))
        + 1
    )

    cal_rows.append(z)

calibration = pd.concat(
    cal_rows,
    ignore_index=True
)


# ============================================================
# Bootstrap internal validation
#
# Primary models A and B only.
# B = 1000 resamples.
#
# Optimism:
#   apparent performance in bootstrap sample
#   minus performance of same model on original cohort.
# ============================================================

N_BOOT = 1000
rng = np.random.default_rng(20260919)

boot_rows = []

for b in range(N_BOOT):

    idx = rng.integers(
        0,
        len(df),
        size=len(df)
    )

    yb = y[idx]

    # Extremely unlikely, but guard against one-class sample.
    if len(np.unique(yb)) < 2:
        continue

    XA_b = X_A[idx]
    XB_b = X_B[idx]

    fA = fit_logistic(
        XA_b,
        yb
    )

    fB = fit_logistic(
        XB_b,
        yb
    )

    # Bootstrap apparent predictions.
    pA_app = fA["pred"]
    pB_app = fB["pred"]

    # Apply bootstrap model to original data.
    pA_test = sigmoid(
        X_A @ fA["beta"]
    )

    pB_test = sigmoid(
        X_B @ fB["beta"]
    )

    # Calibration slope/intercept on original data
    # for the bootstrap-fitted models.
    cal_A_int, cal_A_slope = (
        calibration_on_predictions(
            y,
            pA_test
        )
    )

    cal_B_int, cal_B_slope = (
        calibration_on_predictions(
            y,
            pB_test
        )
    )

    boot_rows.append({
        "iteration": b + 1,

        "auc_A_app":
            auc_score(
                yb,
                pA_app
            ),

        "auc_A_test":
            auc_score(
                y,
                pA_test
            ),

        "auc_B_app":
            auc_score(
                yb,
                pB_app
            ),

        "auc_B_test":
            auc_score(
                y,
                pB_test
            ),

        "brier_A_app":
            brier_score(
                yb,
                pA_app
            ),

        "brier_A_test":
            brier_score(
                y,
                pA_test
            ),

        "brier_B_app":
            brier_score(
                yb,
                pB_app
            ),

        "brier_B_test":
            brier_score(
                y,
                pB_test
            ),

        "cal_A_intercept_test":
            cal_A_int,

        "cal_A_slope_test":
            cal_A_slope,

        "cal_B_intercept_test":
            cal_B_int,

        "cal_B_slope_test":
            cal_B_slope,
    })

    if (b + 1) % 100 == 0:
        print(
            f"Bootstrap {b+1}/{N_BOOT}"
        )


boot = pd.DataFrame(
    boot_rows
)


def corrected_metrics(
    model,
    apparent_auc,
    apparent_brier,
    app_auc_col,
    test_auc_col,
    app_brier_col,
    test_brier_col,
    cal_int_col,
    cal_slope_col
):

    auc_optimism = (
        boot[app_auc_col]
        - boot[test_auc_col]
    ).mean()

    brier_optimism = (
        boot[app_brier_col]
        - boot[test_brier_col]
    ).mean()

    corrected_auc = (
        apparent_auc
        - auc_optimism
    )

    corrected_brier = (
        apparent_brier
        - brier_optimism
    )

    # In an apparent dataset the fitted model has
    # calibration intercept ~0 and slope ~1.
    # Bootstrap test values provide an optimism-corrected
    # estimate of calibration in new-but-similar patients.
    corrected_intercept = (
        boot[cal_int_col].mean()
    )

    corrected_slope = (
        boot[cal_slope_col].mean()
    )

    return {
        "model":
            model,

        "bootstrap_reps":
            len(boot),

        "apparent_auc":
            apparent_auc,

        "mean_auc_optimism":
            auc_optimism,

        "optimism_corrected_auc":
            corrected_auc,

        "apparent_brier":
            apparent_brier,

        "mean_brier_optimism":
            brier_optimism,

        "optimism_corrected_brier":
            corrected_brier,

        "bootstrap_corrected_calibration_intercept":
            corrected_intercept,

        "bootstrap_corrected_calibration_slope":
            corrected_slope,

        "calibration_slope_p025":
            boot[
                cal_slope_col
            ].quantile(0.025),

        "calibration_slope_p975":
            boot[
                cal_slope_col
            ].quantile(0.975),
    }


bootstrap_summary = pd.DataFrame([
    corrected_metrics(
        "A_baseline_meld",
        auc_A,
        brier_A,
        "auc_A_app",
        "auc_A_test",
        "brier_A_app",
        "brier_A_test",
        "cal_A_intercept_test",
        "cal_A_slope_test",
    ),

    corrected_metrics(
        "B_baseline_plus_delta",
        auc_B,
        brier_B,
        "auc_B_app",
        "auc_B_test",
        "brier_B_app",
        "brier_B_test",
        "cal_B_intercept_test",
        "cal_B_slope_test",
    ),
])


# ============================================================
# Save predictions
# ============================================================

pred = df[
    [
        "subject_id",
        "hospital_expire_flag",
        "meld3_primary_baseline",
        "meld3_primary_h48",
        "delta_meld3",
    ]
].copy()

pred[
    "pred_A_baseline"
] = fit_A["pred"]

pred[
    "pred_B_baseline_plus_delta"
] = fit_B["pred"]

pred[
    "pred_C_h48_only"
] = fit_C["pred"]

pred[
    "pred_B_strict"
] = fit_B_STRICT["pred"]


# ============================================================
# Save files
# ============================================================

for col in [
    "auc",
    "brier",
    "log_loss",
    "log_likelihood",
    "aic",
    "calibration_intercept",
    "calibration_slope",
    "lr_vs_baseline",
    "lr_p_vs_baseline",
    "delta_auc_vs_A",
    "delta_brier_vs_A",
]:
    if col in model_summary.columns:
        model_summary[col] = (
            model_summary[col]
            .astype(float)
        )

model_summary.to_csv(
    OUT_MODELS,
    index=False
)

coef.to_csv(
    OUT_COEF,
    index=False
)

bootstrap_summary.to_csv(
    OUT_BOOT,
    index=False
)

calibration.to_csv(
    OUT_CAL,
    index=False
)

pred.to_parquet(
    OUT_PRED,
    index=False,
    compression="zstd"
)


# ============================================================
# Console output
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    220
)

print()
print("===== MODEL PERFORMANCE =====")
print(
    model_summary.round(5)
    .to_string(index=False)
)

print()
print("===== MODEL COEFFICIENTS =====")
print(
    coef[
        coef["term"] != "intercept"
    ]
    .round(5)
    .to_string(index=False)
)

print()
print("===== PRIMARY LR TEST =====")
print(
    f"LR chi-square = {lr:.4f}; "
    f"df = 1; "
    f"p = {lr_p:.8g}"
)

print()
print("===== BOOTSTRAP INTERNAL VALIDATION =====")
print(
    bootstrap_summary
    .round(5)
    .to_string(index=False)
)

print()
print("===== LINEAR EQUIVALENCE QC =====")
print(
    f"|LL(B baseline+delta) - "
    f"LL(B baseline+h48)| = "
    f"{equiv_ll_diff:.12g}"
)

print(
    f"max absolute prediction difference = "
    f"{equiv_pred_diff:.12g}"
)

print()
print("=" * 78)
print("CHECKPOINT 13B COMPLETE")
print("=" * 78)

print(f"Model summary : {OUT_MODELS}")
print(f"Coefficients  : {OUT_COEF}")
print(f"Bootstrap     : {OUT_BOOT}")
print(f"Calibration   : {OUT_CAL}")
print(f"Predictions   : {OUT_PRED}")

