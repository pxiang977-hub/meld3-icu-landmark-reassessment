from pathlib import Path
import math

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INFILE = ROOT / "work" / "h48_primary_analysis_cohort.parquet"

OUT_BOOT = ROOT / "work" / "13d_incremental_bootstrap_ci.csv"
OUT_NONLIN = ROOT / "work" / "13d_delta_nonlinearity.csv"
OUT_CURVE = ROOT / "work" / "13d_delta_rcs_curve.csv"
OUT_DCA = ROOT / "work" / "13d_decision_curve.csv"
OUT_CAL = ROOT / "work" / "13d_calibration_bins.csv"
OUT_ADJ = ROOT / "work" / "13d_age_sex_sensitivity.csv"

if not INFILE.exists():
    raise FileNotFoundError(INFILE)


# ============================================================
# Basic logistic utilities
# ============================================================

def sigmoid(x):
    x = np.clip(x, -35, 35)
    return 1.0 / (1.0 + np.exp(-x))


def fit_logistic(X, y, max_iter=100, tol=1e-9):

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    beta = np.zeros(X.shape[1])

    for _ in range(max_iter):

        p = sigmoid(X @ beta)

        w = np.clip(
            p * (1-p),
            1e-9,
            None
        )

        grad = X.T @ (y-p)

        info = X.T @ (
            X * w[:, None]
        )

        info += np.eye(
            info.shape[0]
        ) * 1e-10

        step = (
            np.linalg.pinv(info)
            @ grad
        )

        new_beta = beta + step

        if np.max(
            np.abs(new_beta-beta)
        ) < tol:
            beta = new_beta
            break

        beta = new_beta

    p = sigmoid(X @ beta)

    ll = np.sum(
        y*np.log(
            np.clip(p, 1e-15, 1)
        )
        +
        (1-y)*np.log(
            np.clip(1-p, 1e-15, 1)
        )
    )

    w = np.clip(
        p*(1-p),
        1e-9,
        None
    )

    info = X.T @ (
        X * w[:, None]
    )

    cov = np.linalg.pinv(info)

    return {
        "beta": beta,
        "pred": p,
        "ll": ll,
        "cov": cov,
    }


def auc_score(y, p):

    y = np.asarray(y)
    p = np.asarray(p)

    n1 = (y == 1).sum()
    n0 = (y == 0).sum()

    if n1 == 0 or n0 == 0:
        return np.nan

    ranks = (
        pd.Series(p)
        .rank(method="average")
        .to_numpy()
    )

    return float(
        (
            ranks[y == 1].sum()
            - n1*(n1+1)/2
        )
        / (n1*n0)
    )


def brier(y, p):

    return float(
        np.mean(
            (
                np.asarray(y)
                - np.asarray(p)
            )**2
        )
    )


# ============================================================
# Restricted cubic spline
# ============================================================

def cube_pos(x, knot):
    return np.maximum(
        x-knot,
        0.0
    ) ** 3


def rcs_basis(x, knots):

    x = np.asarray(
        x,
        dtype=float
    )

    knots = np.asarray(
        knots,
        dtype=float
    )

    if len(knots) < 3:
        raise ValueError(
            "At least 3 unique knots required."
        )

    last = knots[-1]
    penult = knots[-2]

    denom = (
        last - penult
    )

    scale = (
        last - knots[0]
    ) ** 2

    if scale == 0:
        scale = 1.0

    cols = [x]

    for kj in knots[:-2]:

        h = (
            cube_pos(x, kj)

            - cube_pos(x, penult)
            * (
                (last-kj)
                / denom
            )

            + cube_pos(x, last)
            * (
                (penult-kj)
                / denom
            )
        ) / scale

        cols.append(h)

    return np.column_stack(cols)


def chi2_sf_lr(lr, df):

    if df == 1:
        return math.erfc(
            math.sqrt(lr/2)
        )

    if df == 2:
        return math.exp(
            -lr/2
        )

    return np.nan


# ============================================================
# Load cohort
# ============================================================

df = pd.read_parquet(
    INFILE
).copy()

if len(df) != 904:
    raise RuntimeError(
        f"Expected 904 patients; found {len(df)}"
    )

y = (
    df["hospital_expire_flag"]
    .astype(int)
    .to_numpy()
)

baseline = (
    df["meld3_primary_baseline"]
    .astype(float)
    .to_numpy()
)

h48 = (
    df["meld3_primary_h48"]
    .astype(float)
    .to_numpy()
)

delta = h48 - baseline

age = (
    df["age_at_admission"]
    .astype(float)
    .to_numpy()
)

female = (
    df["gender"]
    .astype(str)
    .str.upper()
    .eq("F")
    .astype(float)
    .to_numpy()
)


print("=" * 78)
print("CHECKPOINT 13D - UNCERTAINTY / NONLINEARITY / DCA")
print("=" * 78)

print(
    f"n={len(df)}; deaths={int(y.sum())}; "
    f"mortality={100*y.mean():.2f}%"
)


# ============================================================
# Primary models
# ============================================================

X_A = np.column_stack([
    np.ones(len(df)),
    baseline,
])

X_B = np.column_stack([
    np.ones(len(df)),
    baseline,
    delta,
])

X_C = np.column_stack([
    np.ones(len(df)),
    h48,
])

fit_A = fit_logistic(
    X_A,
    y
)

fit_B = fit_logistic(
    X_B,
    y
)

fit_C = fit_logistic(
    X_C,
    y
)

auc_A = auc_score(
    y,
    fit_A["pred"]
)

auc_B = auc_score(
    y,
    fit_B["pred"]
)

brier_A = brier(
    y,
    fit_A["pred"]
)

brier_B = brier(
    y,
    fit_B["pred"]
)


# ============================================================
# A. Paired bootstrap CI for incremental performance
# ============================================================

N_BOOT = 2000

rng = np.random.default_rng(
    20260919
)

boot_rows = []

for i in range(N_BOOT):

    idx = rng.integers(
        0,
        len(df),
        len(df)
    )

    yb = y[idx]

    if len(
        np.unique(yb)
    ) < 2:
        continue

    XA = X_A[idx]
    XB = X_B[idx]

    fa = fit_logistic(
        XA,
        yb
    )

    fb = fit_logistic(
        XB,
        yb
    )

    auc_a = auc_score(
        yb,
        fa["pred"]
    )

    auc_b = auc_score(
        yb,
        fb["pred"]
    )

    br_a = brier(
        yb,
        fa["pred"]
    )

    br_b = brier(
        yb,
        fb["pred"]
    )

    boot_rows.append({
        "iteration": i+1,
        "auc_A": auc_a,
        "auc_B": auc_b,
        "delta_auc": auc_b-auc_a,
        "brier_A": br_a,
        "brier_B": br_b,
        "delta_brier": br_b-br_a,
    })

    if (i+1) % 200 == 0:
        print(
            f"Bootstrap {i+1}/{N_BOOT}"
        )


boot = pd.DataFrame(
    boot_rows
)

boot_summary = pd.DataFrame([{
    "bootstrap_reps":
        len(boot),

    "apparent_auc_A":
        auc_A,

    "apparent_auc_B":
        auc_B,

    "apparent_delta_auc":
        auc_B-auc_A,

    "delta_auc_ci_low":
        boot["delta_auc"]
        .quantile(0.025),

    "delta_auc_ci_high":
        boot["delta_auc"]
        .quantile(0.975),

    "prob_delta_auc_gt_0":
        (
            boot["delta_auc"] > 0
        ).mean(),

    "apparent_brier_A":
        brier_A,

    "apparent_brier_B":
        brier_B,

    "apparent_delta_brier":
        brier_B-brier_A,

    "delta_brier_ci_low":
        boot["delta_brier"]
        .quantile(0.025),

    "delta_brier_ci_high":
        boot["delta_brier"]
        .quantile(0.975),

    "prob_delta_brier_lt_0":
        (
            boot["delta_brier"] < 0
        ).mean(),
}])

boot_summary.to_csv(
    OUT_BOOT,
    index=False
)


# ============================================================
# B. Nonlinearity of delta MELD
#
# Four-knot restricted cubic spline:
# 5th / 35th / 65th / 95th percentiles.
# ============================================================

raw_knots = np.quantile(
    delta,
    [0.05, 0.35, 0.65, 0.95]
)

knots = np.unique(
    raw_knots
)

if len(knots) < 4:

    knots = np.unique(
        np.quantile(
            delta,
            [0.05, 0.25, 0.75, 0.95]
        )
    )

if len(knots) < 3:
    raise RuntimeError(
        "Insufficient unique delta values for RCS."
    )


B = rcs_basis(
    delta,
    knots
)

X_RCS = np.column_stack([
    np.ones(len(df)),
    baseline,
    B,
])

fit_RCS = fit_logistic(
    X_RCS,
    y
)


# Linear model already contains baseline + delta.
lr_nonlin = max(
    0.0,
    2 * (
        fit_RCS["ll"]
        - fit_B["ll"]
    )
)

df_nonlin = (
    X_RCS.shape[1]
    - X_B.shape[1]
)

p_nonlin = chi2_sf_lr(
    lr_nonlin,
    df_nonlin
)


nonlin = pd.DataFrame([{
    "knots":
        "|".join(
            f"{x:.3f}"
            for x in knots
        ),

    "linear_loglik":
        fit_B["ll"],

    "rcs_loglik":
        fit_RCS["ll"],

    "lr_nonlinearity":
        lr_nonlin,

    "df":
        df_nonlin,

    "p_nonlinearity":
        p_nonlin,
}])

nonlin.to_csv(
    OUT_NONLIN,
    index=False
)


# Adjusted OR curve relative to delta = 0.
lo = int(
    math.floor(
        np.quantile(
            delta,
            0.01
        )
    )
)

hi = int(
    math.ceil(
        np.quantile(
            delta,
            0.99
        )
    )
)

grid = np.arange(
    lo,
    hi+1,
    1.0
)

base_value = float(
    np.median(
        baseline
    )
)

curve_rows = []

ref_basis = rcs_basis(
    np.array([0.0]),
    knots
)[0]

ref_x = np.concatenate(
    [
        [1.0, base_value],
        ref_basis,
    ]
)

for d in grid:

    gb = rcs_basis(
        np.array([d]),
        knots
    )[0]

    x = np.concatenate(
        [
            [1.0, base_value],
            gb,
        ]
    )

    contrast = (
        x - ref_x
    )

    log_or = float(
        contrast
        @ fit_RCS["beta"]
    )

    var = float(
        contrast
        @ fit_RCS["cov"]
        @ contrast
    )

    se = math.sqrt(
        max(
            var,
            0.0
        )
    )

    curve_rows.append({
        "delta_meld":
            d,

        "reference_delta":
            0,

        "baseline_meld_held_at":
            base_value,

        "adjusted_or":
            math.exp(
                log_or
            ),

        "or_95ci_low":
            math.exp(
                log_or
                - 1.96*se
            ),

        "or_95ci_high":
            math.exp(
                log_or
                + 1.96*se
            ),
    })


curve = pd.DataFrame(
    curve_rows
)

curve.to_csv(
    OUT_CURVE,
    index=False
)


# ============================================================
# C. Age + sex adjusted sensitivity
# ============================================================

age_c = (
    age - age.mean()
)

X_A_ADJ = np.column_stack([
    np.ones(len(df)),
    baseline,
    age_c,
    female,
])

X_B_ADJ = np.column_stack([
    np.ones(len(df)),
    baseline,
    delta,
    age_c,
    female,
])

fa = fit_logistic(
    X_A_ADJ,
    y
)

fb = fit_logistic(
    X_B_ADJ,
    y
)

lr_adj = max(
    0.0,
    2 * (
        fb["ll"]
        - fa["ll"]
    )
)

p_adj = math.erfc(
    math.sqrt(
        lr_adj/2
    )
)

adj = pd.DataFrame([{
    "auc_baseline_age_sex":
        auc_score(
            y,
            fa["pred"]
        ),

    "auc_plus_delta":
        auc_score(
            y,
            fb["pred"]
        ),

    "delta_auc":
        auc_score(
            y,
            fb["pred"]
        )
        -
        auc_score(
            y,
            fa["pred"]
        ),

    "brier_baseline_age_sex":
        brier(
            y,
            fa["pred"]
        ),

    "brier_plus_delta":
        brier(
            y,
            fb["pred"]
        ),

    "delta_brier":
        brier(
            y,
            fb["pred"]
        )
        -
        brier(
            y,
            fa["pred"]
        ),

    "lr_delta":
        lr_adj,

    "p_delta":
        p_adj,

    "delta_beta":
        fb["beta"][2],

    "delta_or_per_point":
        math.exp(
            fb["beta"][2]
        ),

    "delta_or_95ci_low":
        math.exp(
            fb["beta"][2]
            - 1.96
            * math.sqrt(
                fb["cov"][2,2]
            )
        ),

    "delta_or_95ci_high":
        math.exp(
            fb["beta"][2]
            + 1.96
            * math.sqrt(
                fb["cov"][2,2]
            )
        ),
}])

adj.to_csv(
    OUT_ADJ,
    index=False
)


# ============================================================
# D. Calibration bins
# ============================================================

def wilson_ci(k, n):

    if n == 0:
        return (
            np.nan,
            np.nan
        )

    z = 1.96
    p = k/n

    den = (
        1
        + z*z/n
    )

    center = (
        p
        + z*z/(2*n)
    ) / den

    half = (
        z
        * math.sqrt(
            p*(1-p)/n
            + z*z/(4*n*n)
        )
        / den
    )

    return (
        max(0.0, center-half),
        min(1.0, center+half),
    )


cal_rows = []

for name, pred in {
    "A_baseline":
        fit_A["pred"],

    "B_baseline_plus_delta":
        fit_B["pred"],

    "C_h48_only":
        fit_C["pred"],
}.items():

    tmp = pd.DataFrame({
        "y": y,
        "pred": pred,
    })

    tmp["bin"] = pd.qcut(
        tmp["pred"],
        q=10,
        duplicates="drop"
    )

    grouped = (
        tmp
        .groupby(
            "bin",
            observed=True
        )
    )

    for i, (_, g) in enumerate(
        grouped,
        start=1
    ):

        n = len(g)
        deaths = int(
            g["y"].sum()
        )

        low, high = wilson_ci(
            deaths,
            n
        )

        cal_rows.append({
            "model":
                name,

            "risk_group":
                i,

            "n":
                n,

            "deaths":
                deaths,

            "mean_predicted":
                g["pred"].mean(),

            "observed_mortality":
                g["y"].mean(),

            "observed_95ci_low":
                low,

            "observed_95ci_high":
                high,
        })


cal = pd.DataFrame(
    cal_rows
)

cal.to_csv(
    OUT_CAL,
    index=False
)


# ============================================================
# E. Decision curve analysis
# ============================================================

def net_benefit(y, p, threshold):

    y = np.asarray(y)
    p = np.asarray(p)

    pred_positive = (
        p >= threshold
    )

    tp = (
        pred_positive
        & (y == 1)
    ).sum()

    fp = (
        pred_positive
        & (y == 0)
    ).sum()

    n = len(y)

    return (
        tp/n
        -
        fp/n
        * threshold
        / (1-threshold)
    )


prevalence = y.mean()

dca_rows = []

for threshold in np.arange(
    0.05,
    0.51,
    0.01
):

    threshold = round(
        float(threshold),
        2
    )

    treat_all = (
        prevalence
        -
        (1-prevalence)
        * threshold
        / (1-threshold)
    )

    nb_A = net_benefit(
        y,
        fit_A["pred"],
        threshold
    )

    nb_B = net_benefit(
        y,
        fit_B["pred"],
        threshold
    )

    nb_C = net_benefit(
        y,
        fit_C["pred"],
        threshold
    )

    dca_rows.append({
        "threshold":
            threshold,

        "treat_none":
            0.0,

        "treat_all":
            treat_all,

        "model_A_baseline":
            nb_A,

        "model_B_baseline_plus_delta":
            nb_B,

        "model_C_h48_only":
            nb_C,

        "net_benefit_gain_B_vs_A":
            nb_B-nb_A,
    })


dca = pd.DataFrame(
    dca_rows
)

dca.to_csv(
    OUT_DCA,
    index=False
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
print(
    "===== BOOTSTRAP INCREMENTAL PERFORMANCE CI ====="
)

print(
    boot_summary
    .round(6)
    .to_string(index=False)
)


print()
print(
    "===== DELTA MELD NONLINEARITY ====="
)

print(
    nonlin
    .round(6)
    .to_string(index=False)
)


print()
print(
    "===== AGE + SEX ADJUSTED SENSITIVITY ====="
)

print(
    adj
    .round(6)
    .to_string(index=False)
)


print()
print(
    "===== DCA SELECTED THRESHOLDS ====="
)

print(
    dca[
        dca["threshold"]
        .isin(
            [
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.40,
            ]
        )
    ]
    .round(5)
    .to_string(index=False)
)


print()
print("=" * 78)
print("CHECKPOINT 13D COMPLETE")
print("=" * 78)

print(f"Bootstrap CI : {OUT_BOOT}")
print(f"Nonlinearity : {OUT_NONLIN}")
print(f"RCS curve    : {OUT_CURVE}")
print(f"Calibration  : {OUT_CAL}")
print(f"DCA          : {OUT_DCA}")
print(f"Age/sex sens.: {OUT_ADJ}")

