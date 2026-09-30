from pathlib import Path
import math

import numpy as np
import pandas as pd


# ============================================================
# Checkpoint 13C
# Inverse-probability-of-observation weighting (IPW)
#
# Target population:
#   48-h landmark survivors with COMPLETE baseline MELD 3.0
#
# n = 1,980 expected
#
# Observed/reassessed:
#   complete 48-h MELD
#   n = 904 expected
#
# Purpose:
#   assess whether the incremental value of 48-h reassessment
#   persists after accounting for informative repeat testing.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INFILE = ROOT / "work" / "meld3_snapshots_wide.parquet"

OUT_WEIGHTS = ROOT / "work" / "13c_ipw_weight_summary.csv"
OUT_BALANCE = ROOT / "work" / "13c_ipw_balance.csv"
OUT_PERF = ROOT / "work" / "13c_ipw_model_performance.csv"
OUT_COEF = ROOT / "work" / "13c_ipw_model_coefficients.csv"
OUT_DATA = ROOT / "work" / "13c_ipw_analysis_dataset.parquet"

if not INFILE.exists():
    raise FileNotFoundError(INFILE)


# ============================================================
# Logistic regression utilities
# ============================================================

def sigmoid(x):
    x = np.clip(x, -35, 35)
    return 1.0 / (1.0 + np.exp(-x))


def fit_logistic(X, y, weights=None, max_iter=100, tol=1e-9):

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if weights is None:
        weights = np.ones(len(y))

    weights = np.asarray(weights, dtype=float)

    beta = np.zeros(X.shape[1])

    for _ in range(max_iter):

        eta = X @ beta
        p = sigmoid(eta)

        v = (
            weights
            * np.clip(p * (1-p), 1e-9, None)
        )

        grad = X.T @ (
            weights * (y-p)
        )

        info = X.T @ (
            X * v[:, None]
        )

        info += (
            np.eye(info.shape[0])
            * 1e-10
        )

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

    v = (
        weights
        * np.clip(p*(1-p), 1e-9, None)
    )

    info = X.T @ (
        X * v[:, None]
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
    }


def auc_unweighted(y, p):

    y = np.asarray(y)
    p = np.asarray(p)

    n1 = (y == 1).sum()
    n0 = (y == 0).sum()

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


def weighted_auc(y, p, w):

    y = np.asarray(y)
    p = np.asarray(p)
    w = np.asarray(w)

    pos = y == 1
    neg = y == 0

    pp = p[pos][:, None]
    pn = p[neg][None, :]

    wp = w[pos][:, None]
    wn = w[neg][None, :]

    pair_w = wp * wn

    concord = (
        (pp > pn).astype(float)
        + 0.5*(pp == pn)
    )

    return float(
        np.sum(
            pair_w * concord
        )
        / np.sum(pair_w)
    )


def weighted_brier(y, p, w):

    y = np.asarray(y)
    p = np.asarray(p)
    w = np.asarray(w)

    return float(
        np.sum(
            w*(y-p)**2
        )
        / np.sum(w)
    )


def effective_sample_size(w):

    w = np.asarray(w)

    return float(
        (w.sum()**2)
        / np.sum(w**2)
    )


# ============================================================
# Load and define target population
# ============================================================

df = pd.read_parquet(
    INFILE
).copy()

target = df[
    (df["eligible_h48"] == 1)
    & df["meld3_primary_baseline"].notna()
].copy()

target["observed_h48"] = (
    target["meld3_primary_h48"]
    .notna()
    .astype(int)
)

print("=" * 78)
print("CHECKPOINT 13C - IPW SELECTION SENSITIVITY")
print("=" * 78)

print(
    f"Target baseline-complete 48h risk set: "
    f"{len(target):,}"
)

print(
    f"Observed complete 48h MELD: "
    f"{target['observed_h48'].sum():,}"
)

print(
    f"Observation rate: "
    f"{100*target['observed_h48'].mean():.2f}%"
)


if len(target) != 1980:
    raise RuntimeError(
        f"Expected target n=1980, found {len(target)}"
    )

if target["observed_h48"].sum() != 904:
    raise RuntimeError(
        "Expected 904 complete reassessments."
    )


# ============================================================
# Selection-model predictors
#
# All are known BEFORE the 48-h reassessment.
# No future outcome is used in the primary observation model.
# ============================================================

target["female"] = (
    target["gender"]
    .astype(str)
    .str.upper()
    .eq("F")
    .astype(int)
)

target["log_bilirubin"] = np.log(
    target[
        "bilirubin_baseline"
    ].clip(lower=1.0)
)

target["log_creatinine"] = np.log(
    target[
        "creatinine_baseline"
    ].clip(lower=1.0)
)

target["log_inr"] = np.log(
    target[
        "inr_baseline"
    ].clip(lower=1.0)
)

predictors = [
    "age_at_admission",
    "female",
    "meld3_primary_baseline",
    "albumin_baseline",
    "log_bilirubin",
    "log_creatinine",
    "sodium_baseline",
    "log_inr",
    "dialysis_flag_primary_baseline",
]


# Standardize continuous predictors in target population.
binary = {
    "female",
    "dialysis_flag_primary_baseline",
}

X_cols = []

for x in predictors:

    if x in binary:

        new = f"sel_{x}"
        target[new] = (
            target[x]
            .astype(float)
        )

    else:

        mu = target[x].mean()
        sd = target[x].std(ddof=0)

        if sd == 0:
            sd = 1.0

        new = f"sel_{x}"

        target[new] = (
            target[x] - mu
        ) / sd

    X_cols.append(new)


# Allow nonlinear relation between baseline severity
# and probability of repeat complete assessment.
target[
    "sel_meld_squared"
] = (
    target[
        "sel_meld3_primary_baseline"
    ] ** 2
)

X_cols.append(
    "sel_meld_squared"
)


X_sel = np.column_stack(
    [
        np.ones(len(target)),
        target[X_cols].to_numpy(float)
    ]
)

r = (
    target["observed_h48"]
    .to_numpy(int)
)

sel_fit = fit_logistic(
    X_sel,
    r
)

prop = np.clip(
    sel_fit["pred"],
    0.01,
    0.99
)

target[
    "prob_h48_complete"
] = prop


# ============================================================
# Stabilized IP weights
# ============================================================

response_rate = r.mean()

target[
    "ipw_untrimmed"
] = np.where(
    r == 1,
    response_rate / prop,
    np.nan
)

obs = target[
    target["observed_h48"] == 1
].copy()

lo = obs[
    "ipw_untrimmed"
].quantile(0.01)

hi = obs[
    "ipw_untrimmed"
].quantile(0.99)

obs[
    "ipw_trimmed"
] = (
    obs["ipw_untrimmed"]
    .clip(lower=lo, upper=hi)
)


# Normalize both sets to mean 1.
for c in [
    "ipw_untrimmed",
    "ipw_trimmed",
]:

    obs[c] = (
        obs[c]
        / obs[c].mean()
    )


# ============================================================
# Weight diagnostics
# ============================================================

weights_summary = pd.DataFrame([
    {
        "metric": "target_n",
        "value": len(target),
    },
    {
        "metric": "observed_n",
        "value": len(obs),
    },
    {
        "metric": "observation_rate_pct",
        "value": 100*response_rate,
    },
    {
        "metric": "propensity_min",
        "value": prop.min(),
    },
    {
        "metric": "propensity_p01",
        "value": np.quantile(prop, 0.01),
    },
    {
        "metric": "propensity_median",
        "value": np.median(prop),
    },
    {
        "metric": "propensity_p99",
        "value": np.quantile(prop, 0.99),
    },
    {
        "metric": "propensity_max",
        "value": prop.max(),
    },
    {
        "metric": "untrimmed_weight_max",
        "value": obs["ipw_untrimmed"].max(),
    },
    {
        "metric": "trimmed_weight_max",
        "value": obs["ipw_trimmed"].max(),
    },
    {
        "metric": "untrimmed_ESS",
        "value": effective_sample_size(
            obs["ipw_untrimmed"]
        ),
    },
    {
        "metric": "trimmed_ESS",
        "value": effective_sample_size(
            obs["ipw_trimmed"]
        ),
    },
    {
        "metric": "selection_model_auc",
        "value": auc_unweighted(
            r,
            prop
        ),
    },
])

weights_summary.to_csv(
    OUT_WEIGHTS,
    index=False
)


# ============================================================
# Baseline-covariate balance
#
# Compare:
#   target population
#   raw 904 reassessed patients
#   IPW-reweighted 904 patients
# ============================================================

balance_vars = [
    "age_at_admission",
    "female",
    "meld3_primary_baseline",
    "albumin_baseline",
    "bilirubin_baseline",
    "creatinine_baseline",
    "sodium_baseline",
    "inr_baseline",
    "dialysis_flag_primary_baseline",
]


def wmean(x, w):
    return float(
        np.sum(x*w)
        / np.sum(w)
    )


balance_rows = []

for var in balance_vars:

    target_x = (
        target[var]
        .astype(float)
        .to_numpy()
    )

    obs_x = (
        obs[var]
        .astype(float)
        .to_numpy()
    )

    target_mean = np.mean(
        target_x
    )

    raw_mean = np.mean(
        obs_x
    )

    weighted_mean = wmean(
        obs_x,
        obs["ipw_trimmed"].to_numpy()
    )

    target_sd = np.std(
        target_x,
        ddof=1
    )

    if target_sd == 0:
        raw_smd = 0
        weighted_smd = 0

    else:
        raw_smd = (
            raw_mean - target_mean
        ) / target_sd

        weighted_smd = (
            weighted_mean - target_mean
        ) / target_sd

    balance_rows.append({
        "variable": var,
        "target_mean": target_mean,
        "observed_raw_mean": raw_mean,
        "ipw_weighted_mean": weighted_mean,
        "raw_smd_vs_target": raw_smd,
        "ipw_smd_vs_target": weighted_smd,
    })


balance = pd.DataFrame(
    balance_rows
)

balance.to_csv(
    OUT_BALANCE,
    index=False
)


# ============================================================
# Outcome models
# ============================================================

# Direct comparator available in ALL 1,980 target patients.
y_target = (
    target[
        "hospital_expire_flag"
    ]
    .astype(int)
    .to_numpy()
)

baseline_target = (
    target[
        "meld3_primary_baseline"
    ]
    .astype(float)
    .to_numpy()
)

X_A_target = np.column_stack(
    [
        np.ones(len(target)),
        baseline_target,
    ]
)

fit_A_target = fit_logistic(
    X_A_target,
    y_target
)


# Reassessed 904 patients.
y_obs = (
    obs[
        "hospital_expire_flag"
    ]
    .astype(int)
    .to_numpy()
)

baseline_obs = (
    obs[
        "meld3_primary_baseline"
    ]
    .astype(float)
    .to_numpy()
)

updated_obs = (
    obs[
        "meld3_primary_h48"
    ]
    .astype(float)
    .to_numpy()
)

delta_obs = (
    updated_obs - baseline_obs
)


X_A = np.column_stack(
    [
        np.ones(len(obs)),
        baseline_obs,
    ]
)

X_B = np.column_stack(
    [
        np.ones(len(obs)),
        baseline_obs,
        delta_obs,
    ]
)


def run_weighted_models(
    label,
    weights
):

    weights = np.asarray(
        weights,
        dtype=float
    )

    fit_A = fit_logistic(
        X_A,
        y_obs,
        weights=weights
    )

    fit_B = fit_logistic(
        X_B,
        y_obs,
        weights=weights
    )

    auc_A = weighted_auc(
        y_obs,
        fit_A["pred"],
        weights
    )

    auc_B = weighted_auc(
        y_obs,
        fit_B["pred"],
        weights
    )

    brier_A = weighted_brier(
        y_obs,
        fit_A["pred"],
        weights
    )

    brier_B = weighted_brier(
        y_obs,
        fit_B["pred"],
        weights
    )

    perf = [
        {
            "analysis":
                label,

            "model":
                "A_baseline",

            "n_observed":
                len(obs),

            "effective_n":
                effective_sample_size(
                    weights
                ),

            "auc":
                auc_A,

            "brier":
                brier_A,

            "delta_auc_vs_A":
                np.nan,

            "delta_brier_vs_A":
                np.nan,
        },

        {
            "analysis":
                label,

            "model":
                "B_baseline_plus_delta",

            "n_observed":
                len(obs),

            "effective_n":
                effective_sample_size(
                    weights
                ),

            "auc":
                auc_B,

            "brier":
                brier_B,

            "delta_auc_vs_A":
                auc_B - auc_A,

            "delta_brier_vs_A":
                brier_B - brier_A,
        },
    ]

    coef = []

    names_A = [
        "intercept",
        "baseline_meld",
    ]

    names_B = [
        "intercept",
        "baseline_meld",
        "delta_meld_48h",
    ]

    for model, fit, names in [
        (
            "A_baseline",
            fit_A,
            names_A
        ),
        (
            "B_baseline_plus_delta",
            fit_B,
            names_B
        ),
    ]:

        for i, term in enumerate(names):

            beta = float(
                fit["beta"][i]
            )

            se = float(
                fit["se"][i]
            )

            coef.append({
                "analysis":
                    label,

                "model":
                    model,

                "term":
                    term,

                "beta":
                    beta,

                "se":
                    se,

                "odds_ratio":
                    math.exp(beta),

                "or_95ci_low":
                    math.exp(
                        beta - 1.96*se
                    ),

                "or_95ci_high":
                    math.exp(
                        beta + 1.96*se
                    ),
            })

    return perf, coef


# Raw complete case
perf_raw, coef_raw = (
    run_weighted_models(
        "complete_case_unweighted",
        np.ones(len(obs))
    )
)

# Untrimmed IPW
perf_ipw, coef_ipw = (
    run_weighted_models(
        "ipw_untrimmed",
        obs["ipw_untrimmed"]
    )
)

# Primary IPW sensitivity
perf_trim, coef_trim = (
    run_weighted_models(
        "ipw_trimmed_1_99",
        obs["ipw_trimmed"]
    )
)


# Direct Model A in complete 1,980 target cohort.
target_auc = auc_unweighted(
    y_target,
    fit_A_target["pred"]
)

target_brier = float(
    np.mean(
        (
            y_target
            - fit_A_target["pred"]
        ) ** 2
    )
)

direct_row = {
    "analysis":
        "direct_full_baseline_complete_target",

    "model":
        "A_baseline",

    "n_observed":
        len(target),

    "effective_n":
        len(target),

    "auc":
        target_auc,

    "brier":
        target_brier,

    "delta_auc_vs_A":
        np.nan,

    "delta_brier_vs_A":
        np.nan,
}


performance = pd.DataFrame(
    [direct_row]
    + perf_raw
    + perf_ipw
    + perf_trim
)

coefficients = pd.DataFrame(
    coef_raw
    + coef_ipw
    + coef_trim
)


performance.to_csv(
    OUT_PERF,
    index=False
)

coefficients.to_csv(
    OUT_COEF,
    index=False
)


# Save analysis dataset.
target = target.merge(
    obs[
        [
            "subject_id",
            "ipw_untrimmed",
            "ipw_trimmed",
        ]
    ],
    on="subject_id",
    how="left",
    suffixes=("", "_obs")
)

target.to_parquet(
    OUT_DATA,
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
print("===== IPW WEIGHT DIAGNOSTICS =====")
print(
    weights_summary
    .round(5)
    .to_string(index=False)
)


print()
print("===== BASELINE COVARIATE BALANCE =====")
print(
    balance
    .round(4)
    .to_string(index=False)
)


print()
print("===== IPW MODEL PERFORMANCE =====")
print(
    performance
    .round(5)
    .to_string(index=False)
)


print()
print("===== IPW MODEL COEFFICIENTS =====")

print(
    coefficients[
        coefficients["term"]
        != "intercept"
    ]
    .round(5)
    .to_string(index=False)
)


print()
print("=" * 78)
print("CHECKPOINT 13C COMPLETE")
print("=" * 78)

print(f"Weights     : {OUT_WEIGHTS}")
print(f"Balance     : {OUT_BALANCE}")
print(f"Performance : {OUT_PERF}")
print(f"Coefficients: {OUT_COEF}")
print(f"Dataset     : {OUT_DATA}")
