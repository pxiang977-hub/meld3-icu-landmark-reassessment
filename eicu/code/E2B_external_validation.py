from pathlib import Path
import math

import numpy as np
import pandas as pd


# ============================================================
# eICU Checkpoint E2B
#
# FIXED-COEFFICIENT EXTERNAL VALIDATION
#
# Model derivation / primary assessment:
#   MIMIC-IV locked paired 48-h cohort
#
# External validation:
#   eICU-CRD
#
# IMPORTANT:
# eICU outcomes are NEVER used to estimate Models A/B/C.
# ============================================================

HOME = Path.home()

MIMIC_WORK = (
    HOME
    / "Desktop"
    / "MIMIC_IV_3_1"
    / "work"
)

EICU_ROOT = (
    HOME
    / "Desktop"
    / "eICU_CRD_2_0"
)

EICU_WORK = EICU_ROOT / "work"

MIMIC_FILE = (
    MIMIC_WORK
    / "h48_primary_analysis_cohort.parquet"
)

EICU_FILE = (
    EICU_WORK
    / "E2A_eicu_external_candidate.parquet"
)


OUT_COEF = (
    EICU_WORK
    / "E2B_mimic_locked_coefficients.csv"
)

OUT_PERF = (
    EICU_WORK
    / "E2B_external_performance.csv"
)

OUT_BOOT = (
    EICU_WORK
    / "E2B_cluster_bootstrap_ci.csv"
)

OUT_CAL = (
    EICU_WORK
    / "E2B_external_calibration.csv"
)

OUT_CAL_BINS = (
    EICU_WORK
    / "E2B_external_calibration_bins.csv"
)

OUT_DCA = (
    EICU_WORK
    / "E2B_external_dca.csv"
)

OUT_PRED = (
    EICU_WORK
    / "E2B_external_predictions.parquet"
)


for p in [
    MIMIC_FILE,
    EICU_FILE,
]:
    if not p.exists():
        raise FileNotFoundError(p)


# ============================================================
# Utility functions
# ============================================================

def sigmoid(x):

    return (
        1.0
        /
        (
            1.0
            +
            np.exp(
                -np.clip(
                    x,
                    -35,
                    35
                )
            )
        )
    )


def fit_logistic(
    X,
    y,
    max_iter=200,
    tol=1e-10
):

    X = np.asarray(
        X,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    beta = np.zeros(
        X.shape[1],
        dtype=float
    )

    for _ in range(max_iter):

        eta = X @ beta

        p = sigmoid(
            eta
        )

        w = np.clip(
            p * (1-p),
            1e-9,
            None
        )

        score = (
            X.T
            @
            (y-p)
        )

        info = (
            X.T
            @
            (
                X
                *
                w[:, None]
            )
        )

        info += (
            np.eye(
                info.shape[0]
            )
            * 1e-10
        )

        step = (
            np.linalg.pinv(
                info
            )
            @ score
        )

        beta_new = (
            beta
            + step
        )

        if np.max(
            np.abs(
                beta_new-beta
            )
        ) < tol:
            beta = beta_new
            break

        beta = beta_new


    pred = sigmoid(
        X @ beta
    )

    return {
        "beta":
            beta,

        "pred":
            pred,
    }


def auc_score(
    y,
    pred
):

    y = np.asarray(y)
    pred = np.asarray(pred)

    n1 = int(
        (y == 1).sum()
    )

    n0 = int(
        (y == 0).sum()
    )

    if (
        n1 == 0
        or n0 == 0
    ):
        return np.nan

    ranks = (
        pd.Series(pred)
        .rank(
            method="average"
        )
        .to_numpy()
    )

    auc = (
        ranks[y == 1].sum()
        -
        n1*(n1+1)/2
    ) / (
        n1*n0
    )

    return float(auc)


def brier_score(
    y,
    pred
):

    y = np.asarray(
        y,
        dtype=float
    )

    pred = np.asarray(
        pred,
        dtype=float
    )

    return float(
        np.mean(
            (y-pred)**2
        )
    )


def log_loss(
    y,
    pred
):

    y = np.asarray(
        y,
        dtype=float
    )

    pred = np.clip(
        np.asarray(
            pred,
            dtype=float
        ),
        1e-15,
        1-1e-15
    )

    return float(
        -np.mean(
            y*np.log(pred)
            +
            (1-y)*np.log(1-pred)
        )
    )


def calibration_intercept_only(
    y,
    lp
):

    y = np.asarray(
        y,
        float
    )

    lp = np.asarray(
        lp,
        float
    )

    alpha = 0.0

    for _ in range(100):

        p = sigmoid(
            alpha + lp
        )

        score = np.sum(
            y-p
        )

        info = np.sum(
            p*(1-p)
        )

        if info <= 0:
            break

        step = score/info

        alpha_new = (
            alpha
            + step
        )

        if abs(
            alpha_new-alpha
        ) < 1e-10:
            alpha = alpha_new
            break

        alpha = alpha_new

    return float(alpha)


def calibration_intercept_slope(
    y,
    lp
):

    X = np.column_stack([
        np.ones(len(y)),
        lp
    ])

    fit = fit_logistic(
        X,
        y
    )

    return (
        float(
            fit["beta"][0]
        ),
        float(
            fit["beta"][1]
        )
    )


def net_benefit(
    y,
    pred,
    threshold
):

    y = np.asarray(
        y,
        dtype=int
    )

    pred = np.asarray(
        pred,
        dtype=float
    )

    treat = (
        pred >= threshold
    )

    tp = np.sum(
        treat
        &
        (y == 1)
    )

    fp = np.sum(
        treat
        &
        (y == 0)
    )

    n = len(y)

    return float(
        tp/n
        -
        fp/n
        *
        threshold
        /
        (1-threshold)
    )


# ============================================================
# 1. Read locked MIMIC primary cohort
# ============================================================

m = pd.read_parquet(
    MIMIC_FILE
)


required_mimic = [
    "meld3_primary_baseline",
    "meld3_primary_h48",
    "hospital_expire_flag",
]


missing = [
    c
    for c in required_mimic
    if c not in m.columns
]


if missing:

    print(
        "MIMIC columns:"
    )

    print(
        list(
            m.columns
        )
    )

    raise RuntimeError(
        f"Missing expected MIMIC columns: {missing}"
    )


m = m[
    m[
        "meld3_primary_baseline"
    ].notna()
    &
    m[
        "meld3_primary_h48"
    ].notna()
].copy()


if len(m) != 904:

    raise RuntimeError(
        f"Expected MIMIC n=904, found {len(m)}"
    )


ym = (
    m[
        "hospital_expire_flag"
    ]
    .astype(int)
    .to_numpy()
)


if int(
    ym.sum()
) != 205:

    raise RuntimeError(
        f"Expected MIMIC deaths=205, found {int(ym.sum())}"
    )


m_base = (
    m[
        "meld3_primary_baseline"
    ]
    .astype(float)
    .to_numpy()
)


m_h48 = (
    m[
        "meld3_primary_h48"
    ]
    .astype(float)
    .to_numpy()
)


m_delta = (
    m_h48
    -
    m_base
)


# ============================================================
# 2. Fit locked MIMIC Models A/B/C
#
# A = baseline MELD
# B = baseline MELD + delta MELD at 48h
# C = updated 48h MELD alone
# ============================================================

X_A_m = np.column_stack([
    np.ones(len(m)),
    m_base
])


X_B_m = np.column_stack([
    np.ones(len(m)),
    m_base,
    m_delta
])


X_C_m = np.column_stack([
    np.ones(len(m)),
    m_h48
])


fit_A = fit_logistic(
    X_A_m,
    ym
)

fit_B = fit_logistic(
    X_B_m,
    ym
)

fit_C = fit_logistic(
    X_C_m,
    ym
)


# Public-release note:
# Dataset-derived fitted coefficient values are not redistributed.
# The fixed external-validation coefficients are fit locally from the authorized
# MIMIC analysis cohort immediately above and then applied unchanged to eICU.


coef = pd.DataFrame([

    {
        "model":
            "A_baseline",

        "term":
            "intercept",

        "beta":
            fit_A["beta"][0],
    },

    {
        "model":
            "A_baseline",

        "term":
            "baseline_meld",

        "beta":
            fit_A["beta"][1],
    },

    {
        "model":
            "B_baseline_plus_delta",

        "term":
            "intercept",

        "beta":
            fit_B["beta"][0],
    },

    {
        "model":
            "B_baseline_plus_delta",

        "term":
            "baseline_meld",

        "beta":
            fit_B["beta"][1],
    },

    {
        "model":
            "B_baseline_plus_delta",

        "term":
            "delta_meld_48h",

        "beta":
            fit_B["beta"][2],
    },

    {
        "model":
            "C_h48_alone",

        "term":
            "intercept",

        "beta":
            fit_C["beta"][0],
    },

    {
        "model":
            "C_h48_alone",

        "term":
            "h48_meld",

        "beta":
            fit_C["beta"][1],
    },
])


coef.to_csv(
    OUT_COEF,
    index=False
)


# ============================================================
# 3. External eICU data
# ============================================================

e = pd.read_parquet(
    EICU_FILE
)


required_external = [
    "patientunitstayid",
    "uniquepid",
    "hospitalid",
    "hospital_death",
    "meld3_observed_primary_baseline",
    "meld3_observed_primary_h48",
    "meld3_observed_strict_baseline",
    "meld3_observed_strict_h48",
    "meld3_pragmatic_baseline",
    "meld3_pragmatic_h48",
    "nonrrt_clean",
]


missing = [
    c
    for c in required_external
    if c not in e.columns
]


if missing:

    raise RuntimeError(
        f"Missing eICU columns: {missing}"
    )


if len(e) != 349:

    raise RuntimeError(
        f"Expected eICU n=349, found {len(e)}"
    )


if int(
    e[
        "hospital_death"
    ].sum()
) != 76:

    raise RuntimeError(
        "Expected eICU deaths=76"
    )


# ============================================================
# 4. Fixed MIMIC coefficients applied to eICU
# ============================================================

strategies = {

    "observed_primary": {
        "baseline":
            "meld3_observed_primary_baseline",

        "h48":
            "meld3_observed_primary_h48",

        "mask":
            np.ones(
                len(e),
                dtype=bool
            ),
    },

    "observed_strict": {
        "baseline":
            "meld3_observed_strict_baseline",

        "h48":
            "meld3_observed_strict_h48",

        "mask":
            np.ones(
                len(e),
                dtype=bool
            ),
    },

    "pragmatic": {
        "baseline":
            "meld3_pragmatic_baseline",

        "h48":
            "meld3_pragmatic_h48",

        "mask":
            np.ones(
                len(e),
                dtype=bool
            ),
    },

    "nonrrt_clean": {
        "baseline":
            "meld3_observed_primary_baseline",

        "h48":
            "meld3_observed_primary_h48",

        "mask":
            (
                e[
                    "nonrrt_clean"
                ]
                == 1
            ).to_numpy(),
    },
}


performance_rows = []
calibration_rows = []


for strategy, cfg in strategies.items():

    mask = cfg[
        "mask"
    ]

    d = e.loc[
        mask
    ].copy()

    y = (
        d[
            "hospital_death"
        ]
        .astype(int)
        .to_numpy()
    )

    base = (
        d[
            cfg["baseline"]
        ]
        .astype(float)
        .to_numpy()
    )

    h48 = (
        d[
            cfg["h48"]
        ]
        .astype(float)
        .to_numpy()
    )

    delta = (
        h48
        -
        base
    )


    lp_A = (
        fit_A["beta"][0]
        +
        fit_A["beta"][1]
        * base
    )


    lp_B = (
        fit_B["beta"][0]
        +
        fit_B["beta"][1]
        * base
        +
        fit_B["beta"][2]
        * delta
    )


    lp_C = (
        fit_C["beta"][0]
        +
        fit_C["beta"][1]
        * h48
    )


    pred_A = sigmoid(
        lp_A
    )

    pred_B = sigmoid(
        lp_B
    )

    pred_C = sigmoid(
        lp_C
    )


    if strategy == "observed_primary":

        e.loc[
            mask,
            "pred_A_fixed_mimic"
        ] = pred_A

        e.loc[
            mask,
            "pred_B_fixed_mimic"
        ] = pred_B

        e.loc[
            mask,
            "pred_C_fixed_mimic"
        ] = pred_C


    metrics = {}

    for model_name, pred, lp in [
        (
            "A_baseline",
            pred_A,
            lp_A
        ),
        (
            "B_baseline_plus_delta",
            pred_B,
            lp_B
        ),
        (
            "C_h48_alone",
            pred_C,
            lp_C
        ),
    ]:

        auc = auc_score(
            y,
            pred
        )

        br = brier_score(
            y,
            pred
        )

        ll = log_loss(
            y,
            pred
        )

        metrics[
            model_name
        ] = {
            "auc":
                auc,

            "brier":
                br,

            "log_loss":
                ll,
        }


        cal_int_only = (
            calibration_intercept_only(
                y,
                lp
            )
        )


        cal_int, cal_slope = (
            calibration_intercept_slope(
                y,
                lp
            )
        )


        calibration_rows.append({

            "strategy":
                strategy,

            "model":
                model_name,

            "n":
                len(d),

            "deaths":
                int(
                    y.sum()
                ),

            "calibration_in_the_large":
                cal_int_only,

            "calibration_intercept":
                cal_int,

            "calibration_slope":
                cal_slope,
        })


        performance_rows.append({

            "strategy":
                strategy,

            "model":
                model_name,

            "n":
                len(d),

            "unique_patients":
                d[
                    "uniquepid"
                ].nunique(),

            "hospitals":
                d[
                    "hospitalid"
                ].nunique(),

            "deaths":
                int(
                    y.sum()
                ),

            "mortality_pct":
                100
                * y.mean(),

            "auc":
                auc,

            "brier":
                br,

            "log_loss":
                ll,

            "delta_auc_vs_A":
                np.nan,

            "delta_brier_vs_A":
                np.nan,
        })


    performance_rows.append({

        "strategy":
            strategy,

        "model":
            "B_minus_A_increment",

        "n":
            len(d),

        "unique_patients":
            d[
                "uniquepid"
            ].nunique(),

        "hospitals":
            d[
                "hospitalid"
            ].nunique(),

        "deaths":
            int(
                y.sum()
            ),

        "mortality_pct":
            100
            * y.mean(),

        "auc":
            np.nan,

        "brier":
            np.nan,

        "log_loss":
            np.nan,

        "delta_auc_vs_A":
            (
                metrics[
                    "B_baseline_plus_delta"
                ]["auc"]
                -
                metrics[
                    "A_baseline"
                ]["auc"]
            ),

        "delta_brier_vs_A":
            (
                metrics[
                    "B_baseline_plus_delta"
                ]["brier"]
                -
                metrics[
                    "A_baseline"
                ]["brier"]
            ),
    })


performance = pd.DataFrame(
    performance_rows
)

calibration = pd.DataFrame(
    calibration_rows
)


performance.to_csv(
    OUT_PERF,
    index=False
)

calibration.to_csv(
    OUT_CAL,
    index=False
)


# ============================================================
# 5. Primary external cluster bootstrap
#
# Cluster = uniquepid
# because some patients contribute >1 hospitalization.
# ============================================================

primary = e.copy()

y = (
    primary[
        "hospital_death"
    ]
    .astype(int)
    .to_numpy()
)

pred_A = (
    primary[
        "pred_A_fixed_mimic"
    ]
    .astype(float)
    .to_numpy()
)

pred_B = (
    primary[
        "pred_B_fixed_mimic"
    ]
    .astype(float)
    .to_numpy()
)

pred_C = (
    primary[
        "pred_C_fixed_mimic"
    ]
    .astype(float)
    .to_numpy()
)


cluster_ids = (
    primary[
        "uniquepid"
    ]
    .astype(str)
    .to_numpy()
)


unique_clusters = np.unique(
    cluster_ids
)


cluster_to_idx = {
    cid:
        np.where(
            cluster_ids == cid
        )[0]

    for cid
    in unique_clusters
}


N_BOOT = 2000

rng = np.random.default_rng(
    20260919
)


boot_rows = []


for b in range(N_BOOT):

    sampled_clusters = (
        rng.choice(
            unique_clusters,
            size=len(
                unique_clusters
            ),
            replace=True
        )
    )


    idx_parts = [
        cluster_to_idx[cid]
        for cid
        in sampled_clusters
    ]


    idx = np.concatenate(
        idx_parts
    )


    yb = y[idx]


    if len(
        np.unique(
            yb
        )
    ) < 2:
        continue


    pa = pred_A[idx]
    pb = pred_B[idx]
    pc = pred_C[idx]


    boot_rows.append({

        "rep":
            b+1,

        "auc_A":
            auc_score(
                yb,
                pa
            ),

        "auc_B":
            auc_score(
                yb,
                pb
            ),

        "auc_C":
            auc_score(
                yb,
                pc
            ),

        "delta_auc_B_vs_A":
            (
                auc_score(
                    yb,
                    pb
                )
                -
                auc_score(
                    yb,
                    pa
                )
            ),

        "brier_A":
            brier_score(
                yb,
                pa
            ),

        "brier_B":
            brier_score(
                yb,
                pb
            ),

        "brier_C":
            brier_score(
                yb,
                pc
            ),

        "delta_brier_B_vs_A":
            (
                brier_score(
                    yb,
                    pb
                )
                -
                brier_score(
                    yb,
                    pa
                )
            ),
    })


    if (
        (b+1)
        % 200
        == 0
    ):
        print(
            f"Cluster bootstrap "
            f"{b+1}/{N_BOOT}"
        )


boot = pd.DataFrame(
    boot_rows
)


def ci(
    series
):

    return (
        float(
            series.quantile(
                0.025
            )
        ),
        float(
            series.quantile(
                0.975
            )
        )
    )


aucA_lo, aucA_hi = ci(
    boot["auc_A"]
)

aucB_lo, aucB_hi = ci(
    boot["auc_B"]
)

aucC_lo, aucC_hi = ci(
    boot["auc_C"]
)

dauc_lo, dauc_hi = ci(
    boot[
        "delta_auc_B_vs_A"
    ]
)

dbrier_lo, dbrier_hi = ci(
    boot[
        "delta_brier_B_vs_A"
    ]
)


boot_summary = pd.DataFrame([{

    "bootstrap_reps":
        len(boot),

    "n_external":
        len(primary),

    "unique_patients":
        len(
            unique_clusters
        ),

    "deaths":
        int(
            y.sum()
        ),

    "auc_A":
        auc_score(
            y,
            pred_A
        ),

    "auc_A_ci_low":
        aucA_lo,

    "auc_A_ci_high":
        aucA_hi,

    "auc_B":
        auc_score(
            y,
            pred_B
        ),

    "auc_B_ci_low":
        aucB_lo,

    "auc_B_ci_high":
        aucB_hi,

    "auc_C":
        auc_score(
            y,
            pred_C
        ),

    "auc_C_ci_low":
        aucC_lo,

    "auc_C_ci_high":
        aucC_hi,

    "delta_auc_B_vs_A":
        (
            auc_score(
                y,
                pred_B
            )
            -
            auc_score(
                y,
                pred_A
            )
        ),

    "delta_auc_ci_low":
        dauc_lo,

    "delta_auc_ci_high":
        dauc_hi,

    "prob_delta_auc_gt_0":
        float(
            (
                boot[
                    "delta_auc_B_vs_A"
                ] > 0
            ).mean()
        ),

    "brier_A":
        brier_score(
            y,
            pred_A
        ),

    "brier_B":
        brier_score(
            y,
            pred_B
        ),

    "delta_brier_B_vs_A":
        (
            brier_score(
                y,
                pred_B
            )
            -
            brier_score(
                y,
                pred_A
            )
        ),

    "delta_brier_ci_low":
        dbrier_lo,

    "delta_brier_ci_high":
        dbrier_hi,

    "prob_delta_brier_lt_0":
        float(
            (
                boot[
                    "delta_brier_B_vs_A"
                ] < 0
            ).mean()
        ),
}])


boot_summary.to_csv(
    OUT_BOOT,
    index=False
)


# ============================================================
# 6. Calibration bins for primary external validation
# ============================================================

cal_rows = []


for model, pred in [
    (
        "A_baseline",
        pred_A
    ),
    (
        "B_baseline_plus_delta",
        pred_B
    ),
    (
        "C_h48_alone",
        pred_C
    ),
]:

    tmp = pd.DataFrame({
        "y":
            y,

        "pred":
            pred,
    })


    tmp["bin"] = pd.qcut(
        tmp["pred"],
        q=10,
        duplicates="drop"
    )


    z = (
        tmp.groupby(
            "bin",
            observed=True
        )
        .agg(
            n=(
                "y",
                "size"
            ),
            observed=(
                "y",
                "mean"
            ),
            predicted=(
                "pred",
                "mean"
            ),
        )
        .reset_index(
            drop=True
        )
    )


    z["model"] = model

    z["bin_number"] = (
        np.arange(
            1,
            len(z)+1
        )
    )


    cal_rows.append(z)


cal_bins = pd.concat(
    cal_rows,
    ignore_index=True
)


cal_bins.to_csv(
    OUT_CAL_BINS,
    index=False
)


# ============================================================
# 7. Decision curve analysis
# ============================================================

dca_rows = []


prevalence = (
    y.mean()
)


for t in np.arange(
    0.05,
    0.501,
    0.01
):

    treat_all = (
        prevalence
        -
        (
            1-prevalence
        )
        * t
        /
        (1-t)
    )


    nb_A = net_benefit(
        y,
        pred_A,
        t
    )

    nb_B = net_benefit(
        y,
        pred_B,
        t
    )

    nb_C = net_benefit(
        y,
        pred_C,
        t
    )


    dca_rows.append({

        "threshold":
            round(
                float(t),
                2
            ),

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
# 8. Save external predictions
# ============================================================

e.to_parquet(
    OUT_PRED,
    index=False,
    compression="zstd"
)


# ============================================================
# 9. Console output
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    260
)


print("=" * 86)
print("eICU CHECKPOINT E2B - FIXED-COEFFICIENT EXTERNAL VALIDATION")
print("=" * 86)


print()
print("===== LOCKED MIMIC COEFFICIENTS =====")

print(
    coef
    .round(6)
    .to_string(
        index=False
    )
)


print()
print("===== EXTERNAL PERFORMANCE =====")

print(
    performance
    .round(6)
    .to_string(
        index=False
    )
)


print()
print("===== PRIMARY CLUSTER-BOOTSTRAP EXTERNAL CI =====")

print(
    boot_summary
    .round(6)
    .to_string(
        index=False
    )
)


print()
print("===== EXTERNAL CALIBRATION =====")

print(
    calibration[
        calibration[
            "strategy"
        ] == "observed_primary"
    ]
    .round(6)
    .to_string(
        index=False
    )
)


print()
print("===== DCA SELECTED THRESHOLDS =====")

print(
    dca[
        dca[
            "threshold"
        ].isin(
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
    .round(6)
    .to_string(
        index=False
    )
)


print()
print("=" * 86)
print("eICU CHECKPOINT E2B COMPLETE")
print("=" * 86)

print(f"MIMIC coefficients : {OUT_COEF}")
print(f"External performance: {OUT_PERF}")
print(f"Cluster bootstrap   : {OUT_BOOT}")
print(f"Calibration         : {OUT_CAL}")
print(f"Calibration bins    : {OUT_CAL_BINS}")
print(f"DCA                 : {OUT_DCA}")
print(f"Predictions         : {OUT_PRED}")

