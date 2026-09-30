from pathlib import Path
import math

import duckdb
import numpy as np
import pandas as pd


HOME = Path.home()

MIMIC_ROOT = HOME / "Desktop" / "MIMIC_IV_3_1"
MIMIC_WORK = MIMIC_ROOT / "work"

EICU_ROOT = HOME / "Desktop" / "eICU_CRD_2_0"
EICU_WORK = EICU_ROOT / "work"

MIMIC_FILE = MIMIC_WORK / "h48_primary_analysis_cohort.parquet"
MIMIC_ICUSTAYS = MIMIC_ROOT / "data" / "icu" / "icustays.csv.gz"

EICU_FILE = EICU_WORK / "E2A_eicu_external_candidate.parquet"

OUT_LOCAL = EICU_WORK / "E2C_eicu_local_diagnostic.csv"
OUT_COEF = EICU_WORK / "E2C_eicu_local_coefficients.csv"
OUT_CHANGE = EICU_WORK / "E2C_delta_change_comparison.csv"
OUT_ICU = EICU_WORK / "E2C_harmonized_icu48_external_validation.csv"
OUT_BOOT = EICU_WORK / "E2C_harmonized_icu48_bootstrap.csv"


for p in [MIMIC_FILE, MIMIC_ICUSTAYS, EICU_FILE]:
    if not p.exists():
        raise FileNotFoundError(p)


def sigmoid(x):
    x = np.clip(np.asarray(x, float), -35, 35)
    return 1.0 / (1.0 + np.exp(-x))


def fit_logistic(X, y, max_iter=200, tol=1e-10):

    X = np.asarray(X, float)
    y = np.asarray(y, float)

    beta = np.zeros(X.shape[1], float)

    for _ in range(max_iter):

        eta = X @ beta
        p = sigmoid(eta)

        w = np.clip(
            p * (1-p),
            1e-9,
            None
        )

        score = X.T @ (y-p)

        info = (
            X.T
            @
            (
                X
                * w[:, None]
            )
        )

        info += np.eye(
            info.shape[0]
        ) * 1e-10

        step = (
            np.linalg.pinv(info)
            @ score
        )

        beta_new = beta + step

        if np.max(
            np.abs(beta_new-beta)
        ) < tol:

            beta = beta_new
            break

        beta = beta_new

    eta = X @ beta
    pred = sigmoid(eta)

    w = np.clip(
        pred * (1-pred),
        1e-9,
        None
    )

    info = (
        X.T
        @
        (
            X
            * w[:, None]
        )
    )

    cov = np.linalg.pinv(info)

    se = np.sqrt(
        np.diag(cov)
    )

    pred_clip = np.clip(
        pred,
        1e-15,
        1-1e-15
    )

    ll = np.sum(
        y*np.log(pred_clip)
        +
        (1-y)*np.log(1-pred_clip)
    )

    return {
        "beta": beta,
        "se": se,
        "pred": pred,
        "ll": float(ll),
    }


def auc_score(y, pred):

    y = np.asarray(y)
    pred = np.asarray(pred)

    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())

    if n1 == 0 or n0 == 0:
        return np.nan

    ranks = (
        pd.Series(pred)
        .rank(method="average")
        .to_numpy()
    )

    return float(
        (
            ranks[y == 1].sum()
            -
            n1*(n1+1)/2
        )
        /
        (n1*n0)
    )


def brier(y, pred):

    y = np.asarray(y, float)
    pred = np.asarray(pred, float)

    return float(
        np.mean(
            (y-pred)**2
        )
    )


def lr_p_df1(lr):

    lr = max(
        float(lr),
        0.0
    )

    return math.erfc(
        math.sqrt(
            lr/2.0
        )
    )


def fit_abc(df, base_col, h48_col, y_col):

    y = (
        df[y_col]
        .astype(int)
        .to_numpy()
    )

    base = (
        df[base_col]
        .astype(float)
        .to_numpy()
    )

    h48 = (
        df[h48_col]
        .astype(float)
        .to_numpy()
    )

    delta = h48 - base

    XA = np.column_stack([
        np.ones(len(df)),
        base,
    ])

    XB = np.column_stack([
        np.ones(len(df)),
        base,
        delta,
    ])

    XC = np.column_stack([
        np.ones(len(df)),
        h48,
    ])

    A = fit_logistic(XA, y)
    B = fit_logistic(XB, y)
    C = fit_logistic(XC, y)

    lr = 2.0 * (
        B["ll"]
        -
        A["ll"]
    )

    return {
        "A": A,
        "B": B,
        "C": C,
        "y": y,
        "base": base,
        "h48": h48,
        "delta": delta,
        "lr": lr,
        "lr_p": lr_p_df1(lr),
    }


def change_summary(
    df,
    database,
    cohort,
    base_col,
    h48_col,
    y_col
):

    z = df[
        [
            base_col,
            h48_col,
            y_col,
        ]
    ].copy()

    z["delta"] = (
        z[h48_col]
        -
        z[base_col]
    )

    z["direction"] = np.select(
        [
            z["delta"] < 0,
            z["delta"] == 0,
            z["delta"] > 0,
        ],
        [
            "improved",
            "unchanged",
            "worsened",
        ],
        default="unknown"
    )

    rows = []

    rows.append({
        "database": database,
        "cohort": cohort,
        "direction": "overall",
        "n": len(z),
        "deaths": int(z[y_col].sum()),
        "mortality_pct": 100*z[y_col].mean(),
        "delta_mean": z["delta"].mean(),
        "delta_sd": z["delta"].std(),
        "delta_q25": z["delta"].quantile(0.25),
        "delta_median": z["delta"].median(),
        "delta_q75": z["delta"].quantile(0.75),
    })

    for direction, g in z.groupby("direction"):

        rows.append({
            "database": database,
            "cohort": cohort,
            "direction": direction,
            "n": len(g),
            "deaths": int(g[y_col].sum()),
            "mortality_pct": 100*g[y_col].mean(),
            "delta_mean": g["delta"].mean(),
            "delta_sd": g["delta"].std(),
            "delta_q25": g["delta"].quantile(0.25),
            "delta_median": g["delta"].median(),
            "delta_q75": g["delta"].quantile(0.75),
        })

    return rows


# ============================================================
# 1. Load primary datasets
# ============================================================

m = pd.read_parquet(
    MIMIC_FILE
)

e = pd.read_parquet(
    EICU_FILE
)


if len(m) != 904:
    raise RuntimeError(
        f"Expected MIMIC n=904, got {len(m)}"
    )

if int(m["hospital_expire_flag"].sum()) != 205:
    raise RuntimeError(
        "Expected MIMIC deaths=205"
    )

if len(e) != 349:
    raise RuntimeError(
        f"Expected eICU n=349, got {len(e)}"
    )

if int(e["hospital_death"].sum()) != 76:
    raise RuntimeError(
        "Expected eICU deaths=76"
    )


print("=" * 86)
print("CHECKPOINT E2C - TRANSPORTABILITY DIAGNOSTIC")
print("=" * 86)


# ============================================================
# 2. LOCAL eICU DIAGNOSTIC
#
# This is NOT external validation.
# It asks whether delta MELD carries a local signal in eICU.
# ============================================================

e_fit = fit_abc(
    e,
    "meld3_observed_primary_baseline",
    "meld3_observed_primary_h48",
    "hospital_death",
)


local_rows = [
    {
        "model": "A_baseline",
        "n": len(e),
        "deaths": int(e_fit["y"].sum()),
        "auc": auc_score(
            e_fit["y"],
            e_fit["A"]["pred"]
        ),
        "brier": brier(
            e_fit["y"],
            e_fit["A"]["pred"]
        ),
        "delta_auc_vs_A": np.nan,
        "delta_brier_vs_A": np.nan,
        "lr_vs_A": np.nan,
        "lr_p": np.nan,
    },
    {
        "model": "B_baseline_plus_delta",
        "n": len(e),
        "deaths": int(e_fit["y"].sum()),
        "auc": auc_score(
            e_fit["y"],
            e_fit["B"]["pred"]
        ),
        "brier": brier(
            e_fit["y"],
            e_fit["B"]["pred"]
        ),
        "delta_auc_vs_A":
            auc_score(
                e_fit["y"],
                e_fit["B"]["pred"]
            )
            -
            auc_score(
                e_fit["y"],
                e_fit["A"]["pred"]
            ),
        "delta_brier_vs_A":
            brier(
                e_fit["y"],
                e_fit["B"]["pred"]
            )
            -
            brier(
                e_fit["y"],
                e_fit["A"]["pred"]
            ),
        "lr_vs_A": e_fit["lr"],
        "lr_p": e_fit["lr_p"],
    },
    {
        "model": "C_h48_alone",
        "n": len(e),
        "deaths": int(e_fit["y"].sum()),
        "auc": auc_score(
            e_fit["y"],
            e_fit["C"]["pred"]
        ),
        "brier": brier(
            e_fit["y"],
            e_fit["C"]["pred"]
        ),
        "delta_auc_vs_A": np.nan,
        "delta_brier_vs_A": np.nan,
        "lr_vs_A": np.nan,
        "lr_p": np.nan,
    },
]


local = pd.DataFrame(
    local_rows
)

local.to_csv(
    OUT_LOCAL,
    index=False
)


coef_rows = []

terms = {
    "A": [
        "intercept",
        "baseline_meld",
    ],
    "B": [
        "intercept",
        "baseline_meld",
        "delta_meld_48h",
    ],
    "C": [
        "intercept",
        "h48_meld",
    ],
}


for key in ["A", "B", "C"]:

    fit = e_fit[key]

    for i, term in enumerate(
        terms[key]
    ):

        beta = float(
            fit["beta"][i]
        )

        se = float(
            fit["se"][i]
        )

        coef_rows.append({
            "model": key,
            "term": term,
            "beta": beta,
            "se": se,
            "odds_ratio":
                math.exp(beta),
            "or_95ci_low":
                math.exp(
                    beta
                    -
                    1.96*se
                ),
            "or_95ci_high":
                math.exp(
                    beta
                    +
                    1.96*se
                ),
        })


coef = pd.DataFrame(
    coef_rows
)

coef.to_csv(
    OUT_COEF,
    index=False
)


# ============================================================
# 3. Derive MIMIC first ICU-stay duration
# ============================================================

required_ids = [
    "subject_id",
    "hadm_id",
]

for c in required_ids:
    if c not in m.columns:
        raise RuntimeError(
            f"MIMIC primary cohort missing {c}"
        )


con = duckdb.connect()

con.register(
    "m_ids",
    m[
        [
            "subject_id",
            "hadm_id",
        ]
    ]
    .drop_duplicates()
)


first_icu = con.execute(f"""
WITH ranked AS (

    SELECT
        i.subject_id,
        i.hadm_id,
        i.stay_id,
        i.intime,
        i.outtime,

        ROW_NUMBER() OVER (
            PARTITION BY
                i.subject_id,
                i.hadm_id

            ORDER BY
                i.intime,
                i.stay_id
        ) AS rn

    FROM read_csv_auto(
        '{MIMIC_ICUSTAYS}',
        header=true
    ) i

    INNER JOIN m_ids m
        ON i.subject_id = m.subject_id
       AND i.hadm_id = m.hadm_id
)

SELECT
    subject_id,
    hadm_id,
    stay_id AS first_stay_id,
    intime AS first_stay_intime,
    outtime AS first_stay_outtime,

    date_diff(
        'minute',
        intime,
        outtime
    ) AS first_stay_duration_min

FROM ranked

WHERE rn = 1
""").fetchdf()


con.close()


m = m.merge(
    first_icu,
    on=[
        "subject_id",
        "hadm_id",
    ],
    how="left"
)


if m[
    "first_stay_duration_min"
].isna().any():

    raise RuntimeError(
        "Some MIMIC primary rows failed ICU-stay linkage"
    )


m_icu = m[
    m[
        "first_stay_duration_min"
    ] > 2880
].copy()


e_icu = e[
    e[
        "unitdischargeoffset"
    ] > 2880
].copy()


# ============================================================
# 4. Compare delta distributions
# ============================================================

change_rows = []

change_rows += change_summary(
    m,
    "MIMIC-IV",
    "hospital48_primary",
    "meld3_primary_baseline",
    "meld3_primary_h48",
    "hospital_expire_flag",
)

change_rows += change_summary(
    e,
    "eICU",
    "hospital48_external",
    "meld3_observed_primary_baseline",
    "meld3_observed_primary_h48",
    "hospital_death",
)

change_rows += change_summary(
    m_icu,
    "MIMIC-IV",
    "icu48_harmonized",
    "meld3_primary_baseline",
    "meld3_primary_h48",
    "hospital_expire_flag",
)

change_rows += change_summary(
    e_icu,
    "eICU",
    "icu48_harmonized",
    "meld3_observed_primary_baseline",
    "meld3_observed_primary_h48",
    "hospital_death",
)


change = pd.DataFrame(
    change_rows
)

change.to_csv(
    OUT_CHANGE,
    index=False
)


# ============================================================
# 5. Harmonized ICU-at-48h development -> external validation
#
# Fit only in MIMIC ICU-at-48h subset.
# Apply fixed coefficients to eICU ICU-at-48h subset.
# ============================================================

m_icu_fit = fit_abc(
    m_icu,
    "meld3_primary_baseline",
    "meld3_primary_h48",
    "hospital_expire_flag",
)


y_ext = (
    e_icu[
        "hospital_death"
    ]
    .astype(int)
    .to_numpy()
)

base_ext = (
    e_icu[
        "meld3_observed_primary_baseline"
    ]
    .astype(float)
    .to_numpy()
)

h48_ext = (
    e_icu[
        "meld3_observed_primary_h48"
    ]
    .astype(float)
    .to_numpy()
)

delta_ext = (
    h48_ext
    -
    base_ext
)


betaA = m_icu_fit["A"]["beta"]
betaB = m_icu_fit["B"]["beta"]
betaC = m_icu_fit["C"]["beta"]


predA = sigmoid(
    betaA[0]
    +
    betaA[1]*base_ext
)

predB = sigmoid(
    betaB[0]
    +
    betaB[1]*base_ext
    +
    betaB[2]*delta_ext
)

predC = sigmoid(
    betaC[0]
    +
    betaC[1]*h48_ext
)


icu_rows = [
    {
        "analysis":
            "MIMIC_icu48_derivation",
        "model":
            "A_baseline",
        "n":
            len(m_icu),
        "deaths":
            int(
                m_icu[
                    "hospital_expire_flag"
                ].sum()
            ),
        "auc":
            auc_score(
                m_icu_fit["y"],
                m_icu_fit["A"]["pred"]
            ),
        "brier":
            brier(
                m_icu_fit["y"],
                m_icu_fit["A"]["pred"]
            ),
        "delta_auc_vs_A":
            np.nan,
        "delta_brier_vs_A":
            np.nan,
    },
    {
        "analysis":
            "MIMIC_icu48_derivation",
        "model":
            "B_baseline_plus_delta",
        "n":
            len(m_icu),
        "deaths":
            int(
                m_icu[
                    "hospital_expire_flag"
                ].sum()
            ),
        "auc":
            auc_score(
                m_icu_fit["y"],
                m_icu_fit["B"]["pred"]
            ),
        "brier":
            brier(
                m_icu_fit["y"],
                m_icu_fit["B"]["pred"]
            ),
        "delta_auc_vs_A":
            auc_score(
                m_icu_fit["y"],
                m_icu_fit["B"]["pred"]
            )
            -
            auc_score(
                m_icu_fit["y"],
                m_icu_fit["A"]["pred"]
            ),
        "delta_brier_vs_A":
            brier(
                m_icu_fit["y"],
                m_icu_fit["B"]["pred"]
            )
            -
            brier(
                m_icu_fit["y"],
                m_icu_fit["A"]["pred"]
            ),
    },
    {
        "analysis":
            "eICU_icu48_external",
        "model":
            "A_baseline",
        "n":
            len(e_icu),
        "deaths":
            int(y_ext.sum()),
        "auc":
            auc_score(
                y_ext,
                predA
            ),
        "brier":
            brier(
                y_ext,
                predA
            ),
        "delta_auc_vs_A":
            np.nan,
        "delta_brier_vs_A":
            np.nan,
    },
    {
        "analysis":
            "eICU_icu48_external",
        "model":
            "B_baseline_plus_delta",
        "n":
            len(e_icu),
        "deaths":
            int(y_ext.sum()),
        "auc":
            auc_score(
                y_ext,
                predB
            ),
        "brier":
            brier(
                y_ext,
                predB
            ),
        "delta_auc_vs_A":
            auc_score(
                y_ext,
                predB
            )
            -
            auc_score(
                y_ext,
                predA
            ),
        "delta_brier_vs_A":
            brier(
                y_ext,
                predB
            )
            -
            brier(
                y_ext,
                predA
            ),
    },
    {
        "analysis":
            "eICU_icu48_external",
        "model":
            "C_h48_alone",
        "n":
            len(e_icu),
        "deaths":
            int(y_ext.sum()),
        "auc":
            auc_score(
                y_ext,
                predC
            ),
        "brier":
            brier(
                y_ext,
                predC
            ),
        "delta_auc_vs_A":
            np.nan,
        "delta_brier_vs_A":
            np.nan,
    },
]


icu_result = pd.DataFrame(
    icu_rows
)

icu_result.to_csv(
    OUT_ICU,
    index=False
)


# ============================================================
# 6. Patient-clustered bootstrap for harmonized eICU ICU48
# ============================================================

cluster_ids = (
    e_icu[
        "uniquepid"
    ]
    .astype(str)
    .to_numpy()
)

unique_clusters = np.unique(
    cluster_ids
)

cluster_map = {
    cid:
        np.where(
            cluster_ids == cid
        )[0]

    for cid in unique_clusters
}


rng = np.random.default_rng(
    20260919
)

boot_rows = []


for b in range(2000):

    sampled = rng.choice(
        unique_clusters,
        size=len(unique_clusters),
        replace=True
    )

    idx = np.concatenate([
        cluster_map[c]
        for c in sampled
    ])

    yb = y_ext[idx]

    if len(
        np.unique(yb)
    ) < 2:
        continue

    pa = predA[idx]
    pb = predB[idx]

    boot_rows.append({
        "rep":
            b+1,

        "delta_auc":
            auc_score(
                yb,
                pb
            )
            -
            auc_score(
                yb,
                pa
            ),

        "delta_brier":
            brier(
                yb,
                pb
            )
            -
            brier(
                yb,
                pa
            ),
    })

    if (b+1) % 200 == 0:
        print(
            f"Harmonized ICU bootstrap "
            f"{b+1}/2000"
        )


boot = pd.DataFrame(
    boot_rows
)


boot_summary = pd.DataFrame([{
    "n_mimic_icu48":
        len(m_icu),

    "mimic_icu48_deaths":
        int(
            m_icu[
                "hospital_expire_flag"
            ].sum()
        ),

    "n_eicu_icu48":
        len(e_icu),

    "eicu_icu48_unique_patients":
        e_icu[
            "uniquepid"
        ].nunique(),

    "eicu_icu48_deaths":
        int(
            y_ext.sum()
        ),

    "delta_auc":
        auc_score(
            y_ext,
            predB
        )
        -
        auc_score(
            y_ext,
            predA
        ),

    "delta_auc_ci_low":
        boot[
            "delta_auc"
        ].quantile(
            0.025
        ),

    "delta_auc_ci_high":
        boot[
            "delta_auc"
        ].quantile(
            0.975
        ),

    "prob_delta_auc_gt_0":
        (
            boot[
                "delta_auc"
            ] > 0
        ).mean(),

    "delta_brier":
        brier(
            y_ext,
            predB
        )
        -
        brier(
            y_ext,
            predA
        ),

    "delta_brier_ci_low":
        boot[
            "delta_brier"
        ].quantile(
            0.025
        ),

    "delta_brier_ci_high":
        boot[
            "delta_brier"
        ].quantile(
            0.975
        ),

    "prob_delta_brier_lt_0":
        (
            boot[
                "delta_brier"
            ] < 0
        ).mean(),
}])


boot_summary.to_csv(
    OUT_BOOT,
    index=False
)


# ============================================================
# 7. Console output
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    260
)


print()
print("===== eICU LOCAL DIAGNOSTIC PERFORMANCE =====")

print(
    local
    .round(6)
    .to_string(
        index=False
    )
)


print()
print("===== eICU LOCAL DIAGNOSTIC COEFFICIENTS =====")

print(
    coef
    .round(6)
    .to_string(
        index=False
    )
)


print()
print("===== DELTA MELD CHANGE PATTERNS =====")

print(
    change
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("===== HARMONIZED ICU-AT-48H ANALYSIS =====")

print(
    icu_result
    .round(6)
    .to_string(
        index=False
    )
)


print()
print("===== HARMONIZED ICU-AT-48H BOOTSTRAP =====")

print(
    boot_summary
    .round(6)
    .to_string(
        index=False
    )
)


print()
print("=" * 86)
print("CHECKPOINT E2C COMPLETE")
print("=" * 86)

print(f"Local diagnostic : {OUT_LOCAL}")
print(f"Local coefficients: {OUT_COEF}")
print(f"Change patterns   : {OUT_CHANGE}")
print(f"ICU48 analysis    : {OUT_ICU}")
print(f"ICU48 bootstrap   : {OUT_BOOT}")
