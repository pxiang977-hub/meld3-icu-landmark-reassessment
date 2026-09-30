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

OUT = EICU_WORK / "E2D_cross_database_heterogeneity.csv"
OUT_BOOT = EICU_WORK / "E2D_heterogeneity_bootstrap.csv"

for p in [MIMIC_FILE, MIMIC_ICUSTAYS, EICU_FILE]:
    if not p.exists():
        raise FileNotFoundError(p)


def sigmoid(x):
    x = np.clip(np.asarray(x, float), -35, 35)
    return 1.0 / (1.0 + np.exp(-x))


def fit_model(base, h48, y):

    base = np.asarray(base, float)
    h48 = np.asarray(h48, float)
    y = np.asarray(y, float)

    delta = h48 - base

    X = np.column_stack([
        np.ones(len(y)),
        base,
        delta
    ])

    beta = np.zeros(3)

    for _ in range(200):

        p = sigmoid(X @ beta)

        w = np.clip(
            p * (1-p),
            1e-9,
            None
        )

        score = X.T @ (y-p)

        info = (
            X.T
            @
            (X * w[:, None])
        )

        step = (
            np.linalg.pinv(info)
            @ score
        )

        new_beta = beta + step

        if np.max(
            np.abs(new_beta-beta)
        ) < 1e-10:
            beta = new_beta
            break

        beta = new_beta

    p = sigmoid(X @ beta)

    w = np.clip(
        p*(1-p),
        1e-9,
        None
    )

    info = (
        X.T
        @
        (X*w[:, None])
    )

    cov = np.linalg.pinv(info)

    se = np.sqrt(
        np.diag(cov)
    )

    return beta, se


def summarize_fit(name, df, base_col, h48_col, y_col):

    beta, se = fit_model(
        df[base_col],
        df[h48_col],
        df[y_col]
    )

    b = float(beta[2])
    s = float(se[2])

    return {
        "cohort": name,
        "n": len(df),
        "deaths": int(df[y_col].sum()),
        "delta_beta": b,
        "delta_se": s,
        "delta_or": math.exp(b),
        "delta_or_ci_low": math.exp(b-1.96*s),
        "delta_or_ci_high": math.exp(b+1.96*s),
    }


# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

m = pd.read_parquet(MIMIC_FILE)
e = pd.read_parquet(EICU_FILE)

if len(m) != 904:
    raise RuntimeError(f"Unexpected MIMIC n={len(m)}")

if len(e) != 349:
    raise RuntimeError(f"Unexpected eICU n={len(e)}")


# ------------------------------------------------------------
# MIMIC first ICU stay linkage
# ------------------------------------------------------------

con = duckdb.connect()

con.register(
    "m_ids",
    m[["subject_id", "hadm_id"]].drop_duplicates()
)

icu = con.execute(f"""
WITH x AS (
    SELECT
        i.subject_id,
        i.hadm_id,
        i.stay_id,
        i.intime,
        i.outtime,
        ROW_NUMBER() OVER (
            PARTITION BY i.subject_id, i.hadm_id
            ORDER BY i.intime, i.stay_id
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
    date_diff(
        'minute',
        intime,
        outtime
    ) AS first_icu_duration_min
FROM x
WHERE rn = 1
""").fetchdf()

con.close()

m = m.merge(
    icu,
    on=["subject_id", "hadm_id"],
    how="left"
)

m_icu = m[
    m["first_icu_duration_min"] > 2880
].copy()

e_icu = e[
    e["unitdischargeoffset"] > 2880
].copy()


# ------------------------------------------------------------
# Point estimates
# ------------------------------------------------------------

summary = []

summary.append(
    summarize_fit(
        "MIMIC_hospital48",
        m,
        "meld3_primary_baseline",
        "meld3_primary_h48",
        "hospital_expire_flag"
    )
)

summary.append(
    summarize_fit(
        "eICU_hospital48",
        e,
        "meld3_observed_primary_baseline",
        "meld3_observed_primary_h48",
        "hospital_death"
    )
)

summary.append(
    summarize_fit(
        "MIMIC_icu48",
        m_icu,
        "meld3_primary_baseline",
        "meld3_primary_h48",
        "hospital_expire_flag"
    )
)

summary.append(
    summarize_fit(
        "eICU_icu48",
        e_icu,
        "meld3_observed_primary_baseline",
        "meld3_observed_primary_h48",
        "hospital_death"
    )
)

summary_df = pd.DataFrame(summary)


def classical_difference(row1, row2, label):

    diff = (
        row2["delta_beta"]
        -
        row1["delta_beta"]
    )

    se = math.sqrt(
        row1["delta_se"]**2
        +
        row2["delta_se"]**2
    )

    z = diff / se

    p = math.erfc(
        abs(z) / math.sqrt(2)
    )

    return {
        "comparison": label,
        "beta_difference_eicu_minus_mimic": diff,
        "difference_se": se,
        "z": z,
        "p_heterogeneity": p,
    }


diff_rows = []

diff_rows.append(
    classical_difference(
        summary[0],
        summary[1],
        "hospital48"
    )
)

diff_rows.append(
    classical_difference(
        summary[2],
        summary[3],
        "icu48_harmonized"
    )
)

diff_df = pd.DataFrame(diff_rows)


# ------------------------------------------------------------
# Independent cluster bootstrap
# ------------------------------------------------------------

def cluster_boot_sample(df, cluster_col, rng):

    clusters = df[cluster_col].astype(str).unique()

    sampled = rng.choice(
        clusters,
        size=len(clusters),
        replace=True
    )

    parts = []

    for i, cid in enumerate(sampled):

        g = df[
            df[cluster_col].astype(str) == cid
        ].copy()

        # New bootstrap cluster identifier not required for fitting.
        parts.append(g)

    return pd.concat(
        parts,
        ignore_index=True
    )


rng = np.random.default_rng(20260919)

boot_rows = []

N_BOOT = 2000

for b in range(N_BOOT):

    mb = cluster_boot_sample(
        m,
        "subject_id",
        rng
    )

    eb = cluster_boot_sample(
        e,
        "uniquepid",
        rng
    )

    mi = cluster_boot_sample(
        m_icu,
        "subject_id",
        rng
    )

    ei = cluster_boot_sample(
        e_icu,
        "uniquepid",
        rng
    )

    try:
        beta_m, _ = fit_model(
            mb["meld3_primary_baseline"],
            mb["meld3_primary_h48"],
            mb["hospital_expire_flag"]
        )

        beta_e, _ = fit_model(
            eb["meld3_observed_primary_baseline"],
            eb["meld3_observed_primary_h48"],
            eb["hospital_death"]
        )

        beta_mi, _ = fit_model(
            mi["meld3_primary_baseline"],
            mi["meld3_primary_h48"],
            mi["hospital_expire_flag"]
        )

        beta_ei, _ = fit_model(
            ei["meld3_observed_primary_baseline"],
            ei["meld3_observed_primary_h48"],
            ei["hospital_death"]
        )

        boot_rows.append({
            "rep": b+1,

            "mimic_hospital48_delta_beta":
                beta_m[2],

            "eicu_hospital48_delta_beta":
                beta_e[2],

            "hospital48_beta_difference":
                beta_e[2] - beta_m[2],

            "mimic_icu48_delta_beta":
                beta_mi[2],

            "eicu_icu48_delta_beta":
                beta_ei[2],

            "icu48_beta_difference":
                beta_ei[2] - beta_mi[2],
        })

    except Exception:
        continue

    if (b+1) % 200 == 0:
        print(
            f"Bootstrap {b+1}/{N_BOOT}"
        )


boot = pd.DataFrame(boot_rows)

boot.to_csv(
    OUT_BOOT,
    index=False
)


heterogeneity_boot = pd.DataFrame([
    {
        "comparison": "hospital48",
        "bootstrap_reps": len(boot),
        "beta_difference":
            summary[1]["delta_beta"]
            -
            summary[0]["delta_beta"],
        "ci_low":
            boot[
                "hospital48_beta_difference"
            ].quantile(0.025),
        "ci_high":
            boot[
                "hospital48_beta_difference"
            ].quantile(0.975),
        "prob_eicu_effect_smaller":
            (
                boot[
                    "hospital48_beta_difference"
                ] < 0
            ).mean(),
    },
    {
        "comparison": "icu48_harmonized",
        "bootstrap_reps": len(boot),
        "beta_difference":
            summary[3]["delta_beta"]
            -
            summary[2]["delta_beta"],
        "ci_low":
            boot[
                "icu48_beta_difference"
            ].quantile(0.025),
        "ci_high":
            boot[
                "icu48_beta_difference"
            ].quantile(0.975),
        "prob_eicu_effect_smaller":
            (
                boot[
                    "icu48_beta_difference"
                ] < 0
            ).mean(),
    }
])


summary_df["section"] = "cohort_effect"
diff_df["section"] = "classical_heterogeneity"
heterogeneity_boot["section"] = "bootstrap_heterogeneity"

combined = pd.concat(
    [
        summary_df,
        diff_df,
        heterogeneity_boot
    ],
    ignore_index=True,
    sort=False
)

combined.to_csv(
    OUT,
    index=False
)


pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    240
)


print()
print("===== DATABASE-SPECIFIC DELTA EFFECTS =====")
print(
    summary_df
    .round(6)
    .to_string(index=False)
)

print()
print("===== CLASSICAL HETEROGENEITY TEST =====")
print(
    diff_df
    .round(6)
    .to_string(index=False)
)

print()
print("===== CLUSTER-BOOTSTRAP HETEROGENEITY =====")
print(
    heterogeneity_boot
    .round(6)
    .to_string(index=False)
)

print()
print("=" * 86)
print("CHECKPOINT E2D COMPLETE")
print("=" * 86)

print(f"Summary  : {OUT}")
print(f"Bootstrap: {OUT_BOOT}")
