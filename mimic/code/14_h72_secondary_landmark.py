from pathlib import Path
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INFILE = ROOT / "work" / "meld3_snapshots_wide.parquet"

OUT_SUMMARY = ROOT / "work" / "14_h72_model_summary.csv"
OUT_COEF = ROOT / "work" / "14_h72_coefficients.csv"
OUT_BOOT = ROOT / "work" / "14_h72_bootstrap_ci.csv"

if not INFILE.exists():
    raise FileNotFoundError(INFILE)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -35, 35)))


def fit_logistic(X, y, max_iter=100, tol=1e-9):

    X = np.asarray(X, float)
    y = np.asarray(y, float)

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

        step = np.linalg.pinv(
            info
        ) @ grad

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
        "ll": ll,
    }


def auc_score(y, p):

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


def brier(y, p):

    return float(
        np.mean(
            (
                np.asarray(y)
                - np.asarray(p)
            )**2
        )
    )


df = pd.read_parquet(
    INFILE
)

cohort = df[
    (df["eligible_h72"] == 1)
    & df["meld3_primary_baseline"].notna()
    & df["meld3_primary_h72"].notna()
].copy()

print("=" * 78)
print("CHECKPOINT 14 - 72H SECONDARY LANDMARK ANALYSIS")
print("=" * 78)

print(
    f"Paired cohort: {len(cohort)}"
)

if len(cohort) != 766:
    raise RuntimeError(
        f"Expected n=766, found {len(cohort)}"
    )

y = (
    cohort["hospital_expire_flag"]
    .astype(int)
    .to_numpy()
)

if y.sum() != 164:
    raise RuntimeError(
        f"Expected 164 deaths, found {y.sum()}"
    )

baseline = (
    cohort["meld3_primary_baseline"]
    .astype(float)
    .to_numpy()
)

h72 = (
    cohort["meld3_primary_h72"]
    .astype(float)
    .to_numpy()
)

delta = h72 - baseline

strict_baseline = (
    cohort["meld3_strict_baseline"]
    .astype(float)
    .to_numpy()
)

strict_h72 = (
    cohort["meld3_strict_h72"]
    .astype(float)
    .to_numpy()
)

strict_delta = (
    strict_h72
    - strict_baseline
)

age = (
    cohort["age_at_admission"]
    .astype(float)
    .to_numpy()
)

female = (
    cohort["gender"]
    .astype(str)
    .str.upper()
    .eq("F")
    .astype(float)
    .to_numpy()
)


X_A = np.column_stack([
    np.ones(len(cohort)),
    baseline
])

X_B = np.column_stack([
    np.ones(len(cohort)),
    baseline,
    delta
])

X_C = np.column_stack([
    np.ones(len(cohort)),
    h72
])

X_STRICT = np.column_stack([
    np.ones(len(cohort)),
    strict_baseline,
    strict_delta
])

X_A_ADJ = np.column_stack([
    np.ones(len(cohort)),
    baseline,
    age-age.mean(),
    female
])

X_B_ADJ = np.column_stack([
    np.ones(len(cohort)),
    baseline,
    delta,
    age-age.mean(),
    female
])


fit_A = fit_logistic(X_A, y)
fit_B = fit_logistic(X_B, y)
fit_C = fit_logistic(X_C, y)
fit_STRICT = fit_logistic(X_STRICT, y)
fit_A_ADJ = fit_logistic(X_A_ADJ, y)
fit_B_ADJ = fit_logistic(X_B_ADJ, y)


lr = max(
    0.0,
    2*(fit_B["ll"]-fit_A["ll"])
)

lr_p = math.erfc(
    math.sqrt(lr/2)
)

lr_adj = max(
    0.0,
    2*(fit_B_ADJ["ll"]-fit_A_ADJ["ll"])
)

lr_adj_p = math.erfc(
    math.sqrt(lr_adj/2)
)


rows = []

for name, fit in [
    ("A_baseline", fit_A),
    ("B_baseline_plus_delta72", fit_B),
    ("C_h72_alone", fit_C),
    ("B_strict_rrt", fit_STRICT),
    ("A_baseline_age_sex", fit_A_ADJ),
    ("B_delta72_age_sex", fit_B_ADJ),
]:

    rows.append({
        "model": name,
        "n": len(cohort),
        "deaths": int(y.sum()),
        "auc": auc_score(
            y,
            fit["pred"]
        ),
        "brier": brier(
            y,
            fit["pred"]
        ),
        "log_likelihood":
            fit["ll"],
    })


summary = pd.DataFrame(rows)

summary["delta_auc_vs_A"] = np.nan
summary["delta_brier_vs_A"] = np.nan

auc_A = auc_score(
    y,
    fit_A["pred"]
)

br_A = brier(
    y,
    fit_A["pred"]
)

m = (
    summary["model"]
    == "B_baseline_plus_delta72"
)

summary.loc[
    m,
    "delta_auc_vs_A"
] = (
    auc_score(
        y,
        fit_B["pred"]
    )
    - auc_A
)

summary.loc[
    m,
    "delta_brier_vs_A"
] = (
    brier(
        y,
        fit_B["pred"]
    )
    - br_A
)


coef_rows = []

for model, fit, names in [
    (
        "B_baseline_plus_delta72",
        fit_B,
        [
            "intercept",
            "baseline_meld",
            "delta_meld_72h",
        ]
    ),
    (
        "B_strict_rrt",
        fit_STRICT,
        [
            "intercept",
            "baseline_meld",
            "delta_meld_72h",
        ]
    ),
    (
        "B_delta72_age_sex",
        fit_B_ADJ,
        [
            "intercept",
            "baseline_meld",
            "delta_meld_72h",
            "age_centered",
            "female",
        ]
    ),
]:

    for i, term in enumerate(names):

        if term == "intercept":
            continue

        beta = float(
            fit["beta"][i]
        )

        se = float(
            fit["se"][i]
        )

        coef_rows.append({
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
                    beta-1.96*se
                ),
            "or_95ci_high":
                math.exp(
                    beta+1.96*se
                ),
        })


coef = pd.DataFrame(
    coef_rows
)


# ============================================================
# Paired bootstrap uncertainty
# ============================================================

N_BOOT = 2000

rng = np.random.default_rng(
    20260919
)

boot_rows = []

for i in range(N_BOOT):

    idx = rng.integers(
        0,
        len(cohort),
        len(cohort)
    )

    yb = y[idx]

    if len(np.unique(yb)) < 2:
        continue

    fA = fit_logistic(
        X_A[idx],
        yb
    )

    fB = fit_logistic(
        X_B[idx],
        yb
    )

    auc_a = auc_score(
        yb,
        fA["pred"]
    )

    auc_b = auc_score(
        yb,
        fB["pred"]
    )

    br_a = brier(
        yb,
        fA["pred"]
    )

    br_b = brier(
        yb,
        fB["pred"]
    )

    boot_rows.append({
        "iteration":
            i+1,

        "delta_auc":
            auc_b-auc_a,

        "delta_brier":
            br_b-br_a,
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

    "delta_auc_apparent":
        auc_score(
            y,
            fit_B["pred"]
        )
        -
        auc_A,

    "delta_auc_ci_low":
        boot[
            "delta_auc"
        ].quantile(0.025),

    "delta_auc_ci_high":
        boot[
            "delta_auc"
        ].quantile(0.975),

    "prob_delta_auc_gt_0":
        (
            boot[
                "delta_auc"
            ] > 0
        ).mean(),

    "delta_brier_apparent":
        brier(
            y,
            fit_B["pred"]
        )
        -
        br_A,

    "delta_brier_ci_low":
        boot[
            "delta_brier"
        ].quantile(0.025),

    "delta_brier_ci_high":
        boot[
            "delta_brier"
        ].quantile(0.975),

    "prob_delta_brier_lt_0":
        (
            boot[
                "delta_brier"
            ] < 0
        ).mean(),

    "lr_chisq":
        lr,

    "lr_p":
        lr_p,

    "age_sex_lr_chisq":
        lr_adj,

    "age_sex_lr_p":
        lr_adj_p,
}])


summary.to_csv(
    OUT_SUMMARY,
    index=False
)

coef.to_csv(
    OUT_COEF,
    index=False
)

boot_summary.to_csv(
    OUT_BOOT,
    index=False
)


pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    220
)


print()
print("===== 72H MODEL PERFORMANCE =====")
print(
    summary
    .round(6)
    .to_string(index=False)
)


print()
print("===== 72H COEFFICIENTS =====")
print(
    coef
    .round(6)
    .to_string(index=False)
)


print()
print("===== 72H BOOTSTRAP INCREMENTAL CI =====")
print(
    boot_summary
    .round(6)
    .to_string(index=False)
)


print()
print("=" * 78)
print("CHECKPOINT 14 COMPLETE")
print("=" * 78)

print(f"Summary     : {OUT_SUMMARY}")
print(f"Coefficients: {OUT_COEF}")
print(f"Bootstrap   : {OUT_BOOT}")

