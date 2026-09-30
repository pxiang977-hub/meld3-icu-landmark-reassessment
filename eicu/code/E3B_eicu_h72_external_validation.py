from pathlib import Path
import math

import duckdb
import numpy as np
import pandas as pd


# ============================================================
# eICU Checkpoint E3B
#
# 72-HOUR SECONDARY FIXED-COEFFICIENT EXTERNAL VALIDATION
#
# Primary external cohort:
#   eICU hospital-72h paired cohort from Checkpoint E3A
#
# Reassessment window:
#   >2880 to <=4320 min from ICU admission (48-72h)
#
# Baseline window:
#   -360 to +720 min from ICU admission
#
# Model coefficients:
#   Locked MIMIC-IV Checkpoint 14 coefficients.
#   eICU outcomes are NEVER used to estimate Models A/B/C.
#
# This script deliberately reproduces the E2A/E2B machinery:
# - latest valid lab within each snapshot window
# - creatinine-test-anchored 7-day dialysis lookback
# - observed-primary / observed-strict / pragmatic RRT strategies
# - official MELD 3.0 formula and bounds used in E2A
# - fixed-coefficient performance, calibration, patient-cluster bootstrap, DCA
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WORK = ROOT / "work"

HOME = Path.home()
MIMIC_WORK = HOME / "Desktop" / "MIMIC_IV_3_1" / "work"

E3A_CANDIDATE = WORK / "E3A_h72_candidate_stays.parquet"
RRT_FILE = WORK / "E1C_rrt_events.parquet"
RRT_AUDIT_FILE = WORK / "E1C_paired_rrt_subject_audit.csv"
LAB_FILE = DATA / "lab.csv.gz"
APACHE_FILE = DATA / "apacheApsVar.csv.gz"

MIMIC_H72_COEF_FILE = MIMIC_WORK / "14b_h72_fixed_coefficients_for_external_validation.csv"

OUT_COHORT = WORK / "E3B_h72_external_candidate.parquet"
OUT_LABS = WORK / "E3B_h72_lab_snapshots.parquet"
OUT_RRT = WORK / "E3B_h72_snapshot_rrt_flags.csv"
OUT_MELD_QC = WORK / "E3B_h72_meld3_strategy_qc.csv"
OUT_COEF = WORK / "E3B_h72_locked_mimic_coefficients.csv"
OUT_PERF = WORK / "E3B_h72_external_performance.csv"
OUT_BOOT = WORK / "E3B_h72_cluster_bootstrap_ci.csv"
OUT_CAL = WORK / "E3B_h72_external_calibration.csv"
OUT_CAL_BINS = WORK / "E3B_h72_external_calibration_bins.csv"
OUT_DCA = WORK / "E3B_h72_external_dca.csv"
OUT_PRED = WORK / "E3B_h72_external_predictions.parquet"
OUT_QC = WORK / "E3B_h72_qc.csv"
OUT_LAB_FAILURES = WORK / "E3B_h72_lab_recheck_failures.csv"

for p in [
    E3A_CANDIDATE,
    RRT_FILE,
    LAB_FILE,
    MIMIC_H72_COEF_FILE,
]:
    if not p.exists():
        raise FileNotFoundError(p)


# ============================================================
# Utility functions (kept aligned with E2B)
# ============================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def fit_logistic(X, y, max_iter=200, tol=1e-10):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.zeros(X.shape[1], dtype=float)

    for _ in range(max_iter):
        eta = X @ beta
        p = sigmoid(eta)
        w = np.clip(p * (1 - p), 1e-9, None)
        score = X.T @ (y - p)
        info = X.T @ (X * w[:, None])
        info += np.eye(info.shape[0]) * 1e-10
        step = np.linalg.pinv(info) @ score
        beta_new = beta + step

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break

        beta = beta_new

    pred = sigmoid(X @ beta)
    return {"beta": beta, "pred": pred}


def auc_score(y, pred):
    y = np.asarray(y)
    pred = np.asarray(pred)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())

    if n1 == 0 or n0 == 0:
        return np.nan

    ranks = pd.Series(pred).rank(method="average").to_numpy()
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(auc)


def brier_score(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean((y - pred) ** 2))


def log_loss(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred)))


def calibration_intercept_only(y, lp):
    y = np.asarray(y, float)
    lp = np.asarray(lp, float)
    alpha = 0.0

    for _ in range(100):
        p = sigmoid(alpha + lp)
        score = np.sum(y - p)
        info = np.sum(p * (1 - p))

        if info <= 0:
            break

        step = score / info
        alpha_new = alpha + step

        if abs(alpha_new - alpha) < 1e-10:
            alpha = alpha_new
            break

        alpha = alpha_new

    return float(alpha)


def calibration_intercept_slope(y, lp):
    X = np.column_stack([np.ones(len(y)), lp])
    fit = fit_logistic(X, y)
    return float(fit["beta"][0]), float(fit["beta"][1])


def net_benefit(y, pred, threshold):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=float)
    treat = pred >= threshold
    tp = np.sum(treat & (y == 1))
    fp = np.sum(treat & (y == 0))
    n = len(y)
    return float(tp / n - fp / n * threshold / (1 - threshold))


def meld3(
    bilirubin,
    sodium,
    inr,
    creatinine,
    albumin,
    female,
    dialysis_flag,
):
    """Exact MELD 3.0 implementation used in E2A."""

    bili = max(float(bilirubin), 1.0)
    inr_v = max(float(inr), 1.0)
    cr = max(float(creatinine), 1.0)

    if dialysis_flag:
        cr = 3.0

    cr = min(cr, 3.0)
    na = min(max(float(sodium), 125.0), 137.0)
    alb = min(max(float(albumin), 1.5), 3.5)

    raw = (
        1.33 * int(female)
        + 4.56 * math.log(bili)
        + 0.82 * (137 - na)
        - 0.24 * (137 - na) * math.log(bili)
        + 9.09 * math.log(inr_v)
        + 11.14 * math.log(cr)
        + 1.85 * (3.5 - alb)
        - 1.83 * (3.5 - alb) * math.log(cr)
        + 6.0
    )

    score = math.floor(raw + 0.5)
    score = max(6, min(40, score))
    return float(score)


# ============================================================
# 1. Locked Hospital72 paired cohort from E3A
# ============================================================

e3a = pd.read_parquet(E3A_CANDIDATE)

required_e3a = [
    "patientunitstayid",
    "patienthealthsystemstayid",
    "uniquepid",
    "hospitalid",
    "gender",
    "hospitaldischargeoffset",
    "hospital_death",
    "hospital72_paired_complete",
]

missing = [c for c in required_e3a if c not in e3a.columns]
if missing:
    raise RuntimeError(f"Missing E3A columns: {missing}")

paired = e3a[e3a["hospital72_paired_complete"].astype(bool)].copy()

# Recompute death from status when available and guard against mismatch.
if "hospitaldischargestatus" in paired.columns:
    death_from_status = (
        paired["hospitaldischargestatus"]
        .astype(str)
        .str.lower()
        .eq("expired")
        .astype(int)
    )
    death_existing = pd.to_numeric(paired["hospital_death"], errors="coerce").fillna(0).astype(int)
    if not np.array_equal(death_from_status.to_numpy(), death_existing.to_numpy()):
        raise RuntimeError("hospital_death mismatch versus hospitaldischargestatus in E3A cohort")
    paired["hospital_death"] = death_from_status
else:
    paired["hospital_death"] = (
        pd.to_numeric(paired["hospital_death"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

if paired["patientunitstayid"].duplicated().any():
    raise RuntimeError("Hospital72 paired cohort contains duplicate patientunitstayid rows")

EXPECTED = {
    "n": 293,
    "unique_patients": 281,
    "hospitals": 80,
    "deaths": 61,
}

actual = {
    "n": len(paired),
    "unique_patients": paired["uniquepid"].nunique(),
    "hospitals": paired["hospitalid"].nunique(),
    "deaths": int(paired["hospital_death"].sum()),
}

for key, value in EXPECTED.items():
    if actual[key] != value:
        raise RuntimeError(
            f"E3A Hospital72 cohort mismatch for {key}: expected {value}, found {actual[key]}"
        )

print("=" * 88)
print("eICU CHECKPOINT E3B - 72H FIXED-COEFFICIENT EXTERNAL VALIDATION")
print("=" * 88)
print(
    f"Hospital72 paired cohort: {len(paired)} stays; "
    f"{paired['uniquepid'].nunique()} unique patients; "
    f"{paired['hospitalid'].nunique()} hospitals; "
    f"{int(paired['hospital_death'].sum())} deaths"
)


# ============================================================
# 2. Re-extract actual baseline and h72 laboratory values
#
# Reproduce E2A selection exactly, except that h48 is replaced by
# the prespecified h72 window >2880 to <=4320 min.
# ============================================================

con = duckdb.connect()
con.register("paired_ids", paired[["patientunitstayid"]])

labs = con.execute(f"""
WITH mapped AS (
    SELECT
        l.labid,
        l.patientunitstayid,
        l.labresultoffset,
        l.labresultrevisedoffset,
        l.labresult,

        CASE
            WHEN lower(trim(l.labname)) = 'albumin' THEN 'albumin'
            WHEN lower(trim(l.labname)) = 'total bilirubin' THEN 'bilirubin'
            WHEN lower(trim(l.labname)) = 'creatinine' THEN 'creatinine'
            WHEN lower(trim(l.labname)) = 'sodium' THEN 'sodium'
            WHEN lower(trim(l.labname)) = 'pt - inr' THEN 'inr'
            ELSE NULL
        END AS component,

        CASE
            WHEN l.labresultoffset BETWEEN -360 AND 720 THEN 'baseline'
            WHEN l.labresultoffset > 2880 AND l.labresultoffset <= 4320 THEN 'h72'
            ELSE NULL
        END AS snapshot

    FROM read_csv_auto('{LAB_FILE}', header=true) l

    INNER JOIN paired_ids p
        ON l.patientunitstayid = p.patientunitstayid

    WHERE
        lower(trim(l.labname)) IN (
            'albumin',
            'total bilirubin',
            'creatinine',
            'sodium',
            'pt - inr'
        )
        AND l.labresult IS NOT NULL
        AND l.labresult > 0
),

ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY patientunitstayid, snapshot, component
            ORDER BY
                labresultoffset DESC,
                labresultrevisedoffset DESC NULLS LAST,
                labid DESC
        ) AS rn
    FROM mapped
    WHERE snapshot IS NOT NULL
)

SELECT
    patientunitstayid,

    MAX(CASE WHEN snapshot='baseline' AND component='albumin'    AND rn=1 THEN labresult END) AS albumin_baseline,
    MAX(CASE WHEN snapshot='baseline' AND component='bilirubin'  AND rn=1 THEN labresult END) AS bilirubin_baseline,
    MAX(CASE WHEN snapshot='baseline' AND component='creatinine' AND rn=1 THEN labresult END) AS creatinine_baseline,
    MAX(CASE WHEN snapshot='baseline' AND component='creatinine' AND rn=1 THEN labresultoffset END) AS creatinine_offset_baseline,
    MAX(CASE WHEN snapshot='baseline' AND component='sodium'     AND rn=1 THEN labresult END) AS sodium_baseline,
    MAX(CASE WHEN snapshot='baseline' AND component='inr'        AND rn=1 THEN labresult END) AS inr_baseline,

    MAX(CASE WHEN snapshot='h72' AND component='albumin'    AND rn=1 THEN labresult END) AS albumin_h72,
    MAX(CASE WHEN snapshot='h72' AND component='bilirubin'  AND rn=1 THEN labresult END) AS bilirubin_h72,
    MAX(CASE WHEN snapshot='h72' AND component='creatinine' AND rn=1 THEN labresult END) AS creatinine_h72,
    MAX(CASE WHEN snapshot='h72' AND component='creatinine' AND rn=1 THEN labresultoffset END) AS creatinine_offset_h72,
    MAX(CASE WHEN snapshot='h72' AND component='sodium'     AND rn=1 THEN labresult END) AS sodium_h72,
    MAX(CASE WHEN snapshot='h72' AND component='inr'        AND rn=1 THEN labresult END) AS inr_h72

FROM ranked
GROUP BY patientunitstayid
""").fetchdf()

labs.to_parquet(OUT_LABS, index=False, compression="zstd")

x = paired.merge(labs, on="patientunitstayid", how="left")

needed = [
    "albumin_baseline",
    "bilirubin_baseline",
    "creatinine_baseline",
    "sodium_baseline",
    "inr_baseline",
    "albumin_h72",
    "bilirubin_h72",
    "creatinine_h72",
    "sodium_h72",
    "inr_h72",
]

missing_lab_mask = x[needed].isna().any(axis=1)
if missing_lab_mask.any():
    failures = x.loc[
        missing_lab_mask,
        ["patientunitstayid", "uniquepid", "hospitalid", "hospital_death"] + needed,
    ].copy()
    failures.to_csv(OUT_LAB_FAILURES, index=False)
    raise RuntimeError(
        f"{int(missing_lab_mask.sum())} E3A-paired rows lack E2A-valid positive selected labs. "
        f"Review {OUT_LAB_FAILURES} before proceeding."
    )

# Audit h72 values against the E3A extraction when available.
h72_value_mismatch_n = 0
for comp in ["albumin", "bilirubin", "creatinine", "sodium", "inr"]:
    old_col = f"h72_{comp}"
    new_col = f"{comp}_h72"
    if old_col in x.columns:
        old = pd.to_numeric(x[old_col], errors="coerce")
        new = pd.to_numeric(x[new_col], errors="coerce")
        mismatch = old.notna() & new.notna() & (~np.isclose(old, new, rtol=0, atol=1e-12))
        h72_value_mismatch_n += int(mismatch.sum())


# ============================================================
# 3. RRT reconstruction aligned with E2A
# ============================================================

rrt = pd.read_parquet(RRT_FILE)
rrt["treatmentoffset"] = pd.to_numeric(rrt["treatmentoffset"], errors="coerce")

# E2A used E1C_paired_rrt_subject_audit.csv for APACHE dialysis.
# That file may have been generated around the 48h paired cohort, so for 72h
# we preserve it when present and supplement missing candidate IDs directly
# from apacheApsVar.csv.gz when possible.
apache_map = {}
audit_direct_ids = set()
raw_fallback_ids = set()

if RRT_AUDIT_FILE.exists():
    audit = pd.read_csv(RRT_AUDIT_FILE, low_memory=False)
    if {"patientunitstayid", "apache_dialysis_positive"}.issubset(audit.columns):
        for _, row in audit[["patientunitstayid", "apache_dialysis_positive"]].dropna(
            subset=["patientunitstayid"]
        ).iterrows():
            sid = int(row["patientunitstayid"])
            val = int(pd.to_numeric(row["apache_dialysis_positive"], errors="coerce") == 1)
            apache_map[sid] = val
            audit_direct_ids.add(sid)

candidate_ids = set(x["patientunitstayid"].astype(int).tolist())
missing_apache_ids = candidate_ids - set(apache_map)

if missing_apache_ids and APACHE_FILE.exists():
    apache_raw = pd.read_csv(APACHE_FILE, compression="gzip", low_memory=False)
    lower_to_actual = {c.lower(): c for c in apache_raw.columns}

    if "patientunitstayid" in lower_to_actual and "dialysis" in lower_to_actual:
        sid_col = lower_to_actual["patientunitstayid"]
        dialysis_col = lower_to_actual["dialysis"]

        z = apache_raw[[sid_col, dialysis_col]].copy()
        z[sid_col] = pd.to_numeric(z[sid_col], errors="coerce")
        z[dialysis_col] = pd.to_numeric(z[dialysis_col], errors="coerce")
        z = z[z[sid_col].isin(missing_apache_ids)].copy()

        if len(z):
            agg = z.groupby(sid_col)[dialysis_col].max()
            for sid, val in agg.items():
                sid = int(sid)
                apache_map[sid] = int(pd.notna(val) and float(val) == 1.0)
                raw_fallback_ids.add(sid)

# Maintain E2A behavior: absent APACHE evidence defaults to 0.
for sid in candidate_ids:
    apache_map.setdefault(int(sid), 0)

ACTIVE_CLASSES = {
    "IHD",
    "CRRT_CVVHD",
    "CRRT_CVVH",
    "SLED",
    "PERITONEAL",
    "OTHER_DIALYSIS",
}


def dialysis_flags_for_snapshot(stayid, creatinine_time):
    g = rrt[
        (rrt["patientunitstayid"] == stayid)
        & (rrt["rrt_class"].isin(ACTIVE_CLASSES))
    ].copy()

    if pd.isna(creatinine_time):
        return {
            "ihd_days": 0,
            "crrt_broad_span_h": np.nan,
            "cvvhd_span_h": np.nan,
            "dialysis_observed_primary": 0,
            "dialysis_observed_strict": 0,
            "dialysis_pragmatic": 0,
            "any_rrt_evidence": 0,
        }

    lo = creatinine_time - 10080
    hi = creatinine_time

    g = g[(g["treatmentoffset"] >= lo) & (g["treatmentoffset"] <= hi)].copy()
    any_rrt = int(len(g) > 0)

    ihd = g[g["rrt_class"] == "IHD"].copy()
    if len(ihd):
        ihd["relative_day"] = np.floor(ihd["treatmentoffset"] / 1440.0)
        ihd_days = int(ihd["relative_day"].nunique())
    else:
        ihd_days = 0

    broad_crrt = g[g["rrt_class"].isin(["CRRT_CVVHD", "CRRT_CVVH"])]
    if len(broad_crrt) >= 2:
        broad_span_h = (
            broad_crrt["treatmentoffset"].max() - broad_crrt["treatmentoffset"].min()
        ) / 60.0
    else:
        broad_span_h = np.nan

    cvvhd = g[g["rrt_class"] == "CRRT_CVVHD"]
    if len(cvvhd) >= 2:
        cvvhd_span_h = (
            cvvhd["treatmentoffset"].max() - cvvhd["treatmentoffset"].min()
        ) / 60.0
    else:
        cvvhd_span_h = np.nan

    observed_primary = int(
        (ihd_days >= 2)
        or (pd.notna(broad_span_h) and broad_span_h >= 24)
    )

    observed_strict = int(
        (ihd_days >= 2)
        or (pd.notna(cvvhd_span_h) and cvvhd_span_h >= 24)
    )

    apache_positive = int(apache_map.get(int(stayid), 0) == 1)
    pragmatic = int(observed_primary or apache_positive)

    return {
        "ihd_days": ihd_days,
        "crrt_broad_span_h": broad_span_h,
        "cvvhd_span_h": cvvhd_span_h,
        "dialysis_observed_primary": observed_primary,
        "dialysis_observed_strict": observed_strict,
        "dialysis_pragmatic": pragmatic,
        "any_rrt_evidence": any_rrt,
    }


rrt_rows = []

for _, row in x.iterrows():
    stayid = int(row["patientunitstayid"])

    for snapshot in ["baseline", "h72"]:
        cr_time = row[f"creatinine_offset_{snapshot}"]
        z = dialysis_flags_for_snapshot(stayid, cr_time)
        z.update(
            {
                "patientunitstayid": stayid,
                "snapshot": snapshot,
                "creatinine_offset": cr_time,
                "apache_dialysis_positive": int(apache_map.get(stayid, 0) == 1),
            }
        )
        rrt_rows.append(z)

rrt_flags = pd.DataFrame(rrt_rows)
rrt_flags.to_csv(OUT_RRT, index=False)

for snapshot in ["baseline", "h72"]:
    z = rrt_flags[rrt_flags["snapshot"] == snapshot].copy()
    keep = [
        "patientunitstayid",
        "ihd_days",
        "crrt_broad_span_h",
        "cvvhd_span_h",
        "dialysis_observed_primary",
        "dialysis_observed_strict",
        "dialysis_pragmatic",
        "any_rrt_evidence",
        "apache_dialysis_positive",
    ]
    z = z[keep]
    z = z.rename(
        columns={
            c: f"{c}_{snapshot}"
            for c in keep
            if c != "patientunitstayid"
        }
    )
    x = x.merge(z, on="patientunitstayid", how="left")


# ============================================================
# 4. MELD 3.0 snapshots under three RRT strategies
# ============================================================

gender_norm = (
    x["gender"]
    .astype("string")
    .str.strip()
    .str.casefold()
)

unexpected_gender = sorted(
    set(gender_norm.dropna().unique())
    - {"female", "male"}
)

if unexpected_gender:
    raise RuntimeError(
        "Unexpected eICU gender values in formal MELD cohort: "
        + ", ".join(map(str, unexpected_gender))
    )

x["female"] = (
    gender_norm
    .eq("female")
    .fillna(False)
    .astype(int)
)

print(
    "Gender mapping QC:",
    {
        "Female": int((gender_norm == "female").sum()),
        "Male": int((gender_norm == "male").sum()),
        "female_flag_1": int(x["female"].sum()),
    }
)

STRATEGIES = {
    "observed_primary": "dialysis_observed_primary",
    "observed_strict": "dialysis_observed_strict",
    "pragmatic": "dialysis_pragmatic",
}

for strategy, flag_base in STRATEGIES.items():
    for snapshot in ["baseline", "h72"]:
        scores = []
        for _, row in x.iterrows():
            scores.append(
                meld3(
                    row[f"bilirubin_{snapshot}"],
                    row[f"sodium_{snapshot}"],
                    row[f"inr_{snapshot}"],
                    row[f"creatinine_{snapshot}"],
                    row[f"albumin_{snapshot}"],
                    row["female"],
                    row[f"{flag_base}_{snapshot}"],
                )
            )
        x[f"meld3_{strategy}_{snapshot}"] = scores

for strategy in STRATEGIES:
    x[f"delta_meld3_{strategy}"] = (
        x[f"meld3_{strategy}_h72"] - x[f"meld3_{strategy}_baseline"]
    )

x["nonrrt_clean"] = (
    (x["any_rrt_evidence_baseline"] == 0)
    & (x["any_rrt_evidence_h72"] == 0)
    & (x["apache_dialysis_positive_baseline"] == 0)
    & (x["apache_dialysis_positive_h72"] == 0)
).astype(int)


# ============================================================
# 5. Locally generated MIMIC 72h coefficients
# ============================================================

# Public releases must not redistribute fitted model artifacts derived from
# credentialed MIMIC data. Run mimic/code/14b_export_h72_fixed_coefficients.py
# under your own authorized MIMIC access before this script. The resulting
# coefficient file is then applied unchanged to eICU.
coef_src = pd.read_csv(MIMIC_H72_COEF_FILE)

required_terms = {
    "A_baseline": ["intercept", "baseline_meld"],
    "B_baseline_plus_delta72": ["intercept", "baseline_meld", "delta_meld_72h"],
    "C_h72_alone": ["intercept", "h72_meld"],
}

LOCKED = {}
for model, terms in required_terms.items():
    LOCKED[model] = {}
    for term in terms:
        z = coef_src[(coef_src["model"] == model) & (coef_src["term"] == term)]
        if len(z) != 1:
            raise RuntimeError(
                f"Could not uniquely load locally generated MIMIC 72h coefficient: {model}/{term}"
            )
        LOCKED[model][term] = float(z.iloc[0]["beta"])

coef_rows = [
    {"model": model, "term": term, "beta": beta}
    for model, terms in LOCKED.items()
    for term, beta in terms.items()
]
coef = pd.DataFrame(coef_rows)


# ============================================================
# 6. MELD/RRT strategy QC
# ============================================================

qc_rows = []


def add_qc(metric, value):
    qc_rows.append({"metric": metric, "value": value})


add_qc("n_external_candidate", len(x))
add_qc("n_unique_patients", x["uniquepid"].nunique())
add_qc("n_hospitals", x["hospitalid"].nunique())
add_qc("hospital_deaths", int(x["hospital_death"].sum()))
add_qc("h72_value_mismatch_vs_e3a_n", h72_value_mismatch_n)
add_qc("apache_audit_direct_candidate_ids", len(candidate_ids & audit_direct_ids))
add_qc("apache_raw_fallback_candidate_ids", len(candidate_ids & raw_fallback_ids))
add_qc("apache_candidate_ids_default_zero", len(candidate_ids - audit_direct_ids - raw_fallback_ids))

for snapshot in ["baseline", "h72"]:
    for strategy in ["observed_primary", "observed_strict", "pragmatic"]:
        add_qc(
            f"{snapshot}_{strategy}_dialysis_positive",
            int(x[f"dialysis_{strategy}_{snapshot}"].sum()),
        )

for snapshot in ["baseline", "h72"]:
    add_qc(
        f"{snapshot}_score_diff_pragmatic_vs_observed_n",
        int((x[f"meld3_pragmatic_{snapshot}"] != x[f"meld3_observed_primary_{snapshot}"]).sum()),
    )
    add_qc(
        f"{snapshot}_score_diff_primary_vs_strict_n",
        int((x[f"meld3_observed_primary_{snapshot}"] != x[f"meld3_observed_strict_{snapshot}"]).sum()),
    )
    add_qc(
        f"{snapshot}_max_abs_score_diff_pragmatic_vs_observed",
        float((x[f"meld3_pragmatic_{snapshot}"] - x[f"meld3_observed_primary_{snapshot}"]).abs().max()),
    )

add_qc("nonrrt_clean_n", int(x["nonrrt_clean"].sum()))
add_qc(
    "nonrrt_clean_deaths",
    int(x.loc[x["nonrrt_clean"] == 1, "hospital_death"].sum()),
)

for strategy in ["observed_primary", "observed_strict", "pragmatic"]:
    add_qc(
        f"{strategy}_baseline_median",
        float(x[f"meld3_{strategy}_baseline"].median()),
    )
    add_qc(
        f"{strategy}_h72_median",
        float(x[f"meld3_{strategy}_h72"].median()),
    )
    add_qc(
        f"{strategy}_delta_median",
        float(x[f"delta_meld3_{strategy}"].median()),
    )

meld_qc = pd.DataFrame(qc_rows)
meld_qc.to_csv(OUT_MELD_QC, index=False)


# ============================================================
# 7. Fixed MIMIC coefficients applied to eICU
# ============================================================

strategies = {
    "observed_primary": {
        "baseline": "meld3_observed_primary_baseline",
        "h72": "meld3_observed_primary_h72",
        "mask": np.ones(len(x), dtype=bool),
    },
    "observed_strict": {
        "baseline": "meld3_observed_strict_baseline",
        "h72": "meld3_observed_strict_h72",
        "mask": np.ones(len(x), dtype=bool),
    },
    "pragmatic": {
        "baseline": "meld3_pragmatic_baseline",
        "h72": "meld3_pragmatic_h72",
        "mask": np.ones(len(x), dtype=bool),
    },
    "nonrrt_clean": {
        "baseline": "meld3_observed_primary_baseline",
        "h72": "meld3_observed_primary_h72",
        "mask": (x["nonrrt_clean"] == 1).to_numpy(),
    },
}

performance_rows = []
calibration_rows = []

for strategy, cfg in strategies.items():
    mask = cfg["mask"]
    d = x.loc[mask].copy()

    if len(d) == 0:
        continue

    y = d["hospital_death"].astype(int).to_numpy()
    base = d[cfg["baseline"]].astype(float).to_numpy()
    h72 = d[cfg["h72"]].astype(float).to_numpy()
    delta = h72 - base

    lp_A = (
        LOCKED["A_baseline"]["intercept"]
        + LOCKED["A_baseline"]["baseline_meld"] * base
    )

    lp_B = (
        LOCKED["B_baseline_plus_delta72"]["intercept"]
        + LOCKED["B_baseline_plus_delta72"]["baseline_meld"] * base
        + LOCKED["B_baseline_plus_delta72"]["delta_meld_72h"] * delta
    )

    lp_C = (
        LOCKED["C_h72_alone"]["intercept"]
        + LOCKED["C_h72_alone"]["h72_meld"] * h72
    )

    pred_A = sigmoid(lp_A)
    pred_B = sigmoid(lp_B)
    pred_C = sigmoid(lp_C)

    if strategy == "observed_primary":
        x.loc[mask, "pred_A_fixed_mimic_h72"] = pred_A
        x.loc[mask, "pred_B_fixed_mimic_h72"] = pred_B
        x.loc[mask, "pred_C_fixed_mimic_h72"] = pred_C
        x.loc[mask, "lp_A_fixed_mimic_h72"] = lp_A
        x.loc[mask, "lp_B_fixed_mimic_h72"] = lp_B
        x.loc[mask, "lp_C_fixed_mimic_h72"] = lp_C

    metrics = {}

    for model_name, pred, lp in [
        ("A_baseline", pred_A, lp_A),
        ("B_baseline_plus_delta72", pred_B, lp_B),
        ("C_h72_alone", pred_C, lp_C),
    ]:
        auc = auc_score(y, pred)
        br = brier_score(y, pred)
        ll = log_loss(y, pred)

        metrics[model_name] = {
            "auc": auc,
            "brier": br,
            "log_loss": ll,
        }

        cal_int_only = calibration_intercept_only(y, lp)
        cal_int, cal_slope = calibration_intercept_slope(y, lp)

        calibration_rows.append(
            {
                "strategy": strategy,
                "model": model_name,
                "n": len(d),
                "deaths": int(y.sum()),
                "calibration_in_the_large": cal_int_only,
                "calibration_intercept": cal_int,
                "calibration_slope": cal_slope,
            }
        )

        performance_rows.append(
            {
                "strategy": strategy,
                "model": model_name,
                "n": len(d),
                "unique_patients": d["uniquepid"].nunique(),
                "hospitals": d["hospitalid"].nunique(),
                "deaths": int(y.sum()),
                "mortality_pct": 100 * y.mean(),
                "auc": auc,
                "brier": br,
                "log_loss": ll,
                "delta_auc_vs_A": np.nan,
                "delta_brier_vs_A": np.nan,
            }
        )

    performance_rows.append(
        {
            "strategy": strategy,
            "model": "B_minus_A_increment",
            "n": len(d),
            "unique_patients": d["uniquepid"].nunique(),
            "hospitals": d["hospitalid"].nunique(),
            "deaths": int(y.sum()),
            "mortality_pct": 100 * y.mean(),
            "auc": np.nan,
            "brier": np.nan,
            "log_loss": np.nan,
            "delta_auc_vs_A": (
                metrics["B_baseline_plus_delta72"]["auc"]
                - metrics["A_baseline"]["auc"]
            ),
            "delta_brier_vs_A": (
                metrics["B_baseline_plus_delta72"]["brier"]
                - metrics["A_baseline"]["brier"]
            ),
        }
    )

performance = pd.DataFrame(performance_rows)
calibration = pd.DataFrame(calibration_rows)
performance.to_csv(OUT_PERF, index=False)
calibration.to_csv(OUT_CAL, index=False)


# ============================================================
# 8. Primary patient-cluster bootstrap (observed-primary)
# ============================================================

primary = x.copy()
y = primary["hospital_death"].astype(int).to_numpy()
pred_A = primary["pred_A_fixed_mimic_h72"].astype(float).to_numpy()
pred_B = primary["pred_B_fixed_mimic_h72"].astype(float).to_numpy()
pred_C = primary["pred_C_fixed_mimic_h72"].astype(float).to_numpy()

cluster_ids = primary["uniquepid"].astype(str).to_numpy()
unique_clusters = np.unique(cluster_ids)
cluster_to_idx = {
    cid: np.where(cluster_ids == cid)[0]
    for cid in unique_clusters
}

N_BOOT = 2000
rng = np.random.default_rng(20260919)
boot_rows = []

for b in range(N_BOOT):
    sampled_clusters = rng.choice(
        unique_clusters,
        size=len(unique_clusters),
        replace=True,
    )
    idx = np.concatenate([cluster_to_idx[cid] for cid in sampled_clusters])
    yb = y[idx]

    if len(np.unique(yb)) < 2:
        continue

    pa = pred_A[idx]
    pb = pred_B[idx]
    pc = pred_C[idx]

    boot_rows.append(
        {
            "rep": b + 1,
            "auc_A": auc_score(yb, pa),
            "auc_B": auc_score(yb, pb),
            "auc_C": auc_score(yb, pc),
            "delta_auc_B_vs_A": auc_score(yb, pb) - auc_score(yb, pa),
            "brier_A": brier_score(yb, pa),
            "brier_B": brier_score(yb, pb),
            "brier_C": brier_score(yb, pc),
            "delta_brier_B_vs_A": brier_score(yb, pb) - brier_score(yb, pa),
        }
    )

    if (b + 1) % 200 == 0:
        print(f"Cluster bootstrap {b + 1}/{N_BOOT}")

boot = pd.DataFrame(boot_rows)


def ci(series):
    return float(series.quantile(0.025)), float(series.quantile(0.975))


aucA_lo, aucA_hi = ci(boot["auc_A"])
aucB_lo, aucB_hi = ci(boot["auc_B"])
aucC_lo, aucC_hi = ci(boot["auc_C"])
dauc_lo, dauc_hi = ci(boot["delta_auc_B_vs_A"])
dbrier_lo, dbrier_hi = ci(boot["delta_brier_B_vs_A"])

boot_summary = pd.DataFrame(
    [
        {
            "bootstrap_reps": len(boot),
            "n_external": len(primary),
            "unique_patients": len(unique_clusters),
            "deaths": int(y.sum()),
            "auc_A": auc_score(y, pred_A),
            "auc_A_ci_low": aucA_lo,
            "auc_A_ci_high": aucA_hi,
            "auc_B": auc_score(y, pred_B),
            "auc_B_ci_low": aucB_lo,
            "auc_B_ci_high": aucB_hi,
            "auc_C": auc_score(y, pred_C),
            "auc_C_ci_low": aucC_lo,
            "auc_C_ci_high": aucC_hi,
            "delta_auc_B_vs_A": auc_score(y, pred_B) - auc_score(y, pred_A),
            "delta_auc_ci_low": dauc_lo,
            "delta_auc_ci_high": dauc_hi,
            "prob_delta_auc_gt_0": float((boot["delta_auc_B_vs_A"] > 0).mean()),
            "brier_A": brier_score(y, pred_A),
            "brier_B": brier_score(y, pred_B),
            "delta_brier_B_vs_A": brier_score(y, pred_B) - brier_score(y, pred_A),
            "delta_brier_ci_low": dbrier_lo,
            "delta_brier_ci_high": dbrier_hi,
            "prob_delta_brier_lt_0": float((boot["delta_brier_B_vs_A"] < 0).mean()),
        }
    ]
)

boot_summary.to_csv(OUT_BOOT, index=False)


# ============================================================
# 9. Calibration bins for primary validation
# ============================================================

cal_rows = []

for model, pred in [
    ("A_baseline", pred_A),
    ("B_baseline_plus_delta72", pred_B),
    ("C_h72_alone", pred_C),
]:
    tmp = pd.DataFrame({"y": y, "pred": pred})
    tmp["bin"] = pd.qcut(tmp["pred"], q=10, duplicates="drop")
    z = (
        tmp.groupby("bin", observed=True)
        .agg(
            n=("y", "size"),
            observed=("y", "mean"),
            predicted=("pred", "mean"),
        )
        .reset_index(drop=True)
    )
    z["model"] = model
    z["bin_number"] = np.arange(1, len(z) + 1)
    cal_rows.append(z)

cal_bins = pd.concat(cal_rows, ignore_index=True)
cal_bins.to_csv(OUT_CAL_BINS, index=False)


# ============================================================
# 10. Decision curve analysis for primary validation
# ============================================================

dca_rows = []
prevalence = y.mean()

for t in np.arange(0.05, 0.501, 0.01):
    treat_all = prevalence - (1 - prevalence) * t / (1 - t)
    nb_A = net_benefit(y, pred_A, t)
    nb_B = net_benefit(y, pred_B, t)
    nb_C = net_benefit(y, pred_C, t)

    dca_rows.append(
        {
            "threshold": round(float(t), 2),
            "treat_none": 0.0,
            "treat_all": treat_all,
            "model_A_baseline": nb_A,
            "model_B_baseline_plus_delta72": nb_B,
            "model_C_h72_only": nb_C,
            "net_benefit_gain_B_vs_A": nb_B - nb_A,
        }
    )

dca = pd.DataFrame(dca_rows)
dca.to_csv(OUT_DCA, index=False)


# ============================================================
# 11. Final hard QC and save predictions
# ============================================================

hard_qc_rows = [
    {"check": "n_external", "value": len(x), "expected": 293, "pass": len(x) == 293},
    {"check": "unique_patients", "value": x["uniquepid"].nunique(), "expected": 281, "pass": x["uniquepid"].nunique() == 281},
    {"check": "hospitals", "value": x["hospitalid"].nunique(), "expected": 80, "pass": x["hospitalid"].nunique() == 80},
    {"check": "deaths", "value": int(x["hospital_death"].sum()), "expected": 61, "pass": int(x["hospital_death"].sum()) == 61},
    {"check": "selected_lab_missing_rows", "value": int(x[needed].isna().any(axis=1).sum()), "expected": 0, "pass": int(x[needed].isna().any(axis=1).sum()) == 0},
    {"check": "h72_creatinine_offset_min", "value": float(x["creatinine_offset_h72"].min()), "expected": ">2880", "pass": float(x["creatinine_offset_h72"].min()) > 2880},
    {"check": "h72_creatinine_offset_max", "value": float(x["creatinine_offset_h72"].max()), "expected": "<=4320", "pass": float(x["creatinine_offset_h72"].max()) <= 4320},
    {"check": "fixed_prediction_missing_rows", "value": int(x[["pred_A_fixed_mimic_h72", "pred_B_fixed_mimic_h72", "pred_C_fixed_mimic_h72"]].isna().any(axis=1).sum()), "expected": 0, "pass": int(x[["pred_A_fixed_mimic_h72", "pred_B_fixed_mimic_h72", "pred_C_fixed_mimic_h72"]].isna().any(axis=1).sum()) == 0},
    {"check": "bootstrap_completed_reps", "value": len(boot), "expected": 2000, "pass": len(boot) == 2000},
]

hard_qc = pd.DataFrame(hard_qc_rows)
hard_qc.to_csv(OUT_QC, index=False)

x.to_parquet(OUT_COHORT, index=False, compression="zstd")
x.to_parquet(OUT_PRED, index=False, compression="zstd")

con.close()


# ============================================================
# 12. Console output
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 300)
pd.set_option("display.width", 280)

print()
print("===== LOCKED MIMIC 72H COEFFICIENTS =====")
print(coef.round(9).to_string(index=False))

print()
print("===== H72 MELD/RRT STRATEGY QC =====")
print(meld_qc.to_string(index=False))

print()
print("===== EXTERNAL PERFORMANCE =====")
print(performance.round(6).to_string(index=False))

print()
print("===== PRIMARY CLUSTER-BOOTSTRAP EXTERNAL CI =====")
print(boot_summary.round(6).to_string(index=False))

print()
print("===== PRIMARY EXTERNAL CALIBRATION =====")
print(
    calibration[calibration["strategy"] == "observed_primary"]
    .round(6)
    .to_string(index=False)
)

print()
print("===== DCA SELECTED THRESHOLDS =====")
print(
    dca[dca["threshold"].isin([0.10, 0.15, 0.20, 0.25, 0.30, 0.40])]
    .round(6)
    .to_string(index=False)
)

print()
print("===== FINAL HARD QC =====")
print(hard_qc.to_string(index=False))

if not bool(hard_qc["pass"].all()):
    print()
    print("WARNING: At least one E3B hard QC check failed. Do not interpret results yet.")
    raise SystemExit(2)

print()
print("=" * 88)
print("eICU CHECKPOINT E3B COMPLETE - ALL HARD QC CHECKS PASS")
print("=" * 88)
print(f"External cohort      : {OUT_COHORT}")
print(f"Lab snapshots        : {OUT_LABS}")
print(f"RRT flags            : {OUT_RRT}")
print(f"MELD/RRT QC          : {OUT_MELD_QC}")
print(f"Locked coefficients  : {OUT_COEF}")
print(f"Performance          : {OUT_PERF}")
print(f"Cluster bootstrap    : {OUT_BOOT}")
print(f"Calibration          : {OUT_CAL}")
print(f"Calibration bins     : {OUT_CAL_BINS}")
print(f"DCA                  : {OUT_DCA}")
print(f"Predictions          : {OUT_PRED}")
print(f"Hard QC              : {OUT_QC}")
