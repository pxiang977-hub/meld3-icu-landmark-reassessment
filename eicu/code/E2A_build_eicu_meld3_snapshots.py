from pathlib import Path
import math

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WORK = ROOT / "work"

COHORT_FILE = WORK / "E1C_harmonized_first_icu_per_hospitalization.csv"
RRT_FILE = WORK / "E1C_rrt_events.parquet"
RRT_AUDIT_FILE = WORK / "E1C_paired_rrt_subject_audit.csv"
LAB_FILE = DATA / "lab.csv.gz"

OUT_COHORT = WORK / "E2A_eicu_external_candidate.parquet"
OUT_LABS = WORK / "E2A_eicu_lab_snapshots.parquet"
OUT_QC = WORK / "E2A_meld3_strategy_qc.csv"
OUT_RRT = WORK / "E2A_snapshot_rrt_flags.csv"


for p in [
    COHORT_FILE,
    RRT_FILE,
    RRT_AUDIT_FILE,
    LAB_FILE,
]:
    if not p.exists():
        raise FileNotFoundError(p)


# ============================================================
# 1. Harmonized external-validation candidate
# ============================================================

cohort = pd.read_csv(
    COHORT_FILE
)

cohort["hospital_death"] = (
    cohort["hospitaldischargestatus"]
    .astype(str)
    .str.lower()
    .eq("expired")
    .astype(int)
)

paired = cohort[
    (cohort["hospitaldischargeoffset"] > 2880)
    &
    (cohort["paired_complete"] == 1)
].copy()


print("=" * 84)
print("eICU CHECKPOINT E2A - FORMAL MELD 3.0 SNAPSHOT CONSTRUCTION")
print("=" * 84)

print(
    f"External candidate: {len(paired)} stays; "
    f"{paired['uniquepid'].nunique()} unique patients; "
    f"{paired['hospitalid'].nunique()} hospitals; "
    f"{int(paired['hospital_death'].sum())} deaths"
)

if len(paired) != 349:
    raise RuntimeError(
        f"Expected harmonized paired n=349, found {len(paired)}"
    )


# ============================================================
# 2. Extract latest lab before each landmark
#
# baseline: -360 to +720 min
# h48:     +1440 to +2880 min
#
# exact eICU lab names:
# albumin
# total bilirubin
# creatinine
# sodium
# PT - INR
# ============================================================

con = duckdb.connect()

con.register(
    "paired_ids",
    paired[["patientunitstayid"]]
)


labs = con.execute(f"""
WITH mapped AS (

    SELECT
        l.labid,
        l.patientunitstayid,
        l.labresultoffset,
        l.labresultrevisedoffset,
        l.labresult,

        CASE
            WHEN lower(trim(l.labname)) = 'albumin'
                THEN 'albumin'

            WHEN lower(trim(l.labname)) = 'total bilirubin'
                THEN 'bilirubin'

            WHEN lower(trim(l.labname)) = 'creatinine'
                THEN 'creatinine'

            WHEN lower(trim(l.labname)) = 'sodium'
                THEN 'sodium'

            WHEN lower(trim(l.labname)) = 'pt - inr'
                THEN 'inr'

            ELSE NULL
        END AS component,

        CASE
            WHEN l.labresultoffset BETWEEN -360 AND 720
                THEN 'baseline'

            WHEN l.labresultoffset BETWEEN 1440 AND 2880
                THEN 'h48'

            ELSE NULL
        END AS snapshot

    FROM read_csv_auto(
        '{LAB_FILE}',
        header=true
    ) l

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
            PARTITION BY
                patientunitstayid,
                snapshot,
                component

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

    MAX(
        CASE
            WHEN snapshot = 'baseline'
             AND component = 'albumin'
             AND rn = 1
            THEN labresult
        END
    ) AS albumin_baseline,

    MAX(
        CASE
            WHEN snapshot = 'baseline'
             AND component = 'bilirubin'
             AND rn = 1
            THEN labresult
        END
    ) AS bilirubin_baseline,

    MAX(
        CASE
            WHEN snapshot = 'baseline'
             AND component = 'creatinine'
             AND rn = 1
            THEN labresult
        END
    ) AS creatinine_baseline,

    MAX(
        CASE
            WHEN snapshot = 'baseline'
             AND component = 'creatinine'
             AND rn = 1
            THEN labresultoffset
        END
    ) AS creatinine_offset_baseline,

    MAX(
        CASE
            WHEN snapshot = 'baseline'
             AND component = 'sodium'
             AND rn = 1
            THEN labresult
        END
    ) AS sodium_baseline,

    MAX(
        CASE
            WHEN snapshot = 'baseline'
             AND component = 'inr'
             AND rn = 1
            THEN labresult
        END
    ) AS inr_baseline,


    MAX(
        CASE
            WHEN snapshot = 'h48'
             AND component = 'albumin'
             AND rn = 1
            THEN labresult
        END
    ) AS albumin_h48,

    MAX(
        CASE
            WHEN snapshot = 'h48'
             AND component = 'bilirubin'
             AND rn = 1
            THEN labresult
        END
    ) AS bilirubin_h48,

    MAX(
        CASE
            WHEN snapshot = 'h48'
             AND component = 'creatinine'
             AND rn = 1
            THEN labresult
        END
    ) AS creatinine_h48,

    MAX(
        CASE
            WHEN snapshot = 'h48'
             AND component = 'creatinine'
             AND rn = 1
            THEN labresultoffset
        END
    ) AS creatinine_offset_h48,

    MAX(
        CASE
            WHEN snapshot = 'h48'
             AND component = 'sodium'
             AND rn = 1
            THEN labresult
        END
    ) AS sodium_h48,

    MAX(
        CASE
            WHEN snapshot = 'h48'
             AND component = 'inr'
             AND rn = 1
            THEN labresult
        END
    ) AS inr_h48

FROM ranked

GROUP BY patientunitstayid
""").fetchdf()


labs.to_parquet(
    OUT_LABS,
    index=False,
    compression="zstd"
)


x = paired.merge(
    labs,
    on="patientunitstayid",
    how="left"
)


needed = [
    "albumin_baseline",
    "bilirubin_baseline",
    "creatinine_baseline",
    "sodium_baseline",
    "inr_baseline",
    "albumin_h48",
    "bilirubin_h48",
    "creatinine_h48",
    "sodium_h48",
    "inr_h48",
]


if x[needed].isna().any().any():
    bad = int(
        x[needed]
        .isna()
        .any(axis=1)
        .sum()
    )

    raise RuntimeError(
        f"{bad} paired rows unexpectedly lack selected labs"
    )


# ============================================================
# 3. RRT data
#
# IMPORTANT:
# The lookback is anchored to the SELECTED CREATININE TEST,
# not simply to the landmark time.
# ============================================================

rrt = pd.read_parquet(
    RRT_FILE
)

rrt["treatmentoffset"] = pd.to_numeric(
    rrt["treatmentoffset"],
    errors="coerce"
)


audit = pd.read_csv(
    RRT_AUDIT_FILE
)


apache_map = dict(
    zip(
        audit["patientunitstayid"],
        audit["apache_dialysis_positive"]
    )
)


ACTIVE_CLASSES = {
    "IHD",
    "CRRT_CVVHD",
    "CRRT_CVVH",
    "SLED",
    "PERITONEAL",
    "OTHER_DIALYSIS",
}


def dialysis_flags_for_snapshot(
    stayid,
    creatinine_time
):

    g = rrt[
        (rrt["patientunitstayid"] == stayid)
        &
        (rrt["rrt_class"].isin(ACTIVE_CLASSES))
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

    g = g[
        (g["treatmentoffset"] >= lo)
        &
        (g["treatmentoffset"] <= hi)
    ].copy()

    any_rrt = int(
        len(g) > 0
    )

    # Conservative IHD proxy:
    # at least 2 distinct ICU-relative treatment days.
    ihd = g[
        g["rrt_class"] == "IHD"
    ].copy()

    if len(ihd):
        ihd["relative_day"] = np.floor(
            ihd["treatmentoffset"] / 1440.0
        )

        ihd_days = int(
            ihd["relative_day"].nunique()
        )
    else:
        ihd_days = 0


    # Broad CKRT mapping:
    # CVVHD + CVVH
    broad_crrt = g[
        g["rrt_class"].isin(
            [
                "CRRT_CVVHD",
                "CRRT_CVVH",
            ]
        )
    ]

    if len(broad_crrt) >= 2:
        broad_span_h = (
            broad_crrt["treatmentoffset"].max()
            -
            broad_crrt["treatmentoffset"].min()
        ) / 60.0
    else:
        broad_span_h = np.nan


    # Strict official modality sensitivity:
    # CVVHD only
    cvvhd = g[
        g["rrt_class"] == "CRRT_CVVHD"
    ]

    if len(cvvhd) >= 2:
        cvvhd_span_h = (
            cvvhd["treatmentoffset"].max()
            -
            cvvhd["treatmentoffset"].min()
        ) / 60.0
    else:
        cvvhd_span_h = np.nan


    observed_primary = int(
        (ihd_days >= 2)
        or
        (
            pd.notna(broad_span_h)
            and broad_span_h >= 24
        )
    )


    observed_strict = int(
        (ihd_days >= 2)
        or
        (
            pd.notna(cvvhd_span_h)
            and cvvhd_span_h >= 24
        )
    )


    apache_positive = int(
        apache_map.get(
            stayid,
            0
        ) == 1
    )


    pragmatic = int(
        observed_primary
        or apache_positive
    )


    return {
        "ihd_days":
            ihd_days,

        "crrt_broad_span_h":
            broad_span_h,

        "cvvhd_span_h":
            cvvhd_span_h,

        "dialysis_observed_primary":
            observed_primary,

        "dialysis_observed_strict":
            observed_strict,

        "dialysis_pragmatic":
            pragmatic,

        "any_rrt_evidence":
            any_rrt,
    }


rrt_rows = []

for _, row in x.iterrows():

    stayid = row["patientunitstayid"]

    for snapshot in [
        "baseline",
        "h48",
    ]:

        cr_time = row[
            f"creatinine_offset_{snapshot}"
        ]

        z = dialysis_flags_for_snapshot(
            stayid,
            cr_time
        )

        z.update({
            "patientunitstayid":
                stayid,

            "snapshot":
                snapshot,

            "creatinine_offset":
                cr_time,

            "apache_dialysis_positive":
                int(
                    apache_map.get(
                        stayid,
                        0
                    ) == 1
                ),
        })

        rrt_rows.append(z)


rrt_flags = pd.DataFrame(
    rrt_rows
)

rrt_flags.to_csv(
    OUT_RRT,
    index=False
)


# ============================================================
# 4. Join RRT flags wide
# ============================================================

for snapshot in [
    "baseline",
    "h48",
]:

    z = rrt_flags[
        rrt_flags["snapshot"] == snapshot
    ].copy()

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

    x = x.merge(
        z,
        on="patientunitstayid",
        how="left"
    )


# ============================================================
# 5. MELD 3.0
#
# Current adult formula:
#
# 1.33*female
# + 4.56*ln(bilirubin)
# + 0.82*(137-Na)
# - 0.24*(137-Na)*ln(bilirubin)
# + 9.09*ln(INR)
# + 11.14*ln(creatinine)
# + 1.85*(3.5-albumin)
# - 1.83*(3.5-albumin)*ln(creatinine)
# + 6
#
# Bounds:
# bilirubin >= 1
# INR >= 1
# creatinine 1-3
# sodium 125-137
# albumin 1.5-3.5
# final MELD 6-40
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


def meld3(
    bilirubin,
    sodium,
    inr,
    creatinine,
    albumin,
    female,
    dialysis_flag
):

    bili = max(
        float(bilirubin),
        1.0
    )

    inr_v = max(
        float(inr),
        1.0
    )

    cr = max(
        float(creatinine),
        1.0
    )

    if dialysis_flag:
        cr = 3.0

    cr = min(
        cr,
        3.0
    )

    na = min(
        max(
            float(sodium),
            125.0
        ),
        137.0
    )

    alb = min(
        max(
            float(albumin),
            1.5
        ),
        3.5
    )

    raw = (
        1.33 * int(female)
        +
        4.56 * math.log(bili)
        +
        0.82 * (137-na)
        -
        0.24
        * (137-na)
        * math.log(bili)
        +
        9.09 * math.log(inr_v)
        +
        11.14 * math.log(cr)
        +
        1.85 * (3.5-alb)
        -
        1.83
        * (3.5-alb)
        * math.log(cr)
        +
        6.0
    )

    score = math.floor(
        raw + 0.5
    )

    score = max(
        6,
        min(
            40,
            score
        )
    )

    return float(score)


STRATEGIES = {
    "observed_primary":
        "dialysis_observed_primary",

    "observed_strict":
        "dialysis_observed_strict",

    "pragmatic":
        "dialysis_pragmatic",
}


for strategy, flag_base in STRATEGIES.items():

    for snapshot in [
        "baseline",
        "h48",
    ]:

        scores = []

        for _, row in x.iterrows():

            scores.append(
                meld3(
                    row[
                        f"bilirubin_{snapshot}"
                    ],
                    row[
                        f"sodium_{snapshot}"
                    ],
                    row[
                        f"inr_{snapshot}"
                    ],
                    row[
                        f"creatinine_{snapshot}"
                    ],
                    row[
                        f"albumin_{snapshot}"
                    ],
                    row["female"],
                    row[
                        f"{flag_base}_{snapshot}"
                    ],
                )
            )

        x[
            f"meld3_{strategy}_{snapshot}"
        ] = scores


for strategy in STRATEGIES:

    x[
        f"delta_meld3_{strategy}"
    ] = (
        x[
            f"meld3_{strategy}_h48"
        ]
        -
        x[
            f"meld3_{strategy}_baseline"
        ]
    )


# ============================================================
# 6. Clean non-RRT subset
# ============================================================

x[
    "nonrrt_clean"
] = (
    (x["any_rrt_evidence_baseline"] == 0)
    &
    (x["any_rrt_evidence_h48"] == 0)
    &
    (x["apache_dialysis_positive_baseline"] == 0)
    &
    (x["apache_dialysis_positive_h48"] == 0)
).astype(int)


# ============================================================
# 7. Strategy QC
# ============================================================

qc_rows = []


def add(metric, value):

    qc_rows.append({
        "metric":
            metric,

        "value":
            value
    })


add(
    "n_external_candidate",
    len(x)
)

add(
    "n_unique_patients",
    x["uniquepid"].nunique()
)

add(
    "n_hospitals",
    x["hospitalid"].nunique()
)

add(
    "hospital_deaths",
    int(
        x["hospital_death"].sum()
    )
)


for snapshot in [
    "baseline",
    "h48",
]:

    for strategy in [
        "observed_primary",
        "observed_strict",
        "pragmatic",
    ]:

        add(
            f"{snapshot}_{strategy}_dialysis_positive",
            int(
                x[
                    f"dialysis_{strategy}_{snapshot}"
                ].sum()
            )
        )


for snapshot in [
    "baseline",
    "h48",
]:

    add(
        f"{snapshot}_score_diff_pragmatic_vs_observed_n",
        int(
            (
                x[
                    f"meld3_pragmatic_{snapshot}"
                ]
                !=
                x[
                    f"meld3_observed_primary_{snapshot}"
                ]
            ).sum()
        )
    )

    add(
        f"{snapshot}_score_diff_primary_vs_strict_n",
        int(
            (
                x[
                    f"meld3_observed_primary_{snapshot}"
                ]
                !=
                x[
                    f"meld3_observed_strict_{snapshot}"
                ]
            ).sum()
        )
    )

    add(
        f"{snapshot}_max_abs_score_diff_pragmatic_vs_observed",
        float(
            (
                x[
                    f"meld3_pragmatic_{snapshot}"
                ]
                -
                x[
                    f"meld3_observed_primary_{snapshot}"
                ]
            )
            .abs()
            .max()
        )
    )


add(
    "nonrrt_clean_n",
    int(
        x[
            "nonrrt_clean"
        ].sum()
    )
)

add(
    "nonrrt_clean_deaths",
    int(
        x.loc[
            x["nonrrt_clean"] == 1,
            "hospital_death"
        ].sum()
    )
)


for strategy in [
    "observed_primary",
    "observed_strict",
    "pragmatic",
]:

    add(
        f"{strategy}_baseline_median",
        float(
            x[
                f"meld3_{strategy}_baseline"
            ].median()
        )
    )

    add(
        f"{strategy}_h48_median",
        float(
            x[
                f"meld3_{strategy}_h48"
            ].median()
        )
    )

    add(
        f"{strategy}_delta_median",
        float(
            x[
                f"delta_meld3_{strategy}"
            ].median()
        )
    )


qc = pd.DataFrame(
    qc_rows
)

qc.to_csv(
    OUT_QC,
    index=False
)


x.to_parquet(
    OUT_COHORT,
    index=False,
    compression="zstd"
)


# ============================================================
# 8. Console output
# ============================================================

pd.set_option(
    "display.max_rows",
    200
)

pd.set_option(
    "display.width",
    220
)


print()
print("===== SNAPSHOT RRT STRATEGY QC =====")
print(
    qc.to_string(
        index=False
    )
)


print()
print("===== MELD DISTRIBUTION =====")

rows = []

for strategy in [
    "observed_primary",
    "observed_strict",
    "pragmatic",
]:

    rows.append({

        "strategy":
            strategy,

        "baseline_median":
            x[
                f"meld3_{strategy}_baseline"
            ].median(),

        "baseline_q25":
            x[
                f"meld3_{strategy}_baseline"
            ].quantile(0.25),

        "baseline_q75":
            x[
                f"meld3_{strategy}_baseline"
            ].quantile(0.75),

        "h48_median":
            x[
                f"meld3_{strategy}_h48"
            ].median(),

        "h48_q25":
            x[
                f"meld3_{strategy}_h48"
            ].quantile(0.25),

        "h48_q75":
            x[
                f"meld3_{strategy}_h48"
            ].quantile(0.75),

        "delta_median":
            x[
                f"delta_meld3_{strategy}"
            ].median(),
    })


print(
    pd.DataFrame(
        rows
    ).to_string(
        index=False
    )
)


print()
print("=" * 84)
print("eICU CHECKPOINT E2A COMPLETE")
print("=" * 84)

print(f"External cohort : {OUT_COHORT}")
print(f"Lab snapshots   : {OUT_LABS}")
print(f"RRT flags       : {OUT_RRT}")
print(f"QC              : {OUT_QC}")


con.close()
