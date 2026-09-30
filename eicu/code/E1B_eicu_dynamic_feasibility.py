from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


# ============================================================
# eICU Checkpoint E1B
#
# Strict explicit-cirrhosis cohort
# + dynamic five-lab feasibility
#
# Baseline window:
#   ICU -6 h to +12 h
#   = -360 to +720 minutes
#
# 48-h reassessment:
#   ICU +24 h to +48 h
#   = +1440 to +2880 minutes
#
# IMPORTANT:
# eICU laboratory data are ICU encounter-centric.
# Therefore both hospital-at-48h and ICU-at-48h risk sets
# are reported.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WORK = ROOT / "work"

PATIENT = DATA / "patient.csv.gz"
DIAGNOSIS = DATA / "diagnosis.csv.gz"
LAB = DATA / "lab.csv.gz"

for p in [PATIENT, DIAGNOSIS, LAB]:
    if not p.exists():
        raise FileNotFoundError(p)


OUT_COHORT = WORK / "E1B_strict_adult_cirrhosis_stays.csv"
OUT_FLAGS = WORK / "E1B_lab_window_flags.parquet"
OUT_SUMMARY = WORK / "E1B_dynamic_feasibility_summary.csv"
OUT_COMPONENT = WORK / "E1B_component_coverage.csv"
OUT_PAIRED = WORK / "E1B_paired_candidate_stays.csv"


con = duckdb.connect()


# ============================================================
# 1. Source views
# ============================================================

con.execute(f"""
CREATE VIEW patient AS
SELECT *
FROM read_csv_auto(
    '{PATIENT}',
    header=true
)
""")

con.execute(f"""
CREATE VIEW diagnosis AS
SELECT *
FROM read_csv_auto(
    '{DIAGNOSIS}',
    header=true
)
""")

con.execute(f"""
CREATE VIEW lab AS
SELECT *
FROM read_csv_auto(
    '{LAB}',
    header=true
)
""")


print("=" * 82)
print("eICU CHECKPOINT E1B - STRICT CIRRHOSIS + DYNAMIC FEASIBILITY")
print("=" * 82)


# ============================================================
# 2. Strict explicit cirrhosis definition
#
# Use explicit cirrhosis diagnosis codes.
#
# ICD-9:
#   571.2
#   571.5
#   571.6
#
# ICD-10:
#   K70.3   eICU commonly stores this truncated family
#   K71.7
#   K74.3
#   K74.4
#   K74.5
#   K74.60
#   K74.69
#
# We deliberately do NOT include complication-only strings
# such as "hyponatremia due to cirrhosis".
# ============================================================

strict = con.execute("""
WITH dx AS (

    SELECT DISTINCT
        patientunitstayid

    FROM (

        SELECT
            patientunitstayid,

            upper(
                regexp_replace(
                    coalesce(
                        CAST(icd9code AS VARCHAR),
                        ''
                    ),
                    '[^A-Za-z0-9]',
                    '',
                    'g'
                )
            ) AS code_clean

        FROM diagnosis
    )

    WHERE
           code_clean LIKE '%5712%'
        OR code_clean LIKE '%5715%'
        OR code_clean LIKE '%5716%'

        OR code_clean LIKE '%K703%'
        OR code_clean LIKE '%K717%'
        OR code_clean LIKE '%K743%'
        OR code_clean LIKE '%K744%'
        OR code_clean LIKE '%K745%'
        OR code_clean LIKE '%K7460%'
        OR code_clean LIKE '%K7469%'
),

p AS (

    SELECT
        *,

        CASE
            WHEN trim(CAST(age AS VARCHAR))
                 LIKE '> %'
            THEN 90.0

            WHEN trim(CAST(age AS VARCHAR))
                 LIKE '>%'
            THEN 90.0

            ELSE TRY_CAST(
                age AS DOUBLE
            )
        END AS age_num

    FROM patient
)

SELECT
    p.patientunitstayid,
    p.patienthealthsystemstayid,
    p.uniquepid,
    p.hospitalid,

    p.gender,
    p.age,
    p.age_num,

    p.unitvisitnumber,
    p.unitstaytype,

    p.hospitaldischargeoffset,
    p.hospitaldischargestatus,

    p.unitdischargeoffset,
    p.unitdischargestatus,

    p.hospitaldischargeyear

FROM p

INNER JOIN dx
    ON p.patientunitstayid
     = dx.patientunitstayid

WHERE
    p.age_num >= 18

ORDER BY
    p.patientunitstayid
""").fetchdf()


strict.to_csv(
    OUT_COHORT,
    index=False
)


print()
print("===== STRICT ADULT CIRRHOSIS COHORT =====")

print(
    pd.DataFrame([{

        "n_unit_stays":
            len(strict),

        "n_unique_patients":
            strict[
                "uniquepid"
            ].nunique(),

        "n_healthsystem_stays":
            strict[
                "patienthealthsystemstayid"
            ].nunique(),

        "n_hospitals":
            strict[
                "hospitalid"
            ].nunique(),

        "hospital_deaths":
            int(
                strict[
                    "hospitaldischargestatus"
                ]
                .astype(str)
                .str.lower()
                .eq("expired")
                .sum()
            ),

        "mortality_pct":
            round(
                100
                * strict[
                    "hospitaldischargestatus"
                ]
                .astype(str)
                .str.lower()
                .eq("expired")
                .mean(),
                2
            )
    }]).to_string(index=False)
)


# ============================================================
# 3. Register strict cohort
# ============================================================

con.register(
    "strict_df",
    strict[
        [
            "patientunitstayid"
        ]
    ]
)


# ============================================================
# 4. Exact MELD lab names and window availability
#
# Only positive numeric values are counted as valid
# measurements for feasibility.
# ============================================================

flags = con.execute("""
WITH meldlabs AS (

    SELECT
        l.patientunitstayid,
        l.labid,
        l.labresultoffset,
        l.labresult,

        CASE

            WHEN lower(trim(l.labname))
                 = 'albumin'
            THEN 'albumin'

            WHEN lower(trim(l.labname))
                 = 'total bilirubin'
            THEN 'bilirubin'

            WHEN lower(trim(l.labname))
                 = 'creatinine'
            THEN 'creatinine'

            WHEN lower(trim(l.labname))
                 = 'sodium'
            THEN 'sodium'

            WHEN lower(trim(l.labname))
                 = 'pt - inr'
            THEN 'inr'

            ELSE NULL

        END AS component

    FROM lab l

    INNER JOIN strict_df s
        ON l.patientunitstayid
         = s.patientunitstayid

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

agg AS (

    SELECT
        patientunitstayid,

        MAX(
            CASE
                WHEN component = 'albumin'
                 AND labresultoffset
                     BETWEEN -360 AND 720
                THEN 1 ELSE 0
            END
        ) AS baseline_albumin,

        MAX(
            CASE
                WHEN component = 'bilirubin'
                 AND labresultoffset
                     BETWEEN -360 AND 720
                THEN 1 ELSE 0
            END
        ) AS baseline_bilirubin,

        MAX(
            CASE
                WHEN component = 'creatinine'
                 AND labresultoffset
                     BETWEEN -360 AND 720
                THEN 1 ELSE 0
            END
        ) AS baseline_creatinine,

        MAX(
            CASE
                WHEN component = 'sodium'
                 AND labresultoffset
                     BETWEEN -360 AND 720
                THEN 1 ELSE 0
            END
        ) AS baseline_sodium,

        MAX(
            CASE
                WHEN component = 'inr'
                 AND labresultoffset
                     BETWEEN -360 AND 720
                THEN 1 ELSE 0
            END
        ) AS baseline_inr,


        MAX(
            CASE
                WHEN component = 'albumin'
                 AND labresultoffset
                     BETWEEN 1440 AND 2880
                THEN 1 ELSE 0
            END
        ) AS h48_albumin,

        MAX(
            CASE
                WHEN component = 'bilirubin'
                 AND labresultoffset
                     BETWEEN 1440 AND 2880
                THEN 1 ELSE 0
            END
        ) AS h48_bilirubin,

        MAX(
            CASE
                WHEN component = 'creatinine'
                 AND labresultoffset
                     BETWEEN 1440 AND 2880
                THEN 1 ELSE 0
            END
        ) AS h48_creatinine,

        MAX(
            CASE
                WHEN component = 'sodium'
                 AND labresultoffset
                     BETWEEN 1440 AND 2880
                THEN 1 ELSE 0
            END
        ) AS h48_sodium,

        MAX(
            CASE
                WHEN component = 'inr'
                 AND labresultoffset
                     BETWEEN 1440 AND 2880
                THEN 1 ELSE 0
            END
        ) AS h48_inr

    FROM meldlabs

    GROUP BY
        patientunitstayid
)

SELECT
    s.patientunitstayid,

    coalesce(a.baseline_albumin,0)
        AS baseline_albumin,

    coalesce(a.baseline_bilirubin,0)
        AS baseline_bilirubin,

    coalesce(a.baseline_creatinine,0)
        AS baseline_creatinine,

    coalesce(a.baseline_sodium,0)
        AS baseline_sodium,

    coalesce(a.baseline_inr,0)
        AS baseline_inr,


    coalesce(a.h48_albumin,0)
        AS h48_albumin,

    coalesce(a.h48_bilirubin,0)
        AS h48_bilirubin,

    coalesce(a.h48_creatinine,0)
        AS h48_creatinine,

    coalesce(a.h48_sodium,0)
        AS h48_sodium,

    coalesce(a.h48_inr,0)
        AS h48_inr

FROM strict_df s

LEFT JOIN agg a
    ON s.patientunitstayid
     = a.patientunitstayid
""").fetchdf()


flags[
    "baseline_complete"
] = (
    flags[
        [
            "baseline_albumin",
            "baseline_bilirubin",
            "baseline_creatinine",
            "baseline_sodium",
            "baseline_inr",
        ]
    ]
    .eq(1)
    .all(axis=1)
    .astype(int)
)


flags[
    "h48_complete"
] = (
    flags[
        [
            "h48_albumin",
            "h48_bilirubin",
            "h48_creatinine",
            "h48_sodium",
            "h48_inr",
        ]
    ]
    .eq(1)
    .all(axis=1)
    .astype(int)
)


flags[
    "paired_complete"
] = (
    (
        flags[
            "baseline_complete"
        ] == 1
    )
    &
    (
        flags[
            "h48_complete"
        ] == 1
    )
).astype(int)


flags.to_parquet(
    OUT_FLAGS,
    index=False,
    compression="zstd"
)


# ============================================================
# 5. Merge cohort + lab flags
# ============================================================

x = strict.merge(
    flags,
    on="patientunitstayid",
    how="left"
)


x[
    "hospital_death"
] = (
    x[
        "hospitaldischargestatus"
    ]
    .astype(str)
    .str.lower()
    .eq("expired")
    .astype(int)
)


# Risk sets

x[
    "hospital_12h_eligible"
] = (
    x[
        "hospitaldischargeoffset"
    ].notna()
    &
    (
        x[
            "hospitaldischargeoffset"
        ] > 720
    )
).astype(int)


x[
    "icu_12h_eligible"
] = (
    x[
        "unitdischargeoffset"
    ].notna()
    &
    (
        x[
            "unitdischargeoffset"
        ] > 720
    )
).astype(int)


x[
    "hospital_48h_eligible"
] = (
    x[
        "hospitaldischargeoffset"
    ].notna()
    &
    (
        x[
            "hospitaldischargeoffset"
        ] > 2880
    )
).astype(int)


x[
    "icu_48h_eligible"
] = (
    x[
        "unitdischargeoffset"
    ].notna()
    &
    (
        x[
            "unitdischargeoffset"
        ] > 2880
    )
).astype(int)


# ============================================================
# 6. Summary function
# ============================================================

def make_summary(
    label,
    mask
):

    g = x.loc[
        mask
    ].copy()

    paired = g[
        g[
            "paired_complete"
        ] == 1
    ]

    return {
        "risk_set":
            label,

        "n_stays":
            len(g),

        "n_unique_patients":
            g[
                "uniquepid"
            ].nunique(),

        "n_hospitals":
            g[
                "hospitalid"
            ].nunique(),

        "hospital_deaths":
            int(
                g[
                    "hospital_death"
                ].sum()
            ),

        "mortality_pct":
            round(
                100
                * g[
                    "hospital_death"
                ].mean(),
                2
            )
            if len(g)
            else np.nan,

        "baseline_complete_n":
            int(
                g[
                    "baseline_complete"
                ].sum()
            ),

        "baseline_complete_pct":
            round(
                100
                * g[
                    "baseline_complete"
                ].mean(),
                2
            )
            if len(g)
            else np.nan,

        "h48_complete_n":
            int(
                g[
                    "h48_complete"
                ].sum()
            ),

        "h48_complete_pct":
            round(
                100
                * g[
                    "h48_complete"
                ].mean(),
                2
            )
            if len(g)
            else np.nan,

        "paired_complete_n":
            len(paired),

        "paired_complete_pct":
            round(
                100
                * len(paired)
                / len(g),
                2
            )
            if len(g)
            else np.nan,

        "paired_unique_patients":
            paired[
                "uniquepid"
            ].nunique(),

        "paired_hospitals":
            paired[
                "hospitalid"
            ].nunique(),

        "paired_deaths":
            int(
                paired[
                    "hospital_death"
                ].sum()
            ),

        "paired_mortality_pct":
            round(
                100
                * paired[
                    "hospital_death"
                ].mean(),
                2
            )
            if len(paired)
            else np.nan,
    }


summary_rows = [

    make_summary(
        "strict_all_adult_stays",
        np.ones(
            len(x),
            dtype=bool
        )
    ),

    make_summary(
        "hospital_12h_risk_set",
        x[
            "hospital_12h_eligible"
        ] == 1
    ),

    make_summary(
        "icu_12h_risk_set",
        x[
            "icu_12h_eligible"
        ] == 1
    ),

    make_summary(
        "hospital_48h_risk_set",
        x[
            "hospital_48h_eligible"
        ] == 1
    ),

    make_summary(
        "icu_48h_risk_set",
        x[
            "icu_48h_eligible"
        ] == 1
    ),
]


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT_SUMMARY,
    index=False
)


# ============================================================
# 7. Component-level coverage
# ============================================================

component_rows = []

risk_sets = {
    "hospital_48h":
        x[
            "hospital_48h_eligible"
        ] == 1,

    "icu_48h":
        x[
            "icu_48h_eligible"
        ] == 1,
}


for risk_name, mask in risk_sets.items():

    g = x.loc[
        mask
    ]

    for component in [
        "albumin",
        "bilirubin",
        "creatinine",
        "sodium",
        "inr",
    ]:

        for window in [
            "baseline",
            "h48",
        ]:

            col = (
                f"{window}_{component}"
            )

            n = int(
                g[col].sum()
            )

            component_rows.append({

                "risk_set":
                    risk_name,

                "window":
                    window,

                "component":
                    component,

                "available_n":
                    n,

                "denominator":
                    len(g),

                "available_pct":
                    round(
                        100*n/len(g),
                        2
                    )
                    if len(g)
                    else np.nan,
            })


component = pd.DataFrame(
    component_rows
)

component.to_csv(
    OUT_COMPONENT,
    index=False
)


# ============================================================
# 8. Save paired candidate stays
#
# This is NOT yet the final external-validation cohort.
# It is only the feasibility candidate set.
# ============================================================

paired_candidates = x[
    (
        x[
            "hospital_48h_eligible"
        ] == 1
    )
    &
    (
        x[
            "paired_complete"
        ] == 1
    )
].copy()


paired_candidates.to_csv(
    OUT_PAIRED,
    index=False
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
    240
)


print()
print("===== DYNAMIC FEASIBILITY SUMMARY =====")

print(
    summary.to_string(
        index=False
    )
)


print()
print("===== COMPONENT COVERAGE: HOSPITAL 48H RISK SET =====")

print(
    component[
        component[
            "risk_set"
        ] == "hospital_48h"
    ]
    .to_string(
        index=False
    )
)


print()
print("===== COMPONENT COVERAGE: ICU 48H RISK SET =====")

print(
    component[
        component[
            "risk_set"
        ] == "icu_48h"
    ]
    .to_string(
        index=False
    )
)


print()
print("=" * 82)
print("eICU CHECKPOINT E1B COMPLETE")
print("=" * 82)

print(f"Strict cohort : {OUT_COHORT}")
print(f"Lab flags     : {OUT_FLAGS}")
print(f"Summary       : {OUT_SUMMARY}")
print(f"Components    : {OUT_COMPONENT}")
print(f"Paired stays  : {OUT_PAIRED}")


con.close()
