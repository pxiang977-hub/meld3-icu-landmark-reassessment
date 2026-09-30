from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WORK = ROOT / "work"

FILES = {
    "patient": DATA / "patient.csv.gz",
    "diagnosis": DATA / "diagnosis.csv.gz",
    "admissiondx": DATA / "admissionDx.csv.gz",
    "lab": DATA / "lab.csv.gz",
    "treatment": DATA / "treatment.csv.gz",
    "apache": DATA / "apacheApsVar.csv.gz",
}

for name, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")


OUT_SUMMARY = WORK / "E1A_summary.csv"
OUT_STAYS = WORK / "E1A_cirrhosis_candidate_stays.csv"
OUT_DX = WORK / "E1A_cirrhosis_diagnosis_candidates.csv"
OUT_ADMDX = WORK / "E1A_cirrhosis_admissiondx_candidates.csv"
OUT_LABS = WORK / "E1A_meld_labname_candidates.csv"
OUT_RRT = WORK / "E1A_rrt_treatment_candidates.csv"
OUT_APACHE = WORK / "E1A_apache_dialysis_values.csv"


con = duckdb.connect()

# ============================================================
# Views
# ============================================================

con.execute(f"""
CREATE VIEW patient AS
SELECT *
FROM read_csv_auto(
    '{FILES["patient"]}',
    header=true
)
""")

con.execute(f"""
CREATE VIEW diagnosis AS
SELECT *
FROM read_csv_auto(
    '{FILES["diagnosis"]}',
    header=true
)
""")

con.execute(f"""
CREATE VIEW admissiondx AS
SELECT *
FROM read_csv_auto(
    '{FILES["admissiondx"]}',
    header=true
)
""")

con.execute(f"""
CREATE VIEW lab AS
SELECT *
FROM read_csv_auto(
    '{FILES["lab"]}',
    header=true
)
""")

con.execute(f"""
CREATE VIEW treatment AS
SELECT *
FROM read_csv_auto(
    '{FILES["treatment"]}',
    header=true
)
""")

con.execute(f"""
CREATE VIEW apacheaps AS
SELECT *
FROM read_csv_auto(
    '{FILES["apache"]}',
    header=true
)
""")


print("=" * 80)
print("eICU CHECKPOINT E1A - SCHEMA + CANDIDATE AUDIT")
print("=" * 80)


# ============================================================
# 1. Schema
# ============================================================

for table in [
    "patient",
    "diagnosis",
    "admissiondx",
    "lab",
    "treatment",
    "apacheaps",
]:
    print()
    print(f"===== SCHEMA: {table} =====")
    schema = con.execute(
        f"DESCRIBE {table}"
    ).fetchdf()

    print(
        schema.to_string(index=False)
    )


# ============================================================
# 2. Overall patient / ICU structure
# ============================================================

patient_summary = con.execute("""
SELECT
    COUNT(*) AS n_unit_stays,
    COUNT(DISTINCT patientunitstayid) AS n_unique_unit_stays,
    COUNT(DISTINCT uniquepid) AS n_unique_patients,
    COUNT(DISTINCT patienthealthsystemstayid) AS n_healthsystem_stays,
    COUNT(DISTINCT hospitalid) AS n_hospitals,

    SUM(
        CASE
            WHEN lower(CAST(hospitaldischargestatus AS VARCHAR))
                 = 'expired'
            THEN 1 ELSE 0
        END
    ) AS n_hospital_deaths,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN lower(CAST(hospitaldischargestatus AS VARCHAR))
                     = 'expired'
                THEN 1 ELSE 0
            END
        )
        / COUNT(*),
        2
    ) AS hospital_mortality_pct

FROM patient
""").fetchdf()


print()
print("===== OVERALL eICU STRUCTURE =====")
print(
    patient_summary.to_string(index=False)
)


# ============================================================
# 3. Cirrhosis candidates in diagnosis
#
# Strict conceptual target:
# explicit cirrhosis only.
#
# Text:
#   cirrhos*
#
# Codes:
# ICD-9:
#   571.2 / 571.5 / 571.6
#
# ICD-10:
#   K70.30 / K70.31
#   K74.3 / K74.4 / K74.5
#   K74.60 / K74.69
#   K71.7
#
# The eICU column is named icd9code, but we audit actual
# contents rather than assuming it contains ICD-9 only.
# ============================================================

dx_candidates = con.execute("""
WITH x AS (
    SELECT
        patientunitstayid,
        diagnosisoffset,
        diagnosisstring,
        icd9code,

        upper(
            regexp_replace(
                coalesce(CAST(icd9code AS VARCHAR), ''),
                '[^A-Za-z0-9]',
                '',
                'g'
            )
        ) AS code_clean

    FROM diagnosis
)

SELECT
    diagnosisstring,
    icd9code,
    COUNT(*) AS n_rows,
    COUNT(DISTINCT patientunitstayid) AS n_unit_stays

FROM x

WHERE
       lower(coalesce(diagnosisstring, '')) LIKE '%cirrhos%'

    OR code_clean LIKE '%5712%'
    OR code_clean LIKE '%5715%'
    OR code_clean LIKE '%5716%'

    OR code_clean LIKE '%K7030%'
    OR code_clean LIKE '%K7031%'
    OR code_clean LIKE '%K743%'
    OR code_clean LIKE '%K744%'
    OR code_clean LIKE '%K745%'
    OR code_clean LIKE '%K7460%'
    OR code_clean LIKE '%K7469%'
    OR code_clean LIKE '%K717%'

GROUP BY
    diagnosisstring,
    icd9code

ORDER BY
    n_unit_stays DESC,
    n_rows DESC
""").fetchdf()

dx_candidates.to_csv(
    OUT_DX,
    index=False
)


print()
print("===== CIRRHOSIS DIAGNOSIS CANDIDATES =====")
print(
    dx_candidates.head(100).to_string(index=False)
)


# ============================================================
# 4. Cirrhosis candidates in admissionDx
# ============================================================

admdx_candidates = con.execute("""
SELECT
    admitdxpath,
    admitdxname,
    COUNT(*) AS n_rows,
    COUNT(DISTINCT patientunitstayid) AS n_unit_stays

FROM admissiondx

WHERE
       lower(coalesce(admitdxpath, '')) LIKE '%cirrhos%'
    OR lower(coalesce(admitdxname, '')) LIKE '%cirrhos%'

GROUP BY
    admitdxpath,
    admitdxname

ORDER BY
    n_unit_stays DESC,
    n_rows DESC
""").fetchdf()

admdx_candidates.to_csv(
    OUT_ADMDX,
    index=False
)


print()
print("===== CIRRHOSIS ADMISSION-DX CANDIDATES =====")
print(
    admdx_candidates.head(100).to_string(index=False)
)


# ============================================================
# 5. Build broad cirrhosis candidate stay set
#
# This is an AUDIT candidate set only.
# It is NOT yet the final external-validation cohort.
# ============================================================

candidate_stays = con.execute("""
WITH dx AS (
    SELECT DISTINCT
        patientunitstayid

    FROM (
        SELECT
            patientunitstayid,
            diagnosisstring,

            upper(
                regexp_replace(
                    coalesce(CAST(icd9code AS VARCHAR), ''),
                    '[^A-Za-z0-9]',
                    '',
                    'g'
                )
            ) AS code_clean

        FROM diagnosis
    )

    WHERE
           lower(coalesce(diagnosisstring, '')) LIKE '%cirrhos%'

        OR code_clean LIKE '%5712%'
        OR code_clean LIKE '%5715%'
        OR code_clean LIKE '%5716%'

        OR code_clean LIKE '%K7030%'
        OR code_clean LIKE '%K7031%'
        OR code_clean LIKE '%K743%'
        OR code_clean LIKE '%K744%'
        OR code_clean LIKE '%K745%'
        OR code_clean LIKE '%K7460%'
        OR code_clean LIKE '%K7469%'
        OR code_clean LIKE '%K717%'
),

adm AS (
    SELECT DISTINCT
        patientunitstayid

    FROM admissiondx

    WHERE
           lower(coalesce(admitdxpath, '')) LIKE '%cirrhos%'
        OR lower(coalesce(admitdxname, '')) LIKE '%cirrhos%'
)

SELECT DISTINCT
    p.patientunitstayid,
    p.patienthealthsystemstayid,
    p.uniquepid,
    p.hospitalid,
    p.gender,
    p.age,
    p.hospitaldischargestatus,
    p.hospitaldischargeoffset,
    p.unitdischargeoffset,

    CASE
        WHEN dx.patientunitstayid IS NOT NULL
        THEN 1 ELSE 0
    END AS found_in_diagnosis,

    CASE
        WHEN adm.patientunitstayid IS NOT NULL
        THEN 1 ELSE 0
    END AS found_in_admissiondx

FROM patient p

LEFT JOIN dx
    ON p.patientunitstayid = dx.patientunitstayid

LEFT JOIN adm
    ON p.patientunitstayid = adm.patientunitstayid

WHERE
       dx.patientunitstayid IS NOT NULL
    OR adm.patientunitstayid IS NOT NULL

ORDER BY
    p.patientunitstayid
""").fetchdf()

candidate_stays.to_csv(
    OUT_STAYS,
    index=False
)


print()
print("===== BROAD CIRRHOSIS CANDIDATE STAY SET =====")
print(
    pd.DataFrame([{
        "n_candidate_unit_stays":
            len(candidate_stays),

        "n_candidate_unique_patients":
            candidate_stays["uniquepid"].nunique(),

        "n_candidate_hospitals":
            candidate_stays["hospitalid"].nunique(),

        "diagnosis_positive":
            int(
                candidate_stays[
                    "found_in_diagnosis"
                ].sum()
            ),

        "admissiondx_positive":
            int(
                candidate_stays[
                    "found_in_admissiondx"
                ].sum()
            ),

        "hospital_deaths":
            int(
                candidate_stays[
                    "hospitaldischargestatus"
                ]
                .astype(str)
                .str.lower()
                .eq("expired")
                .sum()
            ),
    }]).to_string(index=False)
)


# ============================================================
# 6. MELD 3.0 lab-name candidates
#
# Do not assume exact names yet.
# First inspect the actual eICU vocabulary.
# ============================================================

lab_candidates = con.execute("""
SELECT
    labname,
    COUNT(*) AS n_rows,
    COUNT(DISTINCT patientunitstayid) AS n_unit_stays,

    MIN(labresult) AS min_result,
    quantile_cont(labresult, 0.50) AS median_result,
    MAX(labresult) AS max_result

FROM lab

WHERE
       lower(coalesce(labname, '')) LIKE '%albumin%'
    OR lower(coalesce(labname, '')) LIKE '%bilirubin%'
    OR lower(coalesce(labname, '')) LIKE '%creatin%'
    OR lower(coalesce(labname, '')) LIKE '%sodium%'
    OR lower(coalesce(labname, '')) = 'inr'
    OR lower(coalesce(labname, '')) LIKE '%inr%'
    OR lower(coalesce(labname, '')) LIKE '%prothrombin%'

GROUP BY
    labname

ORDER BY
    labname
""").fetchdf()

lab_candidates.to_csv(
    OUT_LABS,
    index=False
)


print()
print("===== MELD LAB-NAME CANDIDATES =====")
print(
    lab_candidates.to_string(index=False)
)


# ============================================================
# 7. Same lab-name audit restricted to broad cirrhosis stays
# ============================================================

con.register(
    "candidate_stays_df",
    candidate_stays[
        ["patientunitstayid"]
    ]
)

candidate_lab_coverage = con.execute("""
SELECT
    l.labname,
    COUNT(*) AS n_rows,
    COUNT(DISTINCT l.patientunitstayid) AS n_candidate_stays

FROM lab l

INNER JOIN candidate_stays_df c
    ON l.patientunitstayid = c.patientunitstayid

WHERE
       lower(coalesce(l.labname, '')) LIKE '%albumin%'
    OR lower(coalesce(l.labname, '')) LIKE '%bilirubin%'
    OR lower(coalesce(l.labname, '')) LIKE '%creatin%'
    OR lower(coalesce(l.labname, '')) LIKE '%sodium%'
    OR lower(coalesce(l.labname, '')) = 'inr'
    OR lower(coalesce(l.labname, '')) LIKE '%inr%'
    OR lower(coalesce(l.labname, '')) LIKE '%prothrombin%'

GROUP BY
    l.labname

ORDER BY
    l.labname
""").fetchdf()


print()
print("===== MELD LAB COVERAGE IN CIRRHOSIS CANDIDATE STAYS =====")
print(
    candidate_lab_coverage.to_string(index=False)
)


# ============================================================
# 8. RRT / dialysis treatment vocabulary
# ============================================================

rrt_candidates = con.execute("""
SELECT
    treatmentstring,
    COUNT(*) AS n_rows,
    COUNT(DISTINCT patientunitstayid) AS n_unit_stays

FROM treatment

WHERE
       lower(coalesce(treatmentstring, '')) LIKE '%dialys%'
    OR lower(coalesce(treatmentstring, '')) LIKE '%crrt%'
    OR lower(coalesce(treatmentstring, '')) LIKE '%cvvh%'
    OR lower(coalesce(treatmentstring, '')) LIKE '%hemofil%'
    OR lower(coalesce(treatmentstring, '')) LIKE '%renal replacement%'
    OR lower(coalesce(treatmentstring, '')) LIKE '%hemodial%'
    OR lower(coalesce(treatmentstring, '')) LIKE '%peritoneal%'

GROUP BY
    treatmentstring

ORDER BY
    n_unit_stays DESC,
    n_rows DESC
""").fetchdf()

rrt_candidates.to_csv(
    OUT_RRT,
    index=False
)


print()
print("===== RRT / DIALYSIS TREATMENT CANDIDATES =====")
print(
    rrt_candidates.head(150).to_string(index=False)
)


# ============================================================
# 9. apacheApsVar dialysis values
# ============================================================

apache_dialysis = con.execute("""
SELECT
    dialysis,
    COUNT(*) AS n_rows,
    COUNT(DISTINCT patientunitstayid) AS n_unit_stays

FROM apacheaps

GROUP BY dialysis

ORDER BY dialysis
""").fetchdf()

apache_dialysis.to_csv(
    OUT_APACHE,
    index=False
)


print()
print("===== APACHE APS DIALYSIS VALUES =====")
print(
    apache_dialysis.to_string(index=False)
)


# ============================================================
# 10. E1A summary
# ============================================================

summary = pd.DataFrame([
    {
        "metric":
            "overall_unit_stays",

        "value":
            int(
                patient_summary[
                    "n_unique_unit_stays"
                ].iloc[0]
            ),
    },

    {
        "metric":
            "overall_unique_patients",

        "value":
            int(
                patient_summary[
                    "n_unique_patients"
                ].iloc[0]
            ),
    },

    {
        "metric":
            "overall_hospitals",

        "value":
            int(
                patient_summary[
                    "n_hospitals"
                ].iloc[0]
            ),
    },

    {
        "metric":
            "broad_cirrhosis_candidate_stays",

        "value":
            len(candidate_stays),
    },

    {
        "metric":
            "broad_cirrhosis_unique_patients",

        "value":
            candidate_stays[
                "uniquepid"
            ].nunique(),
    },

    {
        "metric":
            "broad_cirrhosis_hospitals",

        "value":
            candidate_stays[
                "hospitalid"
            ].nunique(),
    },

    {
        "metric":
            "broad_cirrhosis_hospital_deaths",

        "value":
            int(
                candidate_stays[
                    "hospitaldischargestatus"
                ]
                .astype(str)
                .str.lower()
                .eq("expired")
                .sum()
            ),
    },

    {
        "metric":
            "meld_lab_candidate_names",

        "value":
            len(lab_candidates),
    },

    {
        "metric":
            "rrt_treatment_candidate_strings",

        "value":
            len(rrt_candidates),
    },
])

summary.to_csv(
    OUT_SUMMARY,
    index=False
)


print()
print("===== E1A SUMMARY =====")
print(
    summary.to_string(index=False)
)


print()
print("=" * 80)
print("eICU CHECKPOINT E1A COMPLETE")
print("=" * 80)

print(f"Summary              : {OUT_SUMMARY}")
print(f"Cirrhosis stays      : {OUT_STAYS}")
print(f"Diagnosis candidates : {OUT_DX}")
print(f"AdmissionDx candidates: {OUT_ADMDX}")
print(f"Lab candidates       : {OUT_LABS}")
print(f"RRT candidates       : {OUT_RRT}")
print(f"Apache dialysis      : {OUT_APACHE}")

con.close()
