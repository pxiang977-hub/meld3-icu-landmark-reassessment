from pathlib import Path
import duckdb

ROOT = Path.home() / "Desktop" / "MIMIC_IV_3_1"
DB = ROOT / "work" / "mimic_feasibility.duckdb"
OUT = ROOT / "work" / "cirrhosis_cohort_summary.csv"
OUT_COHORT = ROOT / "work" / "cirrhosis_adult_admissions.csv"

con = duckdb.connect(str(DB))

# Explicit cirrhosis diagnosis codes in MIMIC-IV format (no decimal point)
icd9 = ["5712", "5715", "5716"]

icd10 = [
    "K7030",
    "K7031",
    "K743",
    "K744",
    "K745",
    "K7460",
    "K7469",
    "K717",
]

codes9 = ",".join([f"'{x}'" for x in icd9])
codes10 = ",".join([f"'{x}'" for x in icd10])

# ------------------------------------------------------------
# 1. One row per cirrhosis hospital admission
# ------------------------------------------------------------
con.execute(f"""
CREATE OR REPLACE TABLE cirrhosis_admissions_all AS
WITH cirrhosis_dx AS (
    SELECT DISTINCT
        subject_id,
        hadm_id
    FROM diagnoses_icd
    WHERE
        (icd_version = 9 AND icd_code IN ({codes9}))
        OR
        (icd_version = 10 AND icd_code IN ({codes10}))
),
icu_flag AS (
    SELECT
        hadm_id,
        COUNT(DISTINCT stay_id) AS n_icu_stays,
        MIN(intime) AS first_icu_intime,
        MAX(outtime) AS last_icu_outtime
    FROM icustays
    GROUP BY hadm_id
)
SELECT
    a.subject_id,
    a.hadm_id,
    a.admittime,
    a.dischtime,
    a.deathtime,
    a.admission_type,
    a.admission_location,
    a.discharge_location,
    a.insurance,
    a.marital_status,
    a.race,
    a.hospital_expire_flag,

    p.gender,
    p.anchor_age,
    p.anchor_year,
    p.dod,

    p.anchor_age
      + (EXTRACT(YEAR FROM a.admittime) - p.anchor_year)
      AS age_at_admission,

    CASE
        WHEN i.hadm_id IS NOT NULL THEN 1
        ELSE 0
    END AS icu_admission,

    COALESCE(i.n_icu_stays, 0) AS n_icu_stays,
    i.first_icu_intime,
    i.last_icu_outtime

FROM cirrhosis_dx c
INNER JOIN admissions a
    ON c.subject_id = a.subject_id
   AND c.hadm_id = a.hadm_id
INNER JOIN patients p
    ON c.subject_id = p.subject_id
LEFT JOIN icu_flag i
    ON c.hadm_id = i.hadm_id;
""")

# ------------------------------------------------------------
# 2. Adult cohort
# ------------------------------------------------------------
con.execute("""
CREATE OR REPLACE TABLE cirrhosis_adult_admissions AS
SELECT *
FROM cirrhosis_admissions_all
WHERE age_at_admission >= 18;
""")

# ------------------------------------------------------------
# 3. First eligible admission for each patient
# ------------------------------------------------------------
con.execute("""
CREATE OR REPLACE TABLE cirrhosis_first_admission AS
SELECT *
FROM cirrhosis_adult_admissions
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY subject_id
    ORDER BY admittime, hadm_id
) = 1;
""")

# ------------------------------------------------------------
# 4. First eligible ICU admission for each patient
# ------------------------------------------------------------
con.execute("""
CREATE OR REPLACE TABLE cirrhosis_first_icu_admission AS
SELECT *
FROM cirrhosis_adult_admissions
WHERE icu_admission = 1
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY subject_id
    ORDER BY first_icu_intime, admittime, hadm_id
) = 1;
""")

# ------------------------------------------------------------
# 5. Summary
# ------------------------------------------------------------
summary = con.execute("""
WITH stats AS (

    SELECT
        'All cirrhosis-coded admissions' AS cohort,
        COUNT(*) AS n_admissions,
        COUNT(DISTINCT subject_id) AS n_patients,
        SUM(hospital_expire_flag) AS hospital_deaths
    FROM cirrhosis_admissions_all

    UNION ALL

    SELECT
        'Adult cirrhosis admissions',
        COUNT(*),
        COUNT(DISTINCT subject_id),
        SUM(hospital_expire_flag)
    FROM cirrhosis_adult_admissions

    UNION ALL

    SELECT
        'Adult cirrhosis admissions with ICU stay',
        COUNT(*),
        COUNT(DISTINCT subject_id),
        SUM(hospital_expire_flag)
    FROM cirrhosis_adult_admissions
    WHERE icu_admission = 1

    UNION ALL

    SELECT
        'First adult cirrhosis admission per patient',
        COUNT(*),
        COUNT(DISTINCT subject_id),
        SUM(hospital_expire_flag)
    FROM cirrhosis_first_admission

    UNION ALL

    SELECT
        'First adult cirrhosis ICU admission per patient',
        COUNT(*),
        COUNT(DISTINCT subject_id),
        SUM(hospital_expire_flag)
    FROM cirrhosis_first_icu_admission
)
SELECT
    cohort,
    n_admissions,
    n_patients,
    hospital_deaths,
    ROUND(
        100.0 * hospital_deaths / NULLIF(n_admissions, 0),
        2
    ) AS hospital_mortality_pct
FROM stats;
""").df()

summary.to_csv(OUT, index=False)

adult = con.execute("""
SELECT *
FROM cirrhosis_adult_admissions
ORDER BY subject_id, admittime;
""").df()

adult.to_csv(OUT_COHORT, index=False)

print("\n" + "=" * 90)
print("MIMIC-IV v3.1 CIRRHOSIS COHORT — FEASIBILITY")
print("=" * 90)
print(summary.to_string(index=False))
print("=" * 90)

print("\nAge distribution among adult cirrhosis admissions:")
age = con.execute("""
SELECT
    COUNT(*) AS n,
    ROUND(MIN(age_at_admission),1) AS min_age,
    ROUND(quantile_cont(age_at_admission, 0.25),1) AS q1,
    ROUND(MEDIAN(age_at_admission),1) AS median,
    ROUND(quantile_cont(age_at_admission, 0.75),1) AS q3,
    ROUND(MAX(age_at_admission),1) AS max_age
FROM cirrhosis_adult_admissions;
""").df()
print(age.to_string(index=False))

print("\nSex distribution:")
sex = con.execute("""
SELECT
    gender,
    COUNT(*) AS n_admissions,
    ROUND(100.0 * COUNT(*) /
        SUM(COUNT(*)) OVER (), 2) AS pct
FROM cirrhosis_adult_admissions
GROUP BY gender
ORDER BY n_admissions DESC;
""").df()
print(sex.to_string(index=False))

print("\nOutputs:")
print(OUT)
print(OUT_COHORT)
print("=" * 90)

con.close()
