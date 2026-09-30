from pathlib import Path
import duckdb

ROOT = Path.home() / "Desktop" / "MIMIC_IV_3_1"
DB = ROOT / "work" / "mimic_feasibility.duckdb"

OUT1 = ROOT / "work" / "rrt_item_audit.csv"
OUT2 = ROOT / "work" / "rrt_cirrhosis_icu_summary.csv"

con = duckdb.connect(str(DB), read_only=True)

# Official MIMIC procedureevents RRT-related itemids
all_rrt_related = [
    225441,  # Hemodialysis
    225802,  # Dialysis - CRRT
    225803,  # Dialysis - CVVHD
    225805,  # Peritoneal Dialysis
    225809,  # Dialysis - CVVHDF
    225955,  # Dialysis - SCUF
    225436,  # CRRT Filter Change
    224270,  # Dialysis Catheter
]

# Actual dialysis treatment itemids
treatment_ids = [
    225441,
    225802,
    225803,
    225805,
    225809,
    225955,
]

ids_all = ",".join(str(x) for x in all_rrt_related)
ids_tx = ",".join(str(x) for x in treatment_ids)

# ----------------------------------------------------------
# 1. Audit item labels and overall usage
# ----------------------------------------------------------
q1 = f"""
SELECT
    d.itemid,
    d.label,
    COUNT(p.itemid) AS n_rows,
    COUNT(DISTINCT p.subject_id) AS n_patients,
    COUNT(DISTINCT p.hadm_id) AS n_admissions,
    MIN(p.starttime) AS first_time,
    MAX(p.endtime) AS last_time
FROM d_items d
LEFT JOIN procedureevents p
    ON d.itemid = p.itemid
WHERE d.itemid IN ({ids_all})
GROUP BY d.itemid, d.label
ORDER BY d.itemid;
"""

audit = con.execute(q1).df()
audit.to_csv(OUT1, index=False)

# ----------------------------------------------------------
# 2. Usage in first cirrhosis ICU cohort
# ----------------------------------------------------------
q2 = f"""
WITH rrt AS (
    SELECT
        c.subject_id,
        c.hadm_id,
        c.first_icu_intime,
        p.itemid,
        p.starttime,
        p.endtime,

        DATE_DIFF(
            'hour',
            c.first_icu_intime,
            p.starttime
        ) AS start_hour_from_icu

    FROM cirrhosis_first_icu_admission c

    INNER JOIN procedureevents p
        ON c.hadm_id = p.hadm_id

    WHERE p.itemid IN ({ids_tx})
),

summary AS (
    SELECT
        COUNT(DISTINCT subject_id) AS any_rrt_patients,

        COUNT(DISTINCT CASE
            WHEN itemid = 225441
            THEN subject_id END
        ) AS ihd_patients,

        COUNT(DISTINCT CASE
            WHEN itemid IN (225802,225803,225809,225955)
            THEN subject_id END
        ) AS continuous_rrt_patients,

        COUNT(DISTINCT CASE
            WHEN start_hour_from_icu BETWEEN 0 AND 72
            THEN subject_id END
        ) AS rrt_first72h_patients,

        COUNT(*) AS rrt_event_rows

    FROM rrt
)

SELECT
    4129 AS cohort_n,
    *,
    ROUND(
        100.0 * any_rrt_patients / 4129,
        2
    ) AS any_rrt_pct
FROM summary;
"""

summary = con.execute(q2).df()
summary.to_csv(OUT2, index=False)

print("\n" + "=" * 100)
print("RRT ITEM AUDIT")
print("=" * 100)
print(audit.to_string(index=False))

print("\n" + "=" * 100)
print("RRT IN FIRST CIRRHOSIS ICU COHORT")
print("=" * 100)
print(summary.to_string(index=False))

print("\nSaved:")
print(OUT1)
print(OUT2)

con.close()
