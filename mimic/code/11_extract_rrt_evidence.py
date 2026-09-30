from pathlib import Path
import os
import duckdb

# ============================================================
# Checkpoint 11A
# Extract RRT evidence relevant to formal MELD 3.0 calculation
# MIMIC-IV v3.1
# Main cohort: first adult cirrhosis ICU admission, n=4,129
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DB = ROOT / "work" / "mimic_feasibility.duckdb"

CHARTEVENTS = ROOT / "data" / "icu" / "chartevents.csv.gz"
INPUTEVENTS = ROOT / "data" / "icu" / "inputevents.csv.gz"
PROCEDUREEVENTS = ROOT / "data" / "icu" / "procedureevents.csv.gz"

OUT = ROOT / "work" / "rrt_active_evidence.parquet"
QC_ITEM = ROOT / "work" / "rrt_active_evidence_qc_by_item.csv"
QC_SOURCE = ROOT / "work" / "rrt_active_evidence_qc_by_source.csv"

for p in [DB, CHARTEVENTS, INPUTEVENTS, PROCEDUREEVENTS]:
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")

# Official MIMIC RRT concept chartevents itemids.
# Includes both active treatment evidence and supportive/context evidence.
CHAR_ITEMIDS = [
    226118, 227357, 225725,
    226499, 224154, 225810, 227639, 225183, 227438,
    224191, 225806, 225807, 228004, 228005, 228006,
    224144, 224145, 224149, 224150, 224151, 224152,
    224153, 224404, 224406, 226457, 225959,
    224135, 224139, 224146, 225323, 225740, 225776,
    225951, 225952, 225953, 225954, 225956, 225958,
    225961, 225963, 225965, 225976, 225977, 227124,
    227290, 227638, 227640, 227753
]

# Official rrt.sql chartevents classified as active dialysis evidence.
CHAR_ACTIVE_ITEMIDS = [
    226499,
    224154,
    225183,
    227438,
    224191,
    225806,
    225807,
    228004,
    228005,
    228006,
    224144,
    224145,
    224153,
    226457
]

# Explicit PD-related items.
PD_ITEMIDS = [
    225810, 225806, 225807, 227639, 225959,
    225951, 225952, 225961, 225953, 225963,
    225965, 227638, 227640
]

# Items used by the official CRRT concept.
CRRT_CHAR_ITEMIDS = [
    227290, 224146, 224149, 224144, 228004, 225183,
    225977, 224154, 224151, 224150, 225958, 224145,
    224191, 228005, 228006, 225976, 224153, 224152,
    226457
]

INPUT_ITEMIDS = [227536, 227525]

PROCEDURE_ITEMIDS = [
    225441,  # Hemodialysis
    225802,  # Dialysis - CRRT
    225803,  # Dialysis - CVVHD
    225805,  # Peritoneal Dialysis
    224270,  # Dialysis Catheter - supportive only
    225809,  # Dialysis - CVVHDF
    225955,  # Dialysis - SCUF
    225436   # CRRT Filter Change - supportive only
]

def ids(values):
    return ", ".join(str(x) for x in values)

char_all = ids(CHAR_ITEMIDS)
char_active = ids(CHAR_ACTIVE_ITEMIDS)
pd_ids = ids(PD_ITEMIDS)
crrt_char = ids(CRRT_CHAR_ITEMIDS)
input_ids = ids(INPUT_ITEMIDS)
procedure_ids = ids(PROCEDURE_ITEMIDS)

def sql_path(path):
    return str(path).replace("'", "''")

# Delete outputs from an incomplete previous run, if any.
for p in [OUT, QC_ITEM, QC_SOURCE]:
    if p.exists():
        p.unlink()

print("=" * 72)
print("CHECKPOINT 11A - RRT EVIDENCE EXTRACTION")
print("=" * 72)
print(f"Database       : {DB}")
print(f"chartevents    : {CHARTEVENTS}")
print(f"inputevents    : {INPUTEVENTS}")
print(f"procedureevents: {PROCEDUREEVENTS}")
print()

con = duckdb.connect(str(DB))

# Use a reasonable number of threads without assuming the Mac specification.
threads = max(1, min(8, os.cpu_count() or 4))
con.execute(f"PRAGMA threads={threads}")
con.execute("PRAGMA enable_progress_bar")

# ------------------------------------------------------------
# 1. Verify cohort and first ICU stay linkage.
# ------------------------------------------------------------
cohort_qc = con.execute("""
WITH ranked_stays AS (
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        intime,
        outtime,
        ROW_NUMBER() OVER (
            PARTITION BY subject_id, hadm_id
            ORDER BY intime, stay_id
        ) AS rn
    FROM icustays
)
SELECT
    COUNT(*) AS n_rows,
    COUNT(DISTINCT c.subject_id) AS n_subjects,
    COUNT(DISTINCT c.hadm_id) AS n_hadm,
    COUNT(rs.stay_id) AS n_linked_first_stays,
    SUM(
        CASE WHEN rs.intime = c.first_icu_intime THEN 1 ELSE 0 END
    ) AS n_exact_intime_matches
FROM cirrhosis_first_icu_admission c
LEFT JOIN ranked_stays rs
    ON c.subject_id = rs.subject_id
   AND c.hadm_id = rs.hadm_id
   AND rs.rn = 1
""").fetchdf()

print("===== COHORT CHECK =====")
print(cohort_qc.to_string(index=False))
print()

row = cohort_qc.iloc[0]

expected = 4129

if int(row["n_rows"]) != expected:
    raise RuntimeError(
        f"Unexpected cohort size: {row['n_rows']} instead of {expected}"
    )

if int(row["n_subjects"]) != expected:
    raise RuntimeError("Cohort is not one row per subject.")

if int(row["n_linked_first_stays"]) != expected:
    raise RuntimeError("Not all cohort records link to a first ICU stay.")

if int(row["n_exact_intime_matches"]) != expected:
    raise RuntimeError(
        "First ICU intime linkage does not match all 4,129 patients."
    )

print("Cohort linkage: PASS")
print()

# ------------------------------------------------------------
# 2. Audit official RRT itemids against MIMIC-IV d_items.
# ------------------------------------------------------------
all_itemids = sorted(
    set(CHAR_ITEMIDS + INPUT_ITEMIDS + PROCEDURE_ITEMIDS)
)

item_audit = con.execute(f"""
SELECT
    x.itemid,
    d.label,
    d.linksto
FROM (
    SELECT UNNEST([{ids(all_itemids)}]) AS itemid
) x
LEFT JOIN d_items d
    ON x.itemid = d.itemid
ORDER BY x.itemid
""").fetchdf()

missing_items = item_audit[item_audit["label"].isna()]

print("===== OFFICIAL ITEMID DICTIONARY AUDIT =====")
print(f"Requested itemids : {len(all_itemids)}")
print(f"Found in d_items  : {len(item_audit) - len(missing_items)}")
print(f"Missing           : {len(missing_items)}")

if len(missing_items):
    print("\nWARNING - itemids not found in local d_items:")
    print(missing_items.to_string(index=False))

print()

# ------------------------------------------------------------
# Extraction window
#
# Earliest baseline lab can occur at ICU intime - 6 h.
# MELD dialysis criterion needs a 7-day lookback.
# Therefore -8 days safely covers the required lookback.
#
# The latest planned reassessment is +72 h.
#
# We extract by SUBJECT + TIME, not only the first ICU stay_id.
# This preserves relevant RRT evidence from another ICU stay /
# adjacent admission if it falls inside the required 7-day
# physiologic lookback.
# ------------------------------------------------------------

query = f"""
COPY (
WITH ranked_stays AS (
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        intime,
        outtime,
        ROW_NUMBER() OVER (
            PARTITION BY subject_id, hadm_id
            ORDER BY intime, stay_id
        ) AS rn
    FROM icustays
),

cohort AS (
    SELECT
        c.subject_id,
        c.hadm_id AS cohort_hadm_id,
        c.first_icu_intime,
        rs.stay_id AS cohort_first_stay_id
    FROM cirrhosis_first_icu_admission c
    INNER JOIN ranked_stays rs
        ON c.subject_id = rs.subject_id
       AND c.hadm_id = rs.hadm_id
       AND rs.rn = 1
),

ce AS (
    SELECT
        c.subject_id,
        c.cohort_hadm_id,
        ce.hadm_id AS source_hadm_id,
        c.cohort_first_stay_id,
        ce.stay_id,
        c.first_icu_intime,

        'chartevents'::VARCHAR AS source_table,
        ce.itemid,
        COALESCE(di.label, 'UNKNOWN')::VARCHAR AS label,

        ce.charttime::TIMESTAMP AS event_time,
        CAST(NULL AS TIMESTAMP) AS starttime,
        CAST(NULL AS TIMESTAMP) AS endtime,

        CAST(ce.value AS VARCHAR) AS value,
        TRY_CAST(ce.valuenum AS DOUBLE) AS valuenum,
        CAST(NULL AS DOUBLE) AS amount,
        CAST(NULL AS DOUBLE) AS rate,
        CAST(NULL AS VARCHAR) AS status_description,

        CASE
            WHEN ce.itemid = 224146
                 AND ce.value IN (
                     'Active',
                     'Initiated',
                     'Reinitiated',
                     'New Filter'
                 )
                THEN 'crrt_state_active'

            WHEN ce.itemid = 224146
                 AND ce.value IN (
                     'Recirculating',
                     'Discontinued'
                 )
                THEN 'crrt_state_inactive'

            WHEN ce.itemid = 227290
                THEN 'modality_point'

            WHEN ce.itemid = 225965
                 AND ce.value = 'In use'
                THEN 'active_point'

            WHEN ce.itemid IN ({char_active})
                THEN 'active_point'

            ELSE 'supportive_point'
        END::VARCHAR AS evidence_class,

        CASE
            WHEN ce.itemid = 224146
                 AND ce.value IN (
                     'Active',
                     'Initiated',
                     'Reinitiated',
                     'New Filter'
                 )
                THEN 1

            WHEN ce.itemid = 224146
                 AND ce.value IN (
                     'Recirculating',
                     'Discontinued'
                 )
                THEN 0

            WHEN ce.itemid = 225965
                 AND ce.value = 'In use'
                THEN 1

            WHEN ce.itemid IN ({char_active})
                THEN 1

            ELSE 0
        END::INTEGER AS active_flag,

        CASE
            WHEN ce.itemid = 224146
                 AND ce.value IN (
                     'Active',
                     'Initiated',
                     'Reinitiated',
                     'New Filter'
                 )
                THEN 1

            WHEN ce.itemid = 224146
                 AND ce.value IN (
                     'Recirculating',
                     'Discontinued'
                 )
                THEN 0

            ELSE NULL
        END::INTEGER AS system_active,

        CASE
            WHEN ce.itemid = 227290
                THEN CAST(ce.value AS VARCHAR)

            WHEN ce.itemid = 226499
                THEN 'IHD'

            WHEN ce.itemid IN ({pd_ids})
                THEN 'Peritoneal'

            WHEN ce.itemid IN ({crrt_char})
                THEN 'CRRT'

            ELSE NULL
        END::VARCHAR AS rrt_type,

        CASE
            WHEN ce.hadm_id = c.cohort_hadm_id THEN 1 ELSE 0
        END::INTEGER AS within_cohort_hadm,

        CASE
            WHEN ce.stay_id = c.cohort_first_stay_id THEN 1 ELSE 0
        END::INTEGER AS is_first_icu_stay,

        ROUND(
            date_diff(
                'second',
                c.first_icu_intime,
                ce.charttime
            ) / 3600.0,
            3
        ) AS hours_from_icu

    FROM read_csv_auto(
        '{sql_path(CHARTEVENTS)}',
        header = true,
        sample_size = 100000
    ) ce

    INNER JOIN cohort c
        ON ce.subject_id = c.subject_id

    LEFT JOIN d_items di
        ON ce.itemid = di.itemid

    WHERE ce.itemid IN ({char_all})
      AND ce.value IS NOT NULL
      AND ce.charttime >= c.first_icu_intime - INTERVAL '8 days'
      AND ce.charttime <= c.first_icu_intime + INTERVAL '72 hours'
),

ie AS (
    SELECT
        c.subject_id,
        c.cohort_hadm_id,
        ie.hadm_id AS source_hadm_id,
        c.cohort_first_stay_id,
        ie.stay_id,
        c.first_icu_intime,

        'inputevents'::VARCHAR AS source_table,
        ie.itemid,
        COALESCE(di.label, 'UNKNOWN')::VARCHAR AS label,

        ie.starttime::TIMESTAMP AS event_time,
        ie.starttime::TIMESTAMP AS starttime,
        ie.endtime::TIMESTAMP AS endtime,

        CAST(NULL AS VARCHAR) AS value,
        CAST(NULL AS DOUBLE) AS valuenum,
        TRY_CAST(ie.amount AS DOUBLE) AS amount,
        TRY_CAST(ie.rate AS DOUBLE) AS rate,
        CAST(ie.statusdescription AS VARCHAR) AS status_description,

        'active_interval'::VARCHAR AS evidence_class,
        1::INTEGER AS active_flag,
        CAST(NULL AS INTEGER) AS system_active,
        'CRRT'::VARCHAR AS rrt_type,

        CASE
            WHEN ie.hadm_id = c.cohort_hadm_id THEN 1 ELSE 0
        END::INTEGER AS within_cohort_hadm,

        CASE
            WHEN ie.stay_id = c.cohort_first_stay_id THEN 1 ELSE 0
        END::INTEGER AS is_first_icu_stay,

        ROUND(
            date_diff(
                'second',
                c.first_icu_intime,
                ie.starttime
            ) / 3600.0,
            3
        ) AS hours_from_icu

    FROM read_csv_auto(
        '{sql_path(INPUTEVENTS)}',
        header = true,
        sample_size = 100000
    ) ie

    INNER JOIN cohort c
        ON ie.subject_id = c.subject_id

    LEFT JOIN d_items di
        ON ie.itemid = di.itemid

    WHERE ie.itemid IN ({input_ids})
      AND ie.amount > 0
      AND ie.endtime >= c.first_icu_intime - INTERVAL '8 days'
      AND ie.starttime <= c.first_icu_intime + INTERVAL '72 hours'
),

pe AS (
    SELECT
        c.subject_id,
        c.cohort_hadm_id,
        pe.hadm_id AS source_hadm_id,
        c.cohort_first_stay_id,
        pe.stay_id,
        c.first_icu_intime,

        'procedureevents'::VARCHAR AS source_table,
        pe.itemid,
        COALESCE(di.label, 'UNKNOWN')::VARCHAR AS label,

        pe.starttime::TIMESTAMP AS event_time,
        pe.starttime::TIMESTAMP AS starttime,
        pe.endtime::TIMESTAMP AS endtime,

        CAST(pe.value AS VARCHAR) AS value,
        TRY_CAST(pe.value AS DOUBLE) AS valuenum,
        CAST(NULL AS DOUBLE) AS amount,
        CAST(NULL AS DOUBLE) AS rate,
        CAST(pe.statusdescription AS VARCHAR) AS status_description,

        CASE
            WHEN pe.itemid IN (224270, 225436)
                THEN 'supportive_interval'
            ELSE 'active_interval'
        END::VARCHAR AS evidence_class,

        CASE
            WHEN pe.itemid IN (224270, 225436)
                THEN 0
            ELSE 1
        END::INTEGER AS active_flag,

        CAST(NULL AS INTEGER) AS system_active,

        CASE
            WHEN pe.itemid = 225441 THEN 'IHD'
            WHEN pe.itemid = 225802 THEN 'CRRT'
            WHEN pe.itemid = 225803 THEN 'CVVHD'
            WHEN pe.itemid = 225805 THEN 'Peritoneal'
            WHEN pe.itemid = 225809 THEN 'CVVHDF'
            WHEN pe.itemid = 225955 THEN 'SCUF'
            ELSE NULL
        END::VARCHAR AS rrt_type,

        CASE
            WHEN pe.hadm_id = c.cohort_hadm_id THEN 1 ELSE 0
        END::INTEGER AS within_cohort_hadm,

        CASE
            WHEN pe.stay_id = c.cohort_first_stay_id THEN 1 ELSE 0
        END::INTEGER AS is_first_icu_stay,

        ROUND(
            date_diff(
                'second',
                c.first_icu_intime,
                pe.starttime
            ) / 3600.0,
            3
        ) AS hours_from_icu

    FROM read_csv_auto(
        '{sql_path(PROCEDUREEVENTS)}',
        header = true,
        sample_size = 100000
    ) pe

    INNER JOIN cohort c
        ON pe.subject_id = c.subject_id

    LEFT JOIN d_items di
        ON pe.itemid = di.itemid

    WHERE pe.itemid IN ({procedure_ids})
      AND pe.value IS NOT NULL
      AND pe.endtime >= c.first_icu_intime - INTERVAL '8 days'
      AND pe.starttime <= c.first_icu_intime + INTERVAL '72 hours'
)

SELECT * FROM ce
UNION ALL
SELECT * FROM ie
UNION ALL
SELECT * FROM pe

)
TO '{sql_path(OUT)}'
(
    FORMAT PARQUET,
    COMPRESSION ZSTD
)
"""

print("===== STARTING LARGE-FILE EXTRACTION =====")
print(
    "Scanning chartevents + inputevents + procedureevents once.\n"
    "Do not close the Terminal while this is running."
)
print()

con.execute(query)

print()
print("Main extraction completed.")
print(f"Saved: {OUT}")
print()

# ------------------------------------------------------------
# 3. QC of extracted evidence.
# ------------------------------------------------------------
summary = con.execute(f"""
SELECT
    COUNT(*) AS n_rows,
    COUNT(DISTINCT subject_id) AS n_subjects_with_rrt_evidence,
    COUNT(DISTINCT CASE WHEN active_flag = 1 THEN subject_id END)
        AS n_subjects_with_active_evidence,
    COUNT(DISTINCT source_hadm_id) AS n_source_hadm,
    COUNT(DISTINCT stay_id) AS n_source_stays,
    SUM(CASE WHEN within_cohort_hadm = 1 THEN 1 ELSE 0 END)
        AS rows_same_hadm,
    SUM(CASE WHEN within_cohort_hadm = 0 THEN 1 ELSE 0 END)
        AS rows_other_hadm
FROM read_parquet('{sql_path(OUT)}')
""").fetchdf()

print("===== OVERALL RRT EVIDENCE QC =====")
print(summary.to_string(index=False))
print()

source_qc = con.execute(f"""
SELECT
    source_table,
    evidence_class,
    active_flag,
    COALESCE(rrt_type, 'Unspecified') AS rrt_type,
    COUNT(*) AS n_rows,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM read_parquet('{sql_path(OUT)}')
GROUP BY
    source_table,
    evidence_class,
    active_flag,
    COALESCE(rrt_type, 'Unspecified')
ORDER BY source_table, evidence_class, rrt_type
""").fetchdf()

source_qc.to_csv(QC_SOURCE, index=False)

print("===== QC BY SOURCE / EVIDENCE CLASS =====")
print(source_qc.to_string(index=False))
print()

item_qc = con.execute(f"""
SELECT
    source_table,
    itemid,
    label,
    evidence_class,
    active_flag,
    COALESCE(rrt_type, 'Unspecified') AS rrt_type,
    COUNT(*) AS n_rows,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM read_parquet('{sql_path(OUT)}')
GROUP BY
    source_table,
    itemid,
    label,
    evidence_class,
    active_flag,
    COALESCE(rrt_type, 'Unspecified')
ORDER BY source_table, itemid, evidence_class
""").fetchdf()

item_qc.to_csv(QC_ITEM, index=False)

print("===== QC BY ITEMID =====")
print(item_qc.to_string(index=False))
print()

timing_qc = con.execute(f"""
SELECT
    CASE
        WHEN hours_from_icu < -168 THEN '< -168 h'
        WHEN hours_from_icu < -6 THEN '-168 to -6 h'
        WHEN hours_from_icu < 0 THEN '-6 to 0 h'
        WHEN hours_from_icu <= 12 THEN '0 to 12 h'
        WHEN hours_from_icu <= 24 THEN '12 to 24 h'
        WHEN hours_from_icu <= 48 THEN '24 to 48 h'
        WHEN hours_from_icu <= 72 THEN '48 to 72 h'
        ELSE '> 72 h'
    END AS timing_window,
    COUNT(*) AS n_rows,
    COUNT(DISTINCT subject_id) AS n_subjects,
    COUNT(DISTINCT CASE WHEN active_flag = 1 THEN subject_id END)
        AS n_subjects_active
FROM read_parquet('{sql_path(OUT)}')
GROUP BY 1
ORDER BY MIN(hours_from_icu)
""").fetchdf()

print("===== TIMING QC =====")
print(timing_qc.to_string(index=False))
print()

print("=" * 72)
print("CHECKPOINT 11A COMPLETE")
print("=" * 72)
print(f"Main evidence file : {OUT}")
print(f"QC by source       : {QC_SOURCE}")
print(f"QC by item         : {QC_ITEM}")
print()
print(
    "IMPORTANT: This file is an evidence layer only.\n"
    "No patient has yet been classified as meeting the formal\n"
    "MELD 3.0 dialysis criterion.\n"
    "That classification will be performed in Checkpoint 11B."
)

con.close()
