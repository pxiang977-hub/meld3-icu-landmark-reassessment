from pathlib import Path
import duckdb

ROOT = Path.home() / "Desktop" / "MIMIC_IV_3_1"
LAB = ROOT / "data" / "hosp" / "labevents.csv.gz"
OUT = ROOT / "work" / "meld_labs.parquet"

con = duckdb.connect()
con.execute("SET threads = 4")

query = f"""
COPY (
    SELECT
        labevent_id,
        subject_id,
        hadm_id,
        specimen_id,
        itemid,
        charttime,
        storetime,
        value,
        valuenum,
        valueuom,
        ref_range_lower,
        ref_range_upper,
        flag,
        priority
    FROM read_csv_auto(
        '{LAB.as_posix()}',
        header=true,
        compression='gzip',
        sample_size=100000
    )
    WHERE itemid IN (
        50862,   -- Albumin
        50885,   -- Bilirubin, Total
        50912,   -- Creatinine
        51237,   -- INR(PT)
        50983    -- Sodium
    )
      AND hadm_id IS NOT NULL
      AND valuenum IS NOT NULL
)
TO '{OUT.as_posix()}'
(FORMAT PARQUET, COMPRESSION ZSTD);
"""

print("Extracting the five MELD 3.0 laboratory variables...")
con.execute(query)

audit = con.execute(f"""
SELECT
    itemid,
    COUNT(*) AS n_rows,
    COUNT(DISTINCT subject_id) AS n_patients,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    MIN(charttime) AS first_time,
    MAX(charttime) AS last_time
FROM read_parquet('{OUT.as_posix()}')
GROUP BY itemid
ORDER BY itemid;
""").df()

print("\n" + "=" * 100)
print("MELD 3.0 LAB EXTRACTION COMPLETE")
print("=" * 100)
print(audit.to_string(index=False))
print("=" * 100)
print(f"\nSaved to:\n{OUT}")

con.close()
