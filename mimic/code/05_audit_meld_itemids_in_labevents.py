from pathlib import Path
import duckdb

ROOT = Path.home() / "Desktop" / "MIMIC_IV_3_1"
LAB = ROOT / "data" / "hosp" / "labevents.csv.gz"
OUT = ROOT / "work" / "meld_itemid_usage_audit.csv"

if not LAB.exists():
    raise FileNotFoundError(f"labevents not found: {LAB}")

candidate_ids = [
    50862, 53085,   # Albumin
    50885, 53089,   # Bilirubin, Total
    50912, 52546,   # Creatinine
    51237, 51675,   # INR(PT)
    50983, 52623    # Sodium
]

ids = ",".join(str(x) for x in candidate_ids)

con = duckdb.connect()
con.execute("SET threads = 4")

query = f"""
SELECT
    itemid,
    valueuom,
    COUNT(*) AS n_rows,
    COUNT(DISTINCT subject_id) AS n_patients,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    MIN(charttime) AS first_charttime,
    MAX(charttime) AS last_charttime,
    MIN(valuenum) AS min_value,
    quantile_cont(valuenum, 0.50) AS median_value,
    MAX(valuenum) AS max_value
FROM read_csv_auto(
    '{LAB.as_posix()}',
    header = true,
    compression = 'gzip',
    sample_size = 100000
)
WHERE itemid IN ({ids})
  AND valuenum IS NOT NULL
GROUP BY
    itemid,
    valueuom
ORDER BY
    itemid,
    n_rows DESC;
"""

print("\nScanning labevents.csv.gz ...")
print("This file is 2.4 GB, so several minutes may be required.\n")

df = con.execute(query).df()
df.to_csv(OUT, index=False)

print("=" * 120)
print("MELD 3.0 CANDIDATE ITEMID USAGE AUDIT")
print("=" * 120)
print(df.to_string(index=False))
print("=" * 120)

print(f"\nSaved to:\n{OUT}")

con.close()
