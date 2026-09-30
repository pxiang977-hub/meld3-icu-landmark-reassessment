from pathlib import Path
import duckdb
import pandas as pd

ROOT = Path.home() / "Desktop" / "MIMIC_IV_3_1"
DB = ROOT / "work" / "mimic_feasibility.duckdb"
OUT = ROOT / "work" / "cirrhosis_code_candidates.csv"

con = duckdb.connect(str(DB), read_only=True)

query = """
SELECT
    d.icd_version,
    d.icd_code,
    dict.long_title,
    COUNT(DISTINCT d.subject_id) AS n_patients,
    COUNT(DISTINCT d.hadm_id) AS n_admissions
FROM diagnoses_icd d
INNER JOIN d_icd_diagnoses dict
    ON d.icd_code = dict.icd_code
   AND d.icd_version = dict.icd_version
WHERE LOWER(dict.long_title) LIKE '%cirrhos%'
GROUP BY
    d.icd_version,
    d.icd_code,
    dict.long_title
ORDER BY
    d.icd_version,
    n_patients DESC,
    d.icd_code;
"""

df = con.execute(query).df()

print("\n" + "=" * 100)
print("MIMIC-IV v3.1 — candidate cirrhosis diagnosis codes")
print("=" * 100)
print(df.to_string(index=False))
print("=" * 100)

df.to_csv(OUT, index=False)

print(f"\nNumber of candidate codes: {len(df)}")
print(f"Saved to: {OUT}")

con.close()
