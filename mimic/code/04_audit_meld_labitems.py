from pathlib import Path
import duckdb

ROOT = Path.home() / "Desktop" / "MIMIC_IV_3_1"
DB = ROOT / "work" / "mimic_feasibility.duckdb"

OUT_ALL = ROOT / "work" / "meld_labitem_candidates_all.csv"
OUT_FOCUSED = ROOT / "work" / "meld_labitem_candidates_focused.csv"

con = duckdb.connect(str(DB), read_only=True)

# Broad search: deliberately sensitive, for audit/QC
q_all = """
SELECT
    itemid,
    label,
    fluid,
    category
FROM d_labitems
WHERE
       LOWER(label) LIKE '%bilirubin%'
    OR LOWER(label) LIKE '%creatinine%'
    OR LOWER(label) LIKE '%sodium%'
    OR LOWER(label) LIKE '%albumin%'
    OR LOWER(label) LIKE '%inr%'
ORDER BY
    CASE
        WHEN LOWER(label) LIKE '%bilirubin%' THEN 1
        WHEN LOWER(label) LIKE '%creatinine%' THEN 2
        WHEN LOWER(label) LIKE '%sodium%' THEN 3
        WHEN LOWER(label) LIKE '%albumin%' THEN 4
        WHEN LOWER(label) LIKE '%inr%' THEN 5
        ELSE 9
    END,
    fluid,
    label,
    itemid;
"""

all_df = con.execute(q_all).df()
all_df.to_csv(OUT_ALL, index=False)

# Focused blood candidates for MELD 3.0
q_focused = """
SELECT
    itemid,
    label,
    fluid,
    category
FROM d_labitems
WHERE LOWER(fluid) = 'blood'
  AND (
       LOWER(label) = 'bilirubin, total'
    OR LOWER(label) = 'creatinine'
    OR LOWER(label) = 'sodium'
    OR LOWER(label) = 'albumin'
    OR LOWER(label) LIKE 'inr%'
  )
ORDER BY label, itemid;
"""

focused_df = con.execute(q_focused).df()
focused_df.to_csv(OUT_FOCUSED, index=False)

print("\n" + "=" * 90)
print("BROAD MELD 3.0 LAB ITEM AUDIT")
print("=" * 90)
print(all_df.to_string(index=False))

print("\n" + "=" * 90)
print("FOCUSED BLOOD CANDIDATES")
print("=" * 90)
print(focused_df.to_string(index=False))

print("\nSaved:")
print(OUT_ALL)
print(OUT_FOCUSED)

con.close()
