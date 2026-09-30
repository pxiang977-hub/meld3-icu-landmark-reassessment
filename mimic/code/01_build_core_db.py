from pathlib import Path
import duckdb

ROOT = Path.home() / "Desktop" / "MIMIC_IV_3_1"
DATA = ROOT / "data"
WORK = ROOT / "work"
DB = WORK / "mimic_feasibility.duckdb"

WORK.mkdir(parents=True, exist_ok=True)

files = {
    "admissions": DATA / "hosp" / "admissions.csv.gz",
    "patients": DATA / "hosp" / "patients.csv.gz",
    "diagnoses_icd": DATA / "hosp" / "diagnoses_icd.csv.gz",
    "d_icd_diagnoses": DATA / "hosp" / "d_icd_diagnoses.csv.gz",
    "d_labitems": DATA / "hosp" / "d_labitems.csv.gz",
    "icustays": DATA / "icu" / "icustays.csv.gz",
    "procedureevents": DATA / "icu" / "procedureevents.csv.gz",
    "d_items": DATA / "icu" / "d_items.csv.gz",
}

print("=" * 70)
print("MIMIC-IV v3.1 core database build")
print("=" * 70)

for name, path in files.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    print(f"FOUND  {name:20s}  {path.stat().st_size / 1024 / 1024:.2f} MB")

con = duckdb.connect(str(DB))
con.execute("SET threads = 4")

for name, path in files.items():
    print(f"\nLoading {name} ...")

    sql = f"""
    CREATE OR REPLACE TABLE {name} AS
    SELECT *
    FROM read_csv_auto(
        '{path.as_posix()}',
        header = true,
        compression = 'gzip',
        sample_size = 100000
    );
    """

    con.execute(sql)

    n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    cols = [x[0] for x in con.execute(f"DESCRIBE {name}").fetchall()]

    print(f"  rows    = {n:,}")
    print(f"  columns = {', '.join(cols)}")

print("\n" + "=" * 70)
print("CORE DATABASE BUILD COMPLETE")
print(f"Database: {DB}")
print("=" * 70)

con.close()
