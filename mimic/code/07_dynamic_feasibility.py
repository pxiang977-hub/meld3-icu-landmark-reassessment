from pathlib import Path
import duckdb
import pandas as pd

ROOT = Path.home() / "Desktop" / "MIMIC_IV_3_1"
DB = ROOT / "work" / "mimic_feasibility.duckdb"
LAB = ROOT / "work" / "meld_labs.parquet"

OUT1 = ROOT / "work" / "dynamic_feasibility_by_window.csv"
OUT2 = ROOT / "work" / "dynamic_sequence_summary.csv"

con = duckdb.connect(str(DB), read_only=True)

windows = [
    ("0-24h", 0, 24),
    ("24-48h", 24, 48),
    ("48-72h", 48, 72),
]

cohorts = [
    {
        "name": "Hospital-first",
        "table": "cirrhosis_first_admission",
        "index_col": "admittime",
        "icu": False,
    },
    {
        "name": "ICU-first",
        "table": "cirrhosis_first_icu_admission",
        "index_col": "first_icu_intime",
        "icu": True,
    },
]

def pct(n, d):
    if d == 0 or d is None:
        return None
    return round(100.0 * n / d, 2)

results = []
sequence_results = []

for cohort in cohorts:

    cname = cohort["name"]
    table = cohort["table"]
    index_col = cohort["index_col"]

    total_n = con.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()[0]

    print("\n" + "=" * 110)
    print(f"COHORT: {cname}   N={total_n:,}")
    print("=" * 110)

    complete_sets = []
    eligible_sets = []

    for wname, start_h, end_h in windows:

        # ------------------------------------------------------
        # Landmark eligible:
        # alive and still hospitalized at start of window
        # ------------------------------------------------------
        eligible_query = f"""
        SELECT hadm_id
        FROM {table}
        WHERE {index_col} IS NOT NULL
          AND dischtime > {index_col} + INTERVAL '{start_h} hours'
          AND (
                deathtime IS NULL
                OR deathtime >= {index_col} + INTERVAL '{start_h} hours'
              );
        """

        eligible_ids = {
            x[0] for x in con.execute(eligible_query).fetchall()
        }

        eligible_sets.append(eligible_ids)
        n_eligible = len(eligible_ids)

        # ------------------------------------------------------
        # Lab availability among landmark-eligible patients
        # ------------------------------------------------------
        lab_query = f"""
        WITH eligible AS (
            SELECT
                subject_id,
                hadm_id,
                {index_col} AS index_time,
                hospital_expire_flag
            FROM {table}
            WHERE {index_col} IS NOT NULL
              AND dischtime > {index_col} + INTERVAL '{start_h} hours'
              AND (
                    deathtime IS NULL
                    OR deathtime >= {index_col} + INTERVAL '{start_h} hours'
                  )
        ),
        presence AS (
            SELECT
                e.hadm_id,
                e.hospital_expire_flag,

                MAX(CASE WHEN l.itemid = 50862 THEN 1 ELSE 0 END) AS albumin,
                MAX(CASE WHEN l.itemid = 50885 THEN 1 ELSE 0 END) AS bilirubin,
                MAX(CASE WHEN l.itemid = 50912 THEN 1 ELSE 0 END) AS creatinine,
                MAX(CASE WHEN l.itemid = 50983 THEN 1 ELSE 0 END) AS sodium,
                MAX(CASE WHEN l.itemid = 51237 THEN 1 ELSE 0 END) AS inr

            FROM eligible e

            LEFT JOIN read_parquet('{LAB.as_posix()}') l
              ON l.hadm_id = e.hadm_id
             AND l.charttime >= e.index_time + INTERVAL '{start_h} hours'
             AND l.charttime <  e.index_time + INTERVAL '{end_h} hours'

            GROUP BY
                e.hadm_id,
                e.hospital_expire_flag
        )
        SELECT
            COUNT(*) AS n_eligible,

            SUM(albumin) AS n_albumin,
            SUM(bilirubin) AS n_bilirubin,
            SUM(creatinine) AS n_creatinine,
            SUM(sodium) AS n_sodium,
            SUM(inr) AS n_inr,

            SUM(
                CASE
                    WHEN albumin + bilirubin + creatinine + sodium + inr = 5
                    THEN 1 ELSE 0
                END
            ) AS n_complete,

            SUM(hospital_expire_flag) AS hospital_deaths_after_landmark

        FROM presence;
        """

        r = con.execute(lab_query).fetchone()

        (
            _,
            n_albumin,
            n_bilirubin,
            n_creatinine,
            n_sodium,
            n_inr,
            n_complete,
            n_deaths
        ) = r

        # ------------------------------------------------------
        # Get complete-case hadm_id set for longitudinal overlap
        # ------------------------------------------------------
        complete_query = f"""
        WITH eligible AS (
            SELECT
                hadm_id,
                {index_col} AS index_time
            FROM {table}
            WHERE {index_col} IS NOT NULL
              AND dischtime > {index_col} + INTERVAL '{start_h} hours'
              AND (
                    deathtime IS NULL
                    OR deathtime >= {index_col} + INTERVAL '{start_h} hours'
                  )
        ),
        presence AS (
            SELECT
                e.hadm_id,

                MAX(CASE WHEN l.itemid = 50862 THEN 1 ELSE 0 END) AS albumin,
                MAX(CASE WHEN l.itemid = 50885 THEN 1 ELSE 0 END) AS bilirubin,
                MAX(CASE WHEN l.itemid = 50912 THEN 1 ELSE 0 END) AS creatinine,
                MAX(CASE WHEN l.itemid = 50983 THEN 1 ELSE 0 END) AS sodium,
                MAX(CASE WHEN l.itemid = 51237 THEN 1 ELSE 0 END) AS inr

            FROM eligible e

            LEFT JOIN read_parquet('{LAB.as_posix()}') l
              ON l.hadm_id = e.hadm_id
             AND l.charttime >= e.index_time + INTERVAL '{start_h} hours'
             AND l.charttime <  e.index_time + INTERVAL '{end_h} hours'

            GROUP BY e.hadm_id
        )

        SELECT hadm_id
        FROM presence
        WHERE albumin + bilirubin + creatinine + sodium + inr = 5;
        """

        complete_ids = {
            x[0] for x in con.execute(complete_query).fetchall()
        }

        complete_sets.append(complete_ids)

        # ------------------------------------------------------
        # ICU-active patients at exact landmark
        # Only reported for ICU-first cohort
        # ------------------------------------------------------
        icu_active_n = None

        if cohort["icu"]:

            icu_active_query = f"""
            WITH eligible AS (
                SELECT
                    hadm_id,
                    {index_col} + INTERVAL '{start_h} hours' AS landmark_time
                FROM {table}
                WHERE {index_col} IS NOT NULL
                  AND dischtime > {index_col} + INTERVAL '{start_h} hours'
                  AND (
                        deathtime IS NULL
                        OR deathtime >= {index_col} + INTERVAL '{start_h} hours'
                      )
            )
            SELECT COUNT(DISTINCT e.hadm_id)
            FROM eligible e
            INNER JOIN icustays i
                ON e.hadm_id = i.hadm_id
               AND i.intime <= e.landmark_time
               AND i.outtime > e.landmark_time;
            """

            icu_active_n = con.execute(
                icu_active_query
            ).fetchone()[0]

        results.append({
            "cohort": cname,
            "window": wname,

            "cohort_n": total_n,
            "landmark_eligible_n": n_eligible,
            "landmark_eligible_pct": pct(n_eligible, total_n),

            "albumin_n": int(n_albumin or 0),
            "albumin_pct": pct(int(n_albumin or 0), n_eligible),

            "bilirubin_n": int(n_bilirubin or 0),
            "bilirubin_pct": pct(int(n_bilirubin or 0), n_eligible),

            "creatinine_n": int(n_creatinine or 0),
            "creatinine_pct": pct(int(n_creatinine or 0), n_eligible),

            "sodium_n": int(n_sodium or 0),
            "sodium_pct": pct(int(n_sodium or 0), n_eligible),

            "inr_n": int(n_inr or 0),
            "inr_pct": pct(int(n_inr or 0), n_eligible),

            "complete_meld_labs_n": int(n_complete or 0),
            "complete_meld_labs_pct": pct(int(n_complete or 0), n_eligible),

            "hospital_deaths_after_landmark": int(n_deaths or 0),
            "hospital_death_pct": pct(int(n_deaths or 0), n_eligible),

            "icu_active_at_landmark_n": icu_active_n,
            "icu_active_at_landmark_pct":
                pct(icu_active_n, n_eligible)
                if icu_active_n is not None else None,
        })

        print(
            f"{wname:8s}  "
            f"eligible={n_eligible:5d}  "
            f"Alb={pct(int(n_albumin or 0), n_eligible):6.2f}%  "
            f"TBil={pct(int(n_bilirubin or 0), n_eligible):6.2f}%  "
            f"Cr={pct(int(n_creatinine or 0), n_eligible):6.2f}%  "
            f"Na={pct(int(n_sodium or 0), n_eligible):6.2f}%  "
            f"INR={pct(int(n_inr or 0), n_eligible):6.2f}%  "
            f"COMPLETE={pct(int(n_complete or 0), n_eligible):6.2f}%"
        )

    # ----------------------------------------------------------
    # Longitudinal overlap across windows
    # ----------------------------------------------------------

    all_complete_ids = set.intersection(*complete_sets)

    at_least_two = set()

    universe = set.union(*complete_sets)

    for hadm_id in universe:
        n_windows = sum(
            hadm_id in s
            for s in complete_sets
        )
        if n_windows >= 2:
            at_least_two.add(hadm_id)

    first_two = complete_sets[0] & complete_sets[1]

    sequence_results.append({
        "cohort": cname,
        "cohort_n": total_n,

        "complete_0_24h_n": len(complete_sets[0]),
        "complete_0_24h_pct":
            pct(len(complete_sets[0]), len(eligible_sets[0])),

        "complete_24_48h_n": len(complete_sets[1]),
        "complete_24_48h_pct":
            pct(len(complete_sets[1]), len(eligible_sets[1])),

        "complete_48_72h_n": len(complete_sets[2]),
        "complete_48_72h_pct":
            pct(len(complete_sets[2]), len(eligible_sets[2])),

        "complete_first_two_windows_n": len(first_two),
        "complete_at_least_two_windows_n": len(at_least_two),
        "complete_all_three_windows_n": len(all_complete_ids),

        "all_three_pct_of_cohort":
            pct(len(all_complete_ids), total_n),
    })

df = pd.DataFrame(results)
seq = pd.DataFrame(sequence_results)

df.to_csv(OUT1, index=False)
seq.to_csv(OUT2, index=False)

print("\n" + "=" * 110)
print("FEASIBILITY BY WINDOW")
print("=" * 110)

show_cols = [
    "cohort",
    "window",
    "cohort_n",
    "landmark_eligible_n",
    "albumin_pct",
    "bilirubin_pct",
    "creatinine_pct",
    "sodium_pct",
    "inr_pct",
    "complete_meld_labs_n",
    "complete_meld_labs_pct",
    "hospital_deaths_after_landmark",
]

print(df[show_cols].to_string(index=False))

print("\n" + "=" * 110)
print("LONGITUDINAL COMPLETENESS")
print("=" * 110)
print(seq.to_string(index=False))

print("\nSaved:")
print(OUT1)
print(OUT2)

con.close()
