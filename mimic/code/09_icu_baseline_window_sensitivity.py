from pathlib import Path
import duckdb
import pandas as pd

ROOT = Path.home() / "Desktop" / "MIMIC_IV_3_1"
DB = ROOT / "work" / "mimic_feasibility.duckdb"
LAB = ROOT / "work" / "meld_labs.parquet"
OUT = ROOT / "work" / "icu_baseline_window_sensitivity.csv"

con = duckdb.connect(str(DB), read_only=True)

baseline_windows = [
    ("-6_to_6h", -6, 6),
    ("-6_to_12h", -6, 12),
    ("0_to_12h", 0, 12),
]

updates = [
    ("48h", 24, 48, 48),
    ("72h", 48, 72, 72),
]

def pct(n, d):
    if not d:
        return None
    return round(100.0 * n / d, 2)

rows = []

for bname, b0, b1 in baseline_windows:

    for uname, u0, u1, landmark_h in updates:

        q = f"""
        WITH eligible AS (
            SELECT
                hadm_id,
                first_icu_intime AS index_time,
                hospital_expire_flag
            FROM cirrhosis_first_icu_admission
            WHERE first_icu_intime IS NOT NULL
              AND dischtime >
                  first_icu_intime + INTERVAL '{landmark_h} hours'
              AND (
                    deathtime IS NULL
                    OR deathtime >=
                       first_icu_intime + INTERVAL '{landmark_h} hours'
                  )
        ),

        presence AS (
            SELECT
                e.hadm_id,
                e.hospital_expire_flag,

                MAX(CASE WHEN l.itemid=50862
                          AND l.charttime >= e.index_time + INTERVAL '{b0} hours'
                          AND l.charttime <  e.index_time + INTERVAL '{b1} hours'
                         THEN 1 ELSE 0 END) AS b_alb,

                MAX(CASE WHEN l.itemid=50885
                          AND l.charttime >= e.index_time + INTERVAL '{b0} hours'
                          AND l.charttime <  e.index_time + INTERVAL '{b1} hours'
                         THEN 1 ELSE 0 END) AS b_bil,

                MAX(CASE WHEN l.itemid=50912
                          AND l.charttime >= e.index_time + INTERVAL '{b0} hours'
                          AND l.charttime <  e.index_time + INTERVAL '{b1} hours'
                         THEN 1 ELSE 0 END) AS b_cr,

                MAX(CASE WHEN l.itemid=50983
                          AND l.charttime >= e.index_time + INTERVAL '{b0} hours'
                          AND l.charttime <  e.index_time + INTERVAL '{b1} hours'
                         THEN 1 ELSE 0 END) AS b_na,

                MAX(CASE WHEN l.itemid=51237
                          AND l.charttime >= e.index_time + INTERVAL '{b0} hours'
                          AND l.charttime <  e.index_time + INTERVAL '{b1} hours'
                         THEN 1 ELSE 0 END) AS b_inr,

                MAX(CASE WHEN l.itemid=50862
                          AND l.charttime >= e.index_time + INTERVAL '{u0} hours'
                          AND l.charttime <  e.index_time + INTERVAL '{u1} hours'
                         THEN 1 ELSE 0 END) AS u_alb,

                MAX(CASE WHEN l.itemid=50885
                          AND l.charttime >= e.index_time + INTERVAL '{u0} hours'
                          AND l.charttime <  e.index_time + INTERVAL '{u1} hours'
                         THEN 1 ELSE 0 END) AS u_bil,

                MAX(CASE WHEN l.itemid=50912
                          AND l.charttime >= e.index_time + INTERVAL '{u0} hours'
                          AND l.charttime <  e.index_time + INTERVAL '{u1} hours'
                         THEN 1 ELSE 0 END) AS u_cr,

                MAX(CASE WHEN l.itemid=50983
                          AND l.charttime >= e.index_time + INTERVAL '{u0} hours'
                          AND l.charttime <  e.index_time + INTERVAL '{u1} hours'
                         THEN 1 ELSE 0 END) AS u_na,

                MAX(CASE WHEN l.itemid=51237
                          AND l.charttime >= e.index_time + INTERVAL '{u0} hours'
                          AND l.charttime <  e.index_time + INTERVAL '{u1} hours'
                         THEN 1 ELSE 0 END) AS u_inr

            FROM eligible e
            LEFT JOIN read_parquet('{LAB.as_posix()}') l
              ON l.hadm_id=e.hadm_id
             AND l.charttime >= e.index_time - INTERVAL '6 hours'
             AND l.charttime <  e.index_time + INTERVAL '72 hours'

            GROUP BY
                e.hadm_id,
                e.hospital_expire_flag
        ),

        status AS (
            SELECT *,
                CASE WHEN
                    b_alb+b_bil+b_cr+b_na+b_inr=5
                    THEN 1 ELSE 0 END AS baseline_complete,

                CASE WHEN
                    u_alb+u_bil+u_cr+u_na+u_inr=5
                    THEN 1 ELSE 0 END AS update_complete
            FROM presence
        )

        SELECT
            COUNT(*) AS eligible_n,
            SUM(baseline_complete) AS baseline_n,
            SUM(update_complete) AS update_n,

            SUM(
                CASE WHEN baseline_complete=1
                      AND update_complete=1
                     THEN 1 ELSE 0 END
            ) AS paired_n,

            SUM(
                CASE WHEN baseline_complete=1
                      AND update_complete=1
                     THEN hospital_expire_flag ELSE 0 END
            ) AS paired_deaths

        FROM status;
        """

        r = con.execute(q).fetchone()

        eligible = int(r[0] or 0)
        baseline = int(r[1] or 0)
        update = int(r[2] or 0)
        paired = int(r[3] or 0)
        deaths = int(r[4] or 0)

        rows.append({
            "baseline_window": bname,
            "landmark": uname,
            "eligible_n": eligible,
            "baseline_complete_n": baseline,
            "baseline_complete_pct": pct(baseline, eligible),
            "update_complete_n": update,
            "update_complete_pct": pct(update, eligible),
            "paired_complete_n": paired,
            "paired_pct": pct(paired, eligible),
            "paired_deaths": deaths,
            "paired_mortality_pct": pct(deaths, paired),
        })

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

print("\n" + "="*110)
print("ICU BASELINE WINDOW SENSITIVITY")
print("="*110)
print(df.to_string(index=False))
print("="*110)

print(f"\nSaved to:\n{OUT}")

con.close()
