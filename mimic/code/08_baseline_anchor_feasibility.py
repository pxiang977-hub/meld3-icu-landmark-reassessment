from pathlib import Path
import duckdb
import pandas as pd

ROOT = Path.home() / "Desktop" / "MIMIC_IV_3_1"
DB = ROOT / "work" / "mimic_feasibility.duckdb"
LAB = ROOT / "work" / "meld_labs.parquet"
OUT = ROOT / "work" / "baseline_anchor_feasibility.csv"

con = duckdb.connect(str(DB), read_only=True)

ITEMS = [50862, 50885, 50912, 50983, 51237]

cohorts = [
    ("Hospital-first", "cirrhosis_first_admission", "admittime"),
    ("ICU-first", "cirrhosis_first_icu_admission", "first_icu_intime"),
]

schemes = [
    {
        "scheme": "Baseline 0-6h",
        "baseline": (0, 6),
        "updates": [
            ("24h", 6, 24, 24),
            ("48h", 24, 48, 48),
            ("72h", 48, 72, 72),
        ],
    },
    {
        "scheme": "Baseline 0-12h",
        "baseline": (0, 12),
        "updates": [
            ("24h", 12, 24, 24),
            ("48h", 24, 48, 48),
            ("72h", 48, 72, 72),
        ],
    },
]

def pct(n, d):
    if not d:
        return None
    return round(100.0 * n / d, 2)

rows = []

for cohort_name, table, index_col in cohorts:

    cohort_n = con.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()[0]

    for scheme in schemes:

        b0, b1 = scheme["baseline"]

        for update_name, u0, u1, landmark_h in scheme["updates"]:

            sql = f"""
            WITH eligible AS (
                SELECT
                    hadm_id,
                    {index_col} AS index_time,
                    hospital_expire_flag
                FROM {table}
                WHERE {index_col} IS NOT NULL
                  AND dischtime >
                      {index_col} + INTERVAL '{landmark_h} hours'
                  AND (
                        deathtime IS NULL
                        OR deathtime >=
                           {index_col} + INTERVAL '{landmark_h} hours'
                      )
            ),

            presence AS (
                SELECT
                    e.hadm_id,
                    e.hospital_expire_flag,

                    MAX(CASE WHEN l.itemid = 50862
                              AND l.charttime >= e.index_time + INTERVAL '{b0} hours'
                              AND l.charttime <  e.index_time + INTERVAL '{b1} hours'
                             THEN 1 ELSE 0 END) AS b_alb,

                    MAX(CASE WHEN l.itemid = 50885
                              AND l.charttime >= e.index_time + INTERVAL '{b0} hours'
                              AND l.charttime <  e.index_time + INTERVAL '{b1} hours'
                             THEN 1 ELSE 0 END) AS b_bil,

                    MAX(CASE WHEN l.itemid = 50912
                              AND l.charttime >= e.index_time + INTERVAL '{b0} hours'
                              AND l.charttime <  e.index_time + INTERVAL '{b1} hours'
                             THEN 1 ELSE 0 END) AS b_cr,

                    MAX(CASE WHEN l.itemid = 50983
                              AND l.charttime >= e.index_time + INTERVAL '{b0} hours'
                              AND l.charttime <  e.index_time + INTERVAL '{b1} hours'
                             THEN 1 ELSE 0 END) AS b_na,

                    MAX(CASE WHEN l.itemid = 51237
                              AND l.charttime >= e.index_time + INTERVAL '{b0} hours'
                              AND l.charttime <  e.index_time + INTERVAL '{b1} hours'
                             THEN 1 ELSE 0 END) AS b_inr,

                    MAX(CASE WHEN l.itemid = 50862
                              AND l.charttime >= e.index_time + INTERVAL '{u0} hours'
                              AND l.charttime <  e.index_time + INTERVAL '{u1} hours'
                             THEN 1 ELSE 0 END) AS u_alb,

                    MAX(CASE WHEN l.itemid = 50885
                              AND l.charttime >= e.index_time + INTERVAL '{u0} hours'
                              AND l.charttime <  e.index_time + INTERVAL '{u1} hours'
                             THEN 1 ELSE 0 END) AS u_bil,

                    MAX(CASE WHEN l.itemid = 50912
                              AND l.charttime >= e.index_time + INTERVAL '{u0} hours'
                              AND l.charttime <  e.index_time + INTERVAL '{u1} hours'
                             THEN 1 ELSE 0 END) AS u_cr,

                    MAX(CASE WHEN l.itemid = 50983
                              AND l.charttime >= e.index_time + INTERVAL '{u0} hours'
                              AND l.charttime <  e.index_time + INTERVAL '{u1} hours'
                             THEN 1 ELSE 0 END) AS u_na,

                    MAX(CASE WHEN l.itemid = 51237
                              AND l.charttime >= e.index_time + INTERVAL '{u0} hours'
                              AND l.charttime <  e.index_time + INTERVAL '{u1} hours'
                             THEN 1 ELSE 0 END) AS u_inr

                FROM eligible e

                LEFT JOIN read_parquet('{LAB.as_posix()}') l
                    ON l.hadm_id = e.hadm_id
                   AND l.charttime >= e.index_time
                   AND l.charttime < e.index_time + INTERVAL '72 hours'

                GROUP BY
                    e.hadm_id,
                    e.hospital_expire_flag
            ),

            status AS (
                SELECT
                    *,
                    CASE
                        WHEN b_alb+b_bil+b_cr+b_na+b_inr = 5
                        THEN 1 ELSE 0
                    END AS baseline_complete,

                    CASE
                        WHEN u_alb+u_bil+u_cr+u_na+u_inr = 5
                        THEN 1 ELSE 0
                    END AS update_complete

                FROM presence
            )

            SELECT
                COUNT(*) AS eligible_n,

                SUM(baseline_complete) AS baseline_complete_n,

                SUM(update_complete) AS update_complete_n,

                SUM(
                    CASE
                        WHEN baseline_complete=1
                         AND update_complete=1
                        THEN 1 ELSE 0
                    END
                ) AS paired_complete_n,

                SUM(
                    CASE
                        WHEN baseline_complete=1
                         AND update_complete=1
                        THEN hospital_expire_flag
                        ELSE 0
                    END
                ) AS paired_deaths

            FROM status;
            """

            r = con.execute(sql).fetchone()

            eligible_n = int(r[0] or 0)
            baseline_n = int(r[1] or 0)
            update_n = int(r[2] or 0)
            paired_n = int(r[3] or 0)
            deaths = int(r[4] or 0)

            rows.append({
                "cohort": cohort_name,
                "scheme": scheme["scheme"],
                "landmark": update_name,
                "cohort_n": cohort_n,
                "eligible_n": eligible_n,
                "baseline_complete_n": baseline_n,
                "baseline_complete_pct": pct(baseline_n, eligible_n),
                "update_complete_n": update_n,
                "update_complete_pct": pct(update_n, eligible_n),
                "paired_complete_n": paired_n,
                "paired_pct_of_eligible": pct(paired_n, eligible_n),
                "paired_deaths": deaths,
                "paired_mortality_pct": pct(deaths, paired_n),
            })

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

print("\n" + "=" * 125)
print("BASELINE + LANDMARK REASSESSMENT FEASIBILITY")
print("=" * 125)

cols = [
    "cohort",
    "scheme",
    "landmark",
    "eligible_n",
    "baseline_complete_n",
    "baseline_complete_pct",
    "update_complete_n",
    "update_complete_pct",
    "paired_complete_n",
    "paired_pct_of_eligible",
    "paired_deaths",
    "paired_mortality_pct",
]

print(df[cols].to_string(index=False))

print("\nSaved to:")
print(OUT)

con.close()
