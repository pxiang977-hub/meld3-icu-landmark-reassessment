from pathlib import Path
import math

import duckdb
import numpy as np
import pandas as pd


# ============================================================
# Checkpoint 12
# Formal MELD 3.0 snapshot construction
#
# Main cohort:
#   first adult cirrhosis ICU admission, n = 4,129
#
# Snapshots:
#   baseline : ICU -6 h to +12 h, anchor +12 h
#   48 h     : ICU +24 h to +48 h, anchor +48 h
#   72 h     : ICU +48 h to +72 h, anchor +72 h
#
# Value rule:
#   latest available value within each window
#
# Dialysis rule:
#   applied relative to the SELECTED creatinine timestamp
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DB = ROOT / "work" / "mimic_feasibility.duckdb"
LABS = ROOT / "work" / "meld_labs.parquet"

IHD = ROOT / "work" / "rrt_ihd_sessions.parquet"
PD = ROOT / "work" / "rrt_pd_sessions.parquet"
CONT = ROOT / "work" / "rrt_continuous_sessions.parquet"
HOSP = ROOT / "work" / "rrt_hospital_date_evidence.parquet"

OUT_LONG = ROOT / "work" / "meld3_snapshots_long.parquet"
OUT_WIDE = ROOT / "work" / "meld3_snapshots_wide.parquet"
OUT_QC = ROOT / "work" / "meld3_snapshot_qc.csv"
OUT_PAIR = ROOT / "work" / "meld3_paired_summary.csv"

for p in [DB, LABS, IHD, PD, CONT, HOSP]:
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")


ITEMS = {
    50862: "albumin",
    50885: "bilirubin",
    50912: "creatinine",
    50983: "sodium",
    51237: "inr",
}


SNAPSHOTS = [
    ("baseline", -6.0, 12.0, 12.0),
    ("h48", 24.0, 48.0, 48.0),
    ("h72", 48.0, 72.0, 72.0),
]


print("=" * 78)
print("CHECKPOINT 12 - FORMAL MELD 3.0 SNAPSHOTS")
print("=" * 78)


# ============================================================
# A. Main cohort
# ============================================================

con = duckdb.connect(str(DB), read_only=True)

cohort = con.execute("""
SELECT
    subject_id,
    hadm_id,
    first_icu_intime,
    dischtime,
    hospital_expire_flag,
    gender,
    age_at_admission
FROM cirrhosis_first_icu_admission
ORDER BY subject_id
""").fetchdf()

con.close()

for c in ["first_icu_intime", "dischtime"]:
    cohort[c] = pd.to_datetime(
        cohort[c],
        errors="coerce"
    )

if len(cohort) != 4129:
    raise RuntimeError(
        f"Unexpected cohort size: {len(cohort)}"
    )

if cohort["subject_id"].nunique() != 4129:
    raise RuntimeError(
        "Main cohort is not one row per subject."
    )

print(f"Main cohort: {len(cohort):,}")


# ============================================================
# B. Load RRT session layers
# ============================================================

ihd = pd.read_parquet(IHD)
pd_sessions = pd.read_parquet(PD)
cont = pd.read_parquet(CONT)
hospital_rrt = pd.read_parquet(HOSP)

for df, cols in [
    (ihd, ["session_start", "session_end"]),
    (pd_sessions, ["session_start", "session_end"]),
    (cont, ["session_start", "session_end"]),
]:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(
                df[c],
                errors="coerce"
            )

hospital_rrt["chartdate"] = pd.to_datetime(
    hospital_rrt["chartdate"],
    errors="coerce"
).dt.date


ihd_map = {
    sid: g.copy()
    for sid, g in ihd.groupby("subject_id")
}

pd_map = {
    sid: g.copy()
    for sid, g in pd_sessions.groupby("subject_id")
}

cont_map = {
    sid: g.copy()
    for sid, g in cont.groupby("subject_id")
}

hospital_map = {
    sid: g.copy()
    for sid, g in hospital_rrt.groupby("subject_id")
}


# Hospital date-level evidence allowed to contribute at most
# one conservative dialysis treatment day.
#
# ContinuousOver18h is NOT counted here as an intermittent
# treatment session; continuous therapy is handled separately.
HOSP_SESSION_CLASSES = {
    "IHD",
    "Peritoneal",
    "UrinaryFiltrationSingle",
    "UrinaryFiltrationMultiple",
    "Intermittent",
    "ProlongedIntermittent",
}


# ============================================================
# C. Latest lab selection in each prespecified window
# ============================================================

def select_snapshot(snapshot, start_h, end_h, anchor_h):
    lower_op = ">" if snapshot == "h72" else ">="

    con = duckdb.connect(
        str(DB),
        read_only=True
    )

    sql = f"""
    WITH cohort AS (
        SELECT
            subject_id,
            hadm_id,
            first_icu_intime
        FROM cirrhosis_first_icu_admission
    ),

    candidates AS (
        SELECT
            c.subject_id,
            c.hadm_id,

            l.itemid,
            l.charttime,
            l.valuenum,

            ROW_NUMBER() OVER (
                PARTITION BY
                    c.subject_id,
                    l.itemid
                ORDER BY
                    l.charttime DESC,
                    l.labevent_id DESC
            ) AS rn

        FROM read_parquet(
            '{str(LABS)}'
        ) l

        INNER JOIN cohort c
            ON l.subject_id = c.subject_id
           AND l.hadm_id = c.hadm_id

        WHERE
            l.itemid IN (
                50862,
                50885,
                50912,
                50983,
                51237
            )

            AND l.valuenum IS NOT NULL

            AND l.charttime {lower_op}
                c.first_icu_intime
                + INTERVAL '{start_h} hours'

            AND l.charttime <=
                c.first_icu_intime
                + INTERVAL '{end_h} hours'
    )

    SELECT
        subject_id,
        itemid,
        charttime,
        valuenum

    FROM candidates

    WHERE rn = 1

    ORDER BY subject_id, itemid
    """

    selected = con.execute(sql).fetchdf()
    con.close()

    selected["charttime"] = pd.to_datetime(
        selected["charttime"],
        errors="coerce"
    )

    out = cohort.copy()

    out["snapshot"] = snapshot
    out["window_start_h"] = start_h
    out["window_end_h"] = end_h
    out["anchor_h"] = anchor_h

    out["anchor_time"] = (
        out["first_icu_intime"]
        + pd.to_timedelta(
            anchor_h,
            unit="h"
        )
    )

    # Landmark risk-set requirement:
    # patient must still be in hospital after landmark.
    out["eligible"] = (
        out["dischtime"].notna()
        & (
            out["dischtime"]
            > out["anchor_time"]
        )
    ).astype(int)

    values = selected.pivot(
        index="subject_id",
        columns="itemid",
        values="valuenum"
    )

    times = selected.pivot(
        index="subject_id",
        columns="itemid",
        values="charttime"
    )

    values = values.rename(
        columns={
            k: v
            for k, v in ITEMS.items()
        }
    )

    times = times.rename(
        columns={
            k: f"{v}_time"
            for k, v in ITEMS.items()
        }
    )

    out = out.merge(
        values,
        how="left",
        left_on="subject_id",
        right_index=True
    )

    out = out.merge(
        times,
        how="left",
        left_on="subject_id",
        right_index=True
    )

    return out


# ============================================================
# D. Timestamp-specific 7-day dialysis reconstruction
# ============================================================

def rrt_status(subject_id, creat_time):

    if pd.isna(creat_time):
        return {
            "dialysis_treatment_days_7d": np.nan,
            "max_continuous_rrt_h_7d_primary": np.nan,
            "max_continuous_rrt_h_7d_strict": np.nan,
            "dialysis_flag_primary": np.nan,
            "dialysis_flag_strict": np.nan,
        }

    creat_time = pd.Timestamp(creat_time)

    lookback_start = (
        creat_time
        - pd.Timedelta(days=7)
    )

    treatment_dates = set()

    # --------------------------------------------------------
    # Exact ICU IHD treatment days
    # --------------------------------------------------------

    g = ihd_map.get(subject_id)

    if g is not None:

        z = g[
            (g["session_start"] >= lookback_start)
            & (g["session_start"] <= creat_time)
        ]

        treatment_dates.update(
            pd.Timestamp(t).date()
            for t in z["session_start"].dropna()
        )

    # --------------------------------------------------------
    # Exact ICU PD treatment days
    # --------------------------------------------------------

    g = pd_map.get(subject_id)

    if g is not None:

        z = g[
            (g["session_start"] >= lookback_start)
            & (g["session_start"] <= creat_time)
        ]

        treatment_dates.update(
            pd.Timestamp(t).date()
            for t in z["session_start"].dropna()
        )

    # --------------------------------------------------------
    # Hospital ICD date-level evidence
    #
    # Only dates STRICTLY before the creatinine calendar date
    # are temporally definite.
    #
    # Same-calendar-day time-unknown records are deliberately
    # excluded from the primary definition.
    # --------------------------------------------------------

    g = hospital_map.get(subject_id)

    if g is not None:

        start_date = lookback_start.date()
        creat_date = creat_time.date()

        z = g[
            g["treatment_class"].isin(
                HOSP_SESSION_CLASSES
            )
            & g["chartdate"].notna()
            & (
                g["chartdate"]
                >= start_date
            )
            & (
                g["chartdate"]
                < creat_date
            )
        ]

        treatment_dates.update(
            z["chartdate"].tolist()
        )

    n_treatment_days = len(
        treatment_dates
    )

    # --------------------------------------------------------
    # Continuous RRT:
    #
    # IMPORTANT:
    # evaluate actual overlap of EACH individual continuous
    # episode with the preceding 7-day window.
    #
    # Do NOT sum separated episodes.
    # --------------------------------------------------------

    max_primary_h = 0.0
    max_strict_h = 0.0

    g = cont_map.get(subject_id)

    if g is not None:

        z = g[
            (g["session_end"] > lookback_start)
            & (
                g["session_start"]
                < creat_time
            )
        ]

        for _, r in z.iterrows():

            start = max(
                pd.Timestamp(
                    r["session_start"]
                ),
                lookback_start
            )

            end = min(
                pd.Timestamp(
                    r["session_end"]
                ),
                creat_time
            )

            overlap_h = max(
                0.0,
                (
                    end - start
                ).total_seconds()
                / 3600.0
            )

            if int(
                r[
                    "primary_continuous_eligible"
                ]
            ) == 1:

                max_primary_h = max(
                    max_primary_h,
                    overlap_h
                )

            if int(
                r["strict_cvvhd_only"]
            ) == 1:

                max_strict_h = max(
                    max_strict_h,
                    overlap_h
                )

    dialysis_primary = int(
        (n_treatment_days >= 2)
        or
        (max_primary_h >= 24.0)
    )

    dialysis_strict = int(
        (n_treatment_days >= 2)
        or
        (max_strict_h >= 24.0)
    )

    return {
        "dialysis_treatment_days_7d":
            int(n_treatment_days),

        "max_continuous_rrt_h_7d_primary":
            float(max_primary_h),

        "max_continuous_rrt_h_7d_strict":
            float(max_strict_h),

        "dialysis_flag_primary":
            dialysis_primary,

        "dialysis_flag_strict":
            dialysis_strict,
    }


# ============================================================
# E. MELD 3.0 calculation
# ============================================================

def calculate_meld3(row, strict=False):

    required = [
        "albumin",
        "bilirubin",
        "creatinine",
        "sodium",
        "inr",
    ]

    if any(
        pd.isna(row.get(x))
        for x in required
    ):
        return np.nan

    # Official MELD 3.0 bounds
    albumin = min(
        max(
            float(row["albumin"]),
            1.5
        ),
        3.5
    )

    bilirubin = max(
        float(row["bilirubin"]),
        1.0
    )

    inr = max(
        float(row["inr"]),
        1.0
    )

    sodium = min(
        max(
            float(row["sodium"]),
            125.0
        ),
        137.0
    )

    dialysis_col = (
        "dialysis_flag_strict"
        if strict
        else "dialysis_flag_primary"
    )

    dialysis_positive = int(
        row[dialysis_col]
    )

    creatinine_raw = float(
        row["creatinine"]
    )

    if dialysis_positive == 1:

        creatinine = 3.0

    else:

        creatinine = min(
            max(
                creatinine_raw,
                1.0
            ),
            3.0
        )

    female = (
        1.0
        if str(
            row["gender"]
        ).upper() == "F"
        else 0.0
    )

    score = (
        1.33 * female

        + 4.56
        * math.log(
            bilirubin
        )

        + 0.82
        * (
            137.0
            - sodium
        )

        - 0.24
        * (
            137.0
            - sodium
        )
        * math.log(
            bilirubin
        )

        + 9.09
        * math.log(
            inr
        )

        + 11.14
        * math.log(
            creatinine
        )

        + 1.85
        * (
            3.5
            - albumin
        )

        - 1.83
        * (
            3.5
            - albumin
        )
        * math.log(
            creatinine
        )

        + 6.0
    )

    # Positive-score round-to-nearest integer.
    score = math.floor(
        score + 0.5
    )

    score = min(
        max(
            score,
            6
        ),
        40
    )

    return int(score)


# ============================================================
# F. Build all three snapshots
# ============================================================

all_snapshots = []

for (
    snapshot,
    start_h,
    end_h,
    anchor_h
) in SNAPSHOTS:

    print()
    print(
        f"Building snapshot: {snapshot}"
    )

    df = select_snapshot(
        snapshot,
        start_h,
        end_h,
        anchor_h
    )

    rrt_rows = [
        rrt_status(
            sid,
            creat_time
        )
        for sid, creat_time
        in zip(
            df["subject_id"],
            df["creatinine_time"]
        )
    ]

    rrt_df = pd.DataFrame(
        rrt_rows
    )

    df = pd.concat(
        [
            df.reset_index(drop=True),
            rrt_df
        ],
        axis=1
    )

    # Formal creatinine actually entering MELD.
    df[
        "creatinine_for_meld_primary"
    ] = np.where(

        df["creatinine"].notna(),

        np.where(
            df[
                "dialysis_flag_primary"
            ] == 1,

            3.0,

            df["creatinine"].clip(
                lower=1.0,
                upper=3.0
            )
        ),

        np.nan
    )

    df[
        "creatinine_for_meld_strict"
    ] = np.where(

        df["creatinine"].notna(),

        np.where(
            df[
                "dialysis_flag_strict"
            ] == 1,

            3.0,

            df["creatinine"].clip(
                lower=1.0,
                upper=3.0
            )
        ),

        np.nan
    )

    df["complete_five_labs"] = (
        df[
            [
                "albumin",
                "bilirubin",
                "creatinine",
                "sodium",
                "inr",
            ]
        ]
        .notna()
        .all(axis=1)
        .astype(int)
    )

    df["meld3_primary"] = df.apply(
        calculate_meld3,
        axis=1,
        strict=False
    )

    df["meld3_strict"] = df.apply(
        calculate_meld3,
        axis=1,
        strict=True
    )

    all_snapshots.append(
        df
    )


long = pd.concat(
    all_snapshots,
    ignore_index=True
)

long.to_parquet(
    OUT_LONG,
    index=False,
    compression="zstd"
)


# ============================================================
# G. Wide-format dataset for dynamic analysis
# ============================================================

keep = [
    "subject_id",
    "snapshot",
    "eligible",

    "albumin",
    "bilirubin",
    "creatinine",
    "sodium",
    "inr",

    "albumin_time",
    "bilirubin_time",
    "creatinine_time",
    "sodium_time",
    "inr_time",

    "complete_five_labs",

    "dialysis_treatment_days_7d",

    "max_continuous_rrt_h_7d_primary",
    "max_continuous_rrt_h_7d_strict",

    "dialysis_flag_primary",
    "dialysis_flag_strict",

    "creatinine_for_meld_primary",
    "creatinine_for_meld_strict",

    "meld3_primary",
    "meld3_strict",
]


wide = long[
    keep
].copy()

wide = wide.pivot(
    index="subject_id",
    columns="snapshot"
)

wide.columns = [
    f"{variable}_{snapshot}"
    for variable, snapshot
    in wide.columns
]

wide = wide.reset_index()


wide = cohort[
    [
        "subject_id",
        "hadm_id",
        "first_icu_intime",
        "dischtime",
        "hospital_expire_flag",
        "gender",
        "age_at_admission",
    ]
].merge(
    wide,
    on="subject_id",
    how="left"
)


# Dynamic change variables
wide[
    "delta_meld3_48_primary"
] = (
    wide["meld3_primary_h48"]
    - wide["meld3_primary_baseline"]
)

wide[
    "delta_meld3_72_primary"
] = (
    wide["meld3_primary_h72"]
    - wide["meld3_primary_baseline"]
)

wide[
    "delta_meld3_48_strict"
] = (
    wide["meld3_strict_h48"]
    - wide["meld3_strict_baseline"]
)

wide[
    "delta_meld3_72_strict"
] = (
    wide["meld3_strict_h72"]
    - wide["meld3_strict_baseline"]
)


wide.to_parquet(
    OUT_WIDE,
    index=False,
    compression="zstd"
)


# ============================================================
# H. Snapshot QC
# ============================================================

qc_rows = []

for snapshot in [
    "baseline",
    "h48",
    "h72",
]:

    g = long[
        long["snapshot"]
        == snapshot
    ]

    eligible = g[
        g["eligible"] == 1
    ]

    complete = eligible[
        eligible[
            "complete_five_labs"
        ] == 1
    ]

    qc_rows.append({
        "snapshot":
            snapshot,

        "n_total":
            len(g),

        "n_eligible":
            len(eligible),

        "n_complete_five_labs":
            len(complete),

        "complete_pct_of_eligible":
            round(
                100
                * len(complete)
                / len(eligible),
                2
            )
            if len(eligible)
            else np.nan,

        "n_dialysis_primary_complete":
            int(
                (
                    complete[
                        "dialysis_flag_primary"
                    ] == 1
                ).sum()
            ),

        "n_dialysis_strict_complete":
            int(
                (
                    complete[
                        "dialysis_flag_strict"
                    ] == 1
                ).sum()
            ),

        "meld3_primary_median":
            float(
                complete[
                    "meld3_primary"
                ].median()
            )
            if len(complete)
            else np.nan,

        "meld3_primary_q25":
            float(
                complete[
                    "meld3_primary"
                ].quantile(0.25)
            )
            if len(complete)
            else np.nan,

        "meld3_primary_q75":
            float(
                complete[
                    "meld3_primary"
                ].quantile(0.75)
            )
            if len(complete)
            else np.nan,
    })


qc = pd.DataFrame(
    qc_rows
)

qc.to_csv(
    OUT_QC,
    index=False
)


# ============================================================
# I. Paired baseline + update summary
# ============================================================

pair_rows = []

for snapshot in [
    "h48",
    "h72",
]:

    eligible_col = (
        f"eligible_{snapshot}"
    )

    baseline_col = (
        "meld3_primary_baseline"
    )

    update_col = (
        f"meld3_primary_{snapshot}"
    )

    strict_baseline_col = (
        "meld3_strict_baseline"
    )

    strict_update_col = (
        f"meld3_strict_{snapshot}"
    )

    paired = wide[
        (wide[eligible_col] == 1)
        & wide[baseline_col].notna()
        & wide[update_col].notna()
    ].copy()

    strict_n = (
        (
            (wide[eligible_col] == 1)
            & wide[
                strict_baseline_col
            ].notna()
            & wide[
                strict_update_col
            ].notna()
        )
        .sum()
    )

    pair_rows.append({
        "pair":
            f"baseline+{snapshot}",

        "n_paired_primary":
            len(paired),

        "hospital_deaths_primary":
            int(
                paired[
                    "hospital_expire_flag"
                ].sum()
            ),

        "mortality_pct_primary":
            round(
                100
                * paired[
                    "hospital_expire_flag"
                ].mean(),
                2
            )
            if len(paired)
            else np.nan,

        "median_baseline_meld3_primary":
            float(
                paired[
                    baseline_col
                ].median()
            )
            if len(paired)
            else np.nan,

        "median_updated_meld3_primary":
            float(
                paired[
                    update_col
                ].median()
            )
            if len(paired)
            else np.nan,

        "median_delta_meld3_primary":
            float(
                (
                    paired[
                        update_col
                    ]
                    - paired[
                        baseline_col
                    ]
                ).median()
            )
            if len(paired)
            else np.nan,

        "n_paired_strict":
            int(strict_n),
    })


pair = pd.DataFrame(
    pair_rows
)

pair.to_csv(
    OUT_PAIR,
    index=False
)


# ============================================================
# J. Console QC
# ============================================================

print()
print(
    "===== SNAPSHOT QC ====="
)

print(
    qc.to_string(
        index=False
    )
)


print()
print(
    "===== PAIRED SUMMARY ====="
)

print(
    pair.to_string(
        index=False
    )
)


print()
print("=" * 78)
print("CHECKPOINT 12 COMPLETE")
print("=" * 78)

print(
    f"Long snapshots : {OUT_LONG}"
)

print(
    f"Wide snapshots : {OUT_WIDE}"
)

print(
    f"Snapshot QC    : {OUT_QC}"
)

print(
    f"Paired summary : {OUT_PAIR}"
)

print()
print(
    "Primary MELD uses the prespecified MIMIC-to-CKRT mapping.\n"
    "Strict MELD uses the explicit CVVHD-only continuous-RRT\n"
    "sensitivity definition. No prognostic modeling is performed\n"
    "in this checkpoint."
)
