from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Checkpoint 13A
# 48-hour MELD 3.0 dynamic change + missingness/selection audit
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INFILE = ROOT / "work" / "meld3_snapshots_wide.parquet"

OUT_CHANGE = ROOT / "work" / "13a_h48_change_summary.csv"
OUT_EXACT = ROOT / "work" / "13a_h48_exact_delta_distribution.csv"
OUT_DIRECTION = ROOT / "work" / "13a_h48_change_direction_mortality.csv"
OUT_PATTERN = ROOT / "work" / "13a_h48_completeness_patterns.csv"
OUT_COMPONENT = ROOT / "work" / "13a_h48_component_availability.csv"
OUT_SELECTION = ROOT / "work" / "13a_h48_selection_comparison.csv"
OUT_RRT = ROOT / "work" / "13a_h48_rrt_sensitivity_summary.csv"
OUT_RRT_SUBJECTS = ROOT / "work" / "13a_h48_rrt_sensitivity_subjects.csv"
OUT_PRIMARY = ROOT / "work" / "h48_primary_analysis_cohort.parquet"


if not INFILE.exists():
    raise FileNotFoundError(INFILE)


df = pd.read_parquet(INFILE)

if len(df) != 4129:
    raise RuntimeError(
        f"Unexpected cohort size: {len(df)}"
    )

print("=" * 78)
print("CHECKPOINT 13A - 48H CHANGE + MISSINGNESS AUDIT")
print("=" * 78)

print(f"Full cohort: {len(df):,}")


# ============================================================
# A. 48-hour landmark risk set
# ============================================================

eligible = df[
    df["eligible_h48"] == 1
].copy()

eligible["baseline_complete"] = (
    eligible["meld3_primary_baseline"].notna()
).astype(int)

eligible["h48_complete"] = (
    eligible["meld3_primary_h48"].notna()
).astype(int)

eligible["paired"] = (
    (eligible["baseline_complete"] == 1)
    & (eligible["h48_complete"] == 1)
).astype(int)


print(f"48h eligible: {len(eligible):,}")
print(f"Paired complete: {eligible['paired'].sum():,}")


if len(eligible) != 3747:
    raise RuntimeError(
        f"Expected 3747 h48-eligible patients, found {len(eligible)}"
    )

if int(eligible["paired"].sum()) != 904:
    raise RuntimeError(
        f"Expected 904 paired patients, found {eligible['paired'].sum()}"
    )


# ============================================================
# B. Completeness pattern
# ============================================================

conditions = [
    (eligible["baseline_complete"] == 1)
    & (eligible["h48_complete"] == 1),

    (eligible["baseline_complete"] == 1)
    & (eligible["h48_complete"] == 0),

    (eligible["baseline_complete"] == 0)
    & (eligible["h48_complete"] == 1),
]

choices = [
    "baseline_complete+h48_complete",
    "baseline_complete+h48_missing",
    "baseline_missing+h48_complete",
]

eligible["completeness_pattern"] = np.select(
    conditions,
    choices,
    default="baseline_missing+h48_missing",
)


pattern = (
    eligible
    .groupby(
        "completeness_pattern",
        as_index=False
    )
    .agg(
        n=("subject_id", "size"),
        deaths=("hospital_expire_flag", "sum"),
        age_mean=("age_at_admission", "mean"),
        age_sd=("age_at_admission", "std"),
        female_n=("gender", lambda x: (x == "F").sum()),
    )
)

pattern["pct_of_h48_eligible"] = (
    100 * pattern["n"] / len(eligible)
)

pattern["mortality_pct"] = (
    100 * pattern["deaths"] / pattern["n"]
)

pattern["female_pct"] = (
    100 * pattern["female_n"] / pattern["n"]
)

for c in [
    "pct_of_h48_eligible",
    "mortality_pct",
    "female_pct",
    "age_mean",
    "age_sd",
]:
    pattern[c] = pattern[c].round(2)

pattern.to_csv(
    OUT_PATTERN,
    index=False
)


# ============================================================
# C. Component-level availability among 48h risk set
# ============================================================

components = [
    "albumin",
    "bilirubin",
    "creatinine",
    "sodium",
    "inr",
]

component_rows = []

for x in components:

    b = f"{x}_baseline"
    h = f"{x}_h48"

    component_rows.append({
        "component": x,

        "baseline_available_n":
            int(eligible[b].notna().sum()),

        "baseline_available_pct":
            round(
                100 * eligible[b].notna().mean(),
                2
            ),

        "h48_available_n":
            int(eligible[h].notna().sum()),

        "h48_available_pct":
            round(
                100 * eligible[h].notna().mean(),
                2
            ),
    })

component = pd.DataFrame(
    component_rows
)

component.to_csv(
    OUT_COMPONENT,
    index=False
)


# ============================================================
# D. Paired vs excluded selection audit
# ============================================================

paired = eligible[
    eligible["paired"] == 1
].copy()

excluded = eligible[
    eligible["paired"] == 0
].copy()


def pooled_smd_cont(x1, x0):

    x1 = pd.to_numeric(
        x1,
        errors="coerce"
    ).dropna()

    x0 = pd.to_numeric(
        x0,
        errors="coerce"
    ).dropna()

    if len(x1) < 2 or len(x0) < 2:
        return np.nan

    s = np.sqrt(
        (
            x1.var(ddof=1)
            + x0.var(ddof=1)
        ) / 2
    )

    if s == 0:
        return 0.0

    return (
        x1.mean()
        - x0.mean()
    ) / s


def smd_binary(p1, p0):

    p = (p1 + p0) / 2

    denom = np.sqrt(
        p * (1 - p)
    )

    if denom == 0:
        return 0.0

    return (
        p1 - p0
    ) / denom


female_paired = (
    paired["gender"] == "F"
).mean()

female_excluded = (
    excluded["gender"] == "F"
).mean()


selection_rows = [
    {
        "metric": "n",
        "paired": len(paired),
        "excluded": len(excluded),
        "smd": np.nan,
    },

    {
        "metric": "age_mean",
        "paired":
            round(
                paired["age_at_admission"].mean(),
                2
            ),
        "excluded":
            round(
                excluded["age_at_admission"].mean(),
                2
            ),
        "smd":
            round(
                pooled_smd_cont(
                    paired["age_at_admission"],
                    excluded["age_at_admission"]
                ),
                3
            ),
    },

    {
        "metric": "female_pct",
        "paired":
            round(
                100 * female_paired,
                2
            ),
        "excluded":
            round(
                100 * female_excluded,
                2
            ),
        "smd":
            round(
                smd_binary(
                    female_paired,
                    female_excluded
                ),
                3
            ),
    },

    {
        "metric": "hospital_mortality_pct",
        "paired":
            round(
                100
                * paired[
                    "hospital_expire_flag"
                ].mean(),
                2
            ),
        "excluded":
            round(
                100
                * excluded[
                    "hospital_expire_flag"
                ].mean(),
                2
            ),
        "smd": np.nan,
    },

    {
        "metric": "baseline_complete_pct",
        "paired":
            100.0,
        "excluded":
            round(
                100
                * excluded[
                    "baseline_complete"
                ].mean(),
                2
            ),
        "smd": np.nan,
    },

    {
        "metric": "h48_complete_pct",
        "paired":
            100.0,
        "excluded":
            round(
                100
                * excluded[
                    "h48_complete"
                ].mean(),
                2
            ),
        "smd": np.nan,
    },
]


# Among people with baseline MELD available,
# compare paired versus baseline-complete but update-missing.
baseline_only = eligible[
    (eligible["baseline_complete"] == 1)
    & (eligible["h48_complete"] == 0)
].copy()

if len(baseline_only):

    selection_rows.append({
        "metric":
            "baseline_meld_median_paired_vs_baseline_complete_h48_missing",

        "paired":
            float(
                paired[
                    "meld3_primary_baseline"
                ].median()
            ),

        "excluded":
            float(
                baseline_only[
                    "meld3_primary_baseline"
                ].median()
            ),

        "smd":
            round(
                pooled_smd_cont(
                    paired[
                        "meld3_primary_baseline"
                    ],
                    baseline_only[
                        "meld3_primary_baseline"
                    ]
                ),
                3
            ),
    })

    selection_rows.append({
        "metric":
            "mortality_pct_paired_vs_baseline_complete_h48_missing",

        "paired":
            round(
                100
                * paired[
                    "hospital_expire_flag"
                ].mean(),
                2
            ),

        "excluded":
            round(
                100
                * baseline_only[
                    "hospital_expire_flag"
                ].mean(),
                2
            ),

        "smd":
            np.nan,
    })


selection = pd.DataFrame(
    selection_rows
)

selection.to_csv(
    OUT_SELECTION,
    index=False
)


# ============================================================
# E. 48-hour MELD change distribution in paired cohort
# ============================================================

paired["delta_meld3"] = (
    paired["meld3_primary_h48"]
    - paired["meld3_primary_baseline"]
)


def direction(x):

    if x < 0:
        return "improved"

    if x > 0:
        return "worsened"

    return "unchanged"


paired["change_direction"] = (
    paired["delta_meld3"]
    .apply(direction)
)


change_summary = pd.DataFrame([{
    "n": len(paired),

    "deaths":
        int(
            paired[
                "hospital_expire_flag"
            ].sum()
        ),

    "mortality_pct":
        round(
            100
            * paired[
                "hospital_expire_flag"
            ].mean(),
            2
        ),

    "delta_mean":
        round(
            paired["delta_meld3"].mean(),
            2
        ),

    "delta_sd":
        round(
            paired["delta_meld3"].std(),
            2
        ),

    "delta_min":
        float(
            paired["delta_meld3"].min()
        ),

    "delta_p10":
        float(
            paired["delta_meld3"].quantile(0.10)
        ),

    "delta_q25":
        float(
            paired["delta_meld3"].quantile(0.25)
        ),

    "delta_median":
        float(
            paired["delta_meld3"].median()
        ),

    "delta_q75":
        float(
            paired["delta_meld3"].quantile(0.75)
        ),

    "delta_p90":
        float(
            paired["delta_meld3"].quantile(0.90)
        ),

    "delta_max":
        float(
            paired["delta_meld3"].max()
        ),
}])

change_summary.to_csv(
    OUT_CHANGE,
    index=False
)


exact = (
    paired
    .groupby(
        "delta_meld3",
        as_index=False
    )
    .agg(
        n=("subject_id", "size"),
        deaths=("hospital_expire_flag", "sum"),
    )
    .sort_values("delta_meld3")
)

exact["pct"] = (
    100 * exact["n"] / len(paired)
).round(2)

exact["mortality_pct"] = (
    100
    * exact["deaths"]
    / exact["n"]
).round(2)

exact.to_csv(
    OUT_EXACT,
    index=False
)


direction_summary = (
    paired
    .groupby(
        "change_direction",
        as_index=False
    )
    .agg(
        n=("subject_id", "size"),
        deaths=("hospital_expire_flag", "sum"),
        median_baseline_meld=(
            "meld3_primary_baseline",
            "median"
        ),
        median_h48_meld=(
            "meld3_primary_h48",
            "median"
        ),
        median_delta=(
            "delta_meld3",
            "median"
        ),
    )
)

direction_summary["pct"] = (
    100
    * direction_summary["n"]
    / len(paired)
).round(2)

direction_summary["mortality_pct"] = (
    100
    * direction_summary["deaths"]
    / direction_summary["n"]
).round(2)

direction_summary.to_csv(
    OUT_DIRECTION,
    index=False
)


# ============================================================
# F. Primary vs strict RRT sensitivity impact
# ============================================================

paired["baseline_primary_minus_strict"] = (
    paired["meld3_primary_baseline"]
    - paired["meld3_strict_baseline"]
)

paired["h48_primary_minus_strict"] = (
    paired["meld3_primary_h48"]
    - paired["meld3_strict_h48"]
)

paired["delta_strict"] = (
    paired["meld3_strict_h48"]
    - paired["meld3_strict_baseline"]
)

paired["delta_primary_minus_strict"] = (
    paired["delta_meld3"]
    - paired["delta_strict"]
)


rrt_summary = pd.DataFrame([{
    "n": len(paired),

    "baseline_score_diff_n":
        int(
            (
                paired[
                    "baseline_primary_minus_strict"
                ] != 0
            ).sum()
        ),

    "h48_score_diff_n":
        int(
            (
                paired[
                    "h48_primary_minus_strict"
                ] != 0
            ).sum()
        ),

    "delta_diff_n":
        int(
            (
                paired[
                    "delta_primary_minus_strict"
                ] != 0
            ).sum()
        ),

    "baseline_max_abs_diff":
        float(
            paired[
                "baseline_primary_minus_strict"
            ].abs().max()
        ),

    "h48_max_abs_diff":
        float(
            paired[
                "h48_primary_minus_strict"
            ].abs().max()
        ),
}])

rrt_summary.to_csv(
    OUT_RRT,
    index=False
)


rrt_subjects = paired[
    (
        paired[
            "baseline_primary_minus_strict"
        ] != 0
    )
    |
    (
        paired[
            "h48_primary_minus_strict"
        ] != 0
    )
][
    [
        "subject_id",
        "hospital_expire_flag",

        "meld3_primary_baseline",
        "meld3_strict_baseline",

        "meld3_primary_h48",
        "meld3_strict_h48",

        "delta_meld3",
        "delta_strict",

        "dialysis_flag_primary_baseline",
        "dialysis_flag_strict_baseline",

        "dialysis_flag_primary_h48",
        "dialysis_flag_strict_h48",
    ]
].copy()

rrt_subjects.to_csv(
    OUT_RRT_SUBJECTS,
    index=False
)


# ============================================================
# G. Freeze primary 48h analysis cohort
# ============================================================

paired.to_parquet(
    OUT_PRIMARY,
    index=False,
    compression="zstd"
)


# ============================================================
# H. Console output
# ============================================================

print()
print("===== COMPLETENESS PATTERNS =====")
print(
    pattern.to_string(
        index=False
    )
)


print()
print("===== COMPONENT AVAILABILITY =====")
print(
    component.to_string(
        index=False
    )
)


print()
print("===== PAIRED VS EXCLUDED =====")
print(
    selection.to_string(
        index=False
    )
)


print()
print("===== 48H MELD CHANGE SUMMARY =====")
print(
    change_summary.to_string(
        index=False
    )
)


print()
print("===== CHANGE DIRECTION + MORTALITY =====")
print(
    direction_summary.to_string(
        index=False
    )
)


print()
print("===== RRT DEFINITION SENSITIVITY =====")
print(
    rrt_summary.to_string(
        index=False
    )
)


print()
print("=" * 78)
print("CHECKPOINT 13A COMPLETE")
print("=" * 78)

print(f"Primary cohort  : {OUT_PRIMARY}")
print(f"Change summary  : {OUT_CHANGE}")
print(f"Exact delta     : {OUT_EXACT}")
print(f"Direction       : {OUT_DIRECTION}")
print(f"Patterns        : {OUT_PATTERN}")
print(f"Components      : {OUT_COMPONENT}")
print(f"Selection audit : {OUT_SELECTION}")
print(f"RRT sensitivity : {OUT_RRT}")
