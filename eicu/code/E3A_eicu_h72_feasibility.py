#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Checkpoint E3A (corrected)
eICU-CRD v2.0: 72-hour secondary landmark feasibility

Important correction:
In E1C_harmonized_first_icu_per_hospitalization.csv,
baseline_albumin / baseline_bilirubin / baseline_creatinine /
baseline_sodium / baseline_inr are 0/1 availability indicators,
NOT raw laboratory values.

Therefore:
- baseline completeness is inherited from the locked E1C baseline_complete flag;
- baseline component coverage is calculated from the 0/1 indicators;
- h72 laboratory values are newly extracted as actual labresult values;
- h72 completeness is defined by all five h72 values being non-missing.
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path.home() / "Desktop" / "eICU_CRD_2_0"
WORK = ROOT / "work"
DATA = ROOT / "data"

COHORT_FILE = WORK / "E1C_harmonized_first_icu_per_hospitalization.csv"
LAB_FILE = DATA / "lab.csv.gz"

OUT_CANDIDATES = WORK / "E3A_h72_candidate_stays.parquet"
OUT_SUMMARY = WORK / "E3A_h72_feasibility_summary.csv"
OUT_COVERAGE = WORK / "E3A_h72_component_coverage.csv"
OUT_PAIRED = WORK / "E3A_h72_paired_candidate_stays.csv"
OUT_QC = WORK / "E3A_h72_qc.csv"

LANDMARK_MIN = 4320
H72_LOWER = 2880
H72_UPPER = 4320

LABNAME_TO_COMPONENT = {
    "albumin": "albumin",
    "total bilirubin": "bilirubin",
    "creatinine": "creatinine",
    "sodium": "sodium",
    "pt - inr": "inr",
}

BASELINE_FLAG_COLS = {
    "albumin": "baseline_albumin",
    "bilirubin": "baseline_bilirubin",
    "creatinine": "baseline_creatinine",
    "sodium": "baseline_sodium",
    "inr": "baseline_inr",
}

H72_VALUE_COLS = {
    "albumin": "h72_albumin",
    "bilirubin": "h72_bilirubin",
    "creatinine": "h72_creatinine",
    "sodium": "h72_sodium",
    "inr": "h72_inr",
}


def die(msg: str) -> None:
    raise SystemExit(msg)


def pct(n: int, d: int) -> float:
    return np.nan if d == 0 else 100.0 * n / d


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def summarize_risk_set(df: pd.DataFrame, risk_col: str, label: str) -> dict:
    rs = df.loc[df[risk_col]].copy()
    paired_col = f"{label}_paired_complete"
    paired = rs.loc[rs[paired_col]].copy()

    deaths = int(num(rs["hospital_death"]).fillna(0).sum())
    paired_deaths = int(num(paired["hospital_death"]).fillna(0).sum())

    return {
        "risk_set": label,
        "n_stays": int(rs["patientunitstayid"].nunique()),
        "unique_patients": int(rs["uniquepid"].nunique()),
        "health_system_stays": int(rs["patienthealthsystemstayid"].nunique()),
        "hospitals": int(rs["hospitalid"].nunique()),
        "hospital_deaths": deaths,
        "mortality_pct": pct(deaths, len(rs)),
        "baseline_complete_n": int(rs["baseline_complete_locked"].sum()),
        "baseline_complete_pct": pct(int(rs["baseline_complete_locked"].sum()), len(rs)),
        "h72_complete_n": int(rs["h72_complete"].sum()),
        "h72_complete_pct": pct(int(rs["h72_complete"].sum()), len(rs)),
        "paired_complete_n": int(rs[paired_col].sum()),
        "paired_complete_pct": pct(int(rs[paired_col].sum()), len(rs)),
        "paired_unique_patients": int(paired["uniquepid"].nunique()),
        "paired_hospitals": int(paired["hospitalid"].nunique()),
        "paired_deaths": paired_deaths,
        "paired_mortality_pct": pct(paired_deaths, len(paired)),
    }


def component_coverage(df: pd.DataFrame, risk_col: str, label: str) -> list[dict]:
    rs = df.loc[df[risk_col]].copy()
    rows = []

    for comp in ["albumin", "bilirubin", "creatinine", "sodium", "inr"]:
        bflag = BASELINE_FLAG_COLS[comp]
        hval = H72_VALUE_COLS[comp]

        baseline_available = num(rs[bflag]).eq(1)
        h72_available = rs[hval].notna()

        bn = int(baseline_available.sum())
        hn = int(h72_available.sum())

        rows.append({
            "risk_set": label,
            "component": comp,
            "n_risk_set": len(rs),
            "baseline_available_n": bn,
            "baseline_available_pct": pct(bn, len(rs)),
            "h72_available_n": hn,
            "h72_available_pct": pct(hn, len(rs)),
        })

    return rows


def main() -> None:
    if not COHORT_FILE.exists():
        die(f"ERROR: cohort file not found: {COHORT_FILE}")
    if not LAB_FILE.exists():
        die(f"ERROR: lab file not found: {LAB_FILE}")

    WORK.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(COHORT_FILE, low_memory=False)

    required = [
        "patientunitstayid",
        "patienthealthsystemstayid",
        "uniquepid",
        "hospitalid",
        "hospitaldischargeoffset",
        "unitdischargeoffset",
        "hospital_death",
        "baseline_complete",
        *BASELINE_FLAG_COLS.values(),
    ]
    missing = [c for c in required if c not in cohort.columns]
    if missing:
        die("ERROR: required cohort columns missing: " + ", ".join(missing))

    for c in [
        "patientunitstayid",
        "patienthealthsystemstayid",
        "hospitalid",
        "hospitaldischargeoffset",
        "unitdischargeoffset",
        "hospital_death",
        "baseline_complete",
        *BASELINE_FLAG_COLS.values(),
    ]:
        cohort[c] = num(cohort[c])

    cohort["uniquepid"] = cohort["uniquepid"].astype("string")

    # Locked baseline completeness from E1C.
    cohort["baseline_complete_locked"] = cohort["baseline_complete"].eq(1)

    # Independent audit using the five 0/1 availability flags.
    cohort["baseline_complete_from_flags"] = pd.concat(
        [cohort[c].eq(1) for c in BASELINE_FLAG_COLS.values()],
        axis=1
    ).all(axis=1)

    unit_ids = set(
        cohort["patientunitstayid"].dropna().astype("int64").tolist()
    )

    usecols = [
        "patientunitstayid",
        "labresultoffset",
        "labname",
        "labresult",
    ]

    filtered_chunks = []
    chunksize = 1_000_000

    print("Reading lab.csv.gz in chunks and extracting 48-72 h MELD labs ...")

    for i, chunk in enumerate(
        pd.read_csv(
            LAB_FILE,
            compression="gzip",
            usecols=usecols,
            chunksize=chunksize,
            low_memory=False,
        ),
        start=1,
    ):
        chunk["patientunitstayid"] = num(chunk["patientunitstayid"])
        chunk["labresultoffset"] = num(chunk["labresultoffset"])
        chunk["labresult"] = num(chunk["labresult"])
        chunk["labname_norm"] = (
            chunk["labname"].astype("string").str.strip().str.lower()
        )

        keep = (
            chunk["patientunitstayid"].isin(unit_ids)
            & chunk["labresultoffset"].gt(H72_LOWER)
            & chunk["labresultoffset"].le(H72_UPPER)
            & chunk["labname_norm"].isin(LABNAME_TO_COMPONENT)
            & chunk["labresult"].notna()
        )

        sub = chunk.loc[
            keep,
            ["patientunitstayid", "labresultoffset", "labname_norm", "labresult"],
        ].copy()

        if not sub.empty:
            sub["component"] = sub["labname_norm"].map(LABNAME_TO_COMPONENT)
            filtered_chunks.append(
                sub[["patientunitstayid", "labresultoffset", "component", "labresult"]]
            )

        if i % 5 == 0:
            print(f"  processed {i:,} chunks")

    if filtered_chunks:
        labs = pd.concat(filtered_chunks, ignore_index=True)
        labs["patientunitstayid"] = labs["patientunitstayid"].astype("int64")

        labs = labs.sort_values(
            ["patientunitstayid", "component", "labresultoffset"],
            kind="mergesort",
        )

        latest = (
            labs.groupby(["patientunitstayid", "component"], as_index=False)
            .tail(1)
        )

        wide = latest.pivot(
            index="patientunitstayid",
            columns="component",
            values="labresult",
        ).reset_index()

        wide = wide.rename(
            columns={comp: H72_VALUE_COLS[comp] for comp in H72_VALUE_COLS}
        )

        offset_wide = latest.pivot(
            index="patientunitstayid",
            columns="component",
            values="labresultoffset",
        ).reset_index()

        offset_wide = offset_wide.rename(
            columns={comp: f"h72_{comp}_offset" for comp in H72_VALUE_COLS}
        )

        cohort = cohort.merge(wide, on="patientunitstayid", how="left")
        cohort = cohort.merge(offset_wide, on="patientunitstayid", how="left")
    else:
        labs = pd.DataFrame(
            columns=["patientunitstayid", "labresultoffset", "component", "labresult"]
        )
        for c in H72_VALUE_COLS.values():
            cohort[c] = np.nan
        for comp in H72_VALUE_COLS:
            cohort[f"h72_{comp}_offset"] = np.nan

    for c in H72_VALUE_COLS.values():
        if c not in cohort.columns:
            cohort[c] = np.nan

    cohort["h72_complete"] = cohort[list(H72_VALUE_COLS.values())].notna().all(axis=1)

    cohort["hospital72_risk"] = cohort["hospitaldischargeoffset"].gt(LANDMARK_MIN)
    cohort["icu72_risk"] = cohort["unitdischargeoffset"].gt(LANDMARK_MIN)

    cohort["hospital72_paired_complete"] = (
        cohort["hospital72_risk"]
        & cohort["baseline_complete_locked"]
        & cohort["h72_complete"]
    )
    cohort["icu72_paired_complete"] = (
        cohort["icu72_risk"]
        & cohort["baseline_complete_locked"]
        & cohort["h72_complete"]
    )

    summary = pd.DataFrame([
        summarize_risk_set(cohort, "hospital72_risk", "hospital72"),
        summarize_risk_set(cohort, "icu72_risk", "icu72"),
    ])

    coverage = pd.DataFrame(
        component_coverage(cohort, "hospital72_risk", "hospital72")
        + component_coverage(cohort, "icu72_risk", "icu72")
    )

    paired_exports = []
    for label in ["hospital72", "icu72"]:
        pcol = f"{label}_paired_complete"
        tmp = cohort.loc[cohort[pcol]].copy()
        tmp.insert(0, "risk_set", label)
        paired_exports.append(tmp)

    paired = pd.concat(paired_exports, ignore_index=True)

    qc_rows = []

    def add_qc(check: str, value, expected, passed: bool, note: str = ""):
        qc_rows.append({
            "check": check,
            "value": value,
            "expected": expected,
            "pass": bool(passed),
            "note": note,
        })

    add_qc("cohort_rows", int(len(cohort)), ">=1", len(cohort) >= 1)

    dup = int(cohort["patientunitstayid"].duplicated().sum())
    add_qc(
        "duplicate_patientunitstayid_rows",
        dup,
        0,
        dup == 0,
    )

    # Confirm that the five E1C baseline_* columns are binary availability flags.
    baseline_flag_valid = all(
        set(num(cohort[c]).dropna().unique()).issubset({0, 1})
        for c in BASELINE_FLAG_COLS.values()
    )
    add_qc(
        "baseline_component_columns_are_binary_flags",
        baseline_flag_valid,
        True,
        baseline_flag_valid,
    )

    mismatch = int(
        (cohort["baseline_complete_locked"] != cohort["baseline_complete_from_flags"]).sum()
    )
    add_qc(
        "baseline_complete_vs_five_flags_mismatch",
        mismatch,
        0,
        mismatch == 0,
        "E1C baseline_complete should equal all five baseline availability flags == 1.",
    )

    add_qc(
        "h72_lab_offset_min_gt_2880",
        float(labs["labresultoffset"].min()) if len(labs) else np.nan,
        ">2880",
        True if len(labs) == 0 else labs["labresultoffset"].min() > H72_LOWER,
    )

    add_qc(
        "h72_lab_offset_max_le_4320",
        float(labs["labresultoffset"].max()) if len(labs) else np.nan,
        "<=4320",
        True if len(labs) == 0 else labs["labresultoffset"].max() <= H72_UPPER,
    )

    hospital_leak = int(
        (
            cohort["hospital72_paired_complete"]
            & cohort["hospitaldischargeoffset"].le(LANDMARK_MIN)
        ).sum()
    )
    add_qc(
        "hospital72_landmark_leakage",
        hospital_leak,
        0,
        hospital_leak == 0,
    )

    icu_leak = int(
        (
            cohort["icu72_paired_complete"]
            & cohort["unitdischargeoffset"].le(LANDMARK_MIN)
        ).sum()
    )
    add_qc(
        "icu72_landmark_leakage",
        icu_leak,
        0,
        icu_leak == 0,
    )

    death_before = int(
        (
            cohort["hospital72_paired_complete"]
            & cohort["hospital_death"].eq(1)
            & cohort["hospitaldischargeoffset"].le(LANDMARK_MIN)
        ).sum()
    )
    add_qc(
        "hospital72_death_before_landmark_in_paired",
        death_before,
        0,
        death_before == 0,
    )

    qc = pd.DataFrame(qc_rows)

    cohort.to_parquet(OUT_CANDIDATES, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    coverage.to_csv(OUT_COVERAGE, index=False)
    paired.to_csv(OUT_PAIRED, index=False)
    qc.to_csv(OUT_QC, index=False)

    print()
    print("===== 72H FEASIBILITY SUMMARY =====")
    print(summary.to_string(index=False))

    print()
    print("===== COMPONENT COVERAGE: HOSPITAL 72H =====")
    print(
        coverage.loc[coverage["risk_set"].eq("hospital72")]
        .to_string(index=False)
    )

    print()
    print("===== COMPONENT COVERAGE: ICU 72H =====")
    print(
        coverage.loc[coverage["risk_set"].eq("icu72")]
        .to_string(index=False)
    )

    print()
    print("===== 72H QC =====")
    print(qc.to_string(index=False))

    print()
    print("Saved:")
    for p in [
        OUT_CANDIDATES,
        OUT_SUMMARY,
        OUT_COVERAGE,
        OUT_PAIRED,
        OUT_QC,
    ]:
        print(f"  {p}")

    if not bool(qc["pass"].all()):
        print()
        print("WARNING: At least one hard QC check failed. Do not proceed to E3B.")
        sys.exit(2)

    print()
    print("E3A completed with all hard QC checks PASS.")


if __name__ == "__main__":
    main()
