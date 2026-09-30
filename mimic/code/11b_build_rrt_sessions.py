from pathlib import Path
from bisect import bisect_right

import duckdb
import pandas as pd


# ============================================================
# Checkpoint 11B
# RRT sessionization for MELD 3.0 reconstruction
#
# Inputs:
#   rrt_active_evidence.parquet
#
# Outputs:
#   rrt_ihd_sessions.parquet
#   rrt_pd_sessions.parquet
#   rrt_continuous_sessions.parquet
#   rrt_hospital_date_evidence.parquet
#   rrt_sessionization_qc.csv
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

EVID = ROOT / "work" / "rrt_active_evidence.parquet"
DB = ROOT / "work" / "mimic_feasibility.duckdb"

PROC_ICD = ROOT / "data" / "hosp" / "procedures_icd.csv.gz"
D_PROC_ICD = ROOT / "data" / "hosp" / "d_icd_procedures.csv.gz"

OUT_IHD = ROOT / "work" / "rrt_ihd_sessions.parquet"
OUT_PD = ROOT / "work" / "rrt_pd_sessions.parquet"
OUT_CONT = ROOT / "work" / "rrt_continuous_sessions.parquet"
OUT_HOSP = ROOT / "work" / "rrt_hospital_date_evidence.parquet"
OUT_QC = ROOT / "work" / "rrt_sessionization_qc.csv"

for p in [EVID, DB, PROC_ICD, D_PROC_ICD]:
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")

print("=" * 76)
print("CHECKPOINT 11B - RRT SESSIONIZATION")
print("=" * 76)

e = pd.read_parquet(EVID)

for col in [
    "event_time",
    "starttime",
    "endtime",
    "first_icu_intime",
]:
    if col in e.columns:
        e[col] = pd.to_datetime(e[col], errors="coerce")

print(f"Evidence rows loaded    : {len(e):,}")
print(f"Subjects with evidence  : {e['subject_id'].nunique():,}")


# ============================================================
# A. IHD
#
# Conservative primary rule:
# maximum one intermittent dialysis treatment session per
# subject per calendar date.
#
# This prevents repeated documentation of one treatment from
# being misclassified as >=2 dialysis treatments.
# ============================================================

ihd = e[
    (e["active_flag"] == 1)
    & (e["rrt_type"] == "IHD")
    & (e["source_table"].isin(["procedureevents", "chartevents"]))
].copy()

ihd["session_time"] = ihd["event_time"]

m = (
    (ihd["source_table"] == "procedureevents")
    & ihd["starttime"].notna()
)

ihd.loc[m, "session_time"] = ihd.loc[m, "starttime"]

ihd = ihd[ihd["session_time"].notna()].copy()
ihd["treatment_date"] = ihd["session_time"].dt.date

ihd["end_candidate"] = ihd["session_time"]

m = ihd["endtime"].notna()
ihd.loc[m, "end_candidate"] = ihd.loc[m, "endtime"]

ihd["procedure_evidence"] = (
    ihd["source_table"] == "procedureevents"
).astype(int)

ihd["chart_evidence"] = (
    ihd["source_table"] == "chartevents"
).astype(int)

ihd_sessions = (
    ihd.groupby(
        ["subject_id", "treatment_date"],
        as_index=False
    )
    .agg(
        session_start=("session_time", "min"),
        session_end=("end_candidate", "max"),
        n_evidence_rows=("session_time", "size"),
        n_procedure_rows=("procedure_evidence", "sum"),
        n_chart_points=("chart_evidence", "sum"),
        n_stays=("stay_id", "nunique"),
    )
    .sort_values(["subject_id", "session_start"])
)

ihd_sessions["session_number"] = (
    ihd_sessions.groupby("subject_id").cumcount() + 1
)

ihd_sessions["modality"] = "IHD"

ihd_sessions.to_parquet(
    OUT_IHD,
    index=False,
    compression="zstd"
)


# ============================================================
# B. Peritoneal dialysis
#
# Same conservative treatment-day approach.
# ============================================================

pd_ev = e[
    (e["active_flag"] == 1)
    & (e["rrt_type"] == "Peritoneal")
].copy()

pd_ev["session_time"] = pd_ev["event_time"]

m = (
    (pd_ev["source_table"] == "procedureevents")
    & pd_ev["starttime"].notna()
)

pd_ev.loc[m, "session_time"] = pd_ev.loc[m, "starttime"]

pd_ev = pd_ev[pd_ev["session_time"].notna()].copy()
pd_ev["treatment_date"] = pd_ev["session_time"].dt.date

pd_ev["end_candidate"] = pd_ev["session_time"]

m = pd_ev["endtime"].notna()
pd_ev.loc[m, "end_candidate"] = pd_ev.loc[m, "endtime"]

pd_ev["procedure_evidence"] = (
    pd_ev["source_table"] == "procedureevents"
).astype(int)

pd_ev["chart_evidence"] = (
    pd_ev["source_table"] == "chartevents"
).astype(int)

pd_sessions = (
    pd_ev.groupby(
        ["subject_id", "treatment_date"],
        as_index=False
    )
    .agg(
        session_start=("session_time", "min"),
        session_end=("end_candidate", "max"),
        n_evidence_rows=("session_time", "size"),
        n_procedure_rows=("procedure_evidence", "sum"),
        n_chart_points=("chart_evidence", "sum"),
        n_stays=("stay_id", "nunique"),
    )
    .sort_values(["subject_id", "session_start"])
)

pd_sessions["session_number"] = (
    pd_sessions.groupby("subject_id").cumcount() + 1
)

pd_sessions["modality"] = "Peritoneal"

pd_sessions.to_parquet(
    OUT_PD,
    index=False,
    compression="zstd"
)


# ============================================================
# C. Continuous RRT
#
# Empirical documentation pattern:
# active CRRT charting:
#   p50 = 1 h
#   p95 = 1 h
#   p99 = 2 h
#
# active system state:
#   p95 = 2 h
#   p99 = 4 h
#
# Therefore:
#   chart-only active runs may bridge <=4 h between active
#   observations, unless an explicit inactive state appears.
#
# Procedure start/end intervals are preserved as documented.
# Positive gaps between procedure intervals are NOT bridged
# merely because they are <4 h.
#
# SCUF is excluded from the primary MELD-equivalent CKRT rule.
# ============================================================

MAX_CHART_GAP = pd.Timedelta(hours=4)
CHART_TAIL = pd.Timedelta(hours=1)

chart_active = e[
    (e["source_table"] == "chartevents")
    & (e["active_flag"] == 1)
    & (e["rrt_type"] == "CRRT")
    & e["event_time"].notna()
][
    ["subject_id", "stay_id", "event_time"]
].drop_duplicates()

chart_active = chart_active.sort_values(
    ["subject_id", "stay_id", "event_time"]
)

inactive = e[
    (e["source_table"] == "chartevents")
    & (e["itemid"] == 224146)
    & (e["system_active"] == 0)
    & e["event_time"].notna()
][
    ["subject_id", "stay_id", "event_time"]
].drop_duplicates()

inactive_map = {}

for key, g in inactive.groupby(["subject_id", "stay_id"]):
    inactive_map[key] = sorted(
        pd.Timestamp(x) for x in g["event_time"]
    )


def first_inactive_after(times, t):
    if not times:
        return None

    i = bisect_right(times, t)

    if i >= len(times):
        return None

    return times[i]


def inactive_between(times, t1, t2):
    x = first_inactive_after(times, t1)
    return x is not None and x <= t2


def chart_run_end(inactive_times, last_active):
    """
    A last hourly active observation is conservatively allowed
    to represent up to one additional hour, unless an explicit
    inactive state occurs first.
    """
    tentative = last_active + CHART_TAIL
    stop = first_inactive_after(inactive_times, last_active)

    if stop is not None and stop < tentative:
        return stop

    return tentative


chart_runs = []

for (sid, stay_id), g in chart_active.groupby(
    ["subject_id", "stay_id"]
):
    times = sorted(
        pd.Timestamp(x)
        for x in g["event_time"].dropna().unique()
    )

    if not times:
        continue

    inactive_times = inactive_map.get(
        (sid, stay_id),
        []
    )

    run_start = times[0]
    prev = times[0]
    n_points = 1

    for t in times[1:]:
        gap_break = (t - prev) > MAX_CHART_GAP

        hard_break = inactive_between(
            inactive_times,
            prev,
            t
        )

        if gap_break or hard_break:
            chart_runs.append(
                {
                    "subject_id": sid,
                    "start": run_start,
                    "end": chart_run_end(
                        inactive_times,
                        prev
                    ),
                    "origin": "chart",
                    "n_chart_points": n_points,
                    "n_procedure_rows": 0,
                    "procedure_modes": set(),
                }
            )

            run_start = t
            n_points = 1

        else:
            n_points += 1

        prev = t

    chart_runs.append(
        {
            "subject_id": sid,
            "start": run_start,
            "end": chart_run_end(
                inactive_times,
                prev
            ),
            "origin": "chart",
            "n_chart_points": n_points,
            "n_procedure_rows": 0,
            "procedure_modes": set(),
        }
    )


# ------------------------------------------------------------
# Procedure-based CKRT intervals.
#
# Include:
#   generic CRRT
#   CVVHD
#   CVVH
#   CVVHDF
#
# Exclude:
#   SCUF
# ------------------------------------------------------------

eligible_modes = {
    "CRRT",
    "CVVHD",
    "CVVH",
    "CVVHDF",
}

proc = e[
    (e["source_table"] == "procedureevents")
    & (e["active_flag"] == 1)
    & (e["rrt_type"].isin(eligible_modes))
    & e["starttime"].notna()
    & e["endtime"].notna()
    & (e["endtime"] >= e["starttime"])
].copy()

proc_intervals = []

for _, r in proc.iterrows():
    proc_modes = set()

    if r["rrt_type"] != "CRRT":
        proc_modes.add(str(r["rrt_type"]))

    proc_intervals.append(
        {
            "subject_id": r["subject_id"],
            "start": pd.Timestamp(r["starttime"]),
            "end": pd.Timestamp(r["endtime"]),
            "origin": "procedure",
            "n_chart_points": 0,
            "n_procedure_rows": 1,
            "procedure_modes": proc_modes,
        }
    )


# ------------------------------------------------------------
# Merge only OVERLAPPING or TOUCHING evidence intervals.
#
# We do NOT bridge a positive undocumented gap between two
# procedure intervals.
#
# Chart runs have already used the empirical 4-h charting rule.
# ------------------------------------------------------------

all_pieces = chart_runs + proc_intervals

all_pieces.sort(
    key=lambda x: (
        x["subject_id"],
        x["start"],
        x["end"]
    )
)

merged_sessions = []

current_sid = None
current = None

for x in all_pieces:
    sid = x["subject_id"]

    if current is None or sid != current_sid:
        if current is not None:
            merged_sessions.append(current)

        current_sid = sid

        current = {
            "subject_id": sid,
            "start": x["start"],
            "end": x["end"],
            "origins": {x["origin"]},
            "n_chart_points": x["n_chart_points"],
            "n_procedure_rows": x["n_procedure_rows"],
            "procedure_modes": set(x["procedure_modes"]),
        }

        continue

    if x["start"] <= current["end"]:
        current["end"] = max(
            current["end"],
            x["end"]
        )

        current["origins"].add(x["origin"])

        current["n_chart_points"] += x["n_chart_points"]
        current["n_procedure_rows"] += x["n_procedure_rows"]

        current["procedure_modes"].update(
            x["procedure_modes"]
        )

    else:
        merged_sessions.append(current)

        current = {
            "subject_id": sid,
            "start": x["start"],
            "end": x["end"],
            "origins": {x["origin"]},
            "n_chart_points": x["n_chart_points"],
            "n_procedure_rows": x["n_procedure_rows"],
            "procedure_modes": set(x["procedure_modes"]),
        }

if current is not None:
    merged_sessions.append(current)


cont_rows = []

for x in merged_sessions:
    cont_rows.append(
        {
            "subject_id": x["subject_id"],
            "session_start": x["start"],
            "session_end": x["end"],
            "procedure_supported":
                int("procedure" in x["origins"]),
            "chart_supported":
                int("chart" in x["origins"]),
            "n_chart_points": x["n_chart_points"],
            "n_procedure_rows": x["n_procedure_rows"],
            "procedure_modes":
                "|".join(sorted(x["procedure_modes"]))
                if x["procedure_modes"]
                else "",
        }
    )

cont = pd.DataFrame(cont_rows)

cont["duration_h"] = (
    cont["session_end"] - cont["session_start"]
).dt.total_seconds() / 3600.0


# ============================================================
# D. Add documented modality observations.
# ============================================================

mode_df = e[
    (e["source_table"] == "chartevents")
    & (e["itemid"] == 227290)
    & e["event_time"].notna()
    & e["value"].notna()
][
    ["subject_id", "event_time", "value"]
].copy()

mode_map = {
    sid: g.sort_values("event_time")
    for sid, g in mode_df.groupby("subject_id")
}

input_crrt = e[
    (e["source_table"] == "inputevents")
    & (e["active_flag"] == 1)
    & (e["rrt_type"] == "CRRT")
    & e["starttime"].notna()
    & e["endtime"].notna()
][
    ["subject_id", "starttime", "endtime"]
].copy()

input_map = {
    sid: g
    for sid, g in input_crrt.groupby("subject_id")
}

documented_modes = []
has_cvvhd = []
has_cvvhdf = []
has_cvvh = []
has_scuf = []
strict_cvvhd_only = []
primary_eligible = []
input_overlap_count = []

for _, r in cont.iterrows():
    sid = r["subject_id"]
    start = r["session_start"]
    end = r["session_end"]

    modes = set()

    if r["procedure_modes"]:
        modes.update(
            x for x in r["procedure_modes"].split("|")
            if x
        )

    mg = mode_map.get(sid)

    if mg is not None:
        z = mg[
            (mg["event_time"] >= start)
            & (mg["event_time"] <= end)
        ]

        modes.update(
            str(x)
            for x in z["value"].dropna()
        )

    documented_modes.append(
        "|".join(sorted(modes))
        if modes else "UNSPECIFIED"
    )

    has_cvvhd.append(int("CVVHD" in modes))
    has_cvvhdf.append(int("CVVHDF" in modes))
    has_cvvh.append(int("CVVH" in modes))
    has_scuf.append(int("SCUF" in modes))

    # Strict sensitivity:
    # every documented modality in the episode is CVVHD.
    strict_cvvhd_only.append(
        int(len(modes) > 0 and modes == {"CVVHD"})
    )

    # Primary CKRT definition:
    # SCUF-only treatment is not considered equivalent.
    primary_eligible.append(
        int(modes != {"SCUF"})
    )

    ig = input_map.get(sid)

    if ig is None:
        input_overlap_count.append(0)

    else:
        n = (
            (ig["starttime"] <= end)
            & (ig["endtime"] >= start)
        ).sum()

        input_overlap_count.append(int(n))

cont["documented_modes"] = documented_modes
cont["has_cvvhd"] = has_cvvhd
cont["has_cvvhdf"] = has_cvvhdf
cont["has_cvvh"] = has_cvvh
cont["has_scuf"] = has_scuf

cont["strict_cvvhd_only"] = strict_cvvhd_only
cont["primary_continuous_eligible"] = primary_eligible
cont["n_inputevent_overlaps"] = input_overlap_count

cont["meets_24h_primary_episode"] = (
    (cont["duration_h"] >= 24)
    & (cont["primary_continuous_eligible"] == 1)
).astype(int)

cont["meets_24h_strict_cvvhd_episode"] = (
    (cont["duration_h"] >= 24)
    & (cont["strict_cvvhd_only"] == 1)
).astype(int)

cont = cont.sort_values(
    ["subject_id", "session_start"]
)

cont["session_number"] = (
    cont.groupby("subject_id").cumcount() + 1
)

cont.to_parquet(
    OUT_CONT,
    index=False,
    compression="zstd"
)


# ============================================================
# E. Hospital date-level RRT evidence
#
# This remains supplemental evidence.
# A date-level ICD record is NOT automatically interpreted as
# an exact ICU dialysis session.
# ============================================================

con = duckdb.connect(
    str(DB),
    read_only=True
)

hospital = con.execute(f"""
WITH treatment_codes(
    icd_version,
    icd_code,
    treatment_class
) AS (
    VALUES
        (9,  '3995',    'IHD'),
        (9,  '5498',    'Peritoneal'),
        (10, '5A1D00Z', 'UrinaryFiltrationSingle'),
        (10, '5A1D60Z', 'UrinaryFiltrationMultiple'),
        (10, '5A1D70Z', 'Intermittent'),
        (10, '5A1D80Z', 'ProlongedIntermittent'),
        (10, '5A1D90Z', 'ContinuousOver18h')
),

p AS (
    SELECT
        TRY_CAST(subject_id AS BIGINT) AS subject_id,
        TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
        TRY_CAST(chartdate AS DATE) AS chartdate,
        icd_code,
        TRY_CAST(icd_version AS INTEGER) AS icd_version
    FROM read_csv_auto(
        '{str(PROC_ICD)}',
        header=true,
        all_varchar=true
    )
),

d AS (
    SELECT
        icd_code,
        TRY_CAST(icd_version AS INTEGER) AS icd_version,
        long_title
    FROM read_csv_auto(
        '{str(D_PROC_ICD)}',
        header=true,
        all_varchar=true
    )
)

SELECT DISTINCT
    c.subject_id,
    c.hadm_id AS cohort_hadm_id,
    c.first_icu_intime,

    p.hadm_id AS source_hadm_id,
    p.chartdate,
    p.icd_version,
    p.icd_code,

    t.treatment_class,
    d.long_title,

    CASE
        WHEN p.hadm_id = c.hadm_id THEN 1
        ELSE 0
    END AS within_cohort_hadm,

    CASE
        WHEN p.chartdate < CAST(c.first_icu_intime AS DATE)
            THEN 'definite_pre_icu_date'

        WHEN p.chartdate = CAST(c.first_icu_intime AS DATE)
            THEN 'same_calendar_day_time_unknown'

        ELSE 'post_icu_date'
    END AS timing_class

FROM cirrhosis_first_icu_admission c

INNER JOIN p
    ON c.subject_id = p.subject_id

INNER JOIN treatment_codes t
    ON p.icd_version = t.icd_version
   AND p.icd_code = t.icd_code

LEFT JOIN d
    ON p.icd_version = d.icd_version
   AND p.icd_code = d.icd_code

WHERE
    p.chartdate >=
        CAST(c.first_icu_intime AS DATE) - INTERVAL '8 days'

AND
    p.chartdate <=
        CAST(c.first_icu_intime AS DATE) + INTERVAL '3 days'

ORDER BY
    c.subject_id,
    p.chartdate,
    p.icd_version,
    p.icd_code
""").fetchdf()

con.close()

hospital.to_parquet(
    OUT_HOSP,
    index=False,
    compression="zstd"
)


# ============================================================
# F. QC
# ============================================================

qc = []


def add_qc(metric, value):
    qc.append(
        {
            "metric": metric,
            "value": value
        }
    )


add_qc(
    "ihd_sessions",
    len(ihd_sessions)
)

add_qc(
    "ihd_subjects",
    ihd_sessions["subject_id"].nunique()
)

ihd_n = (
    ihd_sessions.groupby("subject_id")
    .size()
)

add_qc(
    "ihd_subjects_with_2plus_treatment_days",
    int((ihd_n >= 2).sum())
)

add_qc(
    "pd_sessions",
    len(pd_sessions)
)

add_qc(
    "pd_subjects",
    pd_sessions["subject_id"].nunique()
)

add_qc(
    "continuous_sessions",
    len(cont)
)

add_qc(
    "continuous_subjects",
    cont["subject_id"].nunique()
)

add_qc(
    "continuous_procedure_supported_sessions",
    int(cont["procedure_supported"].sum())
)

add_qc(
    "continuous_chart_only_sessions",
    int(
        (
            (cont["procedure_supported"] == 0)
            & (cont["chart_supported"] == 1)
        ).sum()
    )
)

add_qc(
    "continuous_sessions_ge24h_primary",
    int(cont["meets_24h_primary_episode"].sum())
)

add_qc(
    "continuous_subjects_ge24h_primary",
    cont.loc[
        cont["meets_24h_primary_episode"] == 1,
        "subject_id"
    ].nunique()
)

add_qc(
    "continuous_sessions_ge24h_strict_cvvhd",
    int(cont["meets_24h_strict_cvvhd_episode"].sum())
)

add_qc(
    "continuous_subjects_ge24h_strict_cvvhd",
    cont.loc[
        cont["meets_24h_strict_cvvhd_episode"] == 1,
        "subject_id"
    ].nunique()
)

add_qc(
    "hospital_date_evidence_rows",
    len(hospital)
)

add_qc(
    "hospital_date_evidence_subjects",
    hospital["subject_id"].nunique()
)

qc_df = pd.DataFrame(qc)

qc_df.to_csv(
    OUT_QC,
    index=False
)


print()
print("===== SESSIONIZATION QC =====")
print(qc_df.to_string(index=False))


print()
print("===== CONTINUOUS DURATION DISTRIBUTION =====")

print(
    cont["duration_h"]
    .describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
    .round(2)
    .to_string()
)


print()
print("===== LONGEST CONTINUOUS EPISODES =====")

print(
    cont[
        [
            "subject_id",
            "session_start",
            "session_end",
            "duration_h",
            "procedure_supported",
            "chart_supported",
            "documented_modes",
            "meets_24h_primary_episode",
        ]
    ]
    .sort_values(
        "duration_h",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


print()
print("=" * 76)
print("CHECKPOINT 11B SESSIONIZATION COMPLETE")
print("=" * 76)

print(f"IHD sessions        : {OUT_IHD}")
print(f"PD sessions         : {OUT_PD}")
print(f"Continuous sessions : {OUT_CONT}")
print(f"Hospital evidence   : {OUT_HOSP}")
print(f"QC                  : {OUT_QC}")

print()
print(
    "IMPORTANT:\n"
    "Formal MELD 3.0 dialysis status has NOT yet been assigned.\n"
    "Checkpoint 12 will apply a patient- and timestamp-specific\n"
    "7-day lookback to the selected creatinine measurement."
)
