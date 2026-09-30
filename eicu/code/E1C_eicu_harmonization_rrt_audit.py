from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


# ============================================================
# eICU Checkpoint E1C
#
# 1. Cohort harmonization:
#    - repeated ICU stays within one hospitalization
#    - repeated hospitalizations within one patient
#
# 2. RRT reconstructability:
#    - treatment-offset evidence
#    - definitive active RRT vs access/supportive evidence
#    - APACHE dialysis flag as auxiliary evidence only
#
# NO MELD 3.0 score is calculated here.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WORK = ROOT / "work"

STRICT_FILE = WORK / "E1B_strict_adult_cirrhosis_stays.csv"
FLAGS_FILE = WORK / "E1B_lab_window_flags.parquet"

TREATMENT_FILE = DATA / "treatment.csv.gz"
APACHE_FILE = DATA / "apacheApsVar.csv.gz"

OUT_STRUCTURE = WORK / "E1C_cohort_structure.csv"
OUT_HARMONIZED = WORK / "E1C_harmonized_first_icu_per_hospitalization.csv"
OUT_RRT_EVENTS = WORK / "E1C_rrt_events.parquet"
OUT_RRT_SUMMARY = WORK / "E1C_rrt_reconstructability_summary.csv"
OUT_RRT_PATIENT = WORK / "E1C_paired_rrt_subject_audit.csv"


for p in [
    STRICT_FILE,
    FLAGS_FILE,
    TREATMENT_FILE,
    APACHE_FILE,
]:
    if not p.exists():
        raise FileNotFoundError(p)


# ============================================================
# 1. Load E1B outputs
# ============================================================

strict = pd.read_csv(
    STRICT_FILE
)

flags = pd.read_parquet(
    FLAGS_FILE
)

x = strict.merge(
    flags,
    on="patientunitstayid",
    how="left"
)


x["hospital_death"] = (
    x["hospitaldischargestatus"]
    .astype(str)
    .str.lower()
    .eq("expired")
    .astype(int)
)


x["unitvisitnumber_num"] = (
    pd.to_numeric(
        x["unitvisitnumber"],
        errors="coerce"
    )
)


# ============================================================
# 2. First qualifying cirrhosis ICU stay
#    within each hospitalization
#
# We DO NOT pretend to know chronology across separate
# hospital admissions belonging to one uniquepid.
# ============================================================

harm = (
    x.sort_values(
        [
            "patienthealthsystemstayid",
            "unitvisitnumber_num",
            "patientunitstayid",
        ],
        na_position="last"
    )
    .drop_duplicates(
        subset=[
            "patienthealthsystemstayid"
        ],
        keep="first"
    )
    .copy()
)


harm.to_csv(
    OUT_HARMONIZED,
    index=False
)


# ============================================================
# 3. Repeated-admission structure
# ============================================================

episodes_per_patient = (
    harm.groupby(
        "uniquepid"
    )[
        "patienthealthsystemstayid"
    ]
    .nunique()
)


def cohort_stats(label, g):

    hosp48 = g[
        g["hospitaldischargeoffset"] > 2880
    ].copy()

    icu48 = g[
        g["unitdischargeoffset"] > 2880
    ].copy()

    hosp48_pair = hosp48[
        hosp48["paired_complete"] == 1
    ]

    icu48_pair = icu48[
        icu48["paired_complete"] == 1
    ]

    return {
        "cohort":
            label,

        "n_icu_stays":
            len(g),

        "n_healthsystem_stays":
            g[
                "patienthealthsystemstayid"
            ].nunique(),

        "n_unique_patients":
            g[
                "uniquepid"
            ].nunique(),

        "n_hospitals":
            g[
                "hospitalid"
            ].nunique(),

        "hospital_deaths":
            int(
                g[
                    "hospital_death"
                ].sum()
            ),

        "hospital48_n":
            len(hosp48),

        "hospital48_paired_n":
            len(hosp48_pair),

        "hospital48_paired_unique_patients":
            hosp48_pair[
                "uniquepid"
            ].nunique(),

        "hospital48_paired_hospitals":
            hosp48_pair[
                "hospitalid"
            ].nunique(),

        "hospital48_paired_deaths":
            int(
                hosp48_pair[
                    "hospital_death"
                ].sum()
            ),

        "icu48_n":
            len(icu48),

        "icu48_paired_n":
            len(icu48_pair),

        "icu48_paired_unique_patients":
            icu48_pair[
                "uniquepid"
            ].nunique(),

        "icu48_paired_hospitals":
            icu48_pair[
                "hospitalid"
            ].nunique(),

        "icu48_paired_deaths":
            int(
                icu48_pair[
                    "hospital_death"
                ].sum()
            ),
    }


structure_rows = [
    cohort_stats(
        "all_strict_qualifying_icu_stays",
        x
    ),

    cohort_stats(
        "first_qualifying_icu_per_hospitalization",
        harm
    ),
]


structure_rows += [
    {
        "cohort":
            "patient_repeat_hospitalization_audit",

        "n_icu_stays":
            np.nan,

        "n_healthsystem_stays":
            harm[
                "patienthealthsystemstayid"
            ].nunique(),

        "n_unique_patients":
            harm[
                "uniquepid"
            ].nunique(),

        "n_hospitals":
            harm[
                "hospitalid"
            ].nunique(),

        "hospital_deaths":
            np.nan,

        "hospital48_n":
            np.nan,

        "hospital48_paired_n":
            np.nan,

        "hospital48_paired_unique_patients":
            np.nan,

        "hospital48_paired_hospitals":
            np.nan,

        "hospital48_paired_deaths":
            np.nan,

        "icu48_n":
            np.nan,

        "icu48_paired_n":
            np.nan,

        "icu48_paired_unique_patients":
            np.nan,

        "icu48_paired_hospitals":
            np.nan,

        "icu48_paired_deaths":
            np.nan,

        "patients_with_2plus_hospitalizations":
            int(
                (
                    episodes_per_patient > 1
                ).sum()
            ),

        "max_hospitalizations_per_patient":
            int(
                episodes_per_patient.max()
            ),
    }
]


structure = pd.DataFrame(
    structure_rows
)

structure.to_csv(
    OUT_STRUCTURE,
    index=False
)


# ============================================================
# 4. Harmonized primary feasibility subset
# ============================================================

hosp48 = harm[
    harm[
        "hospitaldischargeoffset"
    ] > 2880
].copy()


paired = hosp48[
    hosp48[
        "paired_complete"
    ] == 1
].copy()


print("=" * 82)
print("eICU CHECKPOINT E1C - COHORT HARMONIZATION + RRT AUDIT")
print("=" * 82)

print()
print("===== COHORT HARMONIZATION =====")

print(
    structure.to_string(
        index=False
    )
)

print()
print(
    f"Harmonized hospital-48h paired cohort: "
    f"{len(paired):,} stays; "
    f"{paired['uniquepid'].nunique():,} unique patients; "
    f"{paired['hospitalid'].nunique():,} hospitals; "
    f"{int(paired['hospital_death'].sum()):,} deaths"
)


# ============================================================
# 5. Extract RRT-relevant treatment rows
# ============================================================

con = duckdb.connect()

con.register(
    "harm_ids",
    harm[
        ["patientunitstayid"]
    ]
)


rrt = con.execute(f"""
SELECT
    t.patientunitstayid,
    t.treatmentoffset,
    t.treatmentstring,
    t.activeupondischarge

FROM read_csv_auto(
    '{TREATMENT_FILE}',
    header=true
) t

INNER JOIN harm_ids h
    ON t.patientunitstayid
     = h.patientunitstayid

WHERE
       lower(coalesce(t.treatmentstring, ''))
           LIKE '%dialys%'

    OR lower(coalesce(t.treatmentstring, ''))
           LIKE '%c v v h%'

    OR lower(coalesce(t.treatmentstring, ''))
           LIKE '%crrt%'

    OR lower(coalesce(t.treatmentstring, ''))
           LIKE '%sled%'

    OR lower(coalesce(t.treatmentstring, ''))
           LIKE '%hemofil%'

ORDER BY
    t.patientunitstayid,
    t.treatmentoffset
""").fetchdf()


# ============================================================
# 6. Classify evidence
# ============================================================

def classify_rrt(value):

    s = str(value).lower()

    supportive_terms = [
        "insertion",
        "catheter",
        "shunt",
        "access surgery",
        "access",
    ]

    if any(
        term in s
        for term in supportive_terms
    ):
        return "SUPPORTIVE_ACCESS"

    if "ultrafiltration" in s:
        return "ULTRAFILTRATION_ONLY"

    if "c v v h d" in s:
        return "CRRT_CVVHD"

    if "c v v h" in s:
        return "CRRT_CVVH"

    if "sled" in s:
        return "SLED"

    if "peritoneal dialysis" in s:
        return "PERITONEAL"

    if "hemodialysis" in s:
        return "IHD"

    if "dialysis" in s:
        return "OTHER_DIALYSIS"

    return "OTHER"


rrt["rrt_class"] = (
    rrt[
        "treatmentstring"
    ]
    .apply(classify_rrt)
)


definitive_classes = {
    "CRRT_CVVHD",
    "CRRT_CVVH",
    "SLED",
    "PERITONEAL",
    "IHD",
}


broad_active_classes = (
    definitive_classes
    | {
        "OTHER_DIALYSIS"
    }
)


rrt[
    "definitive_active"
] = (
    rrt[
        "rrt_class"
    ]
    .isin(
        definitive_classes
    )
    .astype(int)
)


rrt[
    "broad_active"
] = (
    rrt[
        "rrt_class"
    ]
    .isin(
        broad_active_classes
    )
    .astype(int)
)


rrt.to_parquet(
    OUT_RRT_EVENTS,
    index=False,
    compression="zstd"
)


# ============================================================
# 7. APACHE dialysis auxiliary evidence
# ============================================================

apache = pd.read_csv(
    APACHE_FILE,
    usecols=[
        "patientunitstayid",
        "dialysis",
    ]
)


apache["dialysis"] = (
    pd.to_numeric(
        apache[
            "dialysis"
        ],
        errors="coerce"
    )
)


# ============================================================
# 8. Patient/stay-specific RRT evidence
# ============================================================

paired_ids = set(
    paired[
        "patientunitstayid"
    ]
)


r = rrt[
    rrt[
        "patientunitstayid"
    ].isin(
        paired_ids
    )
].copy()


r[
    "treatmentoffset"
] = pd.to_numeric(
    r[
        "treatmentoffset"
    ],
    errors="coerce"
)


audit_rows = []

for stayid in paired[
    "patientunitstayid"
]:

    g = r[
        r[
            "patientunitstayid"
        ] == stayid
    ]

    ga = g[
        g[
            "definitive_active"
        ] == 1
    ]

    gb = g[
        g[
            "broad_active"
        ] == 1
    ]

    # Observed available evidence in the nominal
    # 7-day MELD lookback windows.
    #
    # baseline landmark = +12 h = +720 min
    # h48 landmark      = +48 h = +2880 min
    #
    # NOTE: eICU may not capture the entire pre-ICU
    # hospital period. This is an audit, not a final
    # MELD dialysis definition.
    baseline_def = ga[
        (
            ga["treatmentoffset"]
            >= -9360
        )
        &
        (
            ga["treatmentoffset"]
            <= 720
        )
    ]

    h48_def = ga[
        (
            ga["treatmentoffset"]
            >= -7200
        )
        &
        (
            ga["treatmentoffset"]
            <= 2880
        )
    ]

    baseline_broad = gb[
        (
            gb["treatmentoffset"]
            >= -9360
        )
        &
        (
            gb["treatmentoffset"]
            <= 720
        )
    ]

    h48_broad = gb[
        (
            gb["treatmentoffset"]
            >= -7200
        )
        &
        (
            gb["treatmentoffset"]
            <= 2880
        )
    ]

    cr = h48_def[
        h48_def[
            "rrt_class"
        ].isin(
            [
                "CRRT_CVVHD",
                "CRRT_CVVH",
            ]
        )
    ]

    if len(cr):
        cr_span_h = (
            cr[
                "treatmentoffset"
            ].max()
            -
            cr[
                "treatmentoffset"
            ].min()
        ) / 60.0
    else:
        cr_span_h = np.nan

    ihd = h48_def[
        h48_def[
            "rrt_class"
        ] == "IHD"
    ].copy()

    if len(ihd):
        ihd[
            "icu_relative_day"
        ] = np.floor(
            ihd[
                "treatmentoffset"
            ] / 1440.0
        )

        ihd_days = (
            ihd[
                "icu_relative_day"
            ]
            .nunique()
        )
    else:
        ihd_days = 0

    audit_rows.append({

        "patientunitstayid":
            stayid,

        "baseline_definitive_rrt":
            int(
                len(
                    baseline_def
                ) > 0
            ),

        "h48_definitive_rrt":
            int(
                len(
                    h48_def
                ) > 0
            ),

        "baseline_broad_rrt":
            int(
                len(
                    baseline_broad
                ) > 0
            ),

        "h48_broad_rrt":
            int(
                len(
                    h48_broad
                ) > 0
            ),

        "h48_crrt_rows":
            int(
                len(cr)
            ),

        "h48_crrt_observed_span_h":
            cr_span_h,

        "h48_crrt_span_ge24h":
            int(
                pd.notna(
                    cr_span_h
                )
                and
                cr_span_h >= 24
            ),

        "h48_ihd_rows":
            int(
                len(ihd)
            ),

        "h48_ihd_distinct_relative_days":
            int(
                ihd_days
            ),

        "h48_ihd_2plus_relative_days":
            int(
                ihd_days >= 2
            ),
    })


rrt_audit = pd.DataFrame(
    audit_rows
)


rrt_audit = rrt_audit.merge(
    apache[
        [
            "patientunitstayid",
            "dialysis",
        ]
    ],
    on="patientunitstayid",
    how="left"
)


rrt_audit[
    "apache_dialysis_positive"
] = (
    rrt_audit[
        "dialysis"
    ] == 1
).astype(int)


rrt_audit = (
    paired[
        [
            "patientunitstayid",
            "uniquepid",
            "hospitalid",
            "hospital_death",
        ]
    ]
    .merge(
        rrt_audit,
        on="patientunitstayid",
        how="left"
    )
)


rrt_audit.to_csv(
    OUT_RRT_PATIENT,
    index=False
)


# ============================================================
# 9. RRT reconstructability summary
# ============================================================

summary_rows = []


def add_metric(metric, value):

    summary_rows.append({
        "metric":
            metric,

        "value":
            value,
    })


add_metric(
    "harmonized_hospital48_paired_n",
    len(rrt_audit)
)

add_metric(
    "paired_with_baseline_definitive_rrt",
    int(
        rrt_audit[
            "baseline_definitive_rrt"
        ].sum()
    )
)

add_metric(
    "paired_with_h48_definitive_rrt",
    int(
        rrt_audit[
            "h48_definitive_rrt"
        ].sum()
    )
)

add_metric(
    "paired_with_baseline_broad_rrt",
    int(
        rrt_audit[
            "baseline_broad_rrt"
        ].sum()
    )
)

add_metric(
    "paired_with_h48_broad_rrt",
    int(
        rrt_audit[
            "h48_broad_rrt"
        ].sum()
    )
)

add_metric(
    "paired_apache_dialysis_positive",
    int(
        rrt_audit[
            "apache_dialysis_positive"
        ].sum()
    )
)

add_metric(
    "paired_crrt_observed_span_ge24h",
    int(
        rrt_audit[
            "h48_crrt_span_ge24h"
        ].sum()
    )
)

add_metric(
    "paired_ihd_2plus_relative_days",
    int(
        rrt_audit[
            "h48_ihd_2plus_relative_days"
        ].sum()
    )
)


rrt_summary = pd.DataFrame(
    summary_rows
)

rrt_summary.to_csv(
    OUT_RRT_SUMMARY,
    index=False
)


# ============================================================
# 10. Console output
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    240
)


print()
print("===== RRT CLASS DISTRIBUTION =====")

print(
    rrt.groupby(
        "rrt_class"
    )
    .agg(
        n_rows=(
            "patientunitstayid",
            "size"
        ),
        n_stays=(
            "patientunitstayid",
            "nunique"
        ),
    )
    .reset_index()
    .sort_values(
        "n_stays",
        ascending=False
    )
    .to_string(
        index=False
    )
)


print()
print("===== PAIRED-COHORT RRT RECONSTRUCTABILITY =====")

print(
    rrt_summary.to_string(
        index=False
    )
)


crrt_positive = rrt_audit[
    rrt_audit[
        "h48_crrt_rows"
    ] > 0
]

if len(
    crrt_positive
):

    print()
    print("===== CRRT TIMING AMONG PAIRED COHORT =====")

    print(
        pd.DataFrame([{
            "n_crrt_stays":
                len(
                    crrt_positive
                ),

            "median_crrt_rows":
                crrt_positive[
                    "h48_crrt_rows"
                ].median(),

            "median_observed_span_h":
                crrt_positive[
                    "h48_crrt_observed_span_h"
                ].median(),

            "max_observed_span_h":
                crrt_positive[
                    "h48_crrt_observed_span_h"
                ].max(),

            "span_ge24h_n":
                crrt_positive[
                    "h48_crrt_span_ge24h"
                ].sum(),
        }]).to_string(
            index=False
        )
    )


print()
print("=" * 82)
print("eICU CHECKPOINT E1C COMPLETE")
print("=" * 82)

print(f"Structure     : {OUT_STRUCTURE}")
print(f"Harmonized    : {OUT_HARMONIZED}")
print(f"RRT events    : {OUT_RRT_EVENTS}")
print(f"RRT summary   : {OUT_RRT_SUMMARY}")
print(f"RRT stay audit: {OUT_RRT_PATIENT}")


con.close()
