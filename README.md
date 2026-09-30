# MELD 3.0 ICU landmark reassessment

Public **analysis-code-only** release supporting:

**48-Hour Landmark Reassessment of MELD 3.0 in Critically Ill Patients With Cirrhosis: Incremental Prognostic Value and External Transportability**

## Study design

- Development/internal validation: MIMIC-IV v3.1
- Fixed-coefficient external validation: eICU-CRD v2.0
- Primary reassessment: 48-hour landmark
- Prespecified secondary reassessment: 72-hour landmark
- Outcome: in-hospital mortality after the relevant landmark
- Model A: baseline MELD 3.0
- Model B: baseline MELD 3.0 + change in MELD 3.0
- Model C: reassessment MELD 3.0 alone

## Why this repository contains code only

MIMIC-IV and eICU-CRD are credentialed clinical databases. This repository does not redistribute any source records, patient-level derivatives, derived result datasets, or fitted MIMIC model artifacts. Authorized users generate all required intermediate files locally. See `DATA_ACCESS.md`.

## Contents

- `mimic/code/` — MIMIC pipeline 01–14 plus public helper 14b for local generation of full 72-h coefficient vectors
- `eicu/code/` — eICU pipeline E1A–E4
- `environment/` — clean reproduction/QC environment information
- `CODE_PROVENANCE.md` — analysis-lock hashes and the narrowly scoped public-release adaptations
- `DATA_ACCESS.md` — source database access and non-redistribution rules
- `RELEASE_QC.md` — repository safety/integrity checks

## Expected local layout

The scripts use the following default local structure:

```text
~/Desktop/MIMIC_IV_3_1/
  data/
    hosp/
    icu/
  work/

~/Desktop/eICU_CRD_2_0/
  data/
  work/
```

Put database files obtained under your own authorized access into the corresponding local `data` folders. The `work` folders hold derived files generated locally and must never be committed.

## Run order

### MIMIC-IV

Run in order:

`01_build_core_db.py` → `02_review_cirrhosis_codes.py` → `03_build_cirrhosis_cohort.py` → `04_audit_meld_labitems.py` → `05_audit_meld_itemids_in_labevents.py` → `06_extract_meld_labs.py` → `07_dynamic_feasibility.py` → `08_baseline_anchor_feasibility.py` → `09_icu_baseline_window_sensitivity.py` → `10_audit_rrt.py` → `11_extract_rrt_evidence.py` → `11b_build_rrt_sessions.py` → `12_build_meld3_snapshots.py` → `13a_h48_change_missingness.py` → `13b_incremental_prognostic.py` → `13c_ipw_selection_sensitivity.py` → `13d_uncertainty_nonlinearity_dca.py` → `14_h72_secondary_landmark.py`.

Before the eICU 72-hour external-validation/robustness scripts, also run:

`14b_export_h72_fixed_coefficients.py`

This helper creates the full MIMIC A/B/C 72-hour coefficient file locally. That file is deliberately not distributed in this public repository.

### eICU-CRD

Run in order:

`E1A_eicu_schema_candidate_audit.py` → `E1B_eicu_dynamic_feasibility.py` → `E1C_eicu_harmonization_rrt_audit.py` → `E2A_build_eicu_meld3_snapshots.py` → `E2B_external_validation.py` → `E2C_transportability_diagnostic.py` → `E2D_cross_database_heterogeneity.py` → `E3A_eicu_h72_feasibility.py` → `E3B_eicu_h72_external_validation.py` → `E4_final_robustness_h72.py`.

The eICU formal validation remains fixed-coefficient validation: coefficients are derived from MIMIC locally and then applied to eICU without eICU refitting or recalibration.

## Environment

The exact historical analysis dependency versions were not recoverable. A separate clean reproduction/QC environment is documented under `environment/`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements-clean-reproduction.txt
```

## Release verification

```bash
python verify_release.py
```

## Citation

Citation metadata are provided in `CITATION.cff`. Please cite the associated manuscript and the archived software release when available.

## License

Repository-authored code and documentation are provided under the MIT License. The clinical databases and any locally generated derived data/model artifacts remain governed by their own terms and are not covered by this license.
