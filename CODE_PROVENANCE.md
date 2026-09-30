# Code provenance and public-release adaptations

## Authoritative scientific source

- `MELD3_ANALYSIS_LOCK_v1.zip` SHA-256: `caba41875bcfef2972640b6278a79954d4901e2ffb593bfa2b998ca2df35066f`
- Authoritative inner package `MELD3_F2R_POSTFIX_SAFE_INPUT_v3.zip` SHA-256: `b2ae148d7236df393c11fc1306fe81b30aee631b0db463405df0c08e0fb94c08`
- The inner package's 121-file SHA-256 manifest was independently rechecked before release preparation: 121/121 PASS.

## Public GitHub/Zenodo package

The MIMIC and eICU formal scripts were copied from the locked package. Three eICU scripts required narrowly scoped public-release adaptations so that fitted MIMIC model parameters are generated locally rather than redistributed publicly:

1. `E2B_external_validation.py`: removed only the hard-coded approximate coefficient guard; the script still fits the MIMIC 48-h models locally and applies those fitted coefficients unchanged to eICU.
2. `E3B_eicu_h72_external_validation.py`: replaced embedded 72-h MIMIC coefficients with loading from a locally generated coefficient file.
3. `E4_final_robustness_h72.py`: replaced embedded 72-h MIMIC coefficients/SE with loading from that same local file.
4. Added `14b_export_h72_fixed_coefficients.py`, which reproduces the A/B/C H72 coefficient vectors locally from the authorized MIMIC landmark cohort for use by E3B/E4.

These changes concern distribution/packaging only. They do not change the scientific specification: eICU external validation still uses fixed coefficients derived from MIMIC and applies them without eICU refitting or recalibration.

No source data, derived data tables, fitted model files, row-level predictions, or locked aggregate output CSVs are included in the public repository.
