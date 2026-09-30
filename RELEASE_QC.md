# Public code release QC — v1.0.0

Date: 2026-10-01

## Scientific authority

- Analysis Lock SHA-256: `caba41875bcfef2972640b6278a79954d4901e2ffb593bfa2b998ca2df35066f`
- Authoritative v3 package SHA-256: `b2ae148d7236df393c11fc1306fe81b30aee631b0db463405df0c08e0fb94c08`
- Authoritative v3 internal checksum verification: **121/121 PASS**

## Public release checks

- MIMIC formal scripts from lock: 18
- Added public-only helper script: 1 (`14b_export_h72_fixed_coefficients.py`)
- eICU formal scripts from lock: 10
- Python syntax check: PASS
- Public source/derived data files: 0
- Public aggregate-result CSVs: 0
- Public fitted MIMIC model artifact files: 0
- Embedded locked 72-h MIMIC coefficient values: 0 after public-release adaptation
- Personal absolute filesystem paths: 0
- Patient-level/prediction files: 0

## Adaptation boundary

Only distribution-safety adaptations were made to three eICU scripts: fitted MIMIC coefficients are now generated locally rather than embedded/distributed. Scientific model definitions, landmarks, MIMIC fitting, and fixed-coefficient application to eICU remain unchanged. No results were refitted or recalculated for manuscript purposes.

**Disposition: PASS for public GitHub/Zenodo code release.**
