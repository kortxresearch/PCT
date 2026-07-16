# W1 Stale-Data Audit

Date: 2026-07-10

Scope: top-level repository `.md`, `.txt`, `.yaml`, and `.json` material, excluding `packages/` upstream material and generated/checksummed `outputs/` material. The search terms were `v3`, `NEWSUBMISSION`, `GITHUB_REPOSITORY_FILES`, `v4-draft`, `folder 5`, `folder 6`, and dates before 2026-07.

## Classification Table

| Path | Hit | Classification | Action |
|---|---|---|---|
| `GITHUB_REMOTE_HEAD.txt` | v3-era update narrative and v3 filenames | Stale current-state instruction | Rewritten to current v4 snapshot; 2026-06-05 remote commit retained as provenance; remote-push-pending note added. |
| `README/README_EXECUTION.md` | 2026-06-05, v3 resubmission, `GITHUB_REPOSITORY_FILES/` layout | Stale current-state instruction | Rewritten to v4 execution guide with current capsule list and `2. REPOSITORY/` layout. |
| `README/DELIVERABLES.txt` | v3 scope and 2026-06-05 update line | Stale current-state instruction | Rewritten to v4 deliverables checklist with D/NP/P/T outputs. |
| `README/requirements.txt` | duplicate dependency list dated from old package | Duplicate active instruction | Archived under `9. ARCHIVE\duplicates\repo-readme-requirements-2026-07-10\`; root `requirements.txt` remains authoritative. |
| `README.md` | row describing `MANUSCRIPT_ALIGNMENT.md` as tied to v3 article | Stale current-state description | Corrected to v4 article. |
| `planck_running.yaml` | comment says v3 consistency check | Stale comment only | Updated to v4 consistency check; physics configuration unchanged. |
| `MANUSCRIPT_ALIGNMENT.md` | v3 historical carry-over text | Historical record | Kept; appended W1 hygiene entry and expanded current capsule list. |
| `t1_horizon_profile_spec.md` | `8. HISTORY/fop-v3-submission/...` provenance path | Historical record | Kept. |
| `artifacts/meta.json` | `created_utc` from old artifact utility | Historical artifact metadata | Kept; future utility writes no volatile timestamp by default. |
| `outputs/*.json` | created/execution dates in generated outputs | Excluded generated/checksummed outputs | Kept to avoid changing manuscript checksum tables before an approved release. Future reruns are deterministic where scripts were updated. |
| `packages/**` | upstream Planck/CAMB v3/date strings | Excluded upstream material | Kept. |

## Determinism Pass

- `gw150914_pct_predictions.py`: fallback RNG already used a fixed seed; the wall-clock `executed_utc` payload field was replaced with a stable `run_id` for future output.
- `planck2018_running_inference.py`: scaffold wall-clock `created_utc` was replaced with stable `protocol_id`.
- `hash_artifacts.py`: the manifest header no longer includes a wall-clock generation timestamp, so repeated runs over unchanged artifacts are byte-identical.
- `analogue_ds_artifact.py`: new artifacts no longer receive an automatic wall-clock timestamp; existing historical artifact metadata was not rewritten.
- `t1_step_height_adjudication.py`: left unchanged because its script hash is part of the frozen T1 protocol record. The deterministic T1b script has no volatile timestamp.

## GitHub publication follow-up (2026-07-16)

- Removed the deprecated duplicate plain-text manuscript export from the GitHub payload. The `.tex` source and compiled PDF are canonical.
- Removed the obsolete `outputs/NOT_EXECUTED.md`, empty `outputs/gitkeep.txt`, and ignored placeholder `outputs/run.log`; executed JSON/TXT artifacts and the checksum manifest now describe the directory accurately.
- Rewrote the public README and refreshed the execution, deliverables, and publication-provenance documentation.
