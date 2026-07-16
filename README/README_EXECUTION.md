# PCT Repository Snapshot - Execution Guide

Last updated: 2026-07-16

Repository: https://github.com/kortxresearch/PCT

This snapshot is aligned with the Springer Nature / Foundations of Physics resubmission:

`Projective Correlation Theory: A Pregeometric Correlator Framework for Emergent Spacetime Observables`

The repository is a reproducibility package, not a separate confirmation claim. In the v4 manuscript, Planck 2018 is an archival-data consistency check; the toy spectral-dimension and synthetic change-point outputs are algorithmic checks; the LVK ringdown output is a null-control result; and the new D/NP/P/T capsules are declared support, forecast, null, or future-work adjudication artifacts.

## Quick Start

From the repository root:

```bash
python -m pip install -r requirements.txt
python run_capsules.py
python hash_artifacts.py
```

Subset runs:

```bash
python run_capsules.py --gw-only
python run_capsules.py --lvk-only
python pct_ds.py
python gw_change_point_runner.py
python planck2018_running_inference.py --out outputs/planck2018_running_scaffold.json
python lvk_ringdown_end_to_end.py
python gw150914_pct_predictions.py
```

The full Planck MCMC run requires Cobaya/CAMB plus the bundled or locally configured Planck likelihood package tree. The lightweight `run_capsules.py` refreshes the Planck scaffold only; it does not rerun the multi-hour MCMC chain.

`outputs/capsule_run_report.json` records a historical `--gw-only` runner invocation. The individual executed artifacts and the status table below, not that partial runner report, define the scope of the published package.

## Capsule Status

| Capsule | Script | Status in this snapshot | Primary output |
|---|---|---|---|
| Spectral-dimension toy chain | `pct_ds.py` | Executed algorithmic check | `outputs/pct_ds_output.txt` |
| Synthetic change-point | `gw_change_point_runner.py` | Executed algorithmic check | `outputs/gw_change_points_synth.json` |
| Planck 2018 running inference | `planck2018_running_inference.py` / `planck_running.yaml` | Executed archival-data consistency check plus deterministic scaffold | `outputs/planck2018_running.json`, `outputs/planck2018_running_scaffold.json` |
| LVK ringdown public strain | `lvk_ringdown_end_to_end.py` | Executed public-data null-control check | `outputs/lvk_ringdown_public_run.json` |
| GW150914 posterior-derived prediction values | `gw150914_pct_predictions.py` | Executed parameter propagation | `outputs/gw150914_pct_predictions.json` |
| Scalar-running derivation options | `d1_option_a_threshold_alpha.py`, `d1_option_b_corrected_flow.py` | Executed option checks | `outputs/d1_option_a_threshold_alpha.json`, `outputs/d1_option_b_corrected_flow.json` |
| Ringdown timing corrections | `ringdown_timing_corrections.py` | Executed spin/redshift correction table | `outputs/ringdown_timing_corrections.json` |
| Kappa origin hypothesis | `kappa_origin_hypothesis_check.py` | Executed calibration-origin check | `outputs/kappa_origin_hypothesis_check.json` |
| NP-1 spin regression scaffold | `np1_spin_ringdown_regression.py` | Executed harness/scaffold | `outputs/np1_spin_ringdown_regression.json` |
| NP-2 LISA forecast | `np2_lisa_forecast.py` | Executed forecast calculation | `outputs/np2_lisa_forecast.json` |
| NP-6 propagation null | `np6_propagation_null.py` | Executed null statement check | `outputs/np6_propagation_null.json` |
| NP-9 EHT shadow null | `np9_eht_null.py` | Executed null statement check | `outputs/np9_eht_null.json` |
| P4 continuum refinement | `p4_continuum_refinement_study.py` | Executed draft-support study | `outputs/p4_continuum_refinement_study.json` |
| P6 kernel tower | `p6_kernel_tower_study.py` | Executed draft-support study | `outputs/p6_kernel_tower_study.json` |
| P6b emergence/cosmology probe | `p6b_emergence_cosmology_probe.py` | Executed exploratory probe | `outputs/p6b_emergence_cosmology_probe.json` |
| P6c dense flow probe | `p6c_dense_flow_probe.py` | Executed dense monotonicity probe | `outputs/p6c_dense_flow_probe.json` |
| P2a Bogoliubov pipeline | `p2a_bogoliubov_pipeline.py` | Executed feature/robustness diagnostic | `outputs/p2a_bogoliubov_pipeline.json` |
| A1 scalar-sector attempt | `a1_scalar_sector_attempt.py` | Executed; derivation remains open because required maps are unfixed | `outputs/a1_scalar_sector_attempt.json` |
| R2 flow-clock selection | `r2_flow_clock_selection.py` | Executed conditional selection; downstream P2 gate remains blocked | `outputs/r2_flow_clock_selection.json` |
| B1 update-budget check | `b1_update_budget_check.py` | Executed construction check | `outputs/b1_update_budget_check.json` |
| B2 horizon-criterion demo | `b2_horizon_criterion_demo.py` | Executed fixed-threshold demonstration | `outputs/b2_horizon_criterion_demo.json` |
| T1 step-height adjudication | `t1_step_height_adjudication.py` | Executed frozen adjudication; non-adjudicable | `outputs/t1_step_height_adjudication.json` |
| T1b construction ambiguity sweep | `t1b_construction_sweep.py` | Executed 3x2 screening sweep | `outputs/t1b_construction_sweep.json` |
| T2 horizon-profile campaign | `t2_horizon_profile_campaign.py` | Executed frozen campaign; no refinement-stable step | `outputs/t2_horizon_profile_campaign.json` |
| FW1 SU(2) toy | `fw1_su2_toy.py` | Executed interface diagnostic; field content partial, coupling/back-reaction open | `outputs/fw1_su2_toy.json` |
| FW2 RG commutation | `fw2_rg_commutation.py` | Executed analytic/numerical commutation checks | `outputs/fw2_rg_commutation.json` |
| P9 PiSR pilot | `p9_pisr_pilot.py` | Executed frozen scan; joint register/clock criterion not met | `outputs/p9_pisr_pilot.json` |

## Key Results

### Spectral Dimension

```text
d_s peak:           1.723 at ell = 1.468
Refinement check:   max |Delta d_s| = 0.453
```

### Synthetic Change-Point

```text
Breakpoint:         index 60 (2020-01-01)
Segment means:      [10.110, 12.231]
Mean jump:          Delta mu = 2.121, approximately 21 percent
```

### Planck 2018 Running

```text
Convergence:        R-1 = 0.0079
Samples:            24,080 raw / 40,490 weighted

alpha_s:            -0.0037 +/- 0.0069
68 percent CI:      [-0.0108, +0.0032]
95 percent CI:      [-0.0174, +0.0095]
n_s:                0.9648 +/- 0.0043
H0:                 67.38 km/s/Mpc

PCT locked-band:    alpha_s = -0.012 +/- 0.005
Separation:         0.97 sigma
Interpretation:     compatible as a consistency check; not confirmation
```

### LVK Ringdown

```text
On-source Delta AIC:   -219.27
Off-source Delta AIC:  -23.70, -206.59
Null p-value:          0.12
Outcome:               null-control result, not a detection
```

### T1b Construction Sweep

```text
Cells screened:       3 volume readings x 2 operator conventions
Promoted cells:       none
Conclusion:           no documented construction reproduces the claimed historical regime
```

### T2 Horizon-Profile Campaign

```text
Frozen ladder cells:   12
Refinement-stable step: none
Conclusion:            no robust non-analytic step in this fresh construction
```

### P9 PiSR Pilot

```text
Register criterion:    selects the Gaussian fixed-point flow
Clock criterion:       selects slower scanned flows
Joint criterion:       no scanned flow satisfies both
```

## Directory Structure

```text
PCT/
  configs/
  data/
  outputs/
  chains/
  packages/
  artifacts/
  README/
  *.py capsule scripts
  *.tex article and cover-letter sources
  planck_running.yaml
  run_capsules.py
  hash_artifacts.py
  requirements.txt
  README.md
  MANUSCRIPT_ALIGNMENT.md
```

The canonical public manuscript copies are `PCT_FoP_v4_submission.tex` and `PCT_FoP_v4_submission.pdf`.

The root `requirements.txt` is authoritative. The older duplicate `README/requirements.txt` was archived under `9. ARCHIVE\duplicates\repo-readme-requirements-2026-07-10\`.

## Verification

Run:

```bash
python hash_artifacts.py
```

This refreshes `outputs/sha256SUMS.txt` for the included repository artifacts using deterministic file ordering.

Contact: Contact@kort-x.com

Author: Ciprian Stoichici
