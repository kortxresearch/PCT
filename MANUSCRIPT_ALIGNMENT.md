# Manuscript Alignment Notes

Date: 2026-07-16

This repository is aligned with `PCT_FoP_v4_submission.tex` and `PCT_FoP_v4_submission.pdf` (v4 resubmission). These are the canonical public manuscript copies; the deprecated plain-text export is not part of the GitHub release.

The live GitHub remote was rechecked on 2026-07-16 at commit `12a7b97497f7cbe0896f57f9a2b187f9c51b034e`; this release supersedes that three-file pre-v3 state.

## Alignment decisions carried over from v3

- "Pre-specified" / "declared" language instead of "pre-registration" claims.
- No "Level 2" / "Level 3" evidence-grading framing.
- Planck 2018 framed as an archival-data consistency check: PCT band `alpha_s = -0.012 +/- 0.005` vs Planck posterior `-0.0037 +/- 0.0069`, separation `0.97 sigma`; compatible, not confirmation.
- LVK ringdown output kept as a null-control result; toy and synthetic outputs kept as algorithmic checks.

## New in the v4 alignment (2026-07-08/09)

- Metric reconstruction restated as Euclidean-section (`h_munu`); Lorentzian causal language requires the declared static/stationary analytic continuation in A3 (Bochner argument in the proof sketch).
- Distance surrogate redefined as `d_corr = sqrt(-log Ghat_2)`; the un-rooted form fails the triangle inequality for Gaussian kernels.
- `kappa = 0.80 +/- 0.05` is described as calibrated within the locked instantiation (basis: `outputs/kappa_origin_hypothesis_check.json`), not derived.
- Ringdown timing is the spin-scaled law `tc/M = kappa*(1+sqrt(1-a*^2))`; GW150914 detector-frame value `0.53 +/- 0.04 ms`; stacked forecast `SNR ~ 0.39*sqrt(N)`.
- The scalar-running band stays phenomenological; the four missing ingredients for a first-principles derivation are stated in the Limitations section (basis: `outputs/d1_option_a_threshold_alpha.json`).
- New declared commitments: LISA forecast table, analogue universal-kappa test (protocol in the sandbox analysis area), propagation null (GW170817-consistent), EHT shadow null, and a closing prediction register.
- New capsule scripts: `d1_option_a_threshold_alpha.py`, `d1_option_b_corrected_flow.py`, `ringdown_timing_corrections.py`, `kappa_origin_hypothesis_check.py`, `np1_spin_ringdown_regression.py`, `np2_lisa_forecast.py`, `np6_propagation_null.py`, `np9_eht_null.py`, `planck_summary_from_chain.py`, `p4_continuum_refinement_study.py`, `p6_kernel_tower_study.py`, `p6b_emergence_cosmology_probe.py`, `t1_step_height_adjudication.py`, and `t1b_construction_sweep.py`; the first-party files selected by `hash_artifacts.py` are covered by `outputs/sha256SUMS.txt` after refresh.

## W1 repository hygiene update (2026-07-10)

- Repaired stale v3-era repository instructions in `GITHUB_REMOTE_HEAD.txt`, `README/README_EXECUTION.md`, `README/DELIVERABLES.txt`, `README.md`, and the comment header in `planck_running.yaml`.
- Archived the duplicate `README/requirements.txt`; the root `requirements.txt` is the authoritative dependency list.
- Preserved v3/date references that are historical provenance rather than current-state instructions.

## D5 sensitivity wording release (2026-07-10)

- Applied the author-approved D5 wording patch to remove the dangling implication that a dedicated kappa / Delta d_s sensitivity analysis is in Appendix B.
- Synchronized the repository and submission article sources; SHA256 after sync: `65FFB010F9766A32D768933A28C345DFB591A3C1D4004E5BC81B83D5DFB859BC`.
- Recompiled article and cover letter, then synchronized the repository and journal-package PDFs.
- Refreshed `outputs/sha256SUMS.txt` and rebuilt the byte-identical OSF/Zenodo `PCT_v4_archive_update.zip` package.

## R1 step-to-smooth release (2026-07-12)

- Applied the author-approved R1 patch set (decision D-B4-2, "just call it smooth"): the near-horizon spectral-dimension claim is now a smooth crossover with calibrated location kappa = 0.80 +/- 0.05; the step height Delta d_s, the d_UV ~ 3.21 consequence, and the hard-step contrast with CDT/asymptotic-safety/Horava are retired throughout. The falsification rule is flipped: a robust non-analytic step would now count against the locked instantiation.
- Appendix D retains the conditional threshold theorem with a new executed-status remark citing the checksummed T1/T1b/T2 records.
- The plain-language summary carries the smooth-crossover wording plus the author-approved self-test sentence ("we tested the sharper version ... and found none").
- The analogue protocol is re-scoped to a crossover-location/co-location test (`5. ANALYSIS/analogue-protocol/PROTOCOL.md`).
- Ringdown law, phenomenological alpha_s band, LISA forecast, propagation null, and EHT null are unchanged.
- Article recompiled clean (37 pages); submission and repository LaTeX sources synchronized; SHA256 after sync: `461BA736937FE199` (prefix; full value in `outputs/sha256SUMS.txt`).
- Applied by Claude (author instruction, Codex token budget exhausted); patch previews preserved in `10. FUTURE WORK/PENDING_PATCHES/R1_step_to_smooth/`.

## Submission use

Use this folder as the repository snapshot accompanying the v4 submission. The compiled article and cover letter for upload are in `1. SUBMISSION\Foundations of Physics\`.

## GitHub publication (2026-07-16)

- Published the complete reproducibility tree to the `main` branch, replacing the obsolete three-file remote snapshot.
- Retired the duplicate plain-text manuscript export from the GitHub payload; the LaTeX source and PDF remain canonical.
- Replaced stale placeholder output notes with the executed, checksummed artifact set.
- Refreshed the README, execution guide, deliverables ledger, release provenance, and deterministic checksum manifest.
- Made checksum-manifest line endings platform-independent and verified all entries with both the repository validator and GNU `sha256sum --check`.
- Updated the repository About metadata to describe the reproducibility package and link the Zenodo DOI.
- Included the later A1/R2, B1/B2, P2a/P6c/P9, T2, and FW1/FW2 records with their negative, conditional, or open status left explicit.
