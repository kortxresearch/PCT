# Projective Correlation Theory (PCT)

Reproducibility package for the manuscript **“Projective Correlation Theory: A Pregeometric Correlator Framework for Emergent Spacetime Observables”** (submitted to Foundations of Physics on 2026-07-17 and currently under review).

[Manuscript PDF](PCT_FoP_submission.pdf) · [LaTeX source](PCT_FoP_submission.tex) · [Zenodo DOI 10.5281/zenodo.21396416](https://doi.org/10.5281/zenodo.21396416) · [Current OSF project](https://osf.io/7skhc/) · [Legacy OSF DOI 10.17605/OSF.IO/F4WTD](https://doi.org/10.17605/OSF.IO/F4WTD)

## Scope

This repository contains the executable analysis capsules, frozen configurations, public or derived data, MCMC chain material, and checksummed outputs accompanying the manuscript. The canonical public manuscript files are the LaTeX source and compiled PDF above.

The package preserves the manuscript's claim boundaries:

- Planck 2018 is an archival-data consistency check, not confirmation.
- The spectral-dimension toy chain and synthetic change-point run are algorithm checks, not evidence for the underlying physics.
- The LVK ringdown analysis is a null-control result, not a detection.
- The smooth-crossover location `kappa = 0.80 +/- 0.05` is calibrated within the locked instantiation; it is not derived from every PCT primitive.
- Metric reconstruction is stated in the Euclidean section. Lorentzian causal language requires the declared continuation condition A3.
- The ringdown timing law is spin-scaled: `t_c/M = kappa*(1 + sqrt(1 - a_*^2))`.
- Propagation and EHT statements are declared nulls. The T1/T1b/T2 campaigns do not support a robust non-analytic spectral step in the tested constructions.
- No ultraviolet completeness, intrinsic dynamics, or Standard Model derivation is claimed.

## Quick start

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python run_capsules.py
```

The lightweight runner executes the ordinary capsules and refreshes the deterministic Planck scaffold. Reproducing the full Planck MCMC chain additionally requires Cobaya/CAMB and the bundled or locally configured Planck likelihood material.

See [README/README_EXECUTION.md](README/README_EXECUTION.md) for individual commands and [README/DELIVERABLES.txt](README/DELIVERABLES.txt) for the scope and status ledger.

## Repository map

| Path | Contents |
|---|---|
| `PCT_FoP_submission.tex`, `.pdf` | Canonical public manuscript source and compiled article. |
| `pct_ds.py`, `gw_change_point_runner.py` | Spectral-dimension toy chain and synthetic change-point checks. |
| `planck2018_running_inference.py`, `planck_running.yaml` | Planck running scaffold/configuration and result serialization. |
| `lvk_ringdown_end_to_end.py`, `gw150914_pct_predictions.py` | Public-strain null control and posterior-derived parameter propagation. |
| `configs/` | Frozen capsule configurations. |
| `data/` | Public or derived inputs used by the included runs. |
| `chains/` | Planck chain and configuration artifacts. |
| `outputs/` | Executed results and `sha256SUMS.txt`. |
| `packages/` | Bundled CAMB/Planck support tree retained for reproducibility. |
| `README/DELIVERABLES.txt` | Capsule scope and status ledger. |

Additional D/NP/P/T/FW scripts record derivation checks, forecasts, null tests, construction adjudications, and explicitly exploratory work. Their epistemic status is recorded in their JSON outputs and in the execution guide.

## Integrity checks

The repository disables Git line-ending conversion so the published byte streams match the checksum manifest. On systems with `sha256sum`:

```bash
sha256sum --check outputs/sha256SUMS.txt
```

After an intentional artifact change, regenerate the deterministic manifest with:

```bash
python hash_artifacts.py
```

## Citation, license, and contact

Use the archived release DOI: [10.5281/zenodo.21396416](https://doi.org/10.5281/zenodo.21396416).

Author: Ciprian Stoichici · KORT-X Research, Bucharest, Romania  
Contact: [contact@kort-x.com](mailto:contact@kort-x.com)

First-party repository material is released under [CC0 1.0 Universal](LICENSE). Bundled third-party material under `packages/` retains its upstream licensing and attribution.
