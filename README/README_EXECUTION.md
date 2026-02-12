# PCT DELIVERABLES - Execution Guide (v59)

**Last Updated**: 2026-02-10

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all capsules
python run_capsules.py

# Or run individually:
python pct_ds.py                              # Spectral dimension
python gw_change_point_runner.py              # Change-point detection
python planck2018_running_inference.py        # CMB running scaffold
python lvk_ringdown_end_to_end.py             # LVK ringdown analysis
python gw150914_pct_predictions.py            # GW150914 predictions
```

---

## Capsule Status (All Executed)

| Capsule | Script | Evidence Level | Output |
|---------|--------|----------------|--------|
| Spectral Dimension | `pct_ds.py` | Level 2 | `outputs/pct_ds_output.txt` |
| Change-Point (synthetic) | `gw_change_point_runner.py` | Level 2 | `outputs/gw_change_point_runner.json` |
| Planck Running | `planck2018_running_inference.py` | Level 1 | `outputs/planck2018_running.json` |
| LVK Ringdown | `lvk_ringdown_end_to_end.py` | Level 2 | `outputs/lvk_ringdown_public_run.json` |
| GW150914 Predictions | `gw150914_pct_predictions.py` | Level 2 | `outputs/gw150914_pct_predictions.json` |

---

## Key Results Summary

### 1. Spectral Dimension (`pct_ds.py`)
```
d_s peak:           1.723 at ℓ = 1.468
Refinement check:   max|Δd_s| = 0.453
UV regime:          d_s ≈ 0.024 at ℓ = 0.1
IR regime:          d_s ≈ 0.654 at ℓ = 10.0
```

### 2. Change-Point Detection (`gw_change_point_runner.py`)
```
Input:              gw_synth_monthly.csv (120 points)
Breakpoint:         index 60 (2020-01-01)
Segment means:      [10.11, 12.23]
Mean jump:          Δμ = 2.12 (21% step)
```

### 3. Planck Running Scaffold
```
PCT prediction:     α_s = -0.012 ± 0.005
68% CI:             [-0.017, -0.007]
95% CI:             [-0.022, -0.002]
Planck 2018 obs:    dn_s/d ln k = -0.0045 ± 0.0067 (compatible)
```

### 4. LVK Ringdown (`lvk_ringdown_end_to_end.py`)
```
Event:              GW150914 (H1 detector)
On-source ΔAIC:     -219.27
Off-source ΔAIC:    -23.70, -206.59
Null p-value:       0.12
OUTCOME:            NULL (change-point not unique to ringdown)
```

### 5. GW150914 PCT Predictions (`gw150914_pct_predictions.py`)
```
Final mass:         62.0 ± 2.5 M_sun
Final spin:         0.67 ± 0.03
PCT κ:              0.80
t_c/M:              1.60
t_c (physical):     0.49 ± 0.02 ms
```

---

## Directory Structure

```
DELIVERABLES/
├── configs/
│   ├── cmb_capsule_v1.json
│   ├── ds_capsule_v1.json
│   ├── gw_capsule_v1.json
│   └── lvk_ringdown_capsule_v1.json
├── data/
│   ├── gw_synth_monthly.csv
│   ├── H-H1_LOSC_4_V2-1126259446-32.hdf5
│   └── GW150914_GWTC-1.hdf5
├── outputs/
│   ├── pct_ds_output.txt
│   ├── gw_change_point_runner.json
│   ├── gw_change_points_synth.json
│   ├── planck2018_running.json
│   ├── planck2018_running_scaffold.json
│   ├── lvk_ringdown_public_run.json
│   ├── gw150914_pct_predictions.json
│   └── sha256SUMS.txt
├── chains/                              # Planck MCMC chains (if run)
├── pct_ds.py
├── gw_change_point_runner.py
├── planck2018_running_inference.py
├── lvk_ringdown_end_to_end.py
├── gw150914_pct_predictions.py
├── planck_running.yaml                  # Cobaya config
├── run_capsules.py
├── requirements.txt
├── DELIVERABLES.txt
└── README_EXECUTION.md
```

---

## Verification

```bash
# Verify all output checksums
cd outputs
sha256sum -c sha256SUMS.txt
```

Expected:
```
pct_ds_output.txt: OK
gw_change_point_runner.json: OK
planck2018_running.json: OK
lvk_ringdown_public_run.json: OK
```

---

## Planck Full Inference (Optional)

To run the full MCMC inference (6-24 hours):

```bash
# Install Cobaya and CAMB
pip install cobaya camb

# Download Planck likelihoods (first time only, ~2GB)
cobaya-install planck_running.yaml --packages-path ./packages

# Run MCMC
cobaya-run planck_running.yaml

# Extract results
python extractsamples.py
```

**Note**: For publication, the Planck 2018 published value (dn_s/d ln k = -0.0045 ± 0.0067) 
is sufficient to demonstrate PCT prediction compatibility.

---

## Evidence Level Definitions

| Level | Description |
|-------|-------------|
| 0 | Protocol only (no execution) |
| 1 | Scaffold/configuration generated |
| 2 | Executed on synthetic/toy data with null controls |
| 3 | Executed on public real data with full null suite |

Current status: All capsules at Level 2 (except Planck at Level 1).
Level 3 requires execution on full GWTC catalog with population-level analysis.

---

## Contact

Repository: KORT-X Research / Beyond Frontiers
Author: Ciprian Stoichici
