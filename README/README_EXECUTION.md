# PCT DELIVERABLES - Execution Guide 

**Last Updated**: 2026-02-12

**Repository**: https://github.com/kortxresearch/PCT  
**Archive**: https://osf.io/9vc3x

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

## Capsule Status (All Level 2 - Executed)

| Capsule | Script | Level | Output |
|---------|--------|-------|--------|
| Spectral Dimension | `pct_ds.py` | 2 | `outputs/pct_ds_output.txt` |
| Change-Point | `gw_change_point_runner.py` | 2 | `outputs/gw_change_point_runner.json` |
| Planck MCMC | `planck2018_running_inference.py` | 2 | `outputs/planck2018_running.json` |
| LVK Ringdown | `lvk_ringdown_end_to_end.py` | 2 | `outputs/lvk_ringdown_public_run.json` |
| GW150914 Predictions | `gw150914_pct_predictions.py` | 2 | `outputs/gw150914_pct_predictions.json` |

---

## Key Results Summary

### 1. Spectral Dimension
```
d_s peak:           1.723 at ℓ = 1.468
Refinement check:   max|Δd_s| = 0.453
```

### 2. Change-Point Detection
```
Breakpoint:         index 60 (2020-01-01)
Segment means:      [10.11, 12.23]
Mean jump:          Δμ = 2.12 (21% step)
```

### 3. Planck MCMC Inference (NEW in v60)
```
Convergence:        R-1 = 0.0079 (< 0.01) ✓
Samples:            24,080 raw / 40,490 weighted

α_s (running):      -0.0037 ± 0.0069
68% CI:             [-0.0108, +0.0032]
95% CI:             [-0.0174, +0.0095]
n_s:                0.9648 ± 0.0043
H0:                 67.38 km/s/Mpc

PCT prediction:     α_s = -0.012 ± 0.005
Tension:            0.97σ — COMPATIBLE ✓
```

### 4. LVK Ringdown
```
On-source ΔAIC:     -219.27
Off-source ΔAIC:    -23.70, -206.59
Null p-value:       0.12
OUTCOME:            NULL (correct)
```

### 5. GW150914 PCT Predictions
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
├── data/
│   ├── gw_synth_monthly.csv
│   ├── H-H1_LOSC_4_V2-1126259446-32.hdf5
│   └── GW150914_GWTC-1.hdf5
├── outputs/
│   ├── pct_ds_output.txt
│   ├── gw_change_point_runner.json
│   ├── planck2018_running.json
│   ├── lvk_ringdown_public_run.json
│   ├── gw150914_pct_predictions.json
│   └── sha256SUMS.txt
├── chains/
│   └── planck2018_running_rs.1.txt (11 MB)
├── pct_ds.py
├── gw_change_point_runner.py
├── planck2018_running_inference.py
├── lvk_ringdown_end_to_end.py
├── gw150914_pct_predictions.py
├── planck_running.yaml
├── run_capsules.py
├── requirements.txt
├── DELIVERABLES.txt
└── README_EXECUTION.md
```

---

## Evidence Level Definitions

| Level | Description |
|-------|-------------|
| 0 | Protocol only (no execution) |
| 1 | Scaffold/configuration generated |
| 2 | Executed on synthetic/toy data with null controls |
| 3 | Executed on public real data with full null suite |

**Current status**: All 5 capsules at Level 2.

---

## Contact: Contact@kort-x.com

Repository: KORT-X Research 
Author: Ciprian Stoichici
