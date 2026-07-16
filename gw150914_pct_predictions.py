#!/usr/bin/env python3
"""GW150914 PCT prediction capsule.

Reads GWTC-1 posterior samples if available and computes the corrected PCT
ringdown change-point timing:

    t_c/M = kappa * (1 + sqrt(1-a_*^2)).

The input mass is treated as source-frame, so the detector-frame conversion
retains the supplied (1+z) factor.  If the HDF5 structure does not expose final
mass/spin in a recognizable form, the script falls back to the literature
summary used by the earlier capsule.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


G = 6.67430e-11
C = 299792458.0
M_SUN = 1.98892e30


def mass_to_time_s(m_solar: Any):
    return (m_solar * M_SUN) * G / (C**3)


def percentile_ci(np, arr, level: float = 68.0) -> list[float]:
    lower = (100.0 - level) / 2.0
    upper = 100.0 - lower
    return [float(np.percentile(arr, lower)), float(np.percentile(arr, upper))]


def summarize(np, arr) -> dict[str, Any]:
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "ci68": percentile_ci(np, arr, 68),
        "ci90": percentile_ci(np, arr, 90),
    }


def find_dataset(samples, names: list[str]):
    for name in names:
        if name in samples:
            return name, samples[name][()]
    if "posterior_samples" in samples:
        ps = samples["posterior_samples"]
        if hasattr(ps, "dtype") and ps.dtype.names:
            data = ps[()]
            for name in names:
                if name in data.dtype.names:
                    return name, data[name]
    return None, None


def load_samples(path: str):
    import h5py
    import numpy as np

    with h5py.File(path, "r") as f:
        group_name = None
        for key in f.keys():
            if "posterior" in key.lower() or "imr" in key.lower():
                group_name = key
                break
        if group_name is None:
            for key in f.keys():
                if isinstance(f[key], h5py.Group):
                    group_name = key
                    break
        if group_name is None:
            raise KeyError("No posterior-like group found")

        samples = f[group_name]
        mass_name, final_mass = find_dataset(
            samples,
            ["final_mass_source", "final_mass", "mf", "Mf", "remnant_mass"],
        )
        spin_name, final_spin = find_dataset(
            samples,
            ["final_spin", "af", "chi_f", "remnant_spin", "a_final"],
        )

        if final_mass is None or final_spin is None:
            m1_name, m1 = find_dataset(samples, ["mass_1_source", "mass_1", "m1", "mass1"])
            m2_name, m2 = find_dataset(samples, ["mass_2_source", "mass_2", "m2", "mass2"])
            if m1 is None or m2 is None:
                raise KeyError("Could not find final mass/spin or component masses")
            final_mass = 0.95 * (np.asarray(m1) + np.asarray(m2))
            final_spin = 0.69 * np.ones_like(final_mass)
            mass_name = f"0.95*({m1_name}+{m2_name})"
            spin_name = "fallback_0.69"

    return np.asarray(final_mass, dtype=float), np.asarray(final_spin, dtype=float), mass_name, spin_name


def fallback_samples():
    import numpy as np

    rng = np.random.default_rng(42)
    n_samples = 10000
    final_mass = rng.normal(62.0, 2.5, n_samples)
    final_spin = rng.normal(0.67, 0.03, n_samples)
    return final_mass, final_spin, "literature_fallback_Mf_62_pm_2p5", "literature_fallback_af_0p67_pm_0p03"


def main() -> int:
    parser = argparse.ArgumentParser(description="GW150914 PCT timing prediction")
    parser.add_argument("--input", default="data/GW150914_GWTC-1.hdf5")
    parser.add_argument("--out", default="outputs/gw150914_pct_predictions.json")
    parser.add_argument("--kappa", type=float, default=0.80)
    parser.add_argument("--redshift", type=float, default=0.09)
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError as exc:
        print(f"Error: numpy is required: {exc}")
        return 1

    try:
        final_mass_source, final_spin, mass_source_name, spin_source_name = load_samples(args.input)
        source_status = "hdf5"
    except Exception as exc:
        print(f"Could not use HDF5 posterior samples ({exc}); falling back to literature values.")
        final_mass_source, final_spin, mass_source_name, spin_source_name = fallback_samples()
        source_status = "fallback"

    final_spin = np.clip(final_spin, 0.0, 0.999999)
    final_mass_detector = final_mass_source * (1.0 + args.redshift)
    spin_factor = 1.0 + np.sqrt(1.0 - final_spin * final_spin)
    tc_over_m_spin = args.kappa * spin_factor
    tc_over_m_nonspin = 2.0 * args.kappa

    t_geo = mass_to_time_s(final_mass_detector)
    tc_spin_s = tc_over_m_spin * t_geo
    tc_nonspin_s = tc_over_m_nonspin * t_geo

    payload = {
        "capsule": {
            "name": "GW150914 PCT Event Prediction",
            "version": "2.0-corrected-spin-redshift",
            "run_id": "deterministic-gw150914-v2",
            "input_file": args.input,
            "input_status": source_status,
            "n_samples": int(len(final_mass_source)),
            "pct_kappa": float(args.kappa),
            "redshift": float(args.redshift),
        },
        "event": {
            "name": "GW150914",
            "gps_time": 1126259462.4,
            "detection_date": "2015-09-14",
        },
        "posterior_sources": {
            "final_mass_source": mass_source_name,
            "final_spin": spin_source_name,
        },
        "posteriors": {
            "final_mass_source_Msun": summarize(np, final_mass_source),
            "final_mass_detector_frame_Msun": summarize(np, final_mass_detector),
            "final_spin": summarize(np, final_spin),
        },
        "pct_predictions": {
            "formula": "t_c/M = kappa*(1 + sqrt(1-a_*^2))",
            "nonspinning_limit": "t_c/M = 2*kappa",
            "kappa": float(args.kappa),
            "tc_over_M_spin_scaled": summarize(np, tc_over_m_spin),
            "tc_over_M_nonspinning": float(tc_over_m_nonspin),
            "tc_spin_scaled_ms": summarize(np, tc_spin_s * 1000.0),
            "tc_nonspinning_ms": summarize(np, tc_nonspin_s * 1000.0),
            "interpretation": (
                "For the fallback/literature GW150914 values, the spin-scaled "
                "detector-frame timing is about 0.46 ms and the non-spinning "
                "detector-frame timing is about 0.53 ms."
            ),
        },
        "comparison_to_ringdown_capsule": {
            "note": (
                "The lvk_ringdown_end_to_end.py capsule tests change-point "
                "specificity using AIC-based segmentation on the log-envelope."
            ),
            "on_source_window_ms": "20-300 ms post-merger",
            "predicted_tc_in_window": False,
            "status": "Ringdown capsule returned NULL; current sensitivity is not expected to be decisive.",
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("GW150914 corrected PCT timing summary")
    print(f"input status: {source_status}")
    print(f"final mass source: {float(np.mean(final_mass_source)):.2f} Msun")
    print(f"final spin: {float(np.mean(final_spin)):.3f}")
    print(f"spin-scaled t_c/M: {float(np.mean(tc_over_m_spin)):.3f}")
    print(f"spin-scaled t_c: {float(np.mean(tc_spin_s * 1000.0)):.3f} ms")
    print(f"nonspinning t_c: {float(np.mean(tc_nonspin_s * 1000.0)):.3f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
