#!/usr/bin/env python3
"""End-to-end public-data capsule: LVK (LOSC) ringdown change-point analysis.

Dataset choice (exactly one; public): one LOSC open strain file (H1) shipped in-repo:
  data/H-H1_LOSC_4_V2-1126259446-32.hdf5

This capsule is intentionally dependency-light:
- required: numpy
- optional: h5py (needed to read the HDF5)

What this script does (end-to-end):
- preprocessing: load strain, bandpass, compute amplitude envelope
- priors / pre-registrations: declare analysis windows and a simple change-point prior
- null tests: (i) off-source windows in the same file, (ii) phase-randomized surrogate
- reporting: write one JSON artifact with both detection and null outcomes
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def _try_import_h5py():
    try:
        import h5py  # type: ignore

        return h5py
    except Exception:
        return None


@dataclass(frozen=True)
class CapsuleConfig:
    dataset_name: str
    dataset_citation: str
    input_path: str

    # Preprocessing
    f_lo_hz: float
    f_hi_hz: float

    # Event location (seconds since file start)
    t_event_s: float

    # Analysis windows relative to t_event
    on_src_start_s: float
    on_src_end_s: float

    off_src1_start_s: float
    off_src1_end_s: float

    off_src2_start_s: float
    off_src2_end_s: float

    # Change-point prior (uniform on this window)
    cp_min_s: float
    cp_max_s: float

    # Null controls
    n_phase_null: int
    null_rng_seed: int

    decision_rule: str


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def read_losc_hdf5(path: str) -> Dict[str, Any]:
    """Read a LOSC HDF5 strain file.
    
    Robustly handles different LOSC versions (V1, V2, V4) by checking
    both metadata groups and dataset attributes.
    """
    h5py = _try_import_h5py()
    if h5py is None:
        raise RuntimeError("h5py is required to read the uploaded .hdf5 strain file")

    # Ensure path exists before opening
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found at: {path}")

    with h5py.File(path, "r") as f:
        # 1. Locate Strain Data
        # Common paths: 'strain/Strain' or just 'strain'
        strain_dset = None
        if "strain" in f:
            if isinstance(f["strain"], h5py.Dataset):
                strain_dset = f["strain"]
            elif "Strain" in f["strain"]:
                strain_dset = f["strain"]["Strain"]
        elif "strain/Strain" in f:
             strain_dset = f["strain/Strain"]
        
        if strain_dset is None:
             raise KeyError("Could not find 'strain/Strain' dataset in HDF5 file.")

        strain = np.array(strain_dset, dtype=float)

        # 2. Helper to find metadata
        def _find_meta(keys_meta: List[str], attr_keys: List[str] = None) -> float | None:
            # Try looking in 'meta' group
            for k in keys_meta:
                if "meta" in f and k in f["meta"]:
                    try:
                        val = f["meta"][k][()]
                        # If it's bytes, try to decode, but only if we expect a number
                        if isinstance(val, (bytes, str)):
                             # Skip strings if we are looking for a float (prevents date parsing errors)
                             continue 
                        return float(np.array(val).reshape(()))
                    except (ValueError, TypeError):
                        continue
            
            # Try looking in Dataset Attributes
            if attr_keys and strain_dset:
                for k in attr_keys:
                    if k in strain_dset.attrs:
                        return float(strain_dset.attrs[k])
            return None

        # 3. Get Sampling Rate (fs)
        # V2 uses 'Xspacing' attribute (dt). fs = 1/dt
        fs = _find_meta(["SamplingRate", "SampleRate"])
        if fs is None:
            # Try Xspacing (dt)
            if "Xspacing" in strain_dset.attrs:
                fs = 1.0 / float(strain_dset.attrs["Xspacing"])
        
        if fs is None:
             raise KeyError("Could not determine SamplingRate from file.")

        # 4. Get GPS Start Time (t0)
        # Do NOT read 'UTCstart' here, as it is a string. Look for 'GPSstart' or 'Xstart'.
        t0_gps = _find_meta(["GPSstart", "t0"], attr_keys=["Xstart"])
        if t0_gps is None:
             raise KeyError("Could not determine GPS start time (t0).")

        # 5. Get Duration
        duration = _find_meta(["Duration"])
        if duration is None:
            duration = len(strain) / fs

    return {"strain": strain, "fs": fs, "duration": duration, "t0_gps": t0_gps}


def bandpass_fft(x: np.ndarray, fs: float, f_lo: float, f_hi: float) -> np.ndarray:
    """Simple FFT bandpass (zero out frequencies outside [f_lo,f_hi])."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= float(f_lo)) & (freqs <= float(f_hi))
    X_f = X * mask
    y = np.fft.irfft(X_f, n=n)
    return np.asarray(y, dtype=float)


def analytic_envelope(x: np.ndarray) -> np.ndarray:
    """Amplitude envelope via Hilbert transform implemented in the FFT domain."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    z = np.fft.ifft(X * h)
    return np.abs(z).astype(float)


def _window_indices(fs: float, t_start: float, t_end: float) -> slice:
    i0 = int(max(0, math.floor(t_start * fs)))
    i1 = int(max(i0 + 1, math.floor(t_end * fs)))
    return slice(i0, i1)


def _aic_gaussian(n: int, sse: float, k: int) -> float:
    sse = max(float(sse), 1e-300)
    return float(n * math.log(sse / float(n)) + 2.0 * float(k))


def _fit_piecewise_constant_one_cp(y: np.ndarray, cp_min: int, cp_max: int) -> Dict[str, Any]:
    """Fit mean-only model with 0 CP vs 1 CP; choose best CP by SSE."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 10:
        raise ValueError("window too small")

    mu0 = float(y.mean())
    sse0 = float(((y - mu0) ** 2).sum())
    aic0 = _aic_gaussian(n, sse0, k=1)

    best = {"sse": float("inf")}
    lo = int(max(1, cp_min))
    hi = int(min(n - 2, cp_max))
    for c in range(lo, hi + 1):
        y1 = y[:c]
        y2 = y[c:]
        mu1 = float(y1.mean())
        mu2 = float(y2.mean())
        sse = float(((y1 - mu1) ** 2).sum() + ((y2 - mu2) ** 2).sum())
        if sse < best["sse"]:
            best = {"cp": int(c), "mu1": mu1, "mu2": mu2, "sse": sse}

    aic1 = _aic_gaussian(n, float(best["sse"]), k=3)  # mu1, mu2, cp

    return {
        "model0": {"mu": mu0, "sse": sse0, "aic": aic0},
        "model1": {**best, "aic": aic1, "delta_mu": float(best["mu2"] - best["mu1"])},
        "delta_aic_1_minus_0": float(aic1 - aic0),
    }


def phase_randomized_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase randomization surrogate preserving the power spectrum (approx. null)."""
    x = np.asarray(x, dtype=float)
    X = np.fft.rfft(x)
    mag = np.abs(X)
    phase = np.angle(X)

    # randomize all non-DC, non-Nyquist phases
    rand = rng.uniform(0.0, 2.0 * math.pi, size=len(phase))
    rand[0] = phase[0]
    if len(phase) > 1:
        rand[-1] = phase[-1]

    Xs = mag * np.exp(1j * rand)
    y = np.fft.irfft(Xs, n=len(x))
    return np.asarray(y, dtype=float)


def run_one_window(
    strain: np.ndarray,
    fs: float,
    t_start: float,
    t_end: float,
    cfg: CapsuleConfig,
) -> Dict[str, Any]:
    sl = _window_indices(fs, t_start, t_end)
    x = strain[sl]

    # preprocessing
    x_bp = bandpass_fft(x, fs, cfg.f_lo_hz, cfg.f_hi_hz)
    env = analytic_envelope(x_bp)
    # use log-envelope (robust-ish), add tiny epsilon
    y = np.log(env + 1e-12)

    # cp prior window in samples within this window
    t_rel = np.arange(len(y)) / fs
    cp_min = int(np.searchsorted(t_rel, cfg.cp_min_s, side="left"))
    cp_max = int(np.searchsorted(t_rel, cfg.cp_max_s, side="right"))

    fit = _fit_piecewise_constant_one_cp(y, cp_min=cp_min, cp_max=cp_max)

    return {
        "window": {"t_start": float(t_start), "t_end": float(t_end), "n": int(len(y))},
        "preprocess": {"f_lo_hz": cfg.f_lo_hz, "f_hi_hz": cfg.f_hi_hz},
        "fit": fit,
    }


def run_capsule(cfg: CapsuleConfig) -> Dict[str, Any]:
    raw = read_losc_hdf5(cfg.input_path)
    strain = raw["strain"]
    fs = float(raw["fs"])
    duration = float(raw["duration"])
    t0_gps = float(raw["t0_gps"])

    # time relative to file start
    t_event = float(cfg.t_event_s)

    on = run_one_window(
        strain,
        fs,
        t_start=t_event + cfg.on_src_start_s,
        t_end=t_event + cfg.on_src_end_s,
        cfg=cfg,
    )

    off1 = run_one_window(
        strain,
        fs,
        t_start=t_event + cfg.off_src1_start_s,
        t_end=t_event + cfg.off_src1_end_s,
        cfg=cfg,
    )

    off2 = run_one_window(
        strain,
        fs,
        t_start=t_event + cfg.off_src2_start_s,
        t_end=t_event + cfg.off_src2_end_s,
        cfg=cfg,
    )

    # phase-randomized nulls (on-source window only)
    rng = np.random.default_rng(cfg.null_rng_seed)
    sl_on = _window_indices(fs, t_event + cfg.on_src_start_s, t_event + cfg.on_src_end_s)
    x_on = strain[sl_on]
    x_on_bp = bandpass_fft(x_on, fs, cfg.f_lo_hz, cfg.f_hi_hz)

    phase_null_daic: List[float] = []
    for _ in range(int(cfg.n_phase_null)):
        surr = phase_randomized_surrogate(x_on_bp, rng)
        env = analytic_envelope(surr)
        y = np.log(env + 1e-12)

        t_rel = np.arange(len(y)) / fs
        cp_min = int(np.searchsorted(t_rel, cfg.cp_min_s, side="left"))
        cp_max = int(np.searchsorted(t_rel, cfg.cp_max_s, side="right"))
        fit = _fit_piecewise_constant_one_cp(y, cp_min=cp_min, cp_max=cp_max)
        phase_null_daic.append(float(fit["delta_aic_1_minus_0"]))

    obs_daic = float(on["fit"]["delta_aic_1_minus_0"])
    p_phase = float((np.array(phase_null_daic) <= obs_daic).mean()) if phase_null_daic else float("nan")

    return {
        "capsule": {"config": asdict(cfg)},
        "provenance": {
            "file": {
                "path": cfg.input_path,
                "fs_hz": fs,
                "duration_s": duration,
                "t0_gps": t0_gps,
            }
        },
        "results": {
            "on_source": on,
            "off_source": {"window1": off1, "window2": off2},
            "nulls": {
                "phase_randomized": {
                    "n": int(len(phase_null_daic)),
                    "p_more_cp_like_than_observed": p_phase,
                    "delta_aic_values": phase_null_daic,
                }
            },
        },
    }


def build_default_config() -> CapsuleConfig:
    # File name indicates LOSC open data around GPS start 1126259446 for 32s.
    # For GW150914, the event is near the middle; we keep a declared default of 16 s.
    t_event = 16.0

    # On-source window: start shortly after merger; focus on "ringdown-like" tail.
    on_start = 0.02
    on_end = 0.30

    # Off-source windows: same length, far from event.
    off1_start = -10.0
    off1_end = off1_start + (on_end - on_start)

    off2_start = +10.0
    off2_end = off2_start + (on_end - on_start)

    # Change-point prior inside on-source window (avoid boundary effects)
    cp_min = 0.05
    cp_max = 0.25

    return CapsuleConfig(
        dataset_name="LIGO LOSC open strain (H1)",
        dataset_citation="LOSC (LIGO Open Science Center) open strain data; file label LOSC_4_V2.",
        input_path="data/H-H1_LOSC_4_V2-1126259446-32.hdf5",
        f_lo_hz=30.0,
        f_hi_hz=500.0,
        t_event_s=t_event,
        on_src_start_s=on_start,
        on_src_end_s=on_end,
        off_src1_start_s=off1_start,
        off_src1_end_s=off1_end,
        off_src2_start_s=off2_start,
        off_src2_end_s=off2_end,
        cp_min_s=cp_min,
        cp_max_s=cp_max,
        n_phase_null=200,
        null_rng_seed=12345,
        decision_rule=(
            "Primary statistic: ΔAIC := AIC(1 change-point) − AIC(no change-point) on log-envelope. "
            "Decision: call 'change-point detection' if ΔAIC ≤ −10 and both off-source windows return "
            "ΔAIC > 0 and phase-randomized null p-value < 0.05; otherwise call 'null / non-detection'."
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=str,
        default="outputs/lvk_ringdown_public_run.json",
        help="Output JSON path.",
    )
    args = p.parse_args(argv)

    cfg = build_default_config()
    payload = run_capsule(cfg)

    out_path = Path(args.out)
    _ensure_parent(out_path)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())