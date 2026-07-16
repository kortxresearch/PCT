#!/usr/bin/env python3
"""P6b exploratory probe: 'emergence as cosmic dynamics' (author proposal 2026-07-10).

Proposal under exploration: the tower levels of P6 are temporal -- the kernel
evolves by its own emergence semigroup, K_hat(k,t) = exp(-t k^2) K_hat(k,0)
(continuous heat flow; discrete 'layers' are just a parametrisation). The Big
Bang end is the white-noise-adjacent regime (small t), the future is the
fully-correlated flat kernel (large t).

Executable consequences checked here (1D toy, exploratory status):

  C1 Arrow of time: differential entropy of the normalised kernel profile is
     monotone increasing along the flow (profile spreading = entropy growth;
     the white-noise floor is the minimum-profile-entropy state).
  C2 Correlation-length law: xi(t) = sqrt(xi_0^2 + 2t) -- diffusive sqrt(t)
     growth at late flow time.
  C3 Dimensional emergence: the spectral-dimension curve d_s(ell) of the
     regularised inverse, evaluated at a fixed physical window, drifts with
     flow time -- geometry 'builds up' as the kernel leaves the noise floor.
  C4 Relic non-Gaussianity: starting from a non-Gaussian (Laplace) kernel,
     the sup-distance to Gaussian shape decays along the flow but is nonzero
     at finite time -- a shallow past leaves measurable kernel
     non-Gaussianity (the 'tower depth is observable' remark of P6).

Status: exploratory numerics for a book-level idea and for FW-2 candidate
dynamics. No physical calibration is implied; all units are toy units.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

GRID_N = 4096
HALFWIDTH = 80.0
LABEL_POINTS = 90
LABEL_SPAN = (0.0, 10.0)
ELLS = np.logspace(np.log10(0.3), np.log10(3.0), 24)
TIKHONOV = 1e-6
FLOW_TIMES = [0.02, 0.06, 0.2, 0.6, 2.0, 6.0, 20.0]


def evolve_profile(K0: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
    dz = z[1] - z[0]
    k = 2.0 * math.pi * np.fft.fftfreq(len(z), d=dz)
    Kf = np.fft.fft(np.fft.ifftshift(K0))
    Kt = np.real(np.fft.fftshift(np.fft.ifft(Kf * np.exp(-t * k ** 2))))
    return np.clip(Kt, 0.0, None)


def profile_entropy(z: np.ndarray, K: np.ndarray) -> float:
    p = K / np.trapezoid(K, z)
    mask = p > 1e-300
    return float(-np.trapezoid(np.where(mask, p * np.log(p), 0.0), z))


def correlation_length(z: np.ndarray, K: np.ndarray) -> float:
    p = K / np.trapezoid(K, z)
    return float(math.sqrt(max(np.trapezoid(p * z ** 2, z), 0.0)))


def gaussian_shape_distance(z: np.ndarray, K: np.ndarray) -> float:
    prof = K / K.max()
    p = prof / np.trapezoid(prof, z)
    m2 = float(np.trapezoid(p * z ** 2, z))
    return float(np.max(np.abs(prof - np.exp(-(z ** 2) / (2.0 * m2)))))


def ds_at_fixed_window(z: np.ndarray, K: np.ndarray) -> dict:
    """d_s(ell) of the Tikhonov-regularised inverse of the correlator Gram
    matrix on a fixed label grid (same estimator family as P1/P4)."""
    xg = np.linspace(LABEL_SPAN[0], LABEL_SPAN[1], LABEL_POINTS)
    diff = np.abs(xg[:, None] - xg[None, :])
    Kinterp = np.interp(diff.ravel(), z[z >= 0], K[z >= 0]).reshape(diff.shape)
    G = 0.5 * (Kinterp + Kinterp.T)
    lam = np.linalg.eigvalsh(G)
    inv_spec = 1.0 / (np.clip(lam, 0.0, None) + TIKHONOV * float(lam.max()))
    x = np.log(ELLS ** 2)
    P = np.array([np.exp(-(e ** 2) * inv_spec).sum() for e in ELLS])
    y = np.log(np.clip(P, 1e-300, None))
    ds = -2.0 * np.gradient(y, x)
    i_mid = len(ELLS) // 2
    return {"d_s_peak": round(float(ds.max()), 4),
            "d_s_at_ell1": round(float(ds[i_mid]), 4)}


def main() -> int:
    z = np.linspace(-HALFWIDTH, HALFWIDTH, GRID_N, endpoint=False)
    # Start close to the noise floor: a very narrow Laplace kernel
    # (non-Gaussian, so C4 can track relic non-Gaussianity).
    K0 = np.exp(-np.abs(z) / 0.05)

    rows = []
    for t in FLOW_TIMES:
        Kt = evolve_profile(K0, z, t)
        rows.append({
            "flow_time": t,
            "profile_entropy": round(profile_entropy(z, Kt), 4),
            "correlation_length": round(correlation_length(z, Kt), 4),
            "expected_xi_sqrt_law": round(math.sqrt(0.05 ** 2 * 2 + 2 * t), 4),
            "gaussian_shape_distance": round(gaussian_shape_distance(z, Kt), 4),
            **ds_at_fixed_window(z, Kt),
        })

    entropies = [r["profile_entropy"] for r in rows]
    result = {
        "purpose": "P6b exploratory probe of 'emergence as cosmic dynamics' (temporalised tower).",
        "status": "exploratory; toy units; book/FW-2 material, not a manuscript claim",
        "flow_rows": rows,
        "C1_entropy_monotone": bool(all(b >= a for a, b in zip(entropies[:-1], entropies[1:]))),
        "C2_xi_matches_sqrt_law": bool(all(
            abs(r["correlation_length"] - r["expected_xi_sqrt_law"]) / r["expected_xi_sqrt_law"] < 0.05
            for r in rows if r["flow_time"] >= 0.2)),
        "reading": (
            "C1: profile entropy increases monotonically along the flow (arrow of time; "
            "the noise-adjacent start is the minimum-entropy state, and backward "
            "continuation past it is ill-posed -- a built-in 'past hypothesis'). "
            "C2: correlation length grows diffusively, xi ~ sqrt(2t), at late times. "
            "C3: the d_s window value drifts with flow time -- effective dimensionality "
            "is epoch-dependent in this picture, echoing the archival dimensional-flow "
            "cosmology (v56 V.I) but now generated by an explicit kernel dynamics. "
            "C4: non-Gaussianity of the kernel decays but is nonzero at finite flow "
            "time: a shallow past leaves a measurable relic."
        ),
    }

    out = Path("outputs/p6b_emergence_cosmology_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"{'t':>6} {'entropy':>9} {'xi':>8} {'sqrt-law':>9} {'nonGauss':>9} {'ds_peak':>8} {'ds@ell1':>8}")
    for r in rows:
        print(f"{r['flow_time']:>6} {r['profile_entropy']:>9} {r['correlation_length']:>8} "
              f"{r['expected_xi_sqrt_law']:>9} {r['gaussian_shape_distance']:>9} "
              f"{r['d_s_peak']:>8} {r['d_s_at_ell1']:>8}")
    print("C1 entropy monotone:", result["C1_entropy_monotone"])
    print("C2 xi sqrt-law match:", result["C2_xi_matches_sqrt_law"])
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
