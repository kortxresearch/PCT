#!/usr/bin/env python3
"""P4 continuum-limit refinement study (executed numerics for the P4 draft).

Validates the Riemann-sum half of the continuum-limit theorem on the locked
translation-invariant 1D instantiation: Gaussian kernel K, uniform constraint
density, Gaussian projection Pi, label space = interval.

Three experiments, all with the physical parameters held fixed while the
constraint discretisation N refines:

  E1: sup-norm convergence of the normalised correlator Ghat_2^(N) on a fixed
      evaluation grid, against the finest level as reference; empirical order.
  E2: sup-norm convergence of the distance surrogate d_corr = sqrt(-log Ghat_2).
  E3: stability of the spectral-dimension curve d_s(ell) computed from the
      Tikhonov-regularised pseudoinverse of the G_2 Gram matrix on a FIXED
      label grid with FIXED regularisation (the order of limits N -> inf
      before lambda -> 0 is part of the theorem's hypotheses, so lambda is
      held fixed here by design).

What this script does NOT do: it does not adjudicate the horizon-profile
step-height question (Delta d_s drift reported in the v56 lineage, section
V.J.2). That requires the horizon-profile instantiation specification and is
defined as the Part C protocol in the P4 draft.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# Locked physical parameters (fixed under refinement)
L_DOMAIN = 10.0
SIGMA_K = 1.0
SIGMA_PI = 0.35
TIKHONOV_LAMBDA = 1e-6  # fixed by design; see order-of-limits remark
N_LEVELS = [40, 80, 160, 320, 640, 1280]
N_REF = 2560  # reference level for error measurement
EVAL_POINTS = 25  # fixed evaluation grid, away from the boundary
LABEL_POINTS = 100  # fixed label grid for the d_s experiment
ELL_WINDOW = np.logspace(np.log10(0.3), np.log10(3.0), 30)


def constraint_grid(n: int) -> tuple[np.ndarray, float]:
    """Midpoint grid on [0, L] with spacing du (midpoint rule = O(N^-2))."""
    du = L_DOMAIN / n
    u = (np.arange(n) + 0.5) * du
    return u, du


def gaussian_kernel(u: np.ndarray, v: np.ndarray, sigma: float) -> np.ndarray:
    diff = u[:, None] - v[None, :]
    return np.exp(-(diff ** 2) / (2.0 * sigma ** 2))


def projection_density(x: np.ndarray, u: np.ndarray, sigma: float) -> np.ndarray:
    """pi(x|u): Gaussian density in x centred at u. Shape (len(x), len(u))."""
    diff = x[:, None] - u[None, :]
    norm = 1.0 / (math.sqrt(2.0 * math.pi) * sigma)
    return norm * np.exp(-(diff ** 2) / (2.0 * sigma ** 2))


def g2_matrix(x: np.ndarray, n: int) -> np.ndarray:
    """G_2^(N)(x_a, x_b) by the midpoint rule on the constraint grid."""
    u, du = constraint_grid(n)
    A = projection_density(x, u, SIGMA_PI)          # (m, N)
    K = gaussian_kernel(u, u, SIGMA_K)              # (N, N)
    return (A @ K @ A.T) * (du ** 2)


def normalise(G: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(G), 1e-300, None))
    return np.abs(G) / np.outer(d, d)


def d_corr(Ghat: np.ndarray) -> np.ndarray:
    q = -np.log(np.clip(Ghat, 1e-300, 1.0))
    return np.sqrt(np.clip(q, 0.0, None))


def ds_curve(n: int) -> np.ndarray:
    """d_s(ell) from the Tikhonov pseudoinverse of the G_2 Gram matrix on the
    fixed label grid, at constraint refinement level n."""
    xg = np.linspace(0.0, L_DOMAIN, LABEL_POINTS)
    G = g2_matrix(xg, n)
    G = 0.5 * (G + G.T)
    lam, V = np.linalg.eigh(G)
    scale = float(lam.max())
    # L_rho = (G + lambda*scale*I)^{-1} restricted to the spectral resolution
    inv_spec = 1.0 / (np.clip(lam, 0.0, None) + TIKHONOV_LAMBDA * scale)
    spec_L = inv_spec  # eigenvalues of the regularised inverse
    x = np.log(ELL_WINDOW ** 2)
    P = np.array([np.exp(-(e ** 2) * spec_L).sum() for e in ELL_WINDOW])
    y = np.log(np.clip(P, 1e-300, None))
    return -2.0 * np.gradient(y, x)


def empirical_order(errs: list[float]) -> list[float]:
    """log2(err_i / err_{i+1}) between successive dyadic levels."""
    out = []
    for a, b in zip(errs[:-1], errs[1:]):
        if a > 0 and b > 0:
            out.append(round(math.log2(a / b), 3))
        else:
            out.append(float("nan"))
    return out


def main() -> int:
    x_eval = np.linspace(2.0, 8.0, EVAL_POINTS)

    Ghat_ref = normalise(g2_matrix(x_eval, N_REF))
    dcorr_ref = d_corr(Ghat_ref)
    ds_ref = ds_curve(N_REF)

    # Near-diagonal pairs have q -> 0, where d(sqrt q) = dq/(2 sqrt q) amplifies
    # floating-point rounding to the sqrt(machine-eps) ~ 1e-8 floor. Restrict
    # the d_corr error measurement to separated pairs (d_corr_ref > 0.1), where
    # the discretisation error, not arithmetic, is being measured.
    sep_mask = dcorr_ref > 0.1

    e1, e2, e3 = [], [], []
    for n in N_LEVELS:
        Ghat_n = normalise(g2_matrix(x_eval, n))
        e1.append(float(np.max(np.abs(Ghat_n - Ghat_ref))))
        e2.append(float(np.max(np.abs(d_corr(Ghat_n)[sep_mask] - dcorr_ref[sep_mask]))))
        e3.append(float(np.max(np.abs(ds_curve(n) - ds_ref))))

    result = {
        "purpose": "P4 Part B executed numerics: refinement convergence of the locked 1D instantiation.",
        "locked_parameters": {
            "domain_length": L_DOMAIN,
            "sigma_K": SIGMA_K,
            "sigma_Pi": SIGMA_PI,
            "tikhonov_lambda_relative": TIKHONOV_LAMBDA,
            "label_points_fixed": LABEL_POINTS,
            "eval_points_fixed": EVAL_POINTS,
            "reference_level": N_REF,
            "quadrature": "midpoint rule (expected order 2 for smooth integrands)",
        },
        "levels": N_LEVELS,
        "E1_ghat_sup_error": e1,
        "E1_empirical_order": empirical_order(e1),
        "E2_dcorr_sup_error": e2,
        "E2_empirical_order": empirical_order(e2),
        "E3_ds_curve_sup_error": e3,
        "E3_empirical_order": empirical_order(e3),
        "ds_reference_curve": {
            "ell": [round(float(e), 5) for e in ELL_WINDOW],
            "d_s": [round(float(d), 5) for d in ds_ref],
            "d_s_max": round(float(np.max(ds_ref)), 5),
        },
        "part_c_note": (
            "This study covers the translation-invariant instantiation only. The "
            "horizon-profile step-height refinement question (v56 V.J.2 drift "
            "0.79 -> 1.27) is Part C of the P4 draft and requires the locked "
            "horizon-profile specification before it can be executed."
        ),
    }

    out = Path("outputs/p4_continuum_refinement_study.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("E1 Ghat sup-errors :", ["%.3e" % v for v in e1])
    print("E1 orders          :", result["E1_empirical_order"])
    print("E2 dcorr sup-errors:", ["%.3e" % v for v in e2])
    print("E2 orders          :", result["E2_empirical_order"])
    print("E3 ds sup-errors   :", ["%.3e" % v for v in e3])
    print("E3 orders          :", result["E3_empirical_order"])
    print("ds max on window   :", result["ds_reference_curve"]["d_s_max"])
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
