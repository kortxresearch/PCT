"""pct_ds.py

Reference implementation (non-speculative deliverable): compute spectral dimension d_s(ℓ)
from a weighted graph Laplacian / kernel-defined adjacency.

Design goals:
- Minimal dependencies: numpy only.
- Two estimators for Tr(exp(-t L)):
  (1) exact eigendecomposition (small graphs)
  (2) Hutchinson stochastic trace estimator (larger graphs)
- Basic refinement diagnostics: compare d_s(ℓ) curves under graph coarsening.

This script is deliberately standalone; it is not tied to any PCT-specific claims.
"""

from __future__ import annotations

import numpy as np


def _symmetrize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def laplacian_from_adjacency(W: np.ndarray, normalized: bool = True, eps: float = 1e-12) -> np.ndarray:
    """Build a (combinatorial or normalized) Laplacian from a symmetric weighted adjacency W."""
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix")
    W = _symmetrize(W)
    np.fill_diagonal(W, 0.0)

    d = W.sum(axis=1)
    if not normalized:
        return np.diag(d) - W

    # L = I - D^{-1/2} W D^{-1/2}
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(d, eps)))
    return np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt


def heat_trace_eig(L: np.ndarray, ell: float) -> float:
    """Exact Tr(exp(-ell^2 L)) via eigendecomposition (O(n^3))."""
    lam = np.linalg.eigvalsh(_symmetrize(L))
    return float(np.exp(-(ell ** 2) * lam).sum())


def heat_trace_hutchinson(L: np.ndarray, ell: float, n_samples: int = 64, rng_seed: int = 0) -> float:
    """Approximate Tr(exp(-ell^2 L)) using Hutchinson estimator with a truncated Taylor series.

    Notes:
    - This is a minimal implementation; for serious use you would likely prefer
      Chebyshev/Lanczos methods.
    """
    L = _symmetrize(L)
    n = L.shape[0]
    rng = np.random.default_rng(rng_seed)

    # crude truncation order; increase for accuracy at larger ell
    m = 30
    t = ell ** 2

    def expm_series_apply(v: np.ndarray) -> np.ndarray:
        # exp(-tL)v ≈ Σ_{k=0}^{m} (-t)^k / k! * L^k v
        out = v.copy()
        term = v.copy()
        for k in range(1, m + 1):
            term = L @ term
            out = out + ((-t) ** k / float(np.math.factorial(k))) * term
        return out

    acc = 0.0
    for _ in range(n_samples):
        z = rng.choice([-1.0, 1.0], size=n)  # Rademacher
        y = expm_series_apply(z)
        acc += float(z @ y)

    return acc / float(n_samples)


def spectral_dimension_curve(
    W: np.ndarray,
    ells: np.ndarray,
    normalized_laplacian: bool = True,
    estimator: str = "eig",
    n_samples: int = 64,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (P(ell), d_s(ell)) for a list of length scales ell.

    d_s(ell) := -2 * d ln Tr(exp(-ell^2 L)) / d ln(ell^2)
    computed by finite differences in log-space.
    """
    ells = np.asarray(ells, dtype=float)
    if np.any(ells <= 0):
        raise ValueError("ells must be positive")

    L = laplacian_from_adjacency(W, normalized=normalized_laplacian)

    P = np.empty_like(ells)
    if estimator == "eig":
        for i, ell in enumerate(ells):
            P[i] = heat_trace_eig(L, float(ell))
    elif estimator == "hutch":
        for i, ell in enumerate(ells):
            P[i] = heat_trace_hutchinson(L, float(ell), n_samples=n_samples, rng_seed=rng_seed)
    else:
        raise ValueError("estimator must be 'eig' or 'hutch'")

    # finite differences for d_s
    x = np.log(ells ** 2)
    y = np.log(np.maximum(P, 1e-300))
    dy_dx = np.gradient(y, x)
    d_s = -2.0 * dy_dx

    return P, d_s


def coarsen_by_pairing(W: np.ndarray) -> np.ndarray:
    """Very simple coarsening: pair nodes (0,1), (2,3), ... and sum weights.

    This is *only* a refinement diagnostic placeholder.
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    m = n // 2
    if m < 2:
        raise ValueError("graph too small to coarsen")

    Wc = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(m):
            ii = [2 * i, 2 * i + 1]
            jj = [2 * j, 2 * j + 1]
            Wc[i, j] = W[np.ix_(ii, jj)].sum()
    np.fill_diagonal(Wc, 0.0)
    return _symmetrize(Wc)


def refinement_diagnostic(W: np.ndarray, ells: np.ndarray) -> dict:
    """Compute a minimal stability report comparing d_s curves under coarsening."""
    P0, ds0 = spectral_dimension_curve(W, ells, estimator="eig")
    Wc = coarsen_by_pairing(W)
    P1, ds1 = spectral_dimension_curve(Wc, ells, estimator="eig")

    return {
        "ells": ells,
        "ds_fine": ds0,
        "ds_coarse": ds1,
        "max_abs_diff": float(np.max(np.abs(ds0 - ds1))),
    }


if __name__ == "__main__":
    # Tiny demo: a 1D chain with exponential weights.
    n = 40
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            W[i, j] = np.exp(-abs(i - j))

    ells = np.logspace(-1, 1, 25)

    P, ds = spectral_dimension_curve(W, ells, estimator="eig")
    report = refinement_diagnostic(W, ells)

    print("ell\tP(ell)\td_s(ell)")
    for ell, p, d in zip(ells, P, ds):
        print(f"{ell:.3e}\t{p:.6e}\t{d:.6f}")

    print("\nRefinement diagnostic: max |Delta d_s| =", report["max_abs_diff"])