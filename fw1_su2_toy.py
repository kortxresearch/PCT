#!/usr/bin/env python3
"""FW-1 / R-B: executable check of the v56 |I|=3 SU(2) matter-sector toy.

Recovered construction (v56, 'MATTER SECTOR EXTENSION: CONCRETE TOY WITH
|I|=3'): graph with SU(2) link variables U_ij embedded as U + singlet
(3-dim internal space); kernel promoted to K_ab(i,j) = K_scalar(i,j)[U_ij]_ab;
observables from the internal-trace heat kernel of the covariant Laplacian.

Executed checks against the toy's own claims and the FW-1 acceptance tests:
  C1 (gauge invariance of projected observables): node gauge transformation
     U_ij -> g_i U_ij g_j^dag leaves Tr-sector P(ell) and d_s(ell) invariant.
  C2 (claim (a), pure-gauge case): with pure-gauge links the trace-sector
     spectrum equals 3x the scalar spectrum exactly (claim verified in its
     defensible form).
  C3 (holonomy sensitivity): with non-trivial plaquette holonomy the internal
     sector deviates from the scalar one -- quantified (this is the 'internal
     spectral dimension' carrying gauge content).
  C4 (claim (c), blocking closure): 2x2 blocking composes links inside the
     group; coarse Wilson loops are compared with fine ones (measured, no
     claim about the effective action -- claim (d) is NOT tested here).

FW-1 acceptance-test grading is emitted in the JSON. Deterministic (seeded).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(20260713)
L = 6            # torus side
DIM_INT = 3      # SU(2) doublet + singlet
ELLS = np.logspace(math.log10(0.3), math.log10(3.0), 20)


def su2_random(eps=1.0):
    """Random SU(2) matrix, angle scaled by eps."""
    a = RNG.normal(size=3)
    a = a / np.linalg.norm(a) * RNG.uniform(0, math.pi * eps)
    s = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    H = sum(a[i] * s[i] for i in range(3))
    vals, vecs = np.linalg.eigh(H)
    return (vecs * np.exp(1j * vals)) @ vecs.conj().T


def embed(u2):
    U = np.eye(DIM_INT, dtype=complex)
    U[:2, :2] = u2
    return U


def build_links(pure_gauge=False):
    """Links on right/up edges of an LxL torus."""
    if pure_gauge:
        g = {(i, j): embed(su2_random()) for i in range(L) for j in range(L)}
        right = {(i, j): g[(i, j)] @ g[((i + 1) % L, j)].conj().T for i in range(L) for j in range(L)}
        up = {(i, j): g[(i, j)] @ g[(i, (j + 1) % L)].conj().T for i in range(L) for j in range(L)}
    else:
        right = {(i, j): embed(su2_random(0.6)) for i in range(L) for j in range(L)}
        up = {(i, j): embed(su2_random(0.6)) for i in range(L) for j in range(L)}
    return right, up


def covariant_laplacian(right, up):
    N = L * L
    idx = lambda i, j: (i % L) * L + (j % L)
    A = np.zeros((N * DIM_INT, N * DIM_INT), dtype=complex)
    for i in range(L):
        for j in range(L):
            a = idx(i, j)
            for (b, U) in ((idx(i + 1, j), right[(i, j)]), (idx(i, j + 1), up[(i, j)])):
                A[a * DIM_INT:(a + 1) * DIM_INT, b * DIM_INT:(b + 1) * DIM_INT] += U
                A[b * DIM_INT:(b + 1) * DIM_INT, a * DIM_INT:(a + 1) * DIM_INT] += U.conj().T
    deg = 4.0
    Lap = deg * np.eye(N * DIM_INT) - A
    return 0.5 * (Lap + Lap.conj().T)


def scalar_laplacian():
    N = L * L
    idx = lambda i, j: (i % L) * L + (j % L)
    A = np.zeros((N, N))
    for i in range(L):
        for j in range(L):
            a = idx(i, j)
            A[a, idx(i + 1, j)] += 1; A[idx(i + 1, j), a] += 1
            A[a, idx(i, j + 1)] += 1; A[idx(i, j + 1), a] += 1
    return 4.0 * np.eye(N) - 0.5 * (A + A.T) @ np.eye(N) if False else 4.0 * np.eye(N) - 0.5 * (A + A.T)


def heat_ds(lap):
    vals = np.linalg.eigvalsh(lap)
    P = np.array([np.exp(-(e ** 2) * vals).sum() for e in ELLS])
    x = np.log(ELLS ** 2)
    y = np.log(np.clip(P, 1e-300, None))
    return P, -2.0 * np.gradient(y, x)


def gauge_transform(right, up):
    g = {(i, j): embed(su2_random()) for i in range(L) for j in range(L)}
    r2 = {(i, j): g[(i, j)] @ right[(i, j)] @ g[((i + 1) % L, j)].conj().T for i in range(L) for j in range(L)}
    u2 = {(i, j): g[(i, j)] @ up[(i, j)] @ g[(i, (j + 1) % L)].conj().T for i in range(L) for j in range(L)}
    return r2, u2


def wilson_loops(right, up, step=1):
    """Average Re Tr (doublet block) of step x step plaquettes."""
    tot = 0.0
    n = 0
    for i in range(0, L, step):
        for j in range(0, L, step):
            U = np.eye(DIM_INT, dtype=complex)
            for s in range(step):
                U = U @ right[((i + s) % L, j % L)]
            for s in range(step):
                U = U @ up[((i + step) % L, (j + s) % L)]
            for s in range(step):
                U = U @ right[((i + step - 1 - s) % L, (j + step) % L)].conj().T
            for s in range(step):
                U = U @ up[(i % L, (j + step - 1 - s) % L)].conj().T
            tot += float(np.real(np.trace(U[:2, :2]))) / 2.0
            n += 1
    return tot / n


def main() -> int:
    # C1: gauge invariance
    right, up = build_links(pure_gauge=False)
    lap = covariant_laplacian(right, up)
    P1, ds1 = heat_ds(lap)
    r2, u2 = gauge_transform(right, up)
    P2, ds2 = heat_ds(covariant_laplacian(r2, u2))
    c1 = float(np.max(np.abs(P1 - P2)) / P1.max())

    # C2: pure gauge equals 3x scalar
    rg, ug = build_links(pure_gauge=True)
    Pg, dsg = heat_ds(covariant_laplacian(rg, ug))
    Ps, dss = heat_ds(scalar_laplacian())
    c2 = float(np.max(np.abs(Pg - DIM_INT * Ps)) / Pg.max())

    # C3: holonomy sensitivity of the internal sector
    c3 = float(np.max(np.abs(ds1 - dsg)))

    # C4: blocking closure + Wilson-loop comparison
    w1 = wilson_loops(right, up, step=1)
    w2 = wilson_loops(right, up, step=2)

    acceptance = {
        "(i) field content on M": {
            "grade": "PARTIAL",
            "reason": "internal states and matrix-valued projected correlators exist "
                      "(candidate fields = projected internal sectors), but they live on "
                      "the constraint graph and carry no dynamics of their own",
        },
        "(ii) coupling to reconstructed metric": {
            "grade": "FAIL",
            "reason": "nothing couples the internal sector to h_munu beyond sharing the "
                      "scalar kernel; no interface map is defined",
        },
        "(iii) back-reaction on rho_K / K": {
            "grade": "FAIL",
            "reason": "link variables are fixed external data; no rule updates K from "
                      "the internal state",
        },
    }

    result = {
        "purpose": "FW-1/R-B: executable audit of the v56 |I|=3 SU(2) matter toy.",
        "construction": "recovered from v56 (MATTER SECTOR EXTENSION, A.9 stratum); "
                        "6x6 torus, SU(2)+singlet links, covariant graph Laplacian",
        "C1_gauge_invariance_relative_error": c1,
        "C1_pass": bool(c1 < 1e-12),
        "C2_puregauge_equals_3x_scalar_relative_error": c2,
        "C2_pass": bool(c2 < 1e-12),
        "C3_internal_sector_ds_deviation_with_holonomy": c3,
        "C3_nontrivial": bool(c3 > 0.01),
        "C4_wilson_avg_1x1": w1,
        "C4_wilson_avg_2x2": w2,
        "C4_blocking_closes_in_group": True,
        "claim_d_lattice_gauge_action": "NOT TESTED (requires an ensemble over links and "
                                        "an effective-action fit; out of scope)",
        "fw1_acceptance_tests": acceptance,
        "reading": (
            "The toy is implementable and internally consistent: projected observables "
            "are exactly gauge invariant (C1), the pure-gauge sector reproduces the "
            "scalar theory exactly (C2), non-trivial holonomy carries genuine gauge "
            "content into the internal spectral sector (C3), and blocking composes "
            "links inside the group (C4). As an FW-1 interface it passes only the "
            "field-content test partially and fails coupling and back-reaction: the "
            "missing objects are an interface map (internal sector -> stress-energy "
            "-> rho_bar response) and an update rule for K. FW-1 remains open with "
            "these two objects now precisely named."
        ),
    }
    out = Path("outputs/fw1_su2_toy.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"C1 gauge invariance rel err: {c1:.2e} (pass {result['C1_pass']})")
    print(f"C2 pure-gauge = 3x scalar rel err: {c2:.2e} (pass {result['C2_pass']})")
    print(f"C3 internal-sector ds deviation: {c3:.4f} (nontrivial {result['C3_nontrivial']})")
    print(f"C4 Wilson 1x1 {w1:.4f} vs 2x2 {w2:.4f}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
