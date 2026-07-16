# T2 Fresh Horizon-Profile Construction Spec

Status: pre-registered and frozen before the first T2 refinement-ladder run.

Provenance: a fresh construction designed for T2. It is not a reconstruction
of the historical v56 generator and must not be used to retrofit v4.

## Epistemic ledger

- `[locked]` T2 tests a Euclidean heat-kernel diagnostic only. No Lorentzian
  propagation claim is made.
- `[assumed]` A hard environment-density threshold changes the transverse
  constraint operator from second order to fourth order.
- `[derived]` The resulting finite-grid operator is real, symmetric, and
  positive semidefinite because each transverse Fourier block is a Neumann
  radial Laplacian plus a non-negative diagonal potential.
- `[locked]` All construction parameters and both scaling prescriptions below
  are fixed before the first declared ladder run.

## 2+1-dimensional domain

Use the near-horizon planar chart

`(r, y, tau) in [r_H, R] x [0, L_perp) x [0, L_perp)`, with

- `r_H = 1`, `R = 5`, and `L_perp = 4`;
- `r` and `y` interpreted as the two spatial coordinates;
- `tau` interpreted as Euclidean time;
- reflecting (Neumann) radial boundaries and periodic `y,tau` boundaries.

At ladder level `N`, all three directions have `N` cells, so the operator acts
on `N^3` degrees of freedom. The isotropic spacing is `h = 4/N`. The declared
levels are

`N = 12, 16, 20, 24, 28, 32`.

Every level is divisible by four so that the fixed-physical critical surface
is a cell face rather than moving sub-cell under refinement.

## Genuine discontinuous threshold

The dimensionless horizon profile is

`rho_bar(r)/rho_0 = 1/(1-r_H/r)`.

Define the phase indicator with a literal Heaviside threshold,

`chi_N(r) = 1{rho_bar(r)/rho_0 >= rho_crit(N)}`.

Thus `chi_N` jumps from one to zero at `r_c(N)`; it is not a sigmoid, smooth
width modulation, or probabilistic interpolation. Grid cells are classified
by their midpoint. Because the declared critical surfaces are cell faces, no
midpoint lies exactly at the threshold.

## Threshold-sensitive unnormalised operator

Let `A_perp,N` be the continuum-scaled, **unnormalised** nearest-neighbour
periodic graph Laplacian in the `(y,tau)` plane, and let `L_r,N` be the
continuum-scaled unnormalised nearest-neighbour radial graph Laplacian with
Neumann boundaries. Define

`L_N = L_r,N + (1-chi_N) A_perp,N + chi_N A_perp,N^2/k_0^2`,

where the products act plane-wise and

`k_0 = 2*pi/L_perp = pi/2`.

This is the minimal positive order-switch construction: outside the threshold
the transverse operator is second order; inside it is fourth order. Matching
at the first non-zero continuum transverse wavenumber fixes the coefficient,
so no amplitude is fitted to the eventual `d_s` curve.

The operator is deliberately not degree-normalised. A hard change
`mu -> mu^2/k_0^2` in each transverse Fourier eigenvalue `mu` survives in the
spectrum and cannot cancel against a degree matrix as in T1b.

For transverse mode `(p,q)`,

`mu_pq = 4/h^2 [sin^2(pi p/N) + sin^2(pi q/N)]`,

and the exact radial block is

`L_r,N + diag((1-chi_j) mu_pq + chi_j mu_pq^2/k_0^2)`.

All block eigenvalues are computed by symmetric tridiagonal eigensolution. No
stochastic trace approximation is used for the central curve.

## Scaling prescriptions

Both prescriptions keep the domain, `k_0`, boundary conditions, and stencil
normalisations fixed.

1. `fixed_physical`
   - `[locked]` `rho_crit = 2` at every level;
   - `[derived]` `r_c = 2 r_H`, so the critical layer has physical width one
     and contains `N/4` radial cells.

2. `fixed_critical_cells`
   - `[assumed]` the threshold layer is microscopic and remains exactly three
     radial cells thick: `r_c(N) = r_H + 3h`;
   - `[derived]`
     `rho_crit(N)/rho_0 = r_c(N)/(r_c(N)-r_H)`;
   - this coincides with `fixed_physical` at `N=12` and shrinks to zero
     physical width as `N -> infinity`.

The second prescription is a declared scaling stress test, not a claim that
the microscopic prescription is physically preferred.

## Reproducibility

- Central spectra and heat traces are deterministic and exact up to floating
  point eigensolver arithmetic.
- Fourier-block cluster bootstrap uses `256` resamples and the fixed base seed
  `20260710`.
- JSON payloads contain no wall-clock fields.
- The script must verify this spec, the extraction rule, and its own SHA256
  against `outputs/t2_freeze_manifest.json` before running the ladder.
