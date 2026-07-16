# T1 Horizon-Profile Instantiation Spec

Status: frozen before the T1 refinement ladder.

Provenance label: re-fixed construction, not recovered historical construction.

## Provenance Search

The historical generator for the v56 section V.J.2 refinement table was not
recovered exactly.

Searches performed before freezing this spec:

- `10. FUTURE WORK/PCT_futurework.txt`: sections V.A, V.B, V.G.5, V.J.1,
  V.J.2, and appendix G.2 were found. These give locked parameters and the
  refinement table, but not an executable horizon-profile run card.
- `8. HISTORY/springer-v1-submission/PCT/DELIVERABLES` and
  `8. HISTORY/springer-v1-submission/PCT/HISTORY/Project PCT v55/DELIVERABLES`:
  found the generic `pct_ds.py` spectral-dimension capsule and configs, but no
  horizon-profile implementation or V.J.2 generator.
- `5. ANALYSIS/gw-capsule`: no spectral-dimension horizon-profile generator.
- `9. ARCHIVE/zips`: inspected zip entry names without unpacking; only generic
  DS capsule paths matched (`configs/ds_capsule_v1.json`,
  `outputs/pct_ds_output.txt`, `pct_ds.py`).
- `8. HISTORY/fop-v3-submission/PROPOSED_RESTORATIONS.tex`: found the
  documented restoration statement for the horizon profile and projection
  volume. This is the source used for the re-fixed construction below.

The resulting ladder must therefore be labelled as the re-fixed construction.
It must not be presented as a reproduction of the historical v56 computation.

## Domain And Grid

- Horizon radius: `r_H = 1`.
- Exterior radial interval: `r in (r_H, R]`, with `R = 6 r_H`.
- Constraint grid: midpoint grid with `N` cells in `(r_H, R]`.
- Label grid: fixed midpoint radial label grid with `M = 128` labels on the
  same interval. The label grid is fixed under refinement, matching the P4
  fixed-label-grid convention for the heat-trace diagnostic.
- Boundary at `r -> r_H+`: the first midpoint is used; the singular profile is
  never evaluated at the endpoint. Projection widths are floored as stated
  below to avoid zero-width kernels.

## Profile

The projected environment profile is the static single-source profile

`rho_bar(r) / rho_bar_0 = 1 / (1 - r_H / r)`.

The relative projection volume is

`V_rel(r) = V_Pi(r) / V_Pi(infinity) = exp(-(rho_bar/rho_bar_0 - 1))`.

This is the restoration-file construction, normalized so that `V_rel -> 1`
as `r -> infinity`.

## Projection Kernel

Projection from a constraint-grid point `u` to label `x` is a Gaussian-softmax
density on the fixed label grid:

`Pi(x_i | u) proportional exp(-(x_i - u)^2 / (2 sigma_eff(u)^2))`.

The projection volume modulates the projection width:

`sigma_eff(u) = max(sigma_Pi * V_rel(u), sigma_floor)`.

The numerical floor is

`sigma_floor = 0.05 * label_spacing`.

Each column `Pi(. | u)` is normalized on the label grid, so the projection
kernel is a probability density over labels.

## Constraint Kernel

The constraint-space kernel is Gaussian:

`K(u, v) = exp(-eta * (u - v)^2)`.

The induced correlator on the fixed label grid is

`G_2 = Pi K Pi^T * du^2`.

The symmetric part `(G_2 + G_2^T)/2` is used before eigendecomposition.

## Regularised Inverse

The heat-trace operator is the Tikhonov-regularised inverse of `G_2`.

If `lambda_i` are eigenvalues of `G_2` and `lambda_max = max(lambda_i)`, then

`L_i(lambda_rel) = 1 / (max(lambda_i, 0) + lambda_rel * lambda_max)`.

Central relative regularisation:

`lambda_rel = 1e-6`.

Bootstrap regularisation factors:

`lambda_rel * {0.5, 1, 2}`.

## Scaling Prescriptions

Both scalings are run.

1. Fixed physical parameters:
   - `eta(N) = eta_80`
   - `sigma_Pi(N) = sigma_Pi_80`
   - `rho_crit(N) = rho_crit_80`

2. v56 V.J.1 scaling:
   - `eta(N) * N = eta_80 * 80`
   - `sigma_Pi(N) * sqrt(N) = sigma_Pi_80 * sqrt(80)`
   - `rho_crit(N) * N = rho_crit_80 * 80`

Locked reference values:

- `eta_80 = 2.606e-2`
- `sigma_Pi_80 = 0.35`
- `rho_crit_80 / rho_bar_0 = 5`

`rho_crit` is reported for scaling provenance and for the analytic reference
threshold. The numerical heat-trace construction uses the continuous
projection-volume modulation above, not a hard density cutoff.

## Reference Kappa

The restoration record gives the five-e-folding reference

`kappa_ref = 1 - 1/5 = 0.8`.

The ladder does not force this value. It extracts `kappa = ell_star / r_H`
from the frozen step model.
