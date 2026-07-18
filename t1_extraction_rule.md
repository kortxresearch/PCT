# T1 Extraction Rule

Status: frozen before the T1 refinement ladder.

## Spectral-Dimension Estimator

For each refinement level and scaling prescription:

1. Build `G_2` from the frozen horizon-profile spec.
2. Compute the eigenvalues of the symmetrized `G_2`.
3. Form the Tikhonov-regularised inverse spectrum
   `L_i = 1 / (max(lambda_i, 0) + lambda_rel * lambda_max)`.
4. Compute the heat trace
   `P(ell) = sum_i exp(-ell^2 L_i)`.
5. Compute
   `d_s(ell) = -2 d log(P) / d log(ell^2)`
   by finite differences on the declared log-spaced `ell` grid.

`ell` grid:

- Full diagnostic grid: 96 log-spaced points from `ell = 0.15` to `ell = 6.0`.
- Central fit window: `ell in [0.35, 4.5]`.

## Step Fit

The step model is a two-plateau model on `x = log(ell)`:

- For `x < x0`: `d_s = d_pre`.
- For `x >= x0`: `d_s = d_post`.
- `Delta d_s = d_pre - d_post`.
- `ell_star = exp(x0)`.

The split point is scanned over all interior grid gaps in the fit window with
at least four points on each side. At each split, `d_pre` and `d_post` are the
least-squares means on the two sides. The central split is the one with minimum
sum of squared residuals.

## Smooth Comparator

The smooth model is a sigmoid on `x = log(ell)`:

`d_s = d_post + Delta d_s / (1 + exp((x - x0) / w))`.

The scan uses:

- `x0`: all fit-window grid points excluding the outer four points.
- `w`: 24 log-spaced values from `0.08` to `1.2`.

For each `(x0, w)`, `d_post` and `Delta d_s` are fit by linear least squares.

## Model Selection

`AIC = n log(SSE / n) + 2k`, with `SSE` floored at `1e-300`.

- Step model parameter count: `k = 3`.
- Smooth sigmoid parameter count: `k = 4`.
- `Delta AIC = AIC_smooth - AIC_step`.
- A level is adjudicable only if `Delta AIC >= 10` for the step model and
  `Delta d_s > 0`.
- Otherwise the level is reported as `no adjudicable step at this N`.

## Uncertainties

Uncertainties are deterministic nuisance bootstraps over:

- lower fit-window endpoint shifted by `{-1, 0, +1}` ell-grid point;
- upper fit-window endpoint shifted by `{-1, 0, +1}` ell-grid point;
- regularisation factor `{0.5, 1, 2}` times the central `lambda_rel`.

Reported one-sigma errors are sample standard deviations over the 27 nuisance
fits. The central values remain the central-window, central-regularisation
fit.

## Beta And Continuum Decision Thresholds

For adjacent adjudicable levels,

`beta_hat(N1 -> N2) = (Delta d_s(N2) - Delta d_s(N1)) / log(N2 / N1)`.

Operational beta-to-zero criterion:

- at least four adjudicable levels;
- the final two absolute beta values exist;
- the last absolute beta is less than `0.05`;
- the last absolute beta is not larger than the previous absolute beta.

Locked-value interval:

`0.79 +/- 0.15`, i.e. `[0.64, 0.94]`.

Decision branches:

- If beta-to-zero holds and extrapolated `Delta d_s` is inside `[0.64, 0.94]`,
  the locked value stands for that scaling.
- If beta-to-zero holds and extrapolated `Delta d_s` is outside `[0.64, 0.94]`,
  the register value needs re-scoping for that scaling.
- If at least four levels are adjudicable but beta-to-zero does not hold, the
  step height is not a continuum observable for that scaling.
- If fewer than four levels are adjudicable, the scaling is reported as
  non-adjudicable under this extraction rule.

If the two scaling prescriptions land in different branches, the overall T1
finding is scaling-prescription dependence.
