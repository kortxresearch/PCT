# T2 Pre-Registered Extraction and Adjudication Rule

Status: pre-registered and frozen before the first T2 refinement-ladder run.

## Heat trace and spectral dimension

For each level and scaling prescription, use the complete exact spectrum
`{lambda_a}` of the frozen operator:

`P(ell) = sum_a exp(-ell^2 lambda_a)`,

`d_s(ell) = 2 ell^2 [sum_a lambda_a exp(-ell^2 lambda_a)] / P(ell)`.

The second formula is the analytic log derivative of the first; no numerical
finite difference is used.

- full diagnostic grid: 112 log-spaced points on `ell in [0.30, 2.00]`;
- frozen fit window: `ell in [0.45, 1.35]`.

`[derived]` At every finite `N`, `P` and `d_s` are analytic for `ell>0`.
Consequently, “step exists” below is an operational continuum diagnosis: a
hard two-plateau model must become resolution-limited under refinement. A
finite-grid curve is never called literally discontinuous.

## Competing fits

Write `x = log(ell)` in the fit window.

Hard-step model (`k=3`):

- `d_s=d_UV` left of a scanned interior grid gap;
- `d_s=d_IR` right of it;
- at least four points must lie on each side;
- `Delta d_s = d_UV-d_IR` and `ell_star=exp(x_star)`.

Smooth comparator (`k=4`):

`d_s = d_IR + Delta d_s/[1+exp((x-x_star)/w)]`.

Scan `x_star` over fit-window grid points excluding four points at either end.
Scan 48 log-spaced widths from half a log-grid interval through `w=0.8`.
At each nonlinear grid point, solve the two linear amplitudes exactly by least
squares.

Use

`AIC = n log(max(SSE,1e-300)/n) + 2k`,

and report `Delta AIC = AIC_smooth-AIC_step`; positive values favour the hard
step.

Define

`kappa = ell_star/r_H`.

This diffusion-scale ratio is distinct from the input profile ratio
`r_H/r_c`, which is reported separately and is not counted as an extracted
observable.

## Bootstrap uncertainty

The transverse Fourier pairs `(p,q)` are the deterministic cluster units.
At each level, resample the `N^2` complete radial Fourier blocks with
replacement, keeping the resampled block count equal to `N^2`. Recompute
`P`, `d_s`, and both fits for 256 resamples with the seed declared in the spec.

Report sample standard deviation and the percentile `[2.5%,97.5%]` interval
for `Delta d_s`, `ell_star`, `kappa`, `Delta AIC`, and the fitted smooth width.
These errors quantify transverse-mode/block sensitivity; they are not
observational errors and do not replace refinement drift.

A level is `step_adjudicable` iff all four conditions hold:

1. central `Delta d_s > 0`;
2. central `Delta AIC >= 10`;
3. the bootstrap 2.5th percentile of `Delta d_s` is positive;
4. at least 90% of bootstrap resamples have both positive `Delta d_s` and
   `Delta AIC >= 10`.

## Refinement, beta surrogate, and extrapolation

For every adjacent pair, whether adjudicable or not, report

`beta_hat(N1->N2) = [Delta d_s(N2)-Delta d_s(N1)]/log(N2/N1)`.

For `Delta d_s`, `ell_star`, `kappa`, and smooth width, fit all six central
values to

`q(N)=q_inf+a N^(-p)`,

scanning `p` over 251 equally spaced values in `[0.5,3.0]`. Report the
minimum-SSE limit. Its conservative uncertainty is the maximum of the fit
residual scale, final-level bootstrap standard deviation, and absolute
last-point-to-limit drift.

A scaling prescription has a `refinement_stable_step` iff:

1. its final four levels are all `step_adjudicable`;
2. the final two absolute beta surrogates are below `0.10`, and the last does
   not exceed the preceding one;
3. the final-pair fractional drift of `ell_star` is below 5%;
4. at each of the final two levels, the best smooth-comparator width is no more
   than 1.5 log-grid intervals (resolution-limited sharpness);
5. extrapolated `Delta d_s` is positive at the conservative 95% level,
   `Delta d_s(infinity)-1.96 sigma > 0`.

Overall branches:

- both prescriptions pass: `refinement_stable_step_both_scalings`;
- exactly one passes: `scaling_prescription_dependence`;
- neither passes: `no_refinement_stable_step`.

No historical interval is tested: T2 is a fresh construction. New T2 values
may be called locked only if the relevant branch passes, and even then they
must not be retrofitted into v4 without an author decision.
