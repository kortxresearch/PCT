# P9 PiSR Pilot — pre-registered specification and decision rules

**Status:** frozen before the scan runs (hashes in `outputs/p9_freeze_manifest.json`).
**Question:** within two one-parameter families of admissible kernel dynamics, is the heat flow (the P8 incumbent) the selected dynamics — and can any admissible flow satisfy both register-consistency and an inflation-compatible derived clock?

## Search space (FW-17 scope: the nonlinear/non-stationary and non-Gaussian-linear sectors)

- **Family P (porous medium):** ∂_t κ = Δ(κ^m), m ∈ {1.0, 1.5, 2.0, 3.0}. m = 1 is the heat flow. Fast diffusion (m < 1) is excluded from this pilot for numerical scope (unbounded tail diffusivity under the explicit scheme); its derived clock ε_H = m+1 ∈ (1,2) is intermediate and does not affect the headline question. Recorded as a scope limit.
- **Family S (stable/fractional):** ∂_t κ̂(k) = −|k|^α κ̂(k), α ∈ {0.5, 1.0, 1.5, 2.0}. α = 2 is the heat flow.

Both families have exactly one parameter: parsimony is equal by construction; simplicity does not discriminate within this pilot.

## Tiered constraints (hard filters; fail ⇒ eliminated)

- **T1a profile positivity:** min_z κ(z,t) ≥ −10⁻⁶·max κ throughout the flow (pre-clip diagnostic for the explicit PME scheme, which clips to ≥0 after each step; the tolerance absorbs front-tracking scheme noise, not physics).
- **T1b Bochner/PSD (kernel axiom M1):** min_k κ̂(k,t) ≥ −10⁻⁸·max κ̂ throughout the flow. (Linear stable flows preserve this exactly; for the nonlinear family it is a genuine test.)
- **T1c mass conservation:** |Δ∫κ|/∫κ < 10⁻⁶ over the flow (up to declared boundary truncation).
- **T2 entropy monotonicity:** the differential entropy of the normalised profile is non-decreasing along the flow (tolerance −10⁻⁸ per step).

## Measured properties (soft criteria; no weights — see decision rule)

- **C1 register-consistency:** sup-distance of the flow's late-time rescaled profile to the Gaussian shape (the locked instantiation sits at the dynamics' fixed point iff this → 0). Scale measure: interquartile width (second moments diverge for α < 2).
- **C2 derived clock:** scaling exponent p from ξ(t) ~ t^p (log-log fit over the late decade), giving ε_H = 1/p under identification (I). Inflation-compatible means ε_H ≪ 1. Analytic expectations: ε_H = m+1 (family P), ε_H = α (family S); the fit must confirm.

## Resurrection audit (per the FW-17 protocol; applied to all survivors)

For each Tier-passing flow: build the Gaussian-projected correlator from the evolved kernel at three flow times, compute d_s(ℓ) (Tikhonov-inverse heat-trace estimator, P4 conventions), and run a hard-step vs smooth-sigmoid AIC comparison (T-campaign convention, AIC_smooth − AIC_step, threshold 10). Expected outcome under the mollifier theorem: smooth preferred everywhere. **Any step preference triggers the escalation path (identify violated hypothesis → independent reimplementation → formal reopening), not a result claim.**

## Pre-registered decision rules

1. A flow is **admissible** iff it passes T1a–T2.
2. A flow **dominates** iff it is admissible and strictly better on both C1 and C2 than every other admissible flow. A winner is declared only on dominance.
3. Absent dominance, the deliverable is the **Pareto front** and the explicit statement of the trade-off. In particular, the pilot is designed to answer: *does any admissible flow achieve both C1 ≈ 0 (Gaussian fixed point) and ε_H < 0.1 (inflation-compatible clock)?* A negative answer is a theorem-grade structural finding about the framework (within these families), not a failure of the pilot.
4. Numerical validity gates: fitted exponents must match analytic expectations within 10%; the heat flow (both parametrisations) must reproduce ε_H = 2 within 5% (cross-check against the executed √t law); otherwise the affected cell is reported non-adjudicable.

## Provenance note

Smoke tests during implementation write only to `outputs/p9_pisr_pilot_smoke.json` (declared, excluded from claims). The scan output `outputs/p9_pisr_pilot.json` must not exist at freeze time.
