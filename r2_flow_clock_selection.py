#!/usr/bin/env python3
"""R2 bounded selection test for the FW-2a flow-to-cosmological-clock map.

The executed FW-2a probe supplies a heat-flow parameter ``t_flow`` and the
correlation-width law

    xi(t_flow) = sqrt(xi_0^2 + 2 t_flow).

This executable separates that derived law from three possible physical
identifications of a scale factor.  Only the direct correlation-length
surrogate, together with a proportional physical-clock identification, gives
the late-flow A1 map ``a proportional to t_flow**(1/2)``.  The v56 metric-
renormalisation and projection-capacity readings do not fix a power without a
new normalization/evolution law.

The script then reruns the numerical A1 transfer only for the conditionally
selected q=1/2 map and scans the still-unfixed transition placement.  It is an
identifiability audit, not an observational fit and not a P2 result.

Run from ``2. REPOSITORY`` (or from any directory):

    python r2_flow_clock_selection.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Keep reproducibility runs from depositing interpreter-specific cache files
# beside the archived source artifacts.
sys.dont_write_bytecode = True
import a1_scalar_sector_attempt as a1


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = BASE_DIR / "outputs" / "p6b_emergence_cosmology_probe.json"
DEFAULT_A1_RESULT = BASE_DIR / "outputs" / "a1_scalar_sector_attempt.json"
DEFAULT_OUTPUT = BASE_DIR / "outputs" / "r2_flow_clock_selection.json"
SELECTED_MAP_ID = "diffusive_xi"
XI_LAW_RELATIVE_TOLERANCE = 0.05
ORIGINAL_MATCH_ABSOLUTE_TOLERANCE = 2.0e-11
P6B_INITIAL_LAPLACE_SCALE = 0.05
XI0_SQUARED = 2.0 * P6B_INITIAL_LAPLACE_SCALE**2


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


def local_q(t_flow: float, xi0_squared: float) -> float:
    """Return d ln(xi)/d ln(t_flow) for xi^2=xi0^2+2t_flow."""

    return t_flow / (xi0_squared + 2.0 * t_flow)


def normalized_exact_a(t_flow: float, reference_t: float, xi0_squared: float) -> float:
    return math.sqrt(
        (xi0_squared + 2.0 * t_flow)
        / (xi0_squared + 2.0 * reference_t)
    )


def normalized_asymptotic_a(t_flow: float, reference_t: float) -> float:
    return math.sqrt(t_flow / reference_t)


def span(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "max": max(values),
        "width": max(values) - min(values),
    }


def sign_class(values: list[float], floor: float = 1.0e-5) -> str:
    signs = {
        "positive" if value > floor else "negative" if value < -floor else "zero"
        for value in values
    }
    if len(signs) == 1:
        return next(iter(signs))
    return "mixed"


def relative_errors(measured: list[float], expected: list[float]) -> list[float]:
    return [abs(left - right) / abs(right) for left, right in zip(measured, expected)]


def principle_ledger() -> list[dict[str, Any]]:
    return [
        {
            "id": "P1",
            "name": "direct_correlation_length_surrogate",
            "source": [
                "v56 V.I.0.1 and the cosmology capsule: a(tau) proportional to xi(tau)",
                "CODEX_PLAN.md R2: comoving scale proportional to correlation length",
            ],
            "premises": [
                {
                    "status": "[derived]",
                    "statement": "FW-2a heat evolution gives xi(t_flow)^2=xi_0^2+2t_flow in kernel-label units.",
                },
                {
                    "status": "[assumed]",
                    "statement": "The normalized FLRW surrogate is identified with the normalized correlation width: a/a_ref=xi/xi_ref.",
                },
                {
                    "status": "[assumed]",
                    "statement": "Physical cosmic time is proportional to the heat-flow parameter, tau/tau_ref=t_flow/t_ref; the conversion constant supplies units but not a new exponent.",
                },
            ],
            "derivation": [
                "a(t_flow)/a_ref=sqrt[(xi_0^2+2t_flow)/(xi_0^2+2t_ref)]",
                "q(t_flow):=d ln(a)/d ln(tau)=t_flow/(xi_0^2+2t_flow)",
                "q(t_flow)->1/2 when t_flow>>xi_0^2/2, so a proportional to tau^(1/2) in the late-flow regime",
            ],
            "result_status": "[derived-under-stated-identification]",
            "a1_representative_q": 0.5,
            "selection": "selected conditionally",
        },
        {
            "id": "P2",
            "name": "v56_metric_renormalisation",
            "source": [
                "v56 V.Y.2: expansion is a global rescaling of the correlator-to-distance map",
                "v56 IV.A decay-class remark: for exponential decay d_corr is proportional to r/xi",
            ],
            "premises": [
                {
                    "status": "[derived]",
                    "statement": "For the documented exponential decay class, d_corr is proportional to r/xi for fixed label separation r.",
                },
                {
                    "status": "[assumed]",
                    "statement": "Write the unspecified global metric normalization as N(t_flow), so physical distance is D=N d_corr=a r.",
                },
            ],
            "derivation": [
                "a(t_flow) is proportional to N(t_flow)/xi(t_flow)",
                "if, only as a parameterization, N is proportional to xi^p, then a is proportional to xi^(p-1)",
                "at late flow q=(p-1)/2; expansion requires p>1 and q=1/2 requires the additional choice p=2",
            ],
            "result_status": "[derived]",
            "a1_representative_q": None,
            "selection": "not selected: V.Y.2 supplies no evolution law for N",
        },
        {
            "id": "P3",
            "name": "outward_projection_capacity_volume",
            "source": [
                "v56 V.Y.5: expansion signature d C_out/d tau_int > 0",
                "v56 IV.C.4c/IV.C.8h: C_out is integrated outward-compatible projection capacity",
            ],
            "premises": [
                {
                    "status": "[assumed]",
                    "statement": "On a homogeneous patch, identify normalized physical volume a^3 with normalized outward-compatible capacity C_out.",
                },
                {
                    "status": "[derived]",
                    "statement": "The documented expansion-signature condition fixes only d C_out/d tau_int>0, not C_out(t_flow).",
                },
            ],
            "derivation": [
                "a is proportional to C_out^(1/3)",
                "if C_out is proportional to t_flow^s, then q=s/3",
                "FW-2a did not measure C_out or derive s, so the sign condition can imply q>0 but cannot choose its value",
            ],
            "result_status": "[derived]",
            "a1_representative_q": None,
            "selection": "not selected: the capacity evolution law is absent",
        },
    ]


def assumption_and_derivation_ledger() -> list[dict[str, str]]:
    return [
        {
            "id": "R2-L1",
            "status": "[locked]",
            "statement": "R2 tests documented identifications only and creates no scalar-sector paper.",
            "consequence": "A conditional map selection cannot open the D-B4-1 P2 gate by itself.",
        },
        {
            "id": "R2-L2",
            "status": "[locked]",
            "statement": "The A1 dispersion, background, vacuum, interpolation, numerical tolerances, derivative fits, and five transition offsets are inherited unchanged.",
            "consequence": "The rerun isolates map selection; it does not silently tune another A1 degree of freedom.",
        },
        {
            "id": "R2-D1",
            "status": "[derived]",
            "statement": "The executed P6b expected-width rows imply a constant xi_0^2 via xi^2-2t_flow.",
            "consequence": "The exact finite-width correction to the asymptotic square-root law is computable.",
        },
        {
            "id": "R2-A1",
            "status": "[assumed]",
            "statement": "The correlation width is a comoving scale surrogate, a/a_ref=xi/xi_ref.",
            "consequence": "This selects the scale-factor reading; FW-2a heat dynamics alone does not prove it.",
        },
        {
            "id": "R2-A2",
            "status": "[assumed]",
            "statement": "The physical cosmic clock is proportional to t_flow, with a fixed conversion carrying physical time per flow unit.",
            "consequence": "The heat-flow exponent is preserved, but no seconds, energy scale, or CMB epoch is calibrated.",
        },
        {
            "id": "R2-D2",
            "status": "[derived]",
            "statement": "Under R2-A1 and R2-A2, a/a_ref=sqrt[(xi_0^2+2t)/(xi_0^2+2t_ref)] and q(t)=t/(xi_0^2+2t), approaching q=1/2.",
            "consequence": "The q=1/2 A1 candidate is derived under stated identifications, not an unqualified PCT prediction.",
        },
        {
            "id": "R2-D3",
            "status": "[derived]",
            "statement": "Metric renormalisation gives a proportional to N/xi; without a law for N(t), it permits a continuum of q values.",
            "consequence": "V.Y.2 does not independently corroborate q=1/2; N proportional to xi^2 would be a new assumption.",
        },
        {
            "id": "R2-D4",
            "status": "[derived]",
            "statement": "The projection-capacity criterion gives a proportional to C_out^(1/3) only after a volume identification, and no C_out(t_flow) was produced by FW-2a.",
            "consequence": "It can supply an expansion sign under assumptions but no A1 exponent.",
        },
        {
            "id": "R2-A3",
            "status": "[assumed]",
            "statement": "The exact finite-xi law is represented in A1 by its q=1/2 late-flow power law.",
            "consequence": "The approximation error is reported at every sampled time and at the d_s transition center.",
        },
        {
            "id": "R2-D5",
            "status": "[derived]",
            "statement": "After restricting A1 to q=1/2, the only scanned map/placement identifiability coordinate is the transition-center offset delta N.",
            "consequence": "Any remaining envelope is a placement sensitivity interval, not a theory probability or confidence interval.",
        },
    ]


def build_result(source: Path, previous_a1_path: Path) -> dict[str, Any]:
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    previous_a1 = json.loads(previous_a1_path.read_text(encoding="utf-8"))
    rows = source_payload["flow_rows"]
    times = np.asarray([row["flow_time"] for row in rows], dtype=float)
    dimensions = np.asarray([row["d_s_at_ell1"] for row in rows], dtype=float)
    measured_xi = [float(row["correlation_length"]) for row in rows]
    expected_xi = [float(row["expected_xi_sqrt_law"]) for row in rows]
    xi0_squared_samples = [value * value - 2.0 * time for value, time in zip(expected_xi, times)]
    # P6b defines K0=exp(-|z|/0.05).  A normalized Laplace profile with scale
    # b has variance 2b^2, so the executable's unrounded initial width is
    # xi_0^2=2(0.05)^2=0.005.  The JSON width columns are rounded to four
    # decimals and are audited against, rather than used to redefine, it.
    xi0_squared = XI0_SQUARED
    asymptotic_flow = a1.DimensionFlow(times, dimensions)
    asymptotic_map = next(item for item in a1.FLOW_MAPS if item.map_id == SELECTED_MAP_ID)
    # Using u=xi^2=xi_0^2+2t as the coordinate makes the A1 power map with
    # q=1/2 exactly equal to a proportional to xi.  This is the primary R2
    # rerun.  The original t_flow**(1/2) branch is retained below as a
    # reproducibility and late-flow approximation check.
    exact_width_flow = a1.DimensionFlow(xi0_squared + 2.0 * times, dimensions)
    exact_width_map = a1.FlowMap(
        "diffusive_xi_exact_width",
        0.5,
        "a/a_c=[(xi_0^2+2t_flow)/(xi_0^2+2t_c)]^(1/2).",
        "Directly identifies the normalized scale factor with the FW-2a correlation width.",
    )
    reference_t_flow = (exact_width_flow.t_center - xi0_squared) / 2.0

    baseline = a1.baseline_spectrum()
    amplitude_scale = float(baseline["amplitude_scale"])
    centered = a1.calculate_spectrum(
        exact_width_flow, exact_width_map, 0.0, a1.LOG_K_GRID, amplitude_scale
    )
    tight = a1.calculate_spectrum(
        exact_width_flow,
        exact_width_map,
        0.0,
        a1.LOCAL_LOG_K_GRID,
        amplitude_scale,
        rtol=a1.TIGHT_RTOL,
        atol=a1.TIGHT_ATOL,
    )

    placement_rows: list[dict[str, Any]] = []
    for offset in a1.ALIGNMENT_OFFSETS_EFOLDS:
        result = (
            centered
            if offset == 0.0
            else a1.calculate_spectrum(
                exact_width_flow,
                exact_width_map,
                offset,
                a1.LOCAL_LOG_K_GRID,
                amplitude_scale,
            )
        )
        placement_rows.append(
            {
                "center_offset_efolds": offset,
                "pivot_state": result["pivot_state"],
                "pivot_observables": result["pivot_observables"],
                "derivative_method_check": result["derivative_method_check"],
            }
        )

    asymptotic_centered = a1.calculate_spectrum(
        asymptotic_flow,
        asymptotic_map,
        0.0,
        a1.LOG_K_GRID,
        amplitude_scale,
    )
    asymptotic_placement_rows: list[dict[str, Any]] = []
    for offset in a1.ALIGNMENT_OFFSETS_EFOLDS:
        asymptotic_result = (
            asymptotic_centered
            if offset == 0.0
            else a1.calculate_spectrum(
                asymptotic_flow,
                asymptotic_map,
                offset,
                a1.LOCAL_LOG_K_GRID,
                amplitude_scale,
            )
        )
        asymptotic_placement_rows.append(
            {
                "center_offset_efolds": offset,
                "pivot_observables": asymptotic_result["pivot_observables"],
            }
        )

    original_centered = {
        item["map_id"]: item for item in previous_a1["centered_candidate_results"]
    }[SELECTED_MAP_ID]
    original_placement = {
        float(item["center_offset_efolds"]): item
        for item in previous_a1["identifiability_scan"]["rows"]
        if item["map_id"] == SELECTED_MAP_ID
    }
    match_differences: list[float] = []
    exact_vs_asymptotic_differences: dict[str, list[float]] = {
        key: [] for key in ("n_s", "alpha_s", "beta_s")
    }
    for key in ("n_s", "alpha_s", "beta_s"):
        match_differences.append(
            abs(
                asymptotic_centered["pivot_observables"][key]
                - original_centered["pivot_observables"][key]
            )
        )
        for exact_item, asymptotic_item in zip(placement_rows, asymptotic_placement_rows):
            offset = float(asymptotic_item["center_offset_efolds"])
            exact_vs_asymptotic_differences[key].append(
                abs(
                    exact_item["pivot_observables"][key]
                    - asymptotic_item["pivot_observables"][key]
                )
            )
            match_differences.append(
                abs(
                    asymptotic_item["pivot_observables"][key]
                    - original_placement[offset]["pivot_observables"][key]
                )
            )

    exact_q_rows = []
    for time in times:
        exact = normalized_exact_a(float(time), reference_t_flow, xi0_squared)
        asymptotic = normalized_asymptotic_a(float(time), reference_t_flow)
        exact_q_rows.append(
            {
                "t_flow": time,
                "local_q": local_q(float(time), xi0_squared),
                "exact_a_over_a_at_ds_center": exact,
                "q_half_a_over_a_at_ds_center": asymptotic,
                "relative_a_difference": abs(exact - asymptotic) / exact,
            }
        )

    alpha_values = [item["pivot_observables"]["alpha_s"] for item in placement_rows]
    beta_values = [item["pivot_observables"]["beta_s"] for item in placement_rows]
    ns_values = [item["pivot_observables"]["n_s"] for item in placement_rows]
    old_center_alpha = previous_a1["identifiability_scan"]["centered_map_envelope"]
    old_center_beta = [
        item["pivot_observables"]["beta_s"]
        for item in previous_a1["centered_candidate_results"]
    ]
    old_all_alpha = previous_a1["identifiability_scan"]["map_plus_alignment_envelope"]
    old_all_beta = [
        item["pivot_observables"]["beta_s"]
        for item in previous_a1["identifiability_scan"]["rows"]
    ]
    alpha_span = span(alpha_values)
    beta_span = span(beta_values)
    nonzero_abs_alpha = [abs(value) for value in alpha_values if abs(value) > 1.0e-5]
    original_all_alpha_width = old_all_alpha["alpha_s_max"] - old_all_alpha["alpha_s_min"]
    original_all_beta_span = span(old_all_beta)
    width_samples_relative_error = relative_errors(measured_xi, expected_xi)

    center_convergence = {
        key: abs(centered["pivot_observables"][key] - tight["pivot_observables"][key])
        for key in ("n_s", "alpha_s", "beta_s")
    }
    numerical_pass = bool(
        baseline["absolute_errors"]["n_s"] < 2.0e-6
        and baseline["absolute_errors"]["alpha_s"] < 2.0e-6
        and baseline["absolute_errors"]["beta_s"] < 2.0e-6
        and center_convergence["n_s"] < 2.0e-6
        and center_convergence["alpha_s"] < 1.0e-5
        and center_convergence["beta_s"] < 1.0e-4
    )
    placement_sign = sign_class(alpha_values)
    original_match_max = max(match_differences)

    result: dict[str, Any] = {
        "schema_version": "r2-flow-clock-selection-v1",
        "purpose": "Bounded D-B4-1 test of whether FW-2a selects the A1 flow-clock map, followed by an A1 rerun on selected maps only.",
        "scope": "Theory-identifiability audit in toy units; not an observational fit, physical calibration, P2 result, or manuscript claim.",
        "provenance": {
            "fw2a_source": "outputs/p6b_emergence_cosmology_probe.json",
            "fw2a_source_sha256": a1.sha256(source),
            "previous_a1_result": "outputs/a1_scalar_sector_attempt.json",
            "previous_a1_result_sha256": a1.sha256(previous_a1_path),
            "a1_executable": "a1_scalar_sector_attempt.py",
            "a1_executable_sha256": a1.sha256(BASE_DIR / "a1_scalar_sector_attempt.py"),
            "v56_text_source": "../8. HISTORY/springer-v1-submission/PCT/PCT_researchpaper.txt",
            "documented_sections": ["IV.A decay-class remark", "IV.C.4c", "V.I.0.1", "V.Y.2", "V.Y.5"],
        },
        "units_and_normalizations": {
            "t_flow": "heat-flow units; dimension length^2 relative to the kernel-label coordinate",
            "xi": "kernel-label length units",
            "a": "dimensionless after division by its value at the d_s-flow center",
            "tau": "physical time only after an assumed fixed conversion tau=T_unit*t_flow; T_unit is not calibrated",
            "q_definition": "q=d ln(a)/d ln(tau)",
            "cmb_pivot_placement": "unfixed; represented only by delta N in {-2,-1,0,1,2}",
        },
        "assumption_and_derivation_ledger": assumption_and_derivation_ledger(),
        "identification_principles": principle_ledger(),
        "fw2a_width_law_audit": {
            "xi0_squared_samples_from_expected_rows": xi0_squared_samples,
            "xi0_squared_exact_from_p6b_initial_profile": xi0_squared,
            "p6b_initial_laplace_scale": P6B_INITIAL_LAPLACE_SCALE,
            "maximum_xi0_squared_sample_deviation": max(
                abs(item - xi0_squared) for item in xi0_squared_samples
            ),
            "maximum_measured_vs_expected_relative_error": max(width_samples_relative_error),
            "declared_relative_tolerance": XI_LAW_RELATIVE_TOLERANCE,
            "pass": max(width_samples_relative_error) < XI_LAW_RELATIVE_TOLERANCE,
        },
        "conditional_q_half_derivation": {
            "claim_status": "[derived-under-stated-identification]",
            "exact_scale_factor_law": "a/a_ref=sqrt[(xi_0^2+2t_flow)/(xi_0^2+2t_ref)]",
            "exact_local_exponent": "q(t_flow)=t_flow/(xi_0^2+2t_flow)",
            "asymptotic_exponent": 0.5,
            "reference_t_flow_at_ds_center": reference_t_flow,
            "local_q_at_ds_center": local_q(reference_t_flow, xi0_squared),
            "sample_rows": exact_q_rows,
            "maximum_relative_exact_vs_q_half_a_difference_on_samples": max(
                row["relative_a_difference"] for row in exact_q_rows
            ),
            "relative_exact_vs_q_half_a_difference_at_ds_center": 0.0,
            "interpretation": "q=1/2 is the late-flow representative of the exact finite-width law; the bridge to scale and physical time remains assumed.",
        },
        "map_selection": {
            "status": "conditional_selection_only",
            "selected_a1_branch_id": SELECTED_MAP_ID,
            "executed_exact_map_id": exact_width_map.map_id,
            "selected_q": [exact_width_map.q],
            "a1_candidate_count_before": len(a1.FLOW_MAPS),
            "a1_candidate_count_after": 1,
            "discrete_candidate_fraction_removed": (len(a1.FLOW_MAPS) - 1) / len(a1.FLOW_MAPS),
            "unconditional_theory_selection_achieved": False,
            "reason": "FW-2a fixes xi(t_flow); q=1/2 follows only after the direct xi-to-a and proportional-clock identifications. The other documented principles do not fix their required normalization/evolution functions.",
        },
        "restricted_a1_rerun": {
            "inherited_a1_ledger": previous_a1["assumption_and_derivation_ledger"],
            "primary_mapping": "exact a proportional to xi, implemented with flow coordinate u=xi_0^2+2t_flow and q=1/2",
            "baseline_validation": baseline,
            "selected_centered_result": centered,
            "tight_tolerance_centered_result": {
                "pivot_observables": tight["pivot_observables"],
                "absolute_difference_from_production": center_convergence,
            },
            "transition_placement_scan": {
                "only_free_identifiability_coordinate_scanned": "delta N: d_s transition center relative to k_pivot",
                "rows": placement_rows,
                "n_s": span(ns_values),
                "alpha_s": alpha_span,
                "beta_s": beta_span,
                "alpha_s_sign_class": placement_sign,
                "nonzero_abs_alpha_spread_ratio": max(nonzero_abs_alpha) / min(nonzero_abs_alpha),
                "historical_negative_alpha_band": [-0.02, -0.005],
                "historical_band_status": "not rederived; placement choices enter, overshoot, and leave the historical band",
            },
            "late_flow_q_half_approximation_audit": {
                "mapping": "original A1 diffusive_xi branch a proportional to t_flow^(1/2)",
                "centered_pivot_observables": asymptotic_centered["pivot_observables"],
                "placement_rows": asymptotic_placement_rows,
                "maximum_exact_vs_asymptotic_absolute_observable_differences": {
                    key: max(values)
                    for key, values in exact_vs_asymptotic_differences.items()
                },
                "archived_a1_reproduction_max_abs_difference": original_match_max,
            },
        },
        "identifiability_reduction": {
            "scope": "the frozen A1 map-versus-placement subproblem; other perturbation and physical-calibration assumptions remain open",
            "removed_degree_of_freedom": "discrete A1 power-map choice q in {1/2,1,2}, conditionally",
            "remaining_degree_of_freedom": "continuous absolute transition/pivot placement, sampled at five locked offsets",
            "old_three_map_centered_alpha_s_width": old_center_alpha["alpha_s_max"] - old_center_alpha["alpha_s_min"],
            "old_three_map_centered_beta_s_width": span(old_center_beta)["width"],
            "selected_map_centered_alpha_s": centered["pivot_observables"]["alpha_s"],
            "selected_map_centered_beta_s": centered["pivot_observables"]["beta_s"],
            "selected_map_placement_alpha_s_width": alpha_span["width"],
            "selected_map_placement_beta_s_width": beta_span["width"],
            "selected_placement_fraction_of_original_all_map_alpha_width": alpha_span["width"] / original_all_alpha_width,
            "selected_placement_fraction_of_original_all_map_beta_width": beta_span["width"] / original_all_beta_span["width"],
            "placement_alpha_width_divided_by_removed_centered_map_width": alpha_span["width"] / (old_center_alpha["alpha_s_max"] - old_center_alpha["alpha_s_min"]),
            "map_and_placement_ambiguity_reduced_to_transition_placement": True,
            "prediction_identifiable_after_reduction": False,
        },
        "failure_tests": {
            "fw2a_width_law_pass": max(width_samples_relative_error) < XI_LAW_RELATIVE_TOLERANCE,
            "restricted_rerun_numerical_validation_pass": numerical_pass,
            "asymptotic_q_half_check_matches_archived_a1_max_abs_difference": original_match_max,
            "asymptotic_q_half_check_matches_archived_a1_pass": original_match_max < ORIGINAL_MATCH_ABSOLUTE_TOLERANCE,
            "exact_width_local_q_within_0p001_of_half_at_ds_center": abs(
                local_q(reference_t_flow, xi0_squared) - 0.5
            ) < 0.001,
            "transition_placement_retains_one_alpha_sign": placement_sign != "mixed",
            "transition_placement_identifiability_pass": False,
            "metric_normalisation_independently_selects_q": False,
            "projection_capacity_independently_selects_q": False,
        },
        "gate_outcome": {
            "r2_outcome": "conditional_map_selection_but_scalar_closure_still_fails",
            "p2_gate": "blocked",
            "p2_draft_created": False,
            "what_was_gained": [
                "The three-way A1 map bracket is reduced to q=1/2 under two explicit physical identifications.",
                "The exact finite-width correction and local q(t_flow) are derived from the FW-2a width law.",
                "The A1 failure is localized to transition placement within this conditional branch rather than conflated with the discrete q scan.",
            ],
            "what_remains": [
                "A primitive or calibrated reason that correlation width is the FLRW scale rather than its inverse or a separately normalized distance.",
                "A physical flow-to-proper-time conversion beyond proportional-clock assumption and toy units.",
                "A derived absolute transition energy/epoch fixing the d_s center relative to k_pivot.",
                "The previously recorded perturbation-action, matter/background, and interpolation assumptions.",
            ],
        },
        "conclusion": "FW-2a conditionally selects the q=1/2 A1 branch when a is identified with xi and physical time with t_flow. It does not select that bridge unconditionally. With q fixed, the locked +/-2 e-fold placement scan still changes alpha_s sign and spans a wide range, so no scalar-running band is identifiable and no P2 is opened.",
    }
    return a1.clean_json(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--a1-result", default=str(DEFAULT_A1_RESULT))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    source = resolve_path(args.source)
    previous_a1 = resolve_path(args.a1_result)
    output = resolve_path(args.out)
    result = build_result(source, previous_a1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "out": output.as_posix(),
        "r2_outcome": result["gate_outcome"]["r2_outcome"],
        "selected_q": result["map_selection"]["selected_q"],
        "alpha_s_placement_span": result["restricted_a1_rerun"]["transition_placement_scan"]["alpha_s"],
        "p2_gate": result["gate_outcome"]["p2_gate"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
