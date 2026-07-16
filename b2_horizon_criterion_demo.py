#!/usr/bin/env python3
"""Deterministic critical-surface locator for the frozen T1 radial profile.

This script locates the level set rho_bar/rho_bar_0 = q on the T1 domain.
The fixed-physical q=5 result is the executed illustration for P5.  The v56
scaling column is reported separately to expose, rather than hide, that a
resolution-dependent threshold does not define one fixed physical surface.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "t1_horizon_profile_spec.md"
OUTPUT = ROOT / "outputs" / "b2_horizon_criterion_demo.json"

R_H = 1.0
R_OUT = 6.0
RHO_CRIT_FIXED = 5.0
N_REF = 80
LABEL_POINTS = 128
LEVELS = (80, 160, 320, 640, 1280, 2560)


def rho_ratio(r: float) -> float:
    if not r > R_H:
        raise ValueError("T1 profile is defined only for r > r_H")
    return 1.0 / (1.0 - R_H / r)


def analytic_radius(q: float) -> float | None:
    """Return the interior level-set radius, or None if none lies in (r_H,R]."""
    q_outer = rho_ratio(R_OUT)
    if q < q_outer or q <= 1.0:
        return None
    r = R_H * q / (q - 1.0)
    if r > R_OUT + 1.0e-14:
        return None
    return r


def bisection_radius(q: float, iterations: int = 120) -> float | None:
    exact = analytic_radius(q)
    if exact is None:
        return None
    if math.isclose(exact, R_OUT, rel_tol=0.0, abs_tol=1.0e-14):
        return R_OUT
    lo = math.nextafter(R_H, math.inf)
    hi = R_OUT
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if rho_ratio(mid) > q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def midpoint_grid_locator(n: int, q: float) -> dict[str, float | int | str | None]:
    spacing = (R_OUT - R_H) / n
    radii = [R_H + (i + 0.5) * spacing for i in range(n)]
    values = [rho_ratio(r) for r in radii]
    for i in range(n - 1):
        if values[i] >= q >= values[i + 1]:
            r0, r1 = radii[i], radii[i + 1]
            y0, y1 = values[i], values[i + 1]
            estimate = r0 + (q - y0) * (r1 - r0) / (y1 - y0)
            exact = analytic_radius(q)
            return {
                "points": n,
                "spacing": spacing,
                "status": "bracketed interior crossing",
                "bracket_r_low": r0,
                "bracket_r_high": r1,
                "linear_interpolant_radius": estimate,
                "abs_error_against_analytic": abs(estimate - exact) if exact is not None else None,
            }
    exact = analytic_radius(q)
    if exact is not None and math.isclose(exact, R_OUT, abs_tol=1.0e-14):
        return {
            "points": n,
            "spacing": spacing,
            "status": "crossing at outer boundary",
            "bracket_r_low": None,
            "bracket_r_high": R_OUT,
            "linear_interpolant_radius": R_OUT,
            "abs_error_against_analytic": 0.0,
        }
    relation = "entire sampled domain above threshold" if values[-1] > q else "entire sampled domain below threshold"
    return {
        "points": n,
        "spacing": spacing,
        "status": f"no crossing: {relation}",
        "bracket_r_low": None,
        "bracket_r_high": None,
        "linear_interpolant_radius": None,
        "abs_error_against_analytic": None,
    }


def scaling_row(n: int) -> dict[str, float | int | str | None]:
    q = RHO_CRIT_FIXED * N_REF / n
    root = analytic_radius(q)
    if root is None:
        status = "no crossing; whole T1 domain is above the scaled threshold"
    elif math.isclose(root, R_OUT, abs_tol=1.0e-14):
        status = "critical level occurs at the outer boundary"
    else:
        status = "one interior critical surface"
    return {
        "N": n,
        "scaled_rho_crit_over_rho_0": q,
        "analytic_radius_over_r_H": root,
        "status": status,
    }


def main() -> int:
    if not SPEC.exists():
        raise FileNotFoundError(SPEC)

    exact = analytic_radius(RHO_CRIT_FIXED)
    numeric = bisection_radius(RHO_CRIT_FIXED)
    if exact is None or numeric is None:
        raise RuntimeError("fixed T1 threshold should have one interior root")

    refinement = [midpoint_grid_locator(n, RHO_CRIT_FIXED) for n in LEVELS]
    label_grid = midpoint_grid_locator(LABEL_POINTS, RHO_CRIT_FIXED)
    derivative_at_root = -R_H / (exact - R_H) ** 2
    result = {
        "purpose": "Executed P5 illustration: locate the principal-symbol critical surface on the frozen T1 profile.",
        "status_ledger": {
            "locked": [
                "r_H=1, R=6 r_H, rho_crit/rho_0=5, midpoint grids, M=128 fixed labels",
                "rho_bar/rho_0 = 1/(1-r_H/r) from t1_horizon_profile_spec.md",
            ],
            "derived": [
                "the profile is strictly decreasing for r>r_H",
                "r_critical/r_H = q/(q-1) when q=rho_crit/rho_0>1",
                "q=5 gives r_critical/r_H=1.25",
            ],
            "assumed": [
                "principal-symbol collapse occurs at the chosen critical density",
                "the continued local Lorentzian principal-symbol description is admissible outside the surface",
            ],
        },
        "fixed_physical_threshold": {
            "rho_crit_over_rho_0": RHO_CRIT_FIXED,
            "analytic_radius_over_r_H": exact,
            "bisection_radius_over_r_H": numeric,
            "bisection_abs_error": abs(numeric - exact),
            "profile_residual_at_bisection": abs(rho_ratio(numeric) - RHO_CRIT_FIXED),
            "dimensionless_profile_derivative_at_surface": derivative_at_root,
            "unique_surface": True,
        },
        "constraint_midpoint_refinement": refinement,
        "fixed_label_grid_locator": label_grid,
        "v56_threshold_scaling_diagnostic": {
            "warning": (
                "The scaled rho_crit column is a provenance diagnostic, not one fixed physical "
                "criterion: its surface moves with N and disappears from the domain by N=640."
            ),
            "rows": [scaling_row(n) for n in LEVELS],
        },
        "scope": (
            "This is a level-set and principal-symbol illustration only. It does not establish "
            "a global event horizon, an interior Lorentzian extension, or a matter-coupling law."
        ),
    }
    result["all_fixed_threshold_checks_pass"] = bool(
        abs(numeric - exact) <= 1.0e-14
        and abs(rho_ratio(numeric) - RHO_CRIT_FIXED) <= 1.0e-12
        and all(row["status"] == "bracketed interior crossing" for row in refinement)
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT}")
    print("fixed root r/r_H:", numeric)
    print("all_fixed_threshold_checks_pass:", result["all_fixed_threshold_checks_pass"])
    return 0 if result["all_fixed_threshold_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
