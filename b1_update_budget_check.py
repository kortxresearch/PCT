#!/usr/bin/env python3
"""Deterministic check of the locked update-budget relation in v56 section V.W.6.

The calculation is deliberately elementary: it audits the three quoted
Schwarzschild-coordinate radii and a dense f-grid.  It does not model an
observable propagation delay and it does not add a matter-coupling law.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "b1_update_budget_check.json"
RADII_OVER_RS = (3.0, 2.0, 1.5)
TOLERANCE = 5.0e-15


def locked_row(radius_over_rs: float) -> dict[str, float | str]:
    """Evaluate all equivalent locked expressions at r/r_s."""
    f = 1.0 / radius_over_rs
    z_s = 1.0 - f
    z_t = 1.0 / z_s
    clock_rate = 1.0 / math.sqrt(z_t)
    v_direct = math.sqrt(z_s / z_t)
    v_budget_rhs = math.sqrt(z_s) * clock_rate
    v_closed_form = 1.0 - f
    return {
        "r_over_r_s": radius_over_rs,
        "f_equals_r_s_over_r": f,
        "Z_s": z_s,
        "Z_t": z_t,
        "d_tau_over_dt": clock_rate,
        "v_char_over_c_direct": v_direct,
        "sqrt_Z_s_times_d_tau_over_dt": v_budget_rhs,
        "one_minus_f": v_closed_form,
        "identity_abs_residual": abs(v_direct - v_budget_rhs),
        "locked_form_abs_residual": abs(v_direct - v_closed_form),
    }


def dense_sweep(samples: int = 10_001) -> dict[str, float | int | bool]:
    """Check the identity and monotonicity on 0 <= f <= 0.999."""
    velocities: list[float] = []
    identity_errors: list[float] = []
    locked_errors: list[float] = []
    for i in range(samples):
        f = 0.999 * i / (samples - 1)
        z_s = 1.0 - f
        z_t = 1.0 / z_s
        clock_rate = 1.0 / math.sqrt(z_t)
        v_direct = math.sqrt(z_s / z_t)
        v_budget_rhs = math.sqrt(z_s) * clock_rate
        velocities.append(v_direct)
        identity_errors.append(abs(v_direct - v_budget_rhs))
        locked_errors.append(abs(v_direct - (1.0 - f)))
    return {
        "samples": samples,
        "f_min": 0.0,
        "f_max": 0.999,
        "max_identity_abs_residual": max(identity_errors),
        "max_locked_form_abs_residual": max(locked_errors),
        "v_char_monotone_nonincreasing": all(
            b <= a + TOLERANCE for a, b in zip(velocities[:-1], velocities[1:])
        ),
    }


def main() -> int:
    rows = [locked_row(value) for value in RADII_OVER_RS]
    sweep = dense_sweep()
    all_point_checks = all(
        row["identity_abs_residual"] <= TOLERANCE
        and row["locked_form_abs_residual"] <= TOLERANCE
        for row in rows
    )
    result = {
        "purpose": "Audit the quantitative values and algebraic identity in v56 V.W.6.",
        "status_ledger": {
            "derived": [
                "v_char/c = sqrt(Z_s/Z_t)",
                "d_tau/dt = 1/sqrt(Z_t)",
                "v_char/c = sqrt(Z_s) (d_tau/dt)",
            ],
            "locked": [
                "Z_s = 1-f and Z_t = 1/(1-f)",
                "f(r) = r_s/r for the Schwarzschild correspondence check",
            ],
            "not_tested": [
                "matter-to-environment coupling (FW-1)",
                "an operational time-of-flight observable",
                "any discrete-substructure correction",
            ],
        },
        "coordinate_statement": (
            "v_char/c is the radial characteristic speed in the declared static "
            "proxy chart; it is not a locally measured superluminal or subluminal speed."
        ),
        "tolerance": TOLERANCE,
        "quoted_radius_checks": rows,
        "dense_sweep": sweep,
        "checks": {
            "quoted_points_pass": all_point_checks,
            "dense_identity_pass": sweep["max_identity_abs_residual"] <= TOLERANCE,
            "dense_locked_form_pass": sweep["max_locked_form_abs_residual"] <= TOLERANCE,
            "dense_monotonicity_pass": sweep["v_char_monotone_nonincreasing"],
        },
    }
    result["all_checks_pass"] = all(result["checks"].values())

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT}")
    print("all_checks_pass:", result["all_checks_pass"])
    for row in rows:
        print(
            "r/r_s={:.1f}: d_tau/dt={:.9f}, v_char/c={:.9f}".format(
                row["r_over_r_s"], row["d_tau_over_dt"], row["v_char_over_c_direct"]
            )
        )
    return 0 if result["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
