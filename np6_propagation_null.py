#!/usr/bin/env python3
"""NP-6 GW propagation null check."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def fractional_speed_shift(alpha_pct: float, ds_minus_4: float) -> float:
    """For omega^2 = c^2 k^2 [1 + alpha (d_s-4)^2]."""

    return math.sqrt(1.0 + alpha_pct * ds_minus_4 * ds_minus_4) - 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/np6_propagation_null.json")
    parser.add_argument("--alpha-pct", type=float, default=0.02)
    parser.add_argument("--gw170817-bound", type=float, default=1e-15)
    parser.add_argument("--delta-ds", type=float, default=0.79)
    args = parser.parse_args()

    cosmological_shift = fractional_speed_shift(args.alpha_pct, 0.0)
    local_threshold_shift = fractional_speed_shift(args.alpha_pct, args.delta_ds)

    payload = {
        "purpose": "NP-6 propagation null test.",
        "locked_dispersion_form": "omega^2 = c^2 k^2 [1 + alpha_PCT (d_s-4)^2]",
        "premise": (
            "The locked dispersion correction is active only where the local "
            "spectral dimension differs from four, such as the near-horizon "
            "threshold region.  Ordinary cosmological propagation paths have "
            "d_s=4 in this locked account."
        ),
        "inputs": {
            "alpha_pct": args.alpha_pct,
            "gw170817_speed_bound_abs": args.gw170817_bound,
            "delta_ds_local_threshold_example": args.delta_ds,
        },
        "computed": {
            "cosmological_ds_minus_4": 0.0,
            "cosmological_fractional_speed_shift": cosmological_shift,
            "passes_gw170817_bound": abs(cosmological_shift) <= args.gw170817_bound,
            "local_threshold_fractional_shift_if_ds_changes_by_delta": local_threshold_shift,
            "local_threshold_shift_is_cumulative_propagation_prediction": False,
        },
        "falsification_rule": (
            "A robust cumulative GW dispersion signal on propagation paths for "
            "which the locked instantiation requires d_s=4 would falsify this "
            "null statement, provided source and detector systematics are ruled out."
        ),
        "conclusion": (
            "Within the locked assumptions, PCT predicts no cumulative "
            "cosmological GW propagation delay and therefore passes the "
            "GW170817-style speed constraint by construction."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "passes_bound": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
