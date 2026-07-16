#!/usr/bin/env python3
"""D1 option (a): attempt to derive alpha_s from the threshold mechanism.

This script is deliberately conservative.  The projection-threshold theorem
fixes a non-analytic step in d_s(ell).  A scalar spectral-index running,

    alpha_s = d^2 log P_R(k) / d(log k)^2,

requires an additional transfer map from the spectral-dimension feature to the
primordial scalar power spectrum.  The current manuscript does not supply that
map.  The script therefore records the formal obstruction and runs a small toy
identifiability scan to show that extra assumptions can move the answer over a
wide range of signs and magnitudes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def sech2(x: float) -> float:
    c = math.cosh(x)
    return 1.0 / (c * c)


def toy_scan(delta_ds: float) -> dict[str, Any]:
    """Scan a smoothed-threshold toy transfer.

    Toy convention:
      n_s(N) - 1 receives a step-like contribution
      sign * C * delta_ds * 0.5 * (1 + tanh((N-N*)/width)).

    Then |alpha_s| is proportional to
      C * delta_ds * sech^2(offset/width) / (2 width),

    up to the sign convention between ln k and e-fold N.  C, width, and pivot
    offset are not fixed by the threshold theorem.
    """

    couplings = [0.001, 0.003, 0.01, 0.03, 0.05]
    widths = [0.25, 0.5, 1.0, 2.0, 5.0]
    offsets = [-3.0, -1.0, 0.0, 1.0, 3.0]

    values: list[float] = []
    examples: list[dict[str, float]] = []
    for sign in (-1.0, 1.0):
        for coupling in couplings:
            for width in widths:
                for offset in offsets:
                    alpha = sign * coupling * delta_ds * sech2(offset / width) / (2.0 * width)
                    values.append(alpha)
                    if abs(alpha) >= 0.005 and len(examples) < 8:
                        examples.append(
                            {
                                "sign": sign,
                                "coupling": coupling,
                                "width_efolds": width,
                                "pivot_offset_efolds": offset,
                                "alpha_s": alpha,
                            }
                        )

    return {
        "toy_transfer": "step in n_s smoothed by tanh; not a PCT derivation",
        "unknowns_scanned": {
            "dimensionless_coupling_C": couplings,
            "smoothing_width_efolds": widths,
            "pivot_offset_efolds": offsets,
            "sign": [-1, 1],
        },
        "alpha_s_min": min(values),
        "alpha_s_max": max(values),
        "alpha_s_abs_min": min(abs(v) for v in values),
        "alpha_s_abs_max": max(abs(v) for v in values),
        "examples_landing_in_old_band": examples,
        "lesson": (
            "The threshold theorem alone does not fix C, the smoothing width, "
            "the pivot offset, or even the transfer sign.  Extra physics can "
            "make alpha_s negligible or order 1e-2."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/d1_option_a_threshold_alpha.json")
    parser.add_argument("--delta-ds", type=float, default=0.79)
    parser.add_argument("--delta-ds-sigma", type=float, default=0.15)
    parser.add_argument("--kappa", type=float, default=0.80)
    parser.add_argument("--kappa-sigma", type=float, default=0.05)
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "purpose": "Inform D1 option (a); this output is not an author decision.",
        "option": "D1(a): threshold-only alpha_s derivation attempt",
        "inputs_fixed_by_locked_threshold_sector": {
            "delta_ds": args.delta_ds,
            "delta_ds_sigma": args.delta_ds_sigma,
            "kappa": args.kappa,
            "kappa_sigma": args.kappa_sigma,
            "threshold_statement": "non-analytic step in d_s(ell) at ell*=kappa*r_H",
        },
        "required_for_alpha_s": {
            "definition": "alpha_s = d^2 log(P_R(k)) / d(log k)^2",
            "missing_transfer_map": "d_s(ell) threshold -> primordial scalar power P_R(k)",
            "missing_scale_map": "diffusion/horizon scale ell* -> CMB pivot k and e-fold N",
            "missing_smoothing_rule": "width/shape of a cosmological threshold crossing",
            "missing_coupling": "amplitude and sign with which d_s affects scalar perturbations",
            "a5_status": "matter/perturbation coupling is not supplied in the current locked instantiation",
        },
        "derivation_status": "not_closed",
        "closed_alpha_s_prediction": None,
        "can_replace_beta_flow_now": False,
        "toy_identifiability_scan": toy_scan(args.delta_ds),
        "conclusion": (
            "Option (a) does not currently yield a unique alpha_s band from the "
            "projection-threshold theorem alone.  A threshold-only route may be "
            "possible, but it requires an explicit scalar-sector transfer map "
            "and a cosmological scale-identification rule not present in the "
            "v4 draft."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "derivation_status": payload["derivation_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
