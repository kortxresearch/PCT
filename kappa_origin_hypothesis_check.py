#!/usr/bin/env python3
"""D2 support: compare simple kappa-origin hypotheses against sweep claims."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def kappa_from_n(n: float) -> float:
    return 1.0 - 1.0 / n


def z_score(observed: float, sigma: float, predicted: float) -> float:
    return abs(observed - predicted) / sigma if sigma else math.inf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/kappa_origin_hypothesis_check.json")
    args = parser.parse_args()

    sweeps = [
        {
            "label": "2+1 dimensional angular-sector sweep cited in PROPOSED_RESTORATIONS",
            "dimension_spacetime": 3,
            "observed_kappa": 0.81,
            "sigma": 0.06,
        },
        {
            "label": "3+1 dimensional tetrahedral-mesh sweep cited in PROPOSED_RESTORATIONS",
            "dimension_spacetime": 4,
            "observed_kappa": 0.79,
            "sigma": 0.07,
        },
    ]

    rows = []
    for sweep in sweeps:
        d = sweep["dimension_spacetime"]
        obs = sweep["observed_kappa"]
        sig = sweep["sigma"]
        pred_universal = kappa_from_n(5.0)
        pred_d_plus_1 = kappa_from_n(d + 1.0)
        rows.append(
            {
                **sweep,
                "universal_n_5_prediction": pred_universal,
                "universal_n_5_z": z_score(obs, sig, pred_universal),
                "n_equals_D_plus_1_prediction": pred_d_plus_1,
                "n_equals_D_plus_1_z": z_score(obs, sig, pred_d_plus_1),
            }
        )

    result = {
        "purpose": "Inform D2; this output does not decide whether kappa is derived or calibrated.",
        "hypotheses": {
            "universal_n_5": "n is universal, so kappa=1-1/5=0.80 in every dimension.",
            "n_equals_D_plus_1": "n=D+1, where D is spacetime dimension, so kappa depends on dimension.",
        },
        "rows": rows,
        "conclusion": (
            "The cited 3+1 sweep is consistent with both hypotheses. The cited "
            "2+1 sweep is central-value closer to universal n=5 (0.80) than to "
            "n=D+1 (0.75), but the difference is only about one sigma. This is "
            "not a derivation of n=5; at most it is weak numerical support for "
            "dimension-independent calibration."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
