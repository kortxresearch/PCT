#!/usr/bin/env python3
"""NP-9 EHT null statement check."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/np9_eht_null.json")
    parser.add_argument("--kappa", type=float, default=0.80)
    args = parser.parse_args()

    photon_sphere_over_rh = 1.5
    schwarzschild_shadow_impact_over_rh = math.sqrt(27.0) / 2.0

    payload = {
        "purpose": "NP-9 EHT null statement.",
        "inputs": {
            "kappa_step_scale_over_horizon_radius": args.kappa,
            "schwarzschild_photon_sphere_radius_over_horizon_radius": photon_sphere_over_rh,
            "schwarzschild_shadow_critical_impact_over_horizon_radius": schwarzschild_shadow_impact_over_rh,
            "representative_eht_fractional_diameter_uncertainties": {
                "M87_2019": "about 7 percent for 42 +/- 3 microarcsec",
                "SgrA_2022": "order 10 percent, depending on ring/shadow observable",
            },
        },
        "reasoning": [
            "The PCT step scale is a diffusion/spectral-dimension scale, not a radial displacement of the photon sphere.",
            "The current locked instantiation supplies no photon-sector coupling and no metric perturbation at the photon sphere.",
            "Therefore the predicted EHT shadow-diameter shift in the present paper is exactly a null prediction, not a small nonzero number.",
        ],
        "computed": {
            "predicted_fractional_shadow_diameter_shift": 0.0,
            "eht_observable_modified_in_current_locked_instantiation": False,
            "step_scale_inside_schwarzschild_photon_sphere_if_misread_as_radius": args.kappa < photon_sphere_over_rh,
        },
        "falsification_rule": (
            "A future PCT extension that supplies a photon-sector coupling would "
            "need its own EHT prediction.  The present locked instantiation is "
            "not falsified by a standard GR-like EHT shadow, because it predicts "
            "no photon-sphere modification."
        ),
        "conclusion": (
            "The EHT statement is a null: current shadow measurements are not "
            "expected to show a PCT-specific deviation in the locked "
            "instantiation."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "predicted_shift": 0.0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
