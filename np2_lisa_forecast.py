#!/usr/bin/env python3
"""NP-2 LISA massive-black-hole ringdown forecast table."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


GMSUN_OVER_C3_S = 4.925490947e-6


def spin_factor(a: float) -> float:
    return 1.0 + math.sqrt(1.0 - a * a)


def pct_snr(epsilon0: float, rho: float, tau_over_m: float, tobs_over_m: float, tc_over_m: float) -> float:
    return epsilon0 * rho * math.sqrt(tau_over_m / tobs_over_m) * math.exp(-tc_over_m / tau_over_m)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/np2_lisa_forecast.json")
    parser.add_argument("--kappa", type=float, default=0.80)
    parser.add_argument("--spin", type=float, default=0.67)
    parser.add_argument("--epsilon0", type=float, default=0.02)
    parser.add_argument("--tau-over-M", type=float, default=55.0)
    parser.add_argument("--tobs-over-M", type=float, default=55.0)
    args = parser.parse_args()

    masses = [1e5, 3e5, 1e6, 3e6, 1e7]
    rho_values = [100.0, 300.0, 1000.0]
    tc_spin = args.kappa * spin_factor(args.spin)
    tc_nonspin = 2.0 * args.kappa

    rows = []
    for mass in masses:
        row = {
            "redshifted_remnant_mass_Msun": mass,
            "tc_seconds_spin_a_%.2f" % args.spin: tc_spin * mass * GMSUN_OVER_C3_S,
            "tc_seconds_nonspinning": tc_nonspin * mass * GMSUN_OVER_C3_S,
            "snr_pct_by_ringdown_snr": {
                str(int(rho)): pct_snr(args.epsilon0, rho, args.tau_over_M, args.tobs_over_M, tc_spin)
                for rho in rho_values
            },
        }
        rows.append(row)

    payload = {
        "purpose": "NP-2 LISA per-event decisiveness forecast.",
        "formulae": {
            "tc": "tc = kappa*(1+sqrt(1-a_*^2))*G*M_z/c^3",
            "snr_pct": "epsilon0*rho*sqrt(tau_d/Tobs)*exp(-tc/tau_d)",
        },
        "inputs": {
            "kappa": args.kappa,
            "spin": args.spin,
            "epsilon0": args.epsilon0,
            "tau_over_M": args.tau_over_M,
            "Tobs_over_M": args.tobs_over_M,
            "rho_values": rho_values,
            "mass_is_redshifted_detector_frame": True,
        },
        "rows": rows,
        "conclusion": (
            "For 10^5-10^7 solar-mass redshifted remnants, the change-point "
            "falls on second-to-minute timescales.  With LISA ringdown SNR of "
            "order 10^2-10^3, the representative PCT component SNR is order "
            "2-20 per event under the same epsilon0 model used for ground-based "
            "stacking."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
