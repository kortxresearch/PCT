#!/usr/bin/env python3
"""C3/C4 ringdown timing corrections for the v4 draft."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


GMSUN_OVER_C3_S = 4.925490947e-6


def spin_scaled_tc_over_m(kappa: float, final_spin: float) -> float:
    """Horizon-scale law: t_c/M = kappa * (1 + sqrt(1-a_*^2))."""

    return kappa * (1.0 + math.sqrt(max(0.0, 1.0 - final_spin * final_spin)))


def snr_pct(
    tc_over_m: float,
    epsilon0: float = 0.02,
    rho: float = 20.0,
    tau_over_m: float = 55.0,
    tobs_over_m: float = 55.0,
) -> float:
    return epsilon0 * rho * math.sqrt(tau_over_m / tobs_over_m) * math.exp(
        -tc_over_m / tau_over_m
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/ringdown_timing_corrections.json")
    args = parser.parse_args()

    kappa = 0.80
    kappa_sigma = 0.05
    mass_source_msun = 62.0
    mass_sigma_msun = 2.5
    redshift = 0.09
    spin = 0.67
    spin_sigma = 0.03

    tc_over_m_nonspin = 2.0 * kappa
    tc_ms_nonspin_source_frame = (
        tc_over_m_nonspin * mass_source_msun * GMSUN_OVER_C3_S * 1000.0
    )
    tc_ms_nonspin_detector_frame = tc_ms_nonspin_source_frame * (1.0 + redshift)
    rel_sigma = math.sqrt((kappa_sigma / kappa) ** 2 + (mass_sigma_msun / mass_source_msun) ** 2)

    tc_over_m_spin = spin_scaled_tc_over_m(kappa, spin)
    tc_ms_spin_detector_frame = (
        tc_over_m_spin * mass_source_msun * (1.0 + redshift) * GMSUN_OVER_C3_S * 1000.0
    )

    restored_tc = 50.0
    restored_snr = snr_pct(restored_tc)
    nonspin_snr = snr_pct(tc_over_m_nonspin)
    spin_snr = snr_pct(tc_over_m_spin)

    result = {
        "purpose": "Correct C3/C4 for v4 draft; no author decision is made here.",
        "inputs": {
            "kappa": kappa,
            "kappa_sigma": kappa_sigma,
            "mass_source_msun": mass_source_msun,
            "mass_sigma_msun": mass_sigma_msun,
            "redshift": redshift,
            "final_spin": spin,
            "final_spin_sigma": spin_sigma,
            "GMsun_over_c3_s": GMSUN_OVER_C3_S,
        },
        "timing": {
            "nonspinning_tc_over_M": tc_over_m_nonspin,
            "nonspinning_source_frame_ms_no_redshift": tc_ms_nonspin_source_frame,
            "nonspinning_detector_frame_ms_with_1_plus_z": tc_ms_nonspin_detector_frame,
            "nonspinning_detector_frame_ms_sigma_from_kappa_and_mass": tc_ms_nonspin_detector_frame
            * rel_sigma,
            "spin_scaled_law": "t_c/M = kappa*(1 + sqrt(1-a_*^2))",
            "spin_scaled_tc_over_M_at_a_0p67": tc_over_m_spin,
            "spin_scaled_detector_frame_ms_at_a_0p67": tc_ms_spin_detector_frame,
        },
        "representative_snr_forecast": {
            "formula": "epsilon0*rho*sqrt(tau_d/T_obs)*exp(-t_c/tau_d)",
            "epsilon0": 0.02,
            "rho": 20.0,
            "tau_d_over_M": 55.0,
            "T_obs_over_M": 55.0,
            "old_restoration_tc_over_M": restored_tc,
            "old_restoration_snr": restored_snr,
            "nonspinning_tc_1p6M_snr": nonspin_snr,
            "spin_scaled_tc_at_a_0p67_snr": spin_snr,
            "nonspinning_stack_law": f"SNR_stack ~= {nonspin_snr:.3f}*sqrt(N)",
            "spin_scaled_stack_law": f"SNR_stack ~= {spin_snr:.3f}*sqrt(N)",
        },
        "conclusion": (
            "With the (1+z) factor retained, the non-spinning GW150914 timing is "
            "0.532 ms, not 0.489 ms. Using the spin-scaled horizon law with "
            "a_*=0.67 gives t_c/M=1.393 and 0.463 ms. The old 50M SNR estimate "
            "is not consistent with the locked t_c/M~1.6 timing; the same "
            "representative formula gives SNR~0.39*sqrt(N), not 0.16*sqrt(N)."
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
