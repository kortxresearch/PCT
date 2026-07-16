#!/usr/bin/env python3
"""D1 option (b): corrected-flow numerical check.

This script does not decide D1. It quantifies what happens if the old smooth
flow with UV endpoint d_s=2 is replaced by an effective interpolation whose
UV endpoint is d_UV = 4 - Delta d_s, using the locked value
Delta d_s = 0.79 +/- 0.15.

Because the manuscript does not specify a transfer functional from the
dimension-flow beta function to alpha_s, the script reports several transparent
normalisations rather than pretending there is a unique recomputation.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class FlowMetrics:
    d_uv: float
    delta_ds: float
    interval_ratio: float
    beta_peak_abs: float
    beta_peak_at_ds: float
    beta_peak_ratio: float
    beta_area_abs: float
    beta_area_ratio: float


def beta(d: float, d_uv: float) -> float:
    """Generalized effective flow beta(d_s)=d_s(d_s-4)(d_s-d_UV)/4."""

    return d * (d - 4.0) * (d - d_uv) / 4.0


def integrate_abs_beta(d_uv: float, n: int = 20000) -> float:
    a = d_uv
    b = 4.0
    if b <= a:
        return 0.0
    h = (b - a) / n
    total = 0.0
    for i in range(n + 1):
        x = a + i * h
        weight = 0.5 if i in (0, n) else 1.0
        total += weight * abs(beta(x, d_uv))
    return total * h


def maximize_abs_beta(d_uv: float, n: int = 20000) -> tuple[float, float]:
    best_x = d_uv
    best_y = 0.0
    for i in range(n + 1):
        x = d_uv + (4.0 - d_uv) * i / n
        y = abs(beta(x, d_uv))
        if y > best_y:
            best_x = x
            best_y = y
    return best_y, best_x


def metrics_for(delta_ds: float, baseline: FlowMetrics | None = None) -> FlowMetrics:
    d_uv = 4.0 - delta_ds
    peak, peak_at = maximize_abs_beta(d_uv)
    area = integrate_abs_beta(d_uv)
    if baseline is None:
        interval_ratio = 1.0
        peak_ratio = 1.0
        area_ratio = 1.0
    else:
        interval_ratio = delta_ds / baseline.delta_ds
        peak_ratio = peak / baseline.beta_peak_abs if baseline.beta_peak_abs else math.nan
        area_ratio = area / baseline.beta_area_abs if baseline.beta_area_abs else math.nan
    return FlowMetrics(
        d_uv=d_uv,
        delta_ds=delta_ds,
        interval_ratio=interval_ratio,
        beta_peak_abs=peak,
        beta_peak_at_ds=peak_at,
        beta_peak_ratio=peak_ratio,
        beta_area_abs=area,
        beta_area_ratio=area_ratio,
    )


def scale_interval(m: FlowMetrics) -> float:
    return m.interval_ratio


def scale_peak(m: FlowMetrics) -> float:
    return m.beta_peak_ratio


def scale_area(m: FlowMetrics) -> float:
    return m.beta_area_ratio


def alpha_band(alpha_central: float, alpha_sigma: float, scale: float) -> dict[str, float]:
    central = alpha_central * scale
    sigma = abs(alpha_sigma * scale)
    return {
        "central": central,
        "sigma": sigma,
        "low_1sigma": central - sigma,
        "high_1sigma": central + sigma,
    }


def in_original_band(value: float) -> bool:
    return -0.02 <= value <= -0.005


def build_result() -> dict[str, object]:
    alpha_old = -0.012
    alpha_old_sigma = 0.005
    delta_old = 2.0
    delta_central = 0.79
    delta_sigma = 0.15

    baseline = metrics_for(delta_old)
    central = metrics_for(delta_central, baseline)
    low_delta = metrics_for(delta_central - delta_sigma, baseline)
    high_delta = metrics_for(delta_central + delta_sigma, baseline)

    scalings: dict[str, Callable[[FlowMetrics], float]] = {
        "linear_in_dimension_excursion": scale_interval,
        "linear_in_beta_peak": scale_peak,
        "linear_in_integrated_abs_beta": scale_area,
    }

    mapped: dict[str, object] = {}
    for name, scale_fn in scalings.items():
        scale_c = scale_fn(central)
        mapped[name] = {
            "scale_central": scale_c,
            "scale_delta_minus_sigma": scale_fn(low_delta),
            "scale_delta_plus_sigma": scale_fn(high_delta),
            "alpha_s": alpha_band(alpha_old, alpha_old_sigma, scale_c),
            "central_survives_original_band": in_original_band(alpha_old * scale_c),
        }

    return {
        "purpose": "Inform D1 option (b); this output is not an author decision.",
        "inputs": {
            "old_flow_d_uv": 2.0,
            "old_flow_delta_ds": delta_old,
            "old_alpha_s_central": alpha_old,
            "old_alpha_s_sigma": alpha_old_sigma,
            "corrected_delta_ds": delta_central,
            "corrected_delta_ds_sigma": delta_sigma,
            "corrected_d_uv": central.d_uv,
            "corrected_d_uv_sigma": delta_sigma,
            "flow_form": "beta(d_s)=d_s(d_s-4)(d_s-d_UV)/4",
        },
        "baseline_metrics": asdict(baseline),
        "corrected_metrics": {
            "delta_central": asdict(central),
            "delta_minus_sigma": asdict(low_delta),
            "delta_plus_sigma": asdict(high_delta),
        },
        "mapped_alpha_s_under_explicit_scalings": mapped,
        "conclusion": (
            "Option (b) is underdetermined unless the manuscript supplies a "
            "transfer functional from the effective d_s flow to alpha_s. Under "
            "simple no-retuning scalings, the corrected endpoint weakens the "
            "old alpha_s magnitude; only the most generous linear-excursion "
            "scaling lands on the edge of the original [-0.02,-0.005] band."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/d1_option_b_corrected_flow.json")
    args = parser.parse_args()

    result = build_result()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["mapped_alpha_s_under_explicit_scalings"], indent=2, sort_keys=True))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
