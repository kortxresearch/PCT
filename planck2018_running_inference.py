#!/usr/bin/env python3
"""Planck 2018 running inference capsule (PCT)

This script is an *execution scaffold* matching the manuscript's V.M.4
"CMB running inference capsule" requirements.

Important scope note:
- This project does not bundle the Planck likelihood, CAMB/CLASS, or a sampler stack.
- Therefore this script:
  (i) validates and records the intended configuration (likelihood components, priors,
      parameter set, convergence thresholds),
  (ii) provides a consistent place to plug in Cobaya/MontePython/CosmoMC/etc., and
  (iii) emits a machine-readable results summary once you paste/return posterior numbers.

No downloads are performed; no external resources are fetched.

Usage:
  python planck2018_running_inference.py --out outputs/planck2018_running.json

Then either:
  - fill the "posterior" block manually in the output JSON, or
  - extend `run_external_inference()` to call your already-installed inference stack.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Configuration dataclasses
# -----------------------------

@dataclass(frozen=True)
class ConvergenceCriteria:
    rhat_max: float = 1.01
    ess_min: int = 1000


@dataclass(frozen=True)
class Prior:
    name: str
    dist: str
    params: Dict[str, float]


@dataclass(frozen=True)
class LikelihoodConfig:
    """High-level commitment only (as in the manuscript).

    Record exact component names/versions in the executed run artifact.
    """

    high_ell: str = "Planck 2018 high-ell TT/TE/EE"
    low_ell: str = "Planck 2018 low-ell (T+E)"
    lensing: str = "Planck 2018 lensing"


@dataclass(frozen=True)
class ModelConfig:
    """Baseline model per the manuscript capsule."""

    model_name: str = "LambdaCDM + running"
    parameters: Tuple[str, ...] = (
        "omega_b",
        "omega_c",
        "theta_MC_100",
        "tau",
        "ln10^{10}A_s",
        "n_s",
        "alpha_s",  # alpha_s := dn_s/d ln k
    )


@dataclass(frozen=True)
class CapsuleConfig:
    """Everything needed to make the capsule reproducible at the protocol level."""

    created_utc: str
    likelihood: LikelihoodConfig
    model: ModelConfig
    priors: Tuple[Prior, ...]
    convergence: ConvergenceCriteria
    notes: str


# -----------------------------
# Defaults matching V.M.4
# -----------------------------

DEFAULT_PRIORS: Tuple[Prior, ...] = (
    Prior("omega_b", "uniform", {"min": 0.005, "max": 0.1}),
    Prior("omega_c", "uniform", {"min": 0.001, "max": 0.99}),
    Prior("theta_MC_100", "uniform", {"min": 0.5, "max": 10.0}),
    Prior("tau", "uniform", {"min": 0.01, "max": 0.8}),
    Prior("ln10^{10}A_s", "uniform", {"min": 1.0, "max": 5.0}),
    Prior("n_s", "uniform", {"min": 0.8, "max": 1.2}),
    # Manuscript default: Uniform[-0.1, 0.1] for running
    Prior("alpha_s", "uniform", {"min": -0.1, "max": 0.1}),
)


# -----------------------------
# Helpers
# -----------------------------


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def pcts_running_band() -> Dict[str, float]:
    """Model-locked prediction band used throughout the manuscript."""
    return {"mu": -0.012, "sigma": 0.005}


def evaluate_band_membership(alpha_s_ci68: Tuple[float, float], alpha_s_ci95: Tuple[float, float]) -> Dict[str, Any]:
    """Return whether PCT's (mu±sigma) band sits inside the posterior CI."""
    band = pcts_running_band()
    lo_band = band["mu"] - band["sigma"]
    hi_band = band["mu"] + band["sigma"]

    def contains(ci: Tuple[float, float]) -> bool:
        lo, hi = ci
        return lo <= lo_band and hi_band <= hi

    return {
        "pct_band": {"lo": lo_band, "hi": hi_band, **band},
        "pct_band_within_ci68": contains(alpha_s_ci68),
        "pct_band_within_ci95": contains(alpha_s_ci95),
    }


# -----------------------------
# External inference hook (intentionally stubbed)
# -----------------------------


def run_external_inference(config: CapsuleConfig) -> Dict[str, Any]:
    """Hook for a real likelihood run.

    Replace this with your local inference stack invocation (e.g., cobaya).

    Must return a dict with at minimum:
      {
        "alpha_s": {
          "median": float,
          "mean": float,
          "ci68": [lo, hi],
          "ci95": [lo, hi],
        },
        "convergence": {"rhat_max": float, "ess_min": int},
        "runtime": {...}
      }

    This repository intentionally ships as a no-external-deps scaffold.
    """
    raise NotImplementedError(
        "No inference backend bundled. Either (i) fill posterior numbers manually in the JSON output, "
        "or (ii) implement run_external_inference() for your environment."
    )


# -----------------------------
# Main
# -----------------------------


def build_default_config() -> CapsuleConfig:
    return CapsuleConfig(
        created_utc=_dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        likelihood=LikelihoodConfig(),
        model=ModelConfig(),
        priors=DEFAULT_PRIORS,
        convergence=ConvergenceCriteria(),
        notes=(
            "Protocol-level capsule only. Record the exact likelihood component names/versions in the executed run artifact. "
            "Running parameter is alpha_s := dn_s/d ln k."
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/planck2018_running.json",
        help="Where to write the capsule configuration + (optional) results JSON.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Attempt to run an external inference backend via run_external_inference() (stub by default).",
    )
    args = parser.parse_args()

    cfg = build_default_config()

    payload: Dict[str, Any] = {
        "capsule": {
            "config": asdict(cfg),
            "pct_prediction_band": pcts_running_band(),
        },
        "posterior": {
            # Fill these once you have an executed likelihood run.
            "alpha_s": {
                "median": None,
                "mean": None,
                "ci68": [None, None],
                "ci95": [None, None],
            },
            "convergence": {
                "rhat_max": None,
                "ess_min": None,
            },
            "provenance": {
                "likelihood_package": None,
                "likelihood_components_exact": {
                    "high_ell": None,
                    "low_ell": None,
                    "lensing": None,
                },
                "boltzmann_code": None,
                "sampler": None,
                "version_notes": None,
            },
        },
    }

    if args.run:
        results = run_external_inference(cfg)
        payload["posterior"].update(results)

        a = payload["posterior"].get("alpha_s", {})
        ci68 = tuple(a.get("ci68", [None, None]))
        ci95 = tuple(a.get("ci95", [None, None]))
        if all(x is not None for x in ci68) and all(x is not None for x in ci95):
            payload["posterior"]["pct_band_check"] = evaluate_band_membership(
                alpha_s_ci68=(float(ci68[0]), float(ci68[1])),
                alpha_s_ci95=(float(ci95[0]), float(ci95[1])),
            )

    out_path = Path(args.out)
    ensure_parent(out_path)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
