#!/usr/bin/env python3
"""One-click runner for the in-repo capsules.

This script is intentionally dependency-light and only runs capsules that can be
executed with the dependencies available in the current environment.

It writes machine-readable JSON outputs under the project-level `outputs/` dir.

Usage (from project folder):
  python run_capsules.py

You can also select subsets:
  python run_capsules.py --gw-only
  python run_capsules.py --lvk-only
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_gw_change_point(outputs_dir: Path) -> Dict[str, Any]:
    from gw_change_point_runner import detect_change_points, load_csv_series

    dates, values = load_csv_series(
        "data/gw_synth_monthly.csv",
        date_col="date",
        value_col="value",
    )
    res = detect_change_points(values, max_cps=3, min_segment_size=12, select="bic")

    bp_dates = [dates[i] if 0 <= i < len(dates) else "" for i in res.breakpoints]

    payload = {
        "input": {
            "path": "data/gw_synth_monthly.csv",
            "date_col": "date",
            "value_col": "value",
            "n_loaded": len(values),
        },
        "params": {
            "max_cps": 3,
            "min_segment_size": 12,
            "select": "bic",
        },
        "result": {
            "n": int(res.n),
            "breakpoints": [int(x) for x in res.breakpoints],
            "breakpoint_dates": bp_dates,
            "segment_means": [float(x) for x in res.segment_means],
            "segment_sse": [float(x) for x in res.segment_sse],
            "total_sse": float(res.total_sse),
        },
    }

    _write_json(outputs_dir / "gw_change_points_synth.json", payload)
    return {"ok": True, "output": str(outputs_dir / "gw_change_points_synth.json")}


def run_lvk_ringdown(outputs_dir: Path) -> Dict[str, Any]:
    try:
        from lvk_ringdown_end_to_end import build_default_config, run_capsule

        cfg = build_default_config()
        payload = run_capsule(cfg)
        _write_json(outputs_dir / "lvk_ringdown_public_run.json", payload)
        return {"ok": True, "output": str(outputs_dir / "lvk_ringdown_public_run.json")}
    except Exception as e:
        # Typically missing optional dependency `h5py`.
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def run_planck_scaffold(outputs_dir: Path) -> Dict[str, Any]:
    from planck2018_running_inference import build_default_config, pcts_running_band

    cfg = build_default_config()
    payload = {
        "capsule": {"config": asdict(cfg), "pct_prediction_band": pcts_running_band()},
        "posterior": {
            "alpha_s": {"median": None, "mean": None, "ci68": [None, None], "ci95": [None, None]}
        },
    }
    _write_json(outputs_dir / "planck2018_running_scaffold.json", payload)
    return {"ok": True, "output": str(outputs_dir / "planck2018_running_scaffold.json")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs", help="Output directory (default: outputs)")
    ap.add_argument("--gw-only", action="store_true", help="Run only the GW change-point capsule")
    ap.add_argument("--lvk-only", action="store_true", help="Run only the LVK ringdown capsule")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs)
    _ensure_dir(outputs_dir)

    report: Dict[str, Any] = {"outputs_dir": str(outputs_dir)}

    if args.gw_only:
        report["gw_change_point"] = run_gw_change_point(outputs_dir)
    elif args.lvk_only:
        report["lvk_ringdown"] = run_lvk_ringdown(outputs_dir)
    else:
        report["gw_change_point"] = run_gw_change_point(outputs_dir)
        report["lvk_ringdown"] = run_lvk_ringdown(outputs_dir)
        report["planck_scaffold"] = run_planck_scaffold(outputs_dir)

    _write_json(outputs_dir / "capsule_run_report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
