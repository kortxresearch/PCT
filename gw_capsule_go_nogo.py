"""gw_capsule_go_nogo.py

GO/NO-GO runner for the GW (groundwater) change-point capsule.

Reads configs/gw_capsule_v1.json, fits k = 0..max_cps, compares by AIC, and
(optionally) computes a null-test p-value using surrogate baselines.

Writes a single JSON report containing a final go/no-go decision.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List


def _try_import_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except Exception as e:
        raise RuntimeError("This runner requires numpy for null surrogates.") from e


def _read_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | os.PathLike[str], obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _aic_gaussian_sse(n: int, sse: float, k_params: int) -> float:
    if n <= 0:
        raise ValueError("n must be > 0")
    sse = float(max(sse, 1e-12))
    return 2.0 * k_params + n * (math.log(sse / n) + 1.0)


def _fit_k_cps(values: List[float], k_cps: int, min_segment_size: int):
    from gw_change_point_runner import detect_change_points

    return detect_change_points(values, max_cps=k_cps, min_segment_size=min_segment_size)


def _breakpoint_dates(dates: List[str], breakpoints: List[int]) -> List[str]:
    out: List[str] = []
    for i in breakpoints:
        out.append(dates[i] if 0 <= i < len(dates) else "")
    return out


def _time_scramble(np, x: List[float], rng) -> List[float]:
    arr = np.array(x, dtype=float)
    rng.shuffle(arr)
    return arr.tolist()


def _phase_randomize(np, x: List[float], rng) -> List[float]:
    arr = np.array(x, dtype=float)
    n = arr.shape[0]
    X = np.fft.rfft(arr)
    mag = np.abs(X)

    kmax = X.shape[0]
    rand_ph = rng.uniform(0.0, 2.0 * np.pi, size=kmax)
    rand_ph[0] = 0.0
    if n % 2 == 0 and kmax > 1:
        rand_ph[-1] = 0.0

    Y = mag * np.exp(1j * rand_ph)
    y = np.fft.irfft(Y, n=n)
    return y.tolist()


def _null_p_value(
    values: List[float],
    min_segment_size: int,
    max_cps: int,
    delta_AIC_obs: float,
    cfg_null: Dict[str, Any],
) -> Dict[str, Any]:
    np = _try_import_numpy()

    alpha = float(cfg_null.get("alpha", 0.05))
    gens = cfg_null.get("generators", [])
    out: Dict[str, Any] = {"alpha": alpha, "by_generator": {}}

    for g in gens:
        gid = str(g.get("id"))
        num = int(g.get("num_draws", 200))
        seed = int(g.get("seed", 0))
        rng = np.random.default_rng(seed)

        exceed = 0
        for _ in range(num):
            if gid == "time_scramble":
                x0 = _time_scramble(np, values, rng)
            elif gid == "phase_randomize":
                x0 = _phase_randomize(np, values, rng)
            else:
                raise ValueError(f"Unknown null generator id={gid!r}")

            aic0 = None
            aic_best_cp = None
            for k in range(0, max_cps + 1):
                res = _fit_k_cps(x0, k_cps=k, min_segment_size=min_segment_size)
                k_params = (len(res.segment_means) + 1)
                aic = _aic_gaussian_sse(res.n, res.total_sse, k_params=k_params)
                if k == 0:
                    aic0 = aic
                else:
                    aic_best_cp = aic if aic_best_cp is None else min(aic_best_cp, aic)

            if aic0 is None or aic_best_cp is None:
                continue

            delta = aic0 - aic_best_cp
            if delta >= delta_AIC_obs:
                exceed += 1

        p = (exceed + 1) / (num + 1)
        out["by_generator"][gid] = {"num_draws": num, "seed": seed, "p_value": float(p)}

    pvals = [v["p_value"] for v in out["by_generator"].values() if isinstance(v, dict) and "p_value" in v]
    out["p_value_combined"] = float(max(pvals)) if pvals else None
    out["pass"] = (out["p_value_combined"] is not None) and (out["p_value_combined"] <= alpha)
    return out


def run_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    from gw_change_point_runner import load_csv_series

    inp = cfg["inputs"]
    csv_path = inp["csv"]
    date_col = inp.get("date_col", "date")
    value_col = inp.get("value_col", "value")

    seg = cfg.get("segmentation", {})
    max_cps = int(seg.get("max_cps", 3))
    min_seg = int(seg.get("min_segment_size", 12))

    dates, values = load_csv_series(csv_path, date_col=date_col, value_col=value_col)

    rows: List[Dict[str, Any]] = []

    for k in range(0, max_cps + 1):
        res = _fit_k_cps(values, k_cps=k, min_segment_size=min_seg)
        k_params = (len(res.segment_means) + 1)
        aic = _aic_gaussian_sse(res.n, res.total_sse, k_params=k_params)
        rows.append({"k": k, "aic": float(aic), "breakpoints": list(res.breakpoints)})

    aic_k0 = next(r["aic"] for r in rows if r["k"] == 0)
    aic_best_cp = min((r["aic"] for r in rows if r["k"] >= 1), default=None)
    delta_aic = float(aic_k0 - aic_best_cp) if aic_best_cp is not None else 0.0

    rule = cfg.get("model_comparison", {})
    delta_go = float(rule.get("delta_AIC_go", 10.0))
    prefers_cp = (aic_best_cp is not None) and (delta_aic >= delta_go)

    best_row = min(rows, key=lambda r: r["aic"])
    best_res = _fit_k_cps(values, k_cps=int(best_row["k"]), min_segment_size=min_seg)

    report: Dict[str, Any] = {
        "input": {"csv": csv_path, "n": len(values)},
        "models": rows,
        "selected": {
            "k": int(best_row["k"]),
            "breakpoints": list(best_res.breakpoints),
            "breakpoint_dates": _breakpoint_dates(dates, list(best_res.breakpoints)),
            "segment_means": list(best_res.segment_means),
            "total_sse": float(best_res.total_sse),
        },
        "decision_inputs": {
            "delta_AIC_k0_vs_best_cp": float(delta_aic),
            "delta_AIC_go_threshold": float(delta_go),
            "prefers_change_point": bool(prefers_cp),
        },
    }

    null_cfg = cfg.get("null_tests", {})
    if bool(null_cfg.get("enabled", False)) and aic_best_cp is not None:
        report["null_tests"] = _null_p_value(
            values=values,
            min_segment_size=min_seg,
            max_cps=max_cps,
            delta_AIC_obs=delta_aic,
            cfg_null=null_cfg,
        )
        null_pass = bool(report["null_tests"].get("pass", False))
    else:
        report["null_tests"] = {"enabled": False}
        null_pass = True

    reasons: List[str] = []
    go = True
    if not prefers_cp:
        go = False
        reasons.append("NO-GO: change-point model not preferred by AIC threshold")
    if not null_pass:
        go = False
        reasons.append("NO-GO: null-test p-value above alpha")

    report["go_no_go"] = {"go": bool(go), "reasons": reasons}
    return report


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run GW capsule GO/NO-GO")
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)

    cfg = _read_json(args.config)
    out = run_from_config(cfg)

    out_path = cfg.get("artifacts", {}).get("out_json", "outputs/gw_capsule_v1/result.json")
    _write_json(out_path, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
