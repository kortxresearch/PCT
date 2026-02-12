"""gw_change_point_pilot.py

Lightweight pilot / notebook-friendly entrypoint for groundwater (GW)
change-point detection.

This module is meant for quick experimentation and reuse from other code.
It shares the same simple DP segmentation algorithm as gw_change_point_runner.py
but exposes a small, testable API and optional plotting (if matplotlib exists).

Typical usage:

    from gw_change_point_pilot import run_pilot

    out = run_pilot(
        input_csv="data/gw.csv",
        date_col="date",
        value_col="gw",
        max_cps=3,
        min_segment_size=12,
        plot=True,
    )

    print(out["result"]["breakpoints"], out["result"]["breakpoint_dates"])

Notes:
- Input ordering is preserved; if your CSV is unsorted, sort it beforehand.
- Rows with non-numeric values are dropped.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _try_import_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def _coerce_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _try_import_pandas():
    try:
        import pandas as pd  # type: ignore

        return pd
    except Exception:
        return None


def load_csv_series(
    input_csv: str,
    date_col: str = "date",
    value_col: str = "value",
) -> Tuple[List[str], List[float]]:
    """Load (date_str, value) from a CSV.

    Uses pandas if available.
    """

    pd = _try_import_pandas()
    if pd is None:
        raise RuntimeError("pandas is required for the pilot loader; use gw_change_point_runner.py for stdlib-only")

    df = pd.read_csv(input_csv)
    if date_col not in df.columns or value_col not in df.columns:
        raise KeyError(f"Missing required columns: {date_col!r}, {value_col!r}")

    dates = df[date_col].astype(str).tolist()
    vals_raw = df[value_col].tolist()
    values: List[float] = []
    out_dates: List[str] = []
    for d, v in zip(dates, vals_raw):
        fv = _coerce_float(v)
        if fv is None:
            continue
        out_dates.append(str(d))
        values.append(fv)

    return out_dates, values


def _prefix_sums(x: Sequence[float]) -> Tuple[List[float], List[float]]:
    s = [0.0]
    s2 = [0.0]
    for v in x:
        s.append(s[-1] + v)
        s2.append(s2[-1] + v * v)
    return s, s2


def _segment_sse(prefix: Tuple[List[float], List[float]], i: int, j: int) -> Tuple[float, float]:
    s, s2 = prefix
    n = j - i
    if n <= 0:
        raise ValueError("Empty segment")
    seg_sum = s[j] - s[i]
    seg_sum2 = s2[j] - s2[i]
    mean = seg_sum / n
    sse = seg_sum2 - 2.0 * mean * seg_sum + n * mean * mean
    return mean, float(max(sse, 0.0))


def detect_change_points(
    x: Sequence[float],
    max_cps: int = 3,
    min_segment_size: int = 12,
) -> Dict[str, Any]:
    """Return a JSON-serializable dict with change-point results."""

    n = len(x)
    if n == 0:
        return {
            "n": 0,
            "breakpoints": [],
            "segment_means": [],
            "segment_sse": [],
            "total_sse": 0.0,
        }

    if min_segment_size <= 0:
        raise ValueError("min_segment_size must be > 0")

    k_max = max(0, int(max_cps)) + 1
    k_max = min(k_max, max(1, n // min_segment_size))

    prefix = _prefix_sums(x)

    inf = float("inf")
    dp: List[List[float]] = [[inf] * (n + 1) for _ in range(k_max + 1)]
    back: List[List[int]] = [[-1] * (n + 1) for _ in range(k_max + 1)]
    dp[0][0] = 0.0

    for k in range(1, k_max + 1):
        for j in range(k * min_segment_size, n + 1):
            best = inf
            best_i = -1
            i_min = (k - 1) * min_segment_size
            i_max = j - min_segment_size
            for i in range(i_min, i_max + 1):
                if dp[k - 1][i] == inf:
                    continue
                _, sse = _segment_sse(prefix, i, j)
                cost = dp[k - 1][i] + sse
                if cost < best:
                    best = cost
                    best_i = i
            dp[k][j] = best
            back[k][j] = best_i

    best_k = 1
    best_cost = dp[1][n]
    for k in range(2, k_max + 1):
        if dp[k][n] < best_cost:
            best_cost = dp[k][n]
            best_k = k

    bps: List[int] = []
    j = n
    for k in range(best_k, 0, -1):
        i = back[k][j]
        if i < 0:
            break
        if i != 0:
            bps.append(i)
        j = i
    bps.sort()

    seg_starts = [0] + bps
    seg_ends = bps + [n]
    means: List[float] = []
    sses: List[float] = []
    for a, b in zip(seg_starts, seg_ends):
        m, sse = _segment_sse(prefix, a, b)
        means.append(m)
        sses.append(sse)

    return {
        "n": int(n),
        "breakpoints": [int(x) for x in bps],
        "segment_means": [float(x) for x in means],
        "segment_sse": [float(x) for x in sses],
        "total_sse": float(sum(sses)),
    }


def plot_series_with_cps(
    dates: Sequence[str],
    values: Sequence[float],
    breakpoints: Sequence[int],
    title: str = "GW change points",
):
    """Plot the time series and vertical lines at change points (if matplotlib exists)."""

    plt = _try_import_matplotlib_pyplot()
    if plt is None:
        raise RuntimeError("matplotlib is not available")

    xs = list(range(len(values)))
    plt.figure(figsize=(10, 4))
    plt.plot(xs, values, linewidth=1.5)
    for bp in breakpoints:
        plt.axvline(bp, color="red", alpha=0.6, linewidth=1)
    plt.title(title)
    plt.xlabel("index")
    plt.ylabel("value")
    plt.tight_layout()
    return plt


def run_pilot(
    input_csv: str,
    date_col: str = "date",
    value_col: str = "value",
    max_cps: int = 3,
    min_segment_size: int = 12,
    plot: bool = False,
) -> Dict[str, Any]:
    """High-level helper that loads CSV, runs detection, and optionally plots."""

    dates, values = load_csv_series(input_csv, date_col=date_col, value_col=value_col)
    if not values:
        raise ValueError("No valid numeric values found")

    res = detect_change_points(values, max_cps=max_cps, min_segment_size=min_segment_size)

    bp_dates: List[str] = []
    for idx in res["breakpoints"]:
        bp_dates.append(dates[idx] if 0 <= idx < len(dates) else "")

    out: Dict[str, Any] = {
        "input": {
            "path": str(input_csv),
            "date_col": date_col,
            "value_col": value_col,
            "n_loaded": len(values),
        },
        "params": {"max_cps": int(max_cps), "min_segment_size": int(min_segment_size)},
        "result": {**res, "breakpoint_dates": bp_dates},
    }

    if plot:
        plot_series_with_cps(dates, values, out["result"]["breakpoints"])

    return out
