"""gw_change_point_runner.py

CLI runner for simple change-point detection on groundwater (GW) time series.

This script is intentionally self-contained (standard library + NumPy/Pandas if
available) so it can run in minimal environments.

Expected input: a CSV with at least two columns:
  - a datetime-like column (default: "date")
  - a numeric value column (default: "value")

It outputs:
  - a JSON summary (to stdout or a file)
  - optionally a CSV with per-point segment ids

Example:
  python -m PCT.gw_change_point_runner \
    --input data/gw.csv --date-col date --value-col gw \
    --max-cps 3 --min-segment-size 12 --out results/cps.json
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChangePointResult:
    """Change-point detection output.

    Indices are 0-based and refer to the input sequence ordering.

    `breakpoints` are the starting indices of new segments, excluding 0.
    Example: breakpoints [10, 25] => segments [0..9], [10..24], [25..end].
    """

    n: int
    breakpoints: List[int]
    segment_means: List[float]
    segment_sse: List[float]
    total_sse: float


def _try_import_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except Exception:
        return None


def _try_import_pandas():
    try:
        import pandas as pd  # type: ignore

        return pd
    except Exception:
        return None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run simple GW change-point detection")
    p.add_argument("--input", required=True, help="Path to input CSV")
    p.add_argument("--date-col", default="date", help="Datetime column name")
    p.add_argument("--value-col", default="value", help="Numeric value column name")
    p.add_argument(
        "--max-cps",
        type=int,
        default=3,
        help="Maximum number of change points (segments = cps + 1)",
    )
    p.add_argument(
        "--min-segment-size",
        type=int,
        default=12,
        help="Minimum points per segment (e.g., 12 for monthly series)",
    )
    p.add_argument(
        "--select",
        default="bic",
        choices=["bic", "sse"],
        help="Model selection for number of segments: BIC (default) or raw SSE.",
    )
    p.add_argument(
        "--out",
        default="-",
        help='Output JSON path ("-" for stdout).',
    )
    p.add_argument(
        "--segments-out",
        default=None,
        help="Optional CSV path with segment id per row.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


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


def load_csv_series(
    path: str | os.PathLike[str],
    date_col: str,
    value_col: str,
) -> Tuple[List[str], List[float]]:
    """Load (date_str, value) from a CSV.

    Dates are kept as strings in the output; ordering follows file order.
    Rows with missing/invalid values are skipped.
    """

    pd = _try_import_pandas()
    if pd is not None:
        df = pd.read_csv(path)
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

    # Standard library fallback
    out_dates: List[str] = []
    values: List[float] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError("CSV appears to have no header")
        if date_col not in r.fieldnames or value_col not in r.fieldnames:
            raise KeyError(f"Missing required columns: {date_col!r}, {value_col!r}")
        for row in r:
            fv = _coerce_float(row.get(value_col))
            if fv is None:
                continue
            out_dates.append(str(row.get(date_col, "")))
            values.append(fv)
    return out_dates, values


def _prefix_sums(x: Sequence[float]) -> Tuple[List[float], List[float]]:
    """Return prefix sums S and S2 such that:

    sum(x[i:j])  = S[j]  - S[i]
    sum(x^2[i:j])= S2[j] - S2[i]

    with S[0]=S2[0]=0.
    """

    s = [0.0]
    s2 = [0.0]
    for v in x:
        s.append(s[-1] + v)
        s2.append(s2[-1] + v * v)
    return s, s2


def _segment_sse(prefix: Tuple[List[float], List[float]], i: int, j: int) -> Tuple[float, float]:
    """Return (mean, SSE) for x[i:j] (j exclusive)."""

    s, s2 = prefix
    n = j - i
    if n <= 0:
        raise ValueError("Empty segment")
    seg_sum = s[j] - s[i]
    seg_sum2 = s2[j] - s2[i]
    mean = seg_sum / n
    # SSE = sum((x-mean)^2) = sum(x^2) - 2*mean*sum(x) + n*mean^2
    sse = seg_sum2 - 2.0 * mean * seg_sum + n * mean * mean
    return mean, float(max(sse, 0.0))


def _bic_piecewise_constant_mean(n: int, total_sse: float, k_segments: int) -> float:
    """BIC for a Gaussian model with piecewise-constant mean.

    Parameters counted (roughly):
    - k mean parameters
    - (k-1) discrete breakpoint locations (counted as parameters for selection)

    BIC := n*log(SSE/n) + p*log(n), with p = 2k-1.
    """

    if n <= 1:
        return float("inf")
    sse = max(float(total_sse), 1e-300)
    p = max(1, 2 * int(k_segments) - 1)
    return float(n * math.log(sse / float(n)) + float(p) * math.log(float(n)))


def detect_change_points(
    x: Sequence[float],
    max_cps: int,
    min_segment_size: int,
    *,
    select: str = "bic",
) -> ChangePointResult:
    """Detect up to `max_cps` change points via DP (piecewise constant mean).

    This solves an L2 piecewise-constant segmentation problem (DP over segments)
    with a post-hoc model-selection rule for the number of segments.

    Selection options:
    - select="bic" (default): choose k minimizing a simple Gaussian BIC.
    - select="sse": choose k minimizing raw SSE (will typically prefer more segments).

    Complexity is O(K*N^2) which is fine for small/medium N.
    """

    n = len(x)
    if n == 0:
        return ChangePointResult(n=0, breakpoints=[], segment_means=[], segment_sse=[], total_sse=0.0)
    if min_segment_size <= 0:
        raise ValueError("min_segment_size must be > 0")

    k_max = max(0, int(max_cps)) + 1  # number of segments
    # You can't have more segments than floor(n/min_segment_size)
    k_max = min(k_max, max(1, n // min_segment_size))

    prefix = _prefix_sums(x)

    # dp[k][j] = best cost using k segments to cover x[0:j]
    # back[k][j] = best previous breakpoint i
    inf = float("inf")
    dp: List[List[float]] = [[inf] * (n + 1) for _ in range(k_max + 1)]
    back: List[List[int]] = [[-1] * (n + 1) for _ in range(k_max + 1)]
    dp[0][0] = 0.0

    for k in range(1, k_max + 1):
        # j is end index
        for j in range(k * min_segment_size, n + 1):
            best = inf
            best_i = -1
            # i is previous end
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

    # Choose best k (up to k_max) for full length.
    best_k = 1
    if select == "sse":
        best_cost = dp[1][n]
        for k in range(2, k_max + 1):
            if dp[k][n] < best_cost:
                best_cost = dp[k][n]
                best_k = k
    elif select == "bic":
        best_cost = _bic_piecewise_constant_mean(n=n, total_sse=dp[1][n], k_segments=1)
        for k in range(2, k_max + 1):
            cost = _bic_piecewise_constant_mean(n=n, total_sse=dp[k][n], k_segments=k)
            if cost < best_cost:
                best_cost = cost
                best_k = k
    else:
        raise ValueError("select must be 'bic' or 'sse'")

    # Reconstruct breakpoints
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

    # Segment stats
    seg_starts = [0] + bps
    seg_ends = bps + [n]
    means: List[float] = []
    sses: List[float] = []
    for a, b in zip(seg_starts, seg_ends):
        m, sse = _segment_sse(prefix, a, b)
        means.append(m)
        sses.append(sse)

    return ChangePointResult(
        n=n,
        breakpoints=bps,
        segment_means=means,
        segment_sse=sses,
        total_sse=float(sum(sses)),
    )


def write_json(obj: Any, out_path: str) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True)
    if out_path == "-":
        print(text)
        return
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(text + "\n", encoding="utf-8")


def write_segments_csv(
    out_path: str,
    dates: Sequence[str],
    values: Sequence[float],
    breakpoints: Sequence[int],
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    bps = list(breakpoints)
    seg_id = 0
    next_bp_idx = 0
    next_bp = bps[next_bp_idx] if bps else None

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "value", "segment_id"])
        for i, (d, v) in enumerate(zip(dates, values)):
            if next_bp is not None and i >= next_bp:
                seg_id += 1
                next_bp_idx += 1
                next_bp = bps[next_bp_idx] if next_bp_idx < len(bps) else None
            w.writerow([d, v, seg_id])


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    dates, values = load_csv_series(args.input, args.date_col, args.value_col)
    if not values:
        raise SystemExit("No valid numeric values found in input")

    logger.info("Loaded %d points from %s", len(values), args.input)

    res = detect_change_points(values, max_cps=args.max_cps, min_segment_size=args.min_segment_size, select=args.select)

    # Convert breakpoint indices to dates (if possible)
    bp_dates: List[str] = []
    for idx in res.breakpoints:
        if 0 <= idx < len(dates):
            bp_dates.append(dates[idx])
        else:
            bp_dates.append("")

    out_obj = {
        "input": {
            "path": str(args.input),
            "date_col": args.date_col,
            "value_col": args.value_col,
            "n_loaded": len(values),
        },
        "params": {
            "max_cps": int(args.max_cps),
            "min_segment_size": int(args.min_segment_size),
            "select": str(args.select),
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

    write_json(out_obj, args.out)

    if args.segments_out:
        write_segments_csv(args.segments_out, dates, values, res.breakpoints)
        logger.info("Wrote segments CSV to %s", args.segments_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())