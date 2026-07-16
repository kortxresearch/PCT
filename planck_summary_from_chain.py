#!/usr/bin/env python3
"""Recompute Planck running summary from the stored Cobaya chain."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def weighted_quantile(values: list[float], weights: list[float], q: float) -> float:
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total = sum(weights)
    if total <= 0:
        raise ValueError("non-positive total weight")
    target = q * total
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= target:
            return value
    return pairs[-1][0]


def read_chain(path: Path) -> tuple[list[str], list[list[float]]]:
    header: list[str] | None = None
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                stripped = line[1:].strip()
                if stripped.startswith("weight"):
                    header = stripped.split()
                continue
            parts = line.split()
            if parts:
                rows.append([float(x) for x in parts])
    if header is None:
        raise ValueError("could not find chain header")
    return header, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain", default="chains/planck2018_running_rs.1.txt")
    parser.add_argument("--reference", default="outputs/planck2018_running.json")
    parser.add_argument("--out", default="outputs/planck2018_running_summary_check.json")
    args = parser.parse_args()

    header, rows = read_chain(Path(args.chain))
    idx = {name: i for i, name in enumerate(header)}
    for required in ("weight", "nrun", "ns", "H0"):
        if required not in idx:
            raise KeyError(f"missing chain column: {required}")

    def summarize_subset(subset: list[list[float]], column: str) -> dict[str, Any]:
        subset_weights = [r[idx["weight"]] for r in subset]
        vals = [r[idx[column]] for r in subset]
        sw = sum(subset_weights)
        mean = sum(w * v for w, v in zip(subset_weights, vals)) / sw
        var = sum(w * (v - mean) ** 2 for w, v in zip(subset_weights, vals)) / sw
        return {
            "mean": mean,
            "std": math.sqrt(var),
            "ci68": [
                weighted_quantile(vals, subset_weights, 0.16),
                weighted_quantile(vals, subset_weights, 0.84),
            ],
            "ci95": [
                weighted_quantile(vals, subset_weights, 0.025),
                weighted_quantile(vals, subset_weights, 0.975),
            ],
            "weighted_sample_count": sw,
            "kish_effective_sample_size": (sw * sw) / sum(w * w for w in subset_weights),
        }

    all_weights = [r[idx["weight"]] for r in rows]
    half = rows[len(rows) // 2 :]
    half_weights = [r[idx["weight"]] for r in half]

    computed_full = {
        "alpha_s": summarize_subset(rows, "nrun"),
        "n_s": summarize_subset(rows, "ns"),
        "H0": summarize_subset(rows, "H0"),
        "n_rows_raw": len(rows),
        "weighted_sample_count": sum(all_weights),
        "kish_effective_sample_size": (sum(all_weights) ** 2) / sum(w * w for w in all_weights),
    }
    computed_second_half = {
        "alpha_s": summarize_subset(half, "nrun"),
        "n_s": summarize_subset(half, "ns"),
        "H0": summarize_subset(half, "H0"),
        "n_rows_raw": len(half),
        "weighted_sample_count": sum(half_weights),
        "kish_effective_sample_size": (sum(half_weights) ** 2) / sum(w * w for w in half_weights),
    }

    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    ref_alpha = reference["posterior"]["alpha_s"]
    ref_ns = reference["posterior"]["n_s"]
    comparison = {
        "reference_file": args.reference,
        "alpha_s_mean_difference": computed_full["alpha_s"]["mean"] - ref_alpha["mean"],
        "alpha_s_std_difference": computed_full["alpha_s"]["std"] - ref_alpha["std"],
        "n_s_mean_difference": computed_full["n_s"]["mean"] - ref_ns["mean"],
        "matches_manuscript_rounding": (
            round(computed_full["alpha_s"]["mean"], 4) == round(ref_alpha["mean"], 4)
            and round(computed_full["alpha_s"]["std"], 4) == round(ref_alpha["std"], 4)
            and round(computed_full["n_s"]["mean"], 4) == round(ref_ns["mean"], 4)
        ),
    }

    payload = {
        "purpose": "Part 4.2 Planck posterior-summary verification from stored chain.",
        "chain": args.chain,
        "interpretation": (
            "The stored chain file matches the recorded posterior summary when "
            "used as the archived post-burn-in chain.  A diagnostic second-half "
            "split is also reported for stability but is not the source of the "
            "published summary numbers."
        ),
        "computed_from_archived_chain": computed_full,
        "diagnostic_second_half_only": computed_second_half,
        "comparison_to_recorded_json": comparison,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "matches_rounding": comparison["matches_manuscript_rounding"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
