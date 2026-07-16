#!/usr/bin/env python3
"""NP-1 spin-dependent ringdown law and stacked-regression harness."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

GWOSC_TEMPLATE_COLUMNS = [
    "event",
    "final_mass_source_msun",
    "final_mass_detector_msun",
    "final_spin",
    "redshift",
    "network_snr",
    "source_url",
    "catalog_version",
    "preferred_pe_record",
]


def spin_factor(a: float) -> float:
    if not 0.0 <= a < 1.0:
        raise ValueError("final spin must satisfy 0 <= a_* < 1")
    return 1.0 + math.sqrt(1.0 - a * a)


def tc_over_m(kappa: float, a: float) -> float:
    return kappa * spin_factor(a)


def weighted_origin_fit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = []
    controls_fail = []
    for row in rows:
        event = row.get("event", "")
        a = float(row["final_spin"])
        y = float(row["tc_over_M"])
        sigma = float(row["sigma_tc_over_M"])
        injection_pass = str(row.get("injection_pass", "true")).lower() in {"true", "1", "yes"}
        null_pass = str(row.get("null_pass", "true")).lower() in {"true", "1", "yes"}
        if sigma <= 0:
            raise ValueError(f"non-positive sigma for event {event}")
        if not injection_pass or not null_pass:
            controls_fail.append(event)
            continue
        usable.append({"event": event, "x": spin_factor(a), "y": y, "sigma": sigma})

    if not usable:
        return {
            "status": "no_usable_events",
            "controls_failed_events": controls_fail,
            "kappa_hat": None,
            "kappa_sigma": None,
            "kappa_ci95": None,
        }

    swxy = sum((r["x"] * r["y"]) / (r["sigma"] ** 2) for r in usable)
    swxx = sum((r["x"] * r["x"]) / (r["sigma"] ** 2) for r in usable)
    k_hat = swxy / swxx
    k_sig = math.sqrt(1.0 / swxx)
    ci95 = [k_hat - 1.96 * k_sig, k_hat + 1.96 * k_sig]
    excludes_locked = ci95[1] < 0.75 or ci95[0] > 0.85
    return {
        "status": "fit_completed",
        "n_usable": len(usable),
        "controls_failed_events": controls_fail,
        "model": "tc_over_M = kappa * (1 + sqrt(1-a_*^2)); weighted fit through origin",
        "kappa_hat": k_hat,
        "kappa_sigma": k_sig,
        "kappa_ci95": ci95,
        "locked_acceptance_interval": [0.75, 0.85],
        "excludes_locked_interval_at_95pct": excludes_locked,
    }


def load_csv_rows(path: str) -> tuple[list[dict[str, Any]], list[str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def load_events(path: str) -> list[dict[str, Any]]:
    rows, fieldnames = load_csv_rows(path)
    required = {"final_spin", "tc_over_M", "sigma_tc_over_M"}
    missing = required.difference(fieldnames)
    if missing:
        raise KeyError(f"events CSV missing columns: {sorted(missing)}")
    return rows


def _float_field(row: dict[str, Any], column: str, event: str) -> float:
    try:
        return float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value for {column} in event {event}") from exc


def validate_event_template(path: str) -> dict[str, Any]:
    rows, fieldnames = load_csv_rows(path)
    missing = [col for col in GWOSC_TEMPLATE_COLUMNS if col not in fieldnames]
    if missing:
        return {
            "status": "template_validation_failed",
            "errors": [f"events CSV missing columns: {missing}"],
        }

    errors = []
    events_checked = []
    seen_events = set()
    max_mass_error = 0.0
    tolerance = 5e-4

    for index, row in enumerate(rows, start=2):
        event = row.get("event", "").strip()
        if not event:
            errors.append(f"line {index}: missing event name")
            continue
        if event in seen_events:
            errors.append(f"line {index}: duplicate event {event}")
        seen_events.add(event)

        try:
            final_mass_source = _float_field(row, "final_mass_source_msun", event)
            final_mass_detector = _float_field(row, "final_mass_detector_msun", event)
            final_spin = _float_field(row, "final_spin", event)
            redshift = _float_field(row, "redshift", event)
            network_snr = _float_field(row, "network_snr", event)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        computed_detector_mass = final_mass_source * (1.0 + redshift)
        detector_mass_error = abs(final_mass_detector - computed_detector_mass)
        max_mass_error = max(max_mass_error, detector_mass_error)

        if final_mass_source <= 0:
            errors.append(f"line {index}: final_mass_source_msun must be positive for {event}")
        if final_mass_detector <= 0:
            errors.append(f"line {index}: final_mass_detector_msun must be positive for {event}")
        if not 0.0 <= final_spin < 1.0:
            errors.append(f"line {index}: final_spin must satisfy 0 <= a_* < 1 for {event}")
        if redshift < 0:
            errors.append(f"line {index}: redshift must be non-negative for {event}")
        if network_snr <= 0:
            errors.append(f"line {index}: network_snr must be positive for {event}")
        if detector_mass_error > tolerance:
            errors.append(
                f"line {index}: detector mass mismatch for {event}: "
                f"{detector_mass_error:.6g} M_sun"
            )
        if not row["source_url"].startswith("https://gwosc.org/eventapi/json/"):
            errors.append(f"line {index}: source_url is not a GWOSC event API URL for {event}")
        if not row["catalog_version"].strip():
            errors.append(f"line {index}: missing catalog_version for {event}")
        if not row["preferred_pe_record"].strip():
            errors.append(f"line {index}: missing preferred_pe_record for {event}")

        events_checked.append(
            {
                "event": event,
                "catalog_version": row["catalog_version"],
                "final_mass_source_msun": final_mass_source,
                "final_mass_detector_msun": final_mass_detector,
                "computed_detector_mass_msun": computed_detector_mass,
                "detector_mass_abs_error_msun": detector_mass_error,
                "final_spin": final_spin,
                "redshift": redshift,
                "network_snr": network_snr,
            }
        )

    return {
        "status": "template_validation_failed" if errors else "template_validation_completed",
        "mode": "official_gwosc_table_column_validation_only",
        "n_rows": len(rows),
        "required_columns": GWOSC_TEMPLATE_COLUMNS,
        "mass_detector_formula": "final_mass_detector_msun = final_mass_source_msun * (1 + redshift)",
        "mass_detector_tolerance_msun": tolerance,
        "max_mass_detector_abs_error_msun": max_mass_error,
        "events_checked": events_checked,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/np1_spin_ringdown_regression.json")
    parser.add_argument("--events-csv", default=None)
    parser.add_argument(
        "--validate-template",
        action="store_true",
        help="Validate a populated GWOSC event table without running the NP-1 regression.",
    )
    parser.add_argument("--kappa", type=float, default=0.80)
    parser.add_argument("--kappa-sigma", type=float, default=0.05)
    args = parser.parse_args()

    spins = [0.0, 0.5, 0.67, 0.69, 0.9]
    predictions = [
        {
            "final_spin": a,
            "x_spin_factor": spin_factor(a),
            "tc_over_M": tc_over_m(args.kappa, a),
        }
        for a in spins
    ]

    regression = {
        "status": "template_only_no_events_csv",
        "required_event_columns": [
            "event",
            "final_spin",
            "tc_over_M",
            "sigma_tc_over_M",
            "injection_pass",
            "null_pass",
        ],
        "workflow": (
            "Run the event-level change-point analysis on each candidate event; "
            "convert each accepted change-point time to tc_over_M using the "
            "event's redshifted remnant mass; then fit tc_over_M against "
            "1+sqrt(1-a_*^2)."
        ),
    }
    if args.validate_template:
        if not args.events_csv:
            raise ValueError("--validate-template requires --events-csv")
        regression = validate_event_template(args.events_csv)
    elif args.events_csv:
        regression = weighted_origin_fit(load_events(args.events_csv))

    payload = {
        "purpose": "NP-1 spin-dependent ringdown law and stacked GWTC regression harness.",
        "formula": "tc/M = kappa * (1 + sqrt(1-a_*^2))",
        "locked_inputs": {
            "kappa": args.kappa,
            "kappa_sigma": args.kappa_sigma,
            "spin_factor_origin": "Kerr outer-horizon radius in units of M",
        },
        "predictions": predictions,
        "falsification_rule": (
            "A stacked posterior/regression excludes kappa in [0.75,0.85] at "
            "95% while event-level injection recovery passes and off-source or "
            "time-slide controls remain null."
        ),
        "regression_harness": regression,
        "conclusion": (
            "The spin law is algebraically fixed once kappa is supplied by the "
            "locked instantiation; public-data adjudication requires event-level "
            "change-point measurements and controls."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "spins_evaluated": spins}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
