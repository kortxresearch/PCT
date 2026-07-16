#!/usr/bin/env python3
"""T1 step-height adjudication for the re-fixed horizon-profile construction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
SPEC_PATH = ROOT / "t1_horizon_profile_spec.md"
RULE_PATH = ROOT / "t1_extraction_rule.md"
SCRIPT_PATH = ROOT / "t1_step_height_adjudication.py"
FREEZE_MANIFEST_PATH = OUTPUT_DIR / "t1_freeze_manifest.json"
LADDER_OUTPUT_PATH = OUTPUT_DIR / "t1_step_height_adjudication.json"

R_H = 1.0
R_OUT = 6.0
ETA_80 = 2.606e-2
SIGMA_PI_80 = 0.35
RHO_CRIT_80 = 5.0
N_REF = 80
LABEL_POINTS = 128
TIKHONOV_LAMBDA_REL = 1e-6
LEVELS = [80, 160, 320, 640, 1280, 2560]
ELL_GRID = np.logspace(math.log10(0.15), math.log10(6.0), 96)
FIT_WINDOW = (0.35, 4.5)
LOCKED_INTERVAL = (0.64, 0.94)


@dataclass(frozen=True)
class ScalingParams:
    eta: float
    sigma_pi: float
    rho_crit: float


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def current_freeze_hashes() -> dict[str, str]:
    return {
        "t1_horizon_profile_spec.md": sha256(SPEC_PATH),
        "t1_extraction_rule.md": sha256(RULE_PATH),
        "t1_step_height_adjudication.py": sha256(SCRIPT_PATH),
    }


def write_freeze_manifest(force: bool = False) -> dict:
    if FREEZE_MANIFEST_PATH.exists() and not force:
        return json.loads(FREEZE_MANIFEST_PATH.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "Freeze T1 spec, extraction rule, and executable before the refinement ladder.",
        "created_utc": utc_now(),
        "hashes": current_freeze_hashes(),
        "ladder_output_existed_at_freeze": LADDER_OUTPUT_PATH.exists(),
        "provenance_label": "re-fixed construction, not recovered historical construction",
    }
    FREEZE_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_and_check_freeze_manifest() -> dict:
    if not FREEZE_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing freeze manifest: {FREEZE_MANIFEST_PATH}. "
            "Run with --freeze-only before the ladder."
        )
    manifest = json.loads(FREEZE_MANIFEST_PATH.read_text(encoding="utf-8"))
    current = current_freeze_hashes()
    frozen = manifest["hashes"]
    mismatches = {
        key: {"frozen": frozen.get(key), "current": current.get(key)}
        for key in sorted(current)
        if frozen.get(key) != current.get(key)
    }
    if mismatches:
        raise RuntimeError(f"Frozen artifact hash mismatch: {mismatches}")
    return manifest


def scaling_params(n: int, prescription: str) -> ScalingParams:
    if prescription == "fixed_physical":
        return ScalingParams(ETA_80, SIGMA_PI_80, RHO_CRIT_80)
    if prescription == "v56_scaling":
        return ScalingParams(
            eta=ETA_80 * N_REF / n,
            sigma_pi=SIGMA_PI_80 * math.sqrt(N_REF / n),
            rho_crit=RHO_CRIT_80 * N_REF / n,
        )
    raise ValueError(f"unknown scaling prescription: {prescription}")


def radial_constraint_grid(n: int) -> tuple[np.ndarray, float]:
    du = (R_OUT - R_H) / n
    r = R_H + (np.arange(n, dtype=float) + 0.5) * du
    return r, du


def radial_label_grid() -> tuple[np.ndarray, float]:
    dx = (R_OUT - R_H) / LABEL_POINTS
    x = R_H + (np.arange(LABEL_POINTS, dtype=float) + 0.5) * dx
    return x, dx


def rho_ratio(r: np.ndarray) -> np.ndarray:
    return 1.0 / np.maximum(1.0 - R_H / r, 1e-300)


def projection_matrix(x: np.ndarray, u: np.ndarray, sigma_pi: float, dx: float) -> np.ndarray:
    profile = rho_ratio(u)
    v_rel = np.exp(-(profile - 1.0))
    sigma_floor = 0.05 * dx
    sigma_eff = np.maximum(sigma_pi * v_rel, sigma_floor)
    z = -0.5 * ((x[:, None] - u[None, :]) / sigma_eff[None, :]) ** 2
    z = z - np.max(z, axis=0, keepdims=True)
    a = np.exp(z)
    col_norm = np.maximum(a.sum(axis=0, keepdims=True) * dx, 1e-300)
    return a / col_norm


def induced_g2(n: int, params: ScalingParams) -> np.ndarray:
    x, dx = radial_label_grid()
    u, du = radial_constraint_grid(n)
    pi = projection_matrix(x, u, params.sigma_pi, dx)
    diff = u[:, None] - u[None, :]
    kernel = np.exp(-params.eta * diff * diff)
    g = (pi @ kernel @ pi.T) * (du * du)
    return 0.5 * (g + g.T)


def inverse_spectrum_from_g2(g: np.ndarray, lambda_factor: float) -> np.ndarray:
    eig = np.linalg.eigvalsh(g)
    eig = np.clip(eig, 0.0, None)
    scale = float(np.max(eig))
    if scale <= 0:
        raise ValueError("G2 has no positive eigenvalue scale")
    lam_rel = TIKHONOV_LAMBDA_REL * lambda_factor
    return 1.0 / (eig + lam_rel * scale)


def ds_curve_from_inverse_spectrum(inv_spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    heat = np.exp(-(ELL_GRID[:, None] ** 2) * inv_spec[None, :]).sum(axis=1)
    x = np.log(ELL_GRID * ELL_GRID)
    y = np.log(np.clip(heat, 1e-300, None))
    ds = -2.0 * np.gradient(y, x)
    return heat, ds


def aic(sse: float, n: int, k: int) -> float:
    return n * math.log(max(sse, 1e-300) / n) + 2.0 * k


def window_indices(lo_shift: int = 0, hi_shift: int = 0) -> np.ndarray:
    lo, hi = FIT_WINDOW
    idx = np.flatnonzero((ELL_GRID >= lo) & (ELL_GRID <= hi))
    lo_i = max(0, idx[0] + lo_shift)
    hi_i = min(len(ELL_GRID) - 1, idx[-1] + hi_shift)
    if hi_i - lo_i + 1 < 12:
        raise ValueError("fit window has too few points")
    return np.arange(lo_i, hi_i + 1)


def fit_step(ds: np.ndarray, indices: np.ndarray) -> dict:
    x = np.log(ELL_GRID[indices])
    y = ds[indices]
    best: dict | None = None
    n = len(y)
    for split in range(4, n - 4):
        left = y[:split]
        right = y[split:]
        d_pre = float(left.mean())
        d_post = float(right.mean())
        pred = np.r_[np.full(split, d_pre), np.full(n - split, d_post)]
        sse = float(np.sum((y - pred) ** 2))
        x0 = 0.5 * (x[split - 1] + x[split])
        row = {
            "sse": sse,
            "aic": aic(sse, n, 3),
            "d_pre": d_pre,
            "d_post": d_post,
            "delta_ds": d_pre - d_post,
            "ell_star": math.exp(float(x0)),
            "split_index_in_window": split,
        }
        if best is None or row["sse"] < best["sse"]:
            best = row
    assert best is not None
    return best


def fit_sigmoid(ds: np.ndarray, indices: np.ndarray) -> dict:
    x = np.log(ELL_GRID[indices])
    y = ds[indices]
    n = len(y)
    widths = np.logspace(math.log10(0.08), math.log10(1.2), 24)
    best: dict | None = None
    for x0 in x[4:-4]:
        for w in widths:
            basis = 1.0 / (1.0 + np.exp((x - x0) / w))
            design = np.column_stack([np.ones_like(basis), basis])
            coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
            pred = design @ coeff
            sse = float(np.sum((y - pred) ** 2))
            row = {
                "sse": sse,
                "aic": aic(sse, n, 4),
                "d_post": float(coeff[0]),
                "delta_ds": float(coeff[1]),
                "ell_star": math.exp(float(x0)),
                "width_log_ell": float(w),
            }
            if best is None or row["sse"] < best["sse"]:
                best = row
    assert best is not None
    return best


def fit_models(ds: np.ndarray, indices: np.ndarray) -> dict:
    step = fit_step(ds, indices)
    sigmoid = fit_sigmoid(ds, indices)
    delta_aic = sigmoid["aic"] - step["aic"]
    return {
        "step": step,
        "sigmoid": sigmoid,
        "delta_aic_step_minus_smooth": delta_aic,
        "adjudicable": bool(delta_aic >= 10.0 and step["delta_ds"] > 0.0),
    }


def nuisance_fits(g: np.ndarray) -> list[dict]:
    rows = []
    for lo_shift in (-1, 0, 1):
        for hi_shift in (-1, 0, 1):
            indices = window_indices(lo_shift, hi_shift)
            for lambda_factor in (0.5, 1.0, 2.0):
                inv = inverse_spectrum_from_g2(g, lambda_factor)
                _, ds = ds_curve_from_inverse_spectrum(inv)
                fit = fit_models(ds, indices)
                rows.append(
                    {
                        "lo_shift": lo_shift,
                        "hi_shift": hi_shift,
                        "lambda_factor": lambda_factor,
                        "delta_ds": fit["step"]["delta_ds"],
                        "ell_star": fit["step"]["ell_star"],
                        "kappa": fit["step"]["ell_star"] / R_H,
                        "delta_aic": fit["delta_aic_step_minus_smooth"],
                        "adjudicable": fit["adjudicable"],
                    }
                )
    return rows


def sample_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=1))


def run_level(n: int, prescription: str) -> dict:
    params = scaling_params(n, prescription)
    g = induced_g2(n, params)
    inv = inverse_spectrum_from_g2(g, 1.0)
    heat, ds = ds_curve_from_inverse_spectrum(inv)
    central = fit_models(ds, window_indices())
    nuis = nuisance_fits(g)
    row = {
        "N": n,
        "scaling": prescription,
        "parameters": {
            "eta": params.eta,
            "sigma_Pi": params.sigma_pi,
            "rho_crit_over_rho0": params.rho_crit,
        },
        "central": {
            "ell_star": central["step"]["ell_star"],
            "kappa": central["step"]["ell_star"] / R_H,
            "delta_ds": central["step"]["delta_ds"],
            "d_pre": central["step"]["d_pre"],
            "d_post": central["step"]["d_post"],
            "delta_aic_step_minus_smooth": central["delta_aic_step_minus_smooth"],
            "adjudicable": central["adjudicable"],
            "step_sse": central["step"]["sse"],
            "smooth_sse": central["sigmoid"]["sse"],
            "smooth_width_log_ell": central["sigmoid"]["width_log_ell"],
        },
        "uncertainty": {
            "ell_star_sigma": sample_std(r["ell_star"] for r in nuis),
            "kappa_sigma": sample_std(r["kappa"] for r in nuis),
            "delta_ds_sigma": sample_std(r["delta_ds"] for r in nuis),
            "nuisance_runs": len(nuis),
            "nuisance_adjudicable_count": int(sum(1 for r in nuis if r["adjudicable"])),
        },
        "ds_curve": {
            "ell": [round(float(v), 8) for v in ELL_GRID],
            "heat_trace": [float(v) for v in heat],
            "d_s": [float(v) for v in ds],
        },
    }
    return row


def beta_rows(level_rows: list[dict]) -> list[dict]:
    out = []
    for a, b in zip(level_rows[:-1], level_rows[1:]):
        if a["central"]["adjudicable"] and b["central"]["adjudicable"]:
            beta = (
                b["central"]["delta_ds"] - a["central"]["delta_ds"]
            ) / math.log(b["N"] / a["N"])
            out.append({"from_N": a["N"], "to_N": b["N"], "beta_hat": beta})
        else:
            out.append(
                {
                    "from_N": a["N"],
                    "to_N": b["N"],
                    "beta_hat": None,
                    "reason": "one or both levels not adjudicable",
                }
            )
    return out


def extrapolate_quantity(level_rows: list[dict], key: str, sigma_key: str) -> dict:
    accepted = [r for r in level_rows if r["central"]["adjudicable"]]
    if len(accepted) < 3:
        return {"available": False, "reason": "fewer than three adjudicable levels"}
    n = np.asarray([r["N"] for r in accepted], dtype=float)
    y = np.asarray([r["central"][key] for r in accepted], dtype=float)
    best: dict | None = None
    for p in np.linspace(0.25, 3.0, 112):
        x = n ** (-p)
        design = np.column_stack([np.ones_like(x), x])
        coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ coeff
        sse = float(np.sum((y - pred) ** 2))
        row = {"p": float(p), "limit": float(coeff[0]), "amplitude": float(coeff[1]), "sse": sse}
        if best is None or row["sse"] < best["sse"]:
            best = row
    assert best is not None
    residual_sigma = math.sqrt(best["sse"] / max(len(y) - 2, 1))
    last_sigma = float(accepted[-1]["uncertainty"][sigma_key])
    err = max(residual_sigma, abs(float(y[-1]) - best["limit"]), last_sigma)
    return {
        "available": True,
        "fit_model": "q(N)=q_inf+a*N^-p, p scanned over [0.25,3.0]",
        "limit": best["limit"],
        "sigma": err,
        "p": best["p"],
        "sse": best["sse"],
        "adjudicable_levels": [int(v) for v in n],
    }


def classify_scaling(level_rows: list[dict], deltas_extrap: dict) -> dict:
    accepted = [r for r in level_rows if r["central"]["adjudicable"]]
    betas = [r["beta_hat"] for r in beta_rows(level_rows) if r["beta_hat"] is not None]
    if len(accepted) < 4:
        return {
            "branch": "non_adjudicable",
            "reason": "fewer than four levels passed Delta AIC >= 10 and positive Delta d_s",
        }
    beta_to_zero = False
    if len(betas) >= 2:
        beta_to_zero = abs(betas[-1]) < 0.05 and abs(betas[-1]) <= abs(betas[-2])
    if not beta_to_zero:
        return {
            "branch": "beta_not_to_zero",
            "reason": "final beta surrogate did not satisfy the frozen beta-to-zero threshold",
        }
    if not deltas_extrap.get("available"):
        return {"branch": "non_adjudicable", "reason": "extrapolation unavailable"}
    lo, hi = LOCKED_INTERVAL
    limit = deltas_extrap["limit"]
    if lo <= limit <= hi:
        return {
            "branch": "locked_value_stands",
            "reason": "beta-to-zero and extrapolated Delta d_s inside 0.79 +/- 0.15",
        }
    return {
        "branch": "register_value_needs_rescoping",
        "reason": "beta-to-zero but extrapolated Delta d_s outside 0.79 +/- 0.15",
    }


def run_ladder() -> dict:
    freeze_manifest = load_and_check_freeze_manifest()
    prescriptions = ["fixed_physical", "v56_scaling"]
    all_rows: dict[str, list[dict]] = {}
    summaries = {}
    for prescription in prescriptions:
        rows = [run_level(n, prescription) for n in LEVELS]
        all_rows[prescription] = rows
        deltas = extrapolate_quantity(rows, "delta_ds", "delta_ds_sigma")
        kappas = extrapolate_quantity(rows, "kappa", "kappa_sigma")
        summaries[prescription] = {
            "beta": beta_rows(rows),
            "delta_ds_extrapolation": deltas,
            "kappa_extrapolation": kappas,
            "decision": classify_scaling(rows, deltas),
        }
    branches = {k: v["decision"]["branch"] for k, v in summaries.items()}
    if len(set(branches.values())) > 1:
        overall = {
            "branch": "scaling_prescription_dependence",
            "reason": "the two frozen scaling prescriptions landed in different branches",
            "per_scaling": branches,
        }
    else:
        only = next(iter(branches.values()))
        overall = {
            "branch": only,
            "reason": "both scaling prescriptions landed in the same branch",
            "per_scaling": branches,
        }
    result = {
        "purpose": "T1/Delta d_s adjudication for the re-fixed horizon-profile construction.",
        "created_utc": utc_now(),
        "freeze_manifest": freeze_manifest,
        "specification": {
            "provenance_label": "re-fixed construction",
            "levels": LEVELS,
            "label_points_fixed": LABEL_POINTS,
            "ell_grid_count": len(ELL_GRID),
            "fit_window": list(FIT_WINDOW),
            "locked_delta_ds_interval": list(LOCKED_INTERVAL),
            "central_tikhonov_lambda_relative": TIKHONOV_LAMBDA_REL,
        },
        "rows": all_rows,
        "summaries": summaries,
        "overall_decision": overall,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LADDER_OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true", help="write freeze manifest and exit")
    parser.add_argument("--force-freeze", action="store_true", help="overwrite the freeze manifest")
    args = parser.parse_args()

    if args.freeze_only:
        manifest = write_freeze_manifest(force=args.force_freeze)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        print(f"Wrote {FREEZE_MANIFEST_PATH}")
        return 0

    result = run_ladder()
    print(json.dumps(result["overall_decision"], indent=2, sort_keys=True))
    print(f"Wrote {LADDER_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
