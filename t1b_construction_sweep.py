#!/usr/bin/env python3
"""T1b construction-ambiguity sweep.

Runs the documented 3 x 2 ambiguity matrix:

- projection-volume reading: width modulation, hard threshold, density weighting
- operator convention: Tikhonov-regularised inverse of G2, normalised graph Laplacian

The extraction rule is intentionally copied from the frozen T1 rule:
same ell grid, fit window, step/sigmoid parameterisations, AIC convention, and
Delta AIC >= 10 plus positive Delta d_s adjudicability test.

This script is deterministic: no wall-clock fields and no RNG are used.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUT = OUTPUT_DIR / "t1b_construction_sweep.json"

R_H = 1.0
R_OUT = 6.0
ETA_80 = 2.606e-2
SIGMA_PI_80 = 0.35
RHO_CRIT_80 = 5.0
N_REF = 80
LABEL_POINTS = 128
TIKHONOV_LAMBDA_REL = 1e-6
SCREEN_LEVELS = [80, 160, 320, 640]
FULL_LEVELS = [80, 160, 320, 640, 1280, 2560]
ELL_GRID = np.logspace(math.log10(0.15), math.log10(6.0), 96)
FIT_WINDOW = (0.35, 4.5)
LOCKED_INTERVAL = (0.64, 0.94)

VOLUME_READINGS = ("width_modulation", "hard_threshold", "density_weighting")
OPERATORS = ("regularized_inverse", "normalized_graph_laplacian")


@dataclass(frozen=True)
class ScalingParams:
    eta: float
    sigma_pi: float
    rho_crit: float


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


def critical_radius(rho_crit: float) -> float | None:
    if rho_crit <= 1.0:
        return None
    return R_H * rho_crit / (rho_crit - 1.0)


def kappa_radius_ratio(rho_crit: float) -> float | None:
    rc = critical_radius(rho_crit)
    if rc is None:
        return None
    return R_H / rc


def gaussian_projection(x: np.ndarray, u: np.ndarray, sigma_eff: np.ndarray, dx: float) -> np.ndarray:
    sigma_floor = 0.05 * dx
    sigma_eff = np.maximum(sigma_eff, sigma_floor)
    z = -0.5 * ((x[:, None] - u[None, :]) / sigma_eff[None, :]) ** 2
    z = z - np.max(z, axis=0, keepdims=True)
    a = np.exp(z)
    col_norm = np.maximum(a.sum(axis=0, keepdims=True) * dx, 1e-300)
    return a / col_norm


def construction_g2(n: int, params: ScalingParams, volume_reading: str) -> np.ndarray:
    x, dx = radial_label_grid()
    u, du = radial_constraint_grid(n)
    profile = rho_ratio(u)
    v_rel = np.exp(-(profile - 1.0))

    sigma_eff = np.full_like(u, params.sigma_pi)
    weights = np.ones_like(u)

    if volume_reading == "width_modulation":
        sigma_eff = params.sigma_pi * v_rel
    elif volume_reading == "hard_threshold":
        rc = critical_radius(params.rho_crit)
        if rc is None:
            weights[:] = 0.0
        else:
            weights = (u >= rc).astype(float)
    elif volume_reading == "density_weighting":
        weights = v_rel
    else:
        raise ValueError(f"unknown volume reading: {volume_reading}")

    pi = gaussian_projection(x, u, sigma_eff, dx)
    diff = u[:, None] - u[None, :]
    kernel = np.exp(-params.eta * diff * diff)
    weighted_pi = pi * weights[None, :]
    g = (weighted_pi @ kernel @ weighted_pi.T) * (du * du)
    return 0.5 * (g + g.T)


def regularized_inverse_spectrum(g: np.ndarray, lambda_factor: float = 1.0) -> np.ndarray:
    eig = np.linalg.eigvalsh(g)
    eig = np.clip(eig, 0.0, None)
    scale = float(np.max(eig))
    if scale <= 0:
        return np.zeros_like(eig)
    return 1.0 / (eig + TIKHONOV_LAMBDA_REL * lambda_factor * scale)


def normalized_graph_laplacian_spectrum(g: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(g, dtype=float), 0.0, None)
    w = 0.5 * (w + w.T)
    np.fill_diagonal(w, 0.0)
    degree = w.sum(axis=1)
    positive = degree > 1e-300
    if not np.any(positive):
        return np.zeros(w.shape[0], dtype=float)
    inv_sqrt = np.zeros_like(degree)
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    norm_adj = inv_sqrt[:, None] * w * inv_sqrt[None, :]
    lap = np.eye(w.shape[0]) - norm_adj
    lap[~positive, ~positive] = 0.0
    eig = np.linalg.eigvalsh(0.5 * (lap + lap.T))
    return np.clip(eig, 0.0, None)


def heat_and_ds_from_spectrum(spec: np.ndarray, operator: str) -> tuple[np.ndarray, np.ndarray]:
    if operator == "regularized_inverse":
        heat = np.exp(-(ELL_GRID[:, None] ** 2) * spec[None, :]).sum(axis=1)
    elif operator == "normalized_graph_laplacian":
        heat = np.exp(-(ELL_GRID[:, None] ** 2) * spec[None, :]).sum(axis=1)
    else:
        raise ValueError(f"unknown operator: {operator}")
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


def sample_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=1))


def spectrum_for(n: int, scaling: str, volume_reading: str, operator: str) -> tuple[np.ndarray, ScalingParams]:
    params = scaling_params(n, scaling)
    g = construction_g2(n, params, volume_reading)
    if operator == "regularized_inverse":
        return regularized_inverse_spectrum(g, 1.0), params
    if operator == "normalized_graph_laplacian":
        return normalized_graph_laplacian_spectrum(g), params
    raise ValueError(f"unknown operator: {operator}")


def run_level(n: int, scaling: str, volume_reading: str, operator: str) -> dict:
    spec, params = spectrum_for(n, scaling, volume_reading, operator)
    heat, ds = heat_and_ds_from_spectrum(spec, operator)
    central = fit_models(ds, window_indices())
    kr = kappa_radius_ratio(params.rho_crit)
    return {
        "N": n,
        "scaling": scaling,
        "volume_reading": volume_reading,
        "operator": operator,
        "parameters": {
            "eta": params.eta,
            "sigma_Pi": params.sigma_pi,
            "rho_crit_over_rho0": params.rho_crit,
        },
        "central": {
            "ell_star": central["step"]["ell_star"],
            "kappa_diffusion": central["step"]["ell_star"] / R_H,
            "kappa_radius_ratio": kr,
            "delta_ds": central["step"]["delta_ds"],
            "d_pre": central["step"]["d_pre"],
            "d_post": central["step"]["d_post"],
            "delta_aic_step_minus_smooth": central["delta_aic_step_minus_smooth"],
            "adjudicable": central["adjudicable"],
            "step_sse": central["step"]["sse"],
            "smooth_sse": central["sigmoid"]["sse"],
        },
        "ds_range": {
            "min": float(np.min(ds)),
            "max": float(np.max(ds)),
            "fit_window_min": float(np.min(ds[window_indices()])),
            "fit_window_max": float(np.max(ds[window_indices()])),
            "has_value_in_2_to_4": bool(np.any((ds >= 2.0) & (ds <= 4.0))),
        },
        "ds_curve": {
            "ell": [round(float(v), 8) for v in ELL_GRID],
            "heat_trace": [float(v) for v in heat],
            "d_s": [float(v) for v in ds],
        },
    }


def nuisance_fits(n: int, scaling: str, volume_reading: str, operator: str) -> list[dict]:
    params = scaling_params(n, scaling)
    g = construction_g2(n, params, volume_reading)
    rows = []
    for lo_shift in (-1, 0, 1):
        for hi_shift in (-1, 0, 1):
            indices = window_indices(lo_shift, hi_shift)
            lambda_factors = (0.5, 1.0, 2.0) if operator == "regularized_inverse" else (1.0,)
            for lambda_factor in lambda_factors:
                if operator == "regularized_inverse":
                    spec = regularized_inverse_spectrum(g, lambda_factor)
                else:
                    spec = normalized_graph_laplacian_spectrum(g)
                _, ds = heat_and_ds_from_spectrum(spec, operator)
                fit = fit_models(ds, indices)
                rows.append(
                    {
                        "lo_shift": lo_shift,
                        "hi_shift": hi_shift,
                        "lambda_factor": lambda_factor,
                        "delta_ds": fit["step"]["delta_ds"],
                        "ell_star": fit["step"]["ell_star"],
                        "kappa_diffusion": fit["step"]["ell_star"] / R_H,
                        "delta_aic": fit["delta_aic_step_minus_smooth"],
                        "adjudicable": fit["adjudicable"],
                    }
                )
    return rows


def run_full_level(n: int, scaling: str, volume_reading: str, operator: str) -> dict:
    row = run_level(n, scaling, volume_reading, operator)
    nuis = nuisance_fits(n, scaling, volume_reading, operator)
    row["uncertainty"] = {
        "ell_star_sigma": sample_std(r["ell_star"] for r in nuis),
        "kappa_diffusion_sigma": sample_std(r["kappa_diffusion"] for r in nuis),
        "delta_ds_sigma": sample_std(r["delta_ds"] for r in nuis),
        "nuisance_runs": len(nuis),
        "nuisance_adjudicable_count": int(sum(1 for r in nuis if r["adjudicable"])),
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
    last_sigma = float(accepted[-1].get("uncertainty", {}).get(sigma_key, 0.0))
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


def screen_cell(rows: list[dict]) -> dict:
    reaches_range = any(r["ds_range"]["has_value_in_2_to_4"] for r in rows)
    positive_step_levels = [
        r["N"]
        for r in rows
        if r["central"]["delta_ds"] > 0.0 and r["central"]["delta_aic_step_minus_smooth"] >= 10.0
    ]
    plausible = reaches_range and len(positive_step_levels) >= 2
    reasons = []
    if not reaches_range:
        reasons.append("d_s curve never reaches the 2-4 historical range")
    if len(positive_step_levels) < 2:
        reasons.append("positive step is not AIC-preferred at at least two levels")
    return {
        "historically_plausible": plausible,
        "reaches_ds_2_to_4_range": reaches_range,
        "positive_step_aic_preferred_levels": positive_step_levels,
        "reason": "; ".join(reasons) if reasons else "passes screening criterion",
    }


def run_screen() -> tuple[list[dict], list[dict]]:
    cells = []
    plausible = []
    for volume_reading in VOLUME_READINGS:
        for operator in OPERATORS:
            rows = [
                run_level(n, "fixed_physical", volume_reading, operator)
                for n in SCREEN_LEVELS
            ]
            screen = screen_cell(rows)
            cell = {
                "volume_reading": volume_reading,
                "operator": operator,
                "levels": rows,
                "screening": screen,
            }
            cells.append(cell)
            if screen["historically_plausible"]:
                plausible.append(cell)
    return cells, plausible


def run_promoted(cell: dict) -> dict | None:
    if cell is None:
        return None
    volume_reading = cell["volume_reading"]
    operator = cell["operator"]
    out = {}
    for scaling in ("fixed_physical", "v56_scaling"):
        rows = [
            run_full_level(n, scaling, volume_reading, operator)
            for n in FULL_LEVELS
        ]
        deltas = extrapolate_quantity(rows, "delta_ds", "delta_ds_sigma")
        kappas = extrapolate_quantity(rows, "kappa_diffusion", "kappa_diffusion_sigma")
        out[scaling] = {
            "levels": rows,
            "beta": beta_rows(rows),
            "delta_ds_extrapolation": deltas,
            "kappa_diffusion_extrapolation": kappas,
            "decision": classify_scaling(rows, deltas),
        }
    return out


def run() -> dict:
    cells, plausible = run_screen()
    promoted = None
    if len(plausible) == 1:
        promoted = run_promoted(plausible[0])
    if len(plausible) == 0:
        conclusion = {
            "branch": "no_documented_construction_reproduces_claimed_regime",
            "reason": "no T1b cell reached the d_s 2-4 range and positive AIC-preferred step criterion",
        }
    elif len(plausible) == 1:
        conclusion = {
            "branch": "single_historically_plausible_cell_promoted",
            "volume_reading": plausible[0]["volume_reading"],
            "operator": plausible[0]["operator"],
        }
    else:
        conclusion = {
            "branch": "multiple_historically_plausible_cells",
            "cells": [
                {"volume_reading": c["volume_reading"], "operator": c["operator"]}
                for c in plausible
            ],
        }
    result = {
        "purpose": "T1b construction-ambiguity sweep for the T1/Delta d_s adjudication.",
        "determinism": {
            "wall_clock_fields_in_hashed_json": False,
            "rng_used": False,
        },
        "screening_criterion": (
            "Historically plausible iff d_s values reach the 2-4 range and a positive "
            "step is AIC-preferred at two or more screen levels."
        ),
        "screen_levels": SCREEN_LEVELS,
        "screen_cells": cells,
        "plausible_cells": [
            {"volume_reading": c["volume_reading"], "operator": c["operator"]}
            for c in plausible
        ],
        "promoted_full_protocol": promoted,
        "conclusion": conclusion,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = run()
    print(json.dumps(result["conclusion"], indent=2, sort_keys=True))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
