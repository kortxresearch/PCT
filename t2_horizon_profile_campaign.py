#!/usr/bin/env python3
"""T2 fresh 2+1D discontinuous horizon-profile refinement campaign.

The executable, specification, and extraction rule are frozen by hash before
the first declared ladder run. The output contains no wall-clock fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import eigh_tridiagonal


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
SPEC_PATH = ROOT / "t2_horizon_profile_spec.md"
RULE_PATH = ROOT / "t2_extraction_rule.md"
SCRIPT_PATH = ROOT / "t2_horizon_profile_campaign.py"
FREEZE_PATH = OUTPUT_DIR / "t2_freeze_manifest.json"
OUTPUT_PATH = OUTPUT_DIR / "t2_horizon_profile_campaign.json"

R_H = 1.0
R_OUT = 5.0
L_PERP = 4.0
K0 = 2.0 * math.pi / L_PERP
RHO_CRIT_FIXED = 2.0
CRITICAL_CELLS = 3
LEVELS = [12, 16, 20, 24, 28, 32]
PRESCRIPTIONS = ("fixed_physical", "fixed_critical_cells")
ELL_GRID = np.logspace(math.log10(0.30), math.log10(2.00), 112)
FIT_WINDOW = (0.45, 1.35)
BOOTSTRAP_REPLICATES = 256
SEED_BASE = 20260710
DELTA_AIC_THRESHOLD = 10.0
BOOTSTRAP_SUPPORT_THRESHOLD = 0.90


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_freeze_hashes() -> dict[str, str]:
    return {
        "t2_extraction_rule.md": sha256(RULE_PATH),
        "t2_horizon_profile_campaign.py": sha256(SCRIPT_PATH),
        "t2_horizon_profile_spec.md": sha256(SPEC_PATH),
    }


def write_freeze_manifest() -> dict:
    if FREEZE_PATH.exists():
        manifest = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        if manifest.get("hashes") != current_freeze_hashes():
            raise RuntimeError("Existing T2 freeze manifest does not match current files")
        return manifest
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed_base": SEED_BASE,
        "deterministic_payload": True,
        "hashes": current_freeze_hashes(),
        "ladder_output_existed_at_freeze": OUTPUT_PATH.exists(),
        "levels": LEVELS,
        "prescriptions": list(PRESCRIPTIONS),
        "purpose": "Freeze the T2 spec, extraction rule, and executable before the first ladder run.",
        "status": "frozen_before_first_ladder_run",
    }
    FREEZE_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_and_check_freeze() -> dict:
    if not FREEZE_PATH.exists():
        raise FileNotFoundError("Run --freeze-only before the T2 ladder")
    manifest = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if manifest.get("hashes") != current_freeze_hashes():
        raise RuntimeError(
            f"Frozen artifact mismatch: frozen={manifest.get('hashes')}, "
            f"current={current_freeze_hashes()}"
        )
    if manifest.get("ladder_output_existed_at_freeze"):
        raise RuntimeError("Freeze manifest says a ladder output already existed")
    return manifest


def threshold_parameters(n: int, prescription: str) -> dict:
    h = (R_OUT - R_H) / n
    if prescription == "fixed_physical":
        rho_crit = RHO_CRIT_FIXED
        r_crit = R_H * rho_crit / (rho_crit - 1.0)
    elif prescription == "fixed_critical_cells":
        r_crit = R_H + CRITICAL_CELLS * h
        rho_crit = r_crit / (r_crit - R_H)
    else:
        raise ValueError(f"Unknown prescription: {prescription}")
    return {
        "h": h,
        "r_crit": r_crit,
        "rho_crit_over_rho0": rho_crit,
        "profile_kappa_input": R_H / r_crit,
    }


def radial_phase(n: int, prescription: str) -> tuple[np.ndarray, np.ndarray, dict]:
    params = threshold_parameters(n, prescription)
    h = params["h"]
    radius = R_H + (np.arange(n, dtype=float) + 0.5) * h
    rho_ratio = 1.0 / (1.0 - R_H / radius)
    phase = rho_ratio >= params["rho_crit_over_rho0"]
    expected = n // 4 if prescription == "fixed_physical" else CRITICAL_CELLS
    if int(np.count_nonzero(phase)) != expected:
        raise RuntimeError(
            f"Threshold alignment failure at N={n}, {prescription}: "
            f"got {np.count_nonzero(phase)}, expected {expected}"
        )
    return radius, phase, params


def radial_tridiagonal(n: int, h: float) -> tuple[np.ndarray, np.ndarray]:
    diagonal = np.full(n, 2.0 / (h * h), dtype=float)
    diagonal[0] = diagonal[-1] = 1.0 / (h * h)
    off_diagonal = np.full(n - 1, -1.0 / (h * h), dtype=float)
    return diagonal, off_diagonal


def transverse_eigenvalues_1d(n: int, h: float) -> np.ndarray:
    modes = np.arange(n, dtype=float)
    return 4.0 * np.sin(math.pi * modes / n) ** 2 / (h * h)


def exact_fourier_blocks(n: int, prescription: str) -> tuple[np.ndarray, dict]:
    radius, phase, params = radial_phase(n, prescription)
    h = params["h"]
    radial_diag, radial_off = radial_tridiagonal(n, h)
    mu_1d = transverse_eigenvalues_1d(n, h)
    blocks = np.empty((n * n, n), dtype=float)
    block_index = 0
    for mu_y in mu_1d:
        for mu_tau in mu_1d:
            mu = float(mu_y + mu_tau)
            transverse = np.where(phase, (mu * mu) / (K0 * K0), mu)
            eig = eigh_tridiagonal(
                radial_diag + transverse,
                radial_off,
                eigvals_only=True,
                check_finite=False,
            )
            blocks[block_index] = np.clip(eig, 0.0, None)
            block_index += 1
    diagnostics = {
        **params,
        "critical_cell_count": int(np.count_nonzero(phase)),
        "degrees_of_freedom": int(n**3),
        "first_radial_midpoint": float(radius[0]),
        "last_radial_midpoint": float(radius[-1]),
        "spectrum_min": float(np.min(blocks)),
        "spectrum_max": float(np.max(blocks)),
        "transverse_block_count": int(n * n),
    }
    return blocks, diagnostics


def block_heat_moments(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = ELL_GRID * ELL_GRID
    heat = np.empty((blocks.shape[0], len(ELL_GRID)), dtype=float)
    moment = np.empty_like(heat)
    for index, eig in enumerate(blocks):
        weights = np.exp(-times[:, None] * eig[None, :])
        heat[index] = np.sum(weights, axis=1)
        moment[index] = weights @ eig
    return heat, moment


FIT_INDICES = np.flatnonzero(
    (ELL_GRID >= FIT_WINDOW[0]) & (ELL_GRID <= FIT_WINDOW[1])
)
FIT_X = np.log(ELL_GRID[FIT_INDICES])
FIT_SPLITS = np.arange(4, len(FIT_X) - 3, dtype=int)
LOG_ELL_SPACING = float(np.median(np.diff(FIT_X)))
SMOOTH_WIDTHS = np.logspace(
    math.log10(0.5 * LOG_ELL_SPACING), math.log10(0.8), 48
)
SMOOTH_X0 = FIT_X[4:-4]
_smooth_x0_mesh, _smooth_width_mesh = np.meshgrid(
    SMOOTH_X0, SMOOTH_WIDTHS, indexing="ij"
)
SMOOTH_X0_FLAT = _smooth_x0_mesh.ravel()
SMOOTH_WIDTH_FLAT = _smooth_width_mesh.ravel()
SMOOTH_BASIS = 1.0 / (
    1.0
    + np.exp(
        (FIT_X[None, :] - SMOOTH_X0_FLAT[:, None])
        / SMOOTH_WIDTH_FLAT[:, None]
    )
)
SMOOTH_BASIS_MEAN = np.mean(SMOOTH_BASIS, axis=1)
SMOOTH_BASIS_CENTERED = SMOOTH_BASIS - SMOOTH_BASIS_MEAN[:, None]
SMOOTH_BASIS_SS = np.sum(SMOOTH_BASIS_CENTERED**2, axis=1)


def aic_array(sse: np.ndarray, points: int, parameters: int) -> np.ndarray:
    return points * np.log(np.maximum(sse, 1e-300) / points) + 2.0 * parameters


def fit_many(curves: np.ndarray) -> dict[str, np.ndarray]:
    y = np.asarray(curves, dtype=float)[:, FIT_INDICES]
    rows, points = y.shape
    cumulative = np.cumsum(y, axis=1)
    cumulative_sq = np.cumsum(y * y, axis=1)
    total = cumulative[:, -1]
    total_sq = cumulative_sq[:, -1]

    left_n = FIT_SPLITS.astype(float)
    right_n = points - left_n
    left_sum = cumulative[:, FIT_SPLITS - 1]
    left_sq = cumulative_sq[:, FIT_SPLITS - 1]
    right_sum = total[:, None] - left_sum
    right_sq = total_sq[:, None] - left_sq
    step_sse = (
        left_sq
        - left_sum * left_sum / left_n[None, :]
        + right_sq
        - right_sum * right_sum / right_n[None, :]
    )
    step_choice = np.argmin(step_sse, axis=1)
    row_index = np.arange(rows)
    split = FIT_SPLITS[step_choice]
    best_step_sse = np.maximum(step_sse[row_index, step_choice], 0.0)
    d_uv = left_sum[row_index, step_choice] / split
    d_ir = right_sum[row_index, step_choice] / (points - split)
    x_star = 0.5 * (FIT_X[split - 1] + FIT_X[split])

    y_mean = np.mean(y, axis=1)
    y_centered = y - y_mean[:, None]
    y_ss = np.sum(y_centered * y_centered, axis=1)
    covariance = y_centered @ SMOOTH_BASIS_CENTERED.T
    smooth_sse = np.maximum(
        y_ss[:, None] - covariance * covariance / SMOOTH_BASIS_SS[None, :],
        0.0,
    )
    smooth_choice = np.argmin(smooth_sse, axis=1)
    best_smooth_sse = smooth_sse[row_index, smooth_choice]
    smooth_delta = (
        covariance[row_index, smooth_choice] / SMOOTH_BASIS_SS[smooth_choice]
    )
    smooth_post = y_mean - smooth_delta * SMOOTH_BASIS_MEAN[smooth_choice]

    step_aic = aic_array(best_step_sse, points, 3)
    smooth_aic = aic_array(best_smooth_sse, points, 4)
    return {
        "d_uv": d_uv,
        "d_ir": d_ir,
        "delta_ds": d_uv - d_ir,
        "ell_star": np.exp(x_star),
        "kappa": np.exp(x_star) / R_H,
        "step_sse": best_step_sse,
        "step_aic": step_aic,
        "smooth_sse": best_smooth_sse,
        "smooth_aic": smooth_aic,
        "smooth_delta_ds": smooth_delta,
        "smooth_d_ir": smooth_post,
        "smooth_ell_star": np.exp(SMOOTH_X0_FLAT[smooth_choice]),
        "smooth_width_log_ell": SMOOTH_WIDTH_FLAT[smooth_choice],
        "delta_aic_step_minus_smooth": smooth_aic - step_aic,
    }


def ds_from_heat_moment(heat: np.ndarray, moment: np.ndarray) -> np.ndarray:
    times = ELL_GRID * ELL_GRID
    return 2.0 * times[None, :] * moment / np.maximum(heat, 1e-300)


def scalar_metric(metrics: dict[str, np.ndarray], key: str) -> float:
    return float(np.asarray(metrics[key]).reshape(-1)[0])


def bootstrap_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "ci95": [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ],
        "sigma": float(np.std(values, ddof=1)),
    }


def run_level(n: int, prescription: str) -> dict:
    blocks, construction = exact_fourier_blocks(n, prescription)
    block_heat, block_moment = block_heat_moments(blocks)
    exact_heat = np.sum(block_heat, axis=0)
    exact_moment = np.sum(block_moment, axis=0)
    exact_ds = ds_from_heat_moment(exact_heat[None, :], exact_moment[None, :])
    central_fit = fit_many(exact_ds)

    seed_offset = 0 if prescription == "fixed_physical" else 1_000_000
    rng = np.random.Generator(np.random.PCG64(SEED_BASE + seed_offset + n))
    block_count = n * n
    counts = rng.multinomial(
        block_count,
        np.full(block_count, 1.0 / block_count),
        size=BOOTSTRAP_REPLICATES,
    )
    bootstrap_heat = counts @ block_heat
    bootstrap_moment = counts @ block_moment
    bootstrap_ds = ds_from_heat_moment(bootstrap_heat, bootstrap_moment)
    bootstrap_fit = fit_many(bootstrap_ds)

    central_delta = scalar_metric(central_fit, "delta_ds")
    central_delta_aic = scalar_metric(
        central_fit, "delta_aic_step_minus_smooth"
    )
    bootstrap_pass = (
        (bootstrap_fit["delta_ds"] > 0.0)
        & (
            bootstrap_fit["delta_aic_step_minus_smooth"]
            >= DELTA_AIC_THRESHOLD
        )
    )
    support = float(np.mean(bootstrap_pass))
    delta_ci = bootstrap_summary(bootstrap_fit["delta_ds"])["ci95"]
    adjudicable = bool(
        central_delta > 0.0
        and central_delta_aic >= DELTA_AIC_THRESHOLD
        and delta_ci[0] > 0.0
        and support >= BOOTSTRAP_SUPPORT_THRESHOLD
    )

    bootstrap = {
        "cluster_unit": "complete transverse Fourier (p,q) radial block",
        "delta_aic_step_minus_smooth": bootstrap_summary(
            bootstrap_fit["delta_aic_step_minus_smooth"]
        ),
        "delta_ds": bootstrap_summary(bootstrap_fit["delta_ds"]),
        "ell_star": bootstrap_summary(bootstrap_fit["ell_star"]),
        "kappa": bootstrap_summary(bootstrap_fit["kappa"]),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": SEED_BASE + seed_offset + n,
        "smooth_width_log_ell": bootstrap_summary(
            bootstrap_fit["smooth_width_log_ell"]
        ),
        "step_support_fraction": support,
    }
    central = {
        key: scalar_metric(central_fit, key)
        for key in (
            "d_uv",
            "d_ir",
            "delta_ds",
            "ell_star",
            "kappa",
            "step_sse",
            "step_aic",
            "smooth_sse",
            "smooth_aic",
            "smooth_delta_ds",
            "smooth_d_ir",
            "smooth_ell_star",
            "smooth_width_log_ell",
            "delta_aic_step_minus_smooth",
        )
    }
    central["smooth_width_in_grid_intervals"] = (
        central["smooth_width_log_ell"] / LOG_ELL_SPACING
    )
    central["step_adjudicable"] = adjudicable
    return {
        "N": n,
        "bootstrap": bootstrap,
        "central": central,
        "construction": construction,
        "curve": {
            "d_s": [float(value) for value in exact_ds[0]],
            "ell": [float(value) for value in ELL_GRID],
            "heat_trace": [float(value) for value in exact_heat],
        },
        "scaling": prescription,
    }


def beta_rows(rows: list[dict]) -> list[dict]:
    result = []
    for first, second in zip(rows[:-1], rows[1:]):
        beta = (
            second["central"]["delta_ds"] - first["central"]["delta_ds"]
        ) / math.log(second["N"] / first["N"])
        result.append(
            {"beta_hat": beta, "from_N": first["N"], "to_N": second["N"]}
        )
    return result


def continuum_extrapolation(rows: list[dict], key: str, bootstrap_key: str) -> dict:
    n_values = np.asarray([row["N"] for row in rows], dtype=float)
    y_values = np.asarray([row["central"][key] for row in rows], dtype=float)
    best = None
    for p in np.linspace(0.5, 3.0, 251):
        x = n_values ** (-p)
        design = np.column_stack([np.ones_like(x), x])
        coefficients, *_ = np.linalg.lstsq(design, y_values, rcond=None)
        residual = y_values - design @ coefficients
        sse = float(residual @ residual)
        candidate = {
            "amplitude": float(coefficients[1]),
            "limit": float(coefficients[0]),
            "p": float(p),
            "sse": sse,
        }
        if best is None or candidate["sse"] < best["sse"]:
            best = candidate
    assert best is not None
    residual_sigma = math.sqrt(best["sse"] / max(len(rows) - 2, 1))
    last_bootstrap_sigma = float(rows[-1]["bootstrap"][bootstrap_key]["sigma"])
    last_drift = abs(float(y_values[-1]) - best["limit"])
    best["sigma"] = max(residual_sigma, last_bootstrap_sigma, last_drift)
    best["uncertainty_rule"] = (
        "max(residual scale, final bootstrap sigma, abs(last-limit))"
    )
    best["fit_model"] = "q(N)=q_inf+a*N^-p; p scan [0.5,3.0]"
    return best


def classify_scaling(rows: list[dict], extrapolations: dict) -> dict:
    adjudicable_final_four = all(
        row["central"]["step_adjudicable"] for row in rows[-4:]
    )
    betas = beta_rows(rows)
    penultimate_beta = abs(betas[-2]["beta_hat"])
    last_beta = abs(betas[-1]["beta_hat"])
    beta_stable = bool(
        penultimate_beta < 0.10
        and last_beta < 0.10
        and last_beta <= penultimate_beta
    )
    previous_ell = rows[-2]["central"]["ell_star"]
    last_ell = rows[-1]["central"]["ell_star"]
    ell_fractional_drift = abs(last_ell - previous_ell) / abs(previous_ell)
    ell_stable = bool(ell_fractional_drift < 0.05)
    sharp = all(
        row["central"]["smooth_width_in_grid_intervals"] <= 1.5
        for row in rows[-2:]
    )
    delta_ext = extrapolations["delta_ds"]
    extrapolated_positive_95 = bool(
        delta_ext["limit"] - 1.96 * delta_ext["sigma"] > 0.0
    )
    criteria = {
        "final_four_levels_step_adjudicable": adjudicable_final_four,
        "final_two_beta_below_0p10_and_nonincreasing": beta_stable,
        "final_beta_absolute": last_beta,
        "final_ell_star_fractional_drift": ell_fractional_drift,
        "final_ell_star_fractional_drift_below_0p05": ell_stable,
        "final_two_widths_resolution_limited": sharp,
        "extrapolated_delta_ds_positive_at_conservative_95pct": extrapolated_positive_95,
        "penultimate_beta_absolute": penultimate_beta,
    }
    stable = all(
        (
            adjudicable_final_four,
            beta_stable,
            ell_stable,
            sharp,
            extrapolated_positive_95,
        )
    )
    return {
        "branch": "refinement_stable_step" if stable else "no_refinement_stable_step",
        "criteria": criteria,
        "failed_criteria": [name for name, passed in criteria.items() if isinstance(passed, bool) and not passed],
    }


def run_ladder() -> dict:
    freeze = load_and_check_freeze()
    rows_by_scaling = {}
    summaries = {}
    for prescription in PRESCRIPTIONS:
        rows = [run_level(n, prescription) for n in LEVELS]
        rows_by_scaling[prescription] = rows
        extrapolations = {
            "delta_ds": continuum_extrapolation(rows, "delta_ds", "delta_ds"),
            "ell_star": continuum_extrapolation(rows, "ell_star", "ell_star"),
            "kappa": continuum_extrapolation(rows, "kappa", "kappa"),
            "smooth_width_log_ell": continuum_extrapolation(
                rows, "smooth_width_log_ell", "smooth_width_log_ell"
            ),
        }
        summaries[prescription] = {
            "beta": beta_rows(rows),
            "decision": classify_scaling(rows, extrapolations),
            "extrapolations": extrapolations,
        }

    passing = [
        name
        for name, summary in summaries.items()
        if summary["decision"]["branch"] == "refinement_stable_step"
    ]
    if len(passing) == 2:
        overall_branch = "refinement_stable_step_both_scalings"
    elif len(passing) == 1:
        overall_branch = "scaling_prescription_dependence"
    else:
        overall_branch = "no_refinement_stable_step"

    result = {
        "assumption_ledger": [
            "[locked] Euclidean 2+1D heat-kernel diagnostic; no Lorentzian propagation claim.",
            "[assumed] The hard density threshold switches the transverse operator from second to fourth order.",
            "[derived] Every finite-grid heat trace and d_s curve is analytic for ell>0.",
            "[derived] Fourier blocks are real symmetric positive semidefinite, so all heat weights are non-negative.",
            "[locked] Construction, extraction, levels, seeds, and decision thresholds were frozen before the ladder.",
        ],
        "freeze_manifest": freeze,
        "method": {
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "dimension": "2 spatial + 1 Euclidean-time",
            "ell_grid": [float(value) for value in ELL_GRID],
            "fit_window": list(FIT_WINDOW),
            "levels": LEVELS,
            "operator": "hard radial second-to-fourth-order transverse switch; unnormalised",
            "prescriptions": list(PRESCRIPTIONS),
        },
        "overall_decision": {
            "branch": overall_branch,
            "passing_scalings": passing,
            "scientific_scope": (
                "Fresh T2 construction only; never a reproduction of the historical v56 generator."
            ),
        },
        "purpose": "T2 fresh discontinuous horizon-profile campaign",
        "rows": rows_by_scaling,
        "summaries": summaries,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def dense_self_test() -> dict:
    n = 4
    blocks, _ = exact_fourier_blocks(n, "fixed_physical")
    block_eigenvalues = np.sort(blocks.ravel())
    h = (R_OUT - R_H) / n
    radial_diag, radial_off = radial_tridiagonal(n, h)
    radial = np.diag(radial_diag) + np.diag(radial_off, 1) + np.diag(radial_off, -1)
    periodic_1d = np.diag(np.full(n, 2.0 / (h * h)))
    periodic_1d += np.diag(np.full(n - 1, -1.0 / (h * h)), 1)
    periodic_1d += np.diag(np.full(n - 1, -1.0 / (h * h)), -1)
    periodic_1d[0, -1] = periodic_1d[-1, 0] = -1.0 / (h * h)
    identity = np.eye(n)
    transverse = np.kron(periodic_1d, identity) + np.kron(identity, periodic_1d)
    _, phase, _ = radial_phase(n, "fixed_physical")
    dense = np.kron(radial, np.eye(n * n))
    dense += np.kron(np.diag((~phase).astype(float)), transverse)
    dense += np.kron(np.diag(phase.astype(float)), transverse @ transverse / (K0 * K0))
    dense_eigenvalues = np.sort(np.clip(np.linalg.eigvalsh(dense), 0.0, None))
    max_difference = float(np.max(np.abs(block_eigenvalues - dense_eigenvalues)))
    if max_difference > 1e-10:
        raise RuntimeError(f"Fourier/dense spectrum mismatch: {max_difference}")
    return {
        "N": n,
        "max_abs_spectrum_difference": max_difference,
        "passed": True,
        "scope": "toy spectrum equivalence only; N=4 is not a declared ladder level",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        print(json.dumps(dense_self_test(), indent=2, sort_keys=True))
        return 0
    if arguments.freeze_only:
        print(json.dumps(write_freeze_manifest(), indent=2, sort_keys=True))
        return 0
    result = run_ladder()
    print(json.dumps(result["overall_decision"], indent=2, sort_keys=True))
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
