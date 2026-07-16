#!/usr/bin/env python3
"""A1 scalar-sector closure attempt using the P6b emergence-flow probe.

This is a conditional Mukhanov--Sasaki calculation, not a fit to CMB data.
It starts from the executed P6b d_s(t_flow) samples and the plan-locked
dispersion

    omega^2 = k^2 [1 + alpha_PCT (d_s - 4)^2],  alpha_PCT = 0.02,

then evolves canonically normalized scalar modes on an explicitly assumed
constant-epsilon background.  Three candidate flow-time/e-fold maps and a
pivot-placement scan expose the remaining identifiability of alpha_s and
beta_s.  Every premise is recorded in the JSON assumption ledger.

Run from ``2. REPOSITORY``:

    python a1_scalar_sector_attempt.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq


ALPHA_PCT = 0.02
K_PIVOT_MPC = 0.05
A_S_BASELINE = 2.10e-9
N_BEFORE_END_AT_PIVOT = 57.5
TARGET_NS_BASELINE = 0.965
EPSILON_H = (1.0 - TARGET_NS_BASELINE) / (2.0 + (1.0 - TARGET_NS_BASELINE))

LOG_K_GRID = np.linspace(-2.5, 2.5, 51)
LOCAL_LOG_K_GRID = np.linspace(-0.6, 0.6, 13)
ALIGNMENT_OFFSETS_EFOLDS = (-2.0, -1.0, 0.0, 1.0, 2.0)
INITIAL_SOUND_HORIZON_RATIO = 160.0
FINAL_SOUND_HORIZON_RATIO = 1.0e-4
RTOL = 2.0e-9
ATOL = 2.0e-11
TIGHT_RTOL = 2.0e-10
TIGHT_ATOL = 2.0e-12


@dataclass(frozen=True)
class FlowMap:
    map_id: str
    q: float
    statement: str
    motivation: str


FLOW_MAPS = (
    FlowMap(
        "diffusive_xi",
        0.5,
        "a/a_c = (t_flow/t_c)^(1/2), hence t_flow=t_c*exp(2*(x-delta)).",
        "Identifies the scale factor with the measured correlation length xi proportional to sqrt(t_flow).",
    ),
    FlowMap(
        "linear_clock",
        1.0,
        "a/a_c = t_flow/t_c, hence t_flow=t_c*exp(x-delta).",
        "Uses flow time itself as the simplest positive expansion clock.",
    ),
    FlowMap(
        "quadratic_clock",
        2.0,
        "a/a_c = (t_flow/t_c)^2, hence t_flow=t_c*exp((x-delta)/2).",
        "A declared slower-flow sensitivity bracket; it is not selected by current PCT dynamics.",
    ),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite value in output: {value}")
    if abs(value) < 5.0e-15:
        return 0.0
    return float(f"{value:.12g}")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return finite_float(float(value))
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


class DimensionFlow:
    """Monotone interpolation of the executed P6b d_s-at-fixed-window rows."""

    def __init__(self, times: np.ndarray, dimensions: np.ndarray) -> None:
        if np.any(np.diff(times) <= 0.0) or np.any(np.diff(dimensions) < 0.0):
            raise ValueError("P6b flow samples must be ordered and monotone")
        self.times = np.asarray(times, dtype=float)
        self.dimensions = np.asarray(dimensions, dtype=float)
        self.log_times = np.log(self.times)
        self._pchip = PchipInterpolator(self.log_times, self.dimensions, extrapolate=False)
        self._pchip_prime = self._pchip.derivative()
        self.d_mid = 0.5 * (float(self.dimensions[0]) + float(self.dimensions[-1]))
        self.log_t_center = float(
            brentq(
                lambda log_t: float(self._pchip(log_t)) - self.d_mid,
                float(self.log_times[0]),
                float(self.log_times[-1]),
            )
        )
        self.t_center = math.exp(self.log_t_center)

    def ds_from_log_t(self, log_t: float) -> float:
        if log_t <= self.log_times[0]:
            return float(self.dimensions[0])
        if log_t >= self.log_times[-1]:
            return float(self.dimensions[-1])
        return float(self._pchip(log_t))

    def dds_dlogt(self, log_t: float) -> float:
        if log_t <= self.log_times[0] or log_t >= self.log_times[-1]:
            return 0.0
        return float(self._pchip_prime(log_t))


def make_sound_speed(
    flow: DimensionFlow, flow_map: FlowMap, center_offset: float, alpha_pct: float = ALPHA_PCT
) -> tuple[Callable[[float], float], Callable[[float], float], Callable[[float], float]]:
    """Return c_s(x), d ln(c_s)/dx, and d_s(x).

    x=ln(a/a_pivot) grows forward in time.  center_offset is the x location
    of the d_s midpoint relative to the pivot x=0.
    """

    def log_t(x: float) -> float:
        return flow.log_t_center + (x - center_offset) / flow_map.q

    def ds(x: float) -> float:
        return flow.ds_from_log_t(log_t(x))

    def cs(x: float) -> float:
        delta = ds(x) - 4.0
        return math.sqrt(1.0 + alpha_pct * delta * delta)

    def dlncs_dx(x: float) -> float:
        lt = log_t(x)
        dimension = flow.ds_from_log_t(lt)
        d_dimension_dx = flow.dds_dlogt(lt) / flow_map.q
        c_squared = 1.0 + alpha_pct * (dimension - 4.0) ** 2
        return alpha_pct * (dimension - 4.0) * d_dimension_dx / c_squared

    return cs, dlncs_dx, ds


def evolve_mode(
    k: float,
    cs: Callable[[float], float],
    dlncs_dx: Callable[[float], float],
    *,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> tuple[float, float, int]:
    """Evolve one canonical scalar mode and return raw P_R, Wronskian error, nfev."""

    def a_hubble(x: float) -> float:
        return math.exp((1.0 - EPSILON_H) * x)

    def sound_horizon_ratio(x: float) -> float:
        return cs(x) * k / a_hubble(x)

    x_initial = brentq(
        lambda x: sound_horizon_ratio(x) - INITIAL_SOUND_HORIZON_RATIO,
        -50.0,
        50.0,
    )
    x_final = brentq(
        lambda x: sound_horizon_ratio(x) - FINAL_SOUND_HORIZON_RATIO,
        -50.0,
        50.0,
    )
    x_wronskian_audit = brentq(
        lambda x: sound_horizon_ratio(x) - 1.0,
        x_initial,
        x_final,
    )
    omega_initial = cs(x_initial) * k
    v_initial = complex(1.0 / math.sqrt(2.0 * omega_initial), 0.0)
    vx_initial = (
        -0.5 * dlncs_dx(x_initial) - 1j * INITIAL_SOUND_HORIZON_RATIO
    ) * v_initial
    y0 = np.array(
        [v_initial.real, v_initial.imag, vx_initial.real, vx_initial.imag], dtype=float
    )

    def rhs(x: float, y: np.ndarray) -> np.ndarray:
        v = complex(y[0], y[1])
        vx = complex(y[2], y[3])
        frequency_term = (cs(x) * k / a_hubble(x)) ** 2 - (2.0 - EPSILON_H)
        vxx = -(1.0 - EPSILON_H) * vx - frequency_term * v
        return np.array([vx.real, vx.imag, vxx.real, vxx.imag], dtype=float)

    solution = solve_ivp(
        rhs,
        (x_initial, x_final),
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        t_eval=(x_wronskian_audit, x_final),
    )
    if not solution.success:
        raise RuntimeError(f"mode integration failed for k={k}: {solution.message}")
    audit = solution.y[:, 0]
    final = solution.y[:, -1]
    v_final = complex(final[0], final[1])
    z_final = math.exp(x_final) * math.sqrt(2.0 * EPSILON_H)
    power = k**3 * abs(v_final / z_final) ** 2 / (2.0 * math.pi**2)
    v_audit = complex(audit[0], audit[1])
    vx_audit = complex(audit[2], audit[3])
    v_eta_audit = a_hubble(x_wronskian_audit) * vx_audit
    wronskian = -2.0 * (np.conjugate(v_audit) * v_eta_audit).imag
    return float(power), abs(float(wronskian) - 1.0), int(solution.nfev)


def local_derivatives(x: np.ndarray, log_power: np.ndarray, window_points: int = 11) -> tuple[np.ndarray, ...]:
    """Local quintic derivatives of ln(P) on a uniform log-k grid."""

    if window_points % 2 == 0 or window_points < 7:
        raise ValueError("window_points must be odd and at least 7")
    n = len(x)
    if n < window_points:
        raise ValueError("grid shorter than derivative window")
    first = np.empty(n)
    second = np.empty(n)
    third = np.empty(n)
    half = window_points // 2
    for i in range(n):
        start = min(max(i - half, 0), n - window_points)
        stop = start + window_points
        dx = x[start:stop] - x[i]
        coeff = np.polynomial.polynomial.polyfit(dx, log_power[start:stop], deg=5)
        first[i] = coeff[1]
        second[i] = 2.0 * coeff[2]
        third[i] = 6.0 * coeff[3]
    return first, second, third


def pivot_fit(x: np.ndarray, log_power: np.ndarray, half_width: float) -> dict[str, float]:
    mask = np.abs(x) <= half_width + 1.0e-12
    coeff = np.polynomial.polynomial.polyfit(x[mask], log_power[mask], deg=5)
    return {
        "n_s": 1.0 + float(coeff[1]),
        "alpha_s": 2.0 * float(coeff[2]),
        "beta_s": 6.0 * float(coeff[3]),
    }


def calculate_spectrum(
    flow: DimensionFlow,
    flow_map: FlowMap,
    center_offset: float,
    log_k_grid: np.ndarray,
    amplitude_scale: float,
    *,
    alpha_pct: float = ALPHA_PCT,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> dict[str, Any]:
    cs, dlncs_dx, ds = make_sound_speed(flow, flow_map, center_offset, alpha_pct)
    k_pivot_code = 1.0 / cs(0.0)
    raw_power: list[float] = []
    wronskian_errors: list[float] = []
    evaluations = 0
    for log_ratio in log_k_grid:
        k_code = k_pivot_code * math.exp(float(log_ratio))
        power, wronskian_error, nfev = evolve_mode(k_code, cs, dlncs_dx, rtol=rtol, atol=atol)
        raw_power.append(power * amplitude_scale)
        wronskian_errors.append(wronskian_error)
        evaluations += nfev
    power_array = np.asarray(raw_power)
    log_power = np.log(power_array)
    first, second, third = local_derivatives(log_k_grid, log_power, window_points=11)
    pivot_main = pivot_fit(log_k_grid, log_power, half_width=0.5)
    pivot_narrow = pivot_fit(log_k_grid, log_power, half_width=0.4)
    pivot_wide = pivot_fit(log_k_grid, log_power, half_width=0.6)
    pivot_index = int(np.argmin(np.abs(log_k_grid)))
    curves = []
    for i, log_ratio in enumerate(log_k_grid):
        curves.append(
            {
                "k_mpc_inverse": K_PIVOT_MPC * math.exp(float(log_ratio)),
                "k_over_k_pivot": math.exp(float(log_ratio)),
                "P_R": power_array[i],
                "n_s": 1.0 + first[i],
                "alpha_s": second[i],
                "beta_s": third[i],
            }
        )
    return {
        "map_id": flow_map.map_id,
        "center_offset_efolds": center_offset,
        "flow_map_q": flow_map.q,
        "pivot_state": {
            "d_s": ds(0.0),
            "c_s": cs(0.0),
            "k_code_at_sound_horizon_exit": k_pivot_code,
            "P_R_at_pivot": power_array[pivot_index],
        },
        "pivot_observables": pivot_main,
        "derivative_method_check": {
            "narrow_half_width_0p4": pivot_narrow,
            "wide_half_width_0p6": pivot_wide,
            "max_abs_difference_from_main": max(
                abs(pivot_main[key] - candidate[key])
                for key in ("n_s", "alpha_s", "beta_s")
                for candidate in (pivot_narrow, pivot_wide)
            ),
        },
        "numerics": {
            "max_wronskian_abs_error_at_c_s_k_over_aH_1": max(wronskian_errors),
            "ode_function_evaluations": evaluations,
        },
        "curves": curves,
    }


def baseline_spectrum() -> dict[str, Any]:
    cs = lambda _x: 1.0
    dlncs = lambda _x: 0.0
    raw: list[float] = []
    wronskian_errors: list[float] = []
    evaluations = 0
    for log_ratio in LOG_K_GRID:
        power, error, nfev = evolve_mode(math.exp(float(log_ratio)), cs, dlncs)
        raw.append(power)
        wronskian_errors.append(error)
        evaluations += nfev
    raw_array = np.asarray(raw)
    pivot_index = int(np.argmin(np.abs(LOG_K_GRID)))
    amplitude_scale = A_S_BASELINE / raw_array[pivot_index]
    log_normalized = np.log(raw_array * amplitude_scale)
    pivot = pivot_fit(LOG_K_GRID, log_normalized, half_width=0.5)
    expected_ns = 1.0 - 2.0 * EPSILON_H / (1.0 - EPSILON_H)
    return {
        "amplitude_scale": amplitude_scale,
        "pivot_observables": pivot,
        "analytic_constant_epsilon": {
            "n_s": expected_ns,
            "alpha_s": 0.0,
            "beta_s": 0.0,
        },
        "absolute_errors": {
            "n_s": abs(pivot["n_s"] - expected_ns),
            "alpha_s": abs(pivot["alpha_s"]),
            "beta_s": abs(pivot["beta_s"]),
        },
        "max_wronskian_abs_error_at_c_s_k_over_aH_1": max(wronskian_errors),
        "ode_function_evaluations": evaluations,
    }


def labeled_ledger() -> list[dict[str, str]]:
    return [
        {
            "id": "L1",
            "status": "[locked]",
            "statement": "Use omega^2=k^2[1+alpha_PCT(d_s-4)^2] with alpha_PCT=0.02 and the positive sign specified by CODEX_PLAN A1.",
            "consequence": "The dispersion amplitude and sign are not scanned or fitted.",
        },
        {
            "id": "L2",
            "status": "[derived]",
            "statement": "The seven d_s(t_flow) values are read from the executed P6b JSON field d_s_at_ell1; their source SHA256 is recorded.",
            "consequence": "The epoch engine is the existing toy heat-flow probe, not the historical cubic beta-flow assertion.",
        },
        {
            "id": "L3",
            "status": "[locked]",
            "statement": "The A1 dispersion is used without an additional k-window: W=1 in the notation of v56 V.I.0.3.",
            "consequence": "No window amplitude, width, or sign is available for post hoc tuning.",
        },
        {
            "id": "A1",
            "status": "[assumed]",
            "statement": "A canonical single scalar mode is valid, v=z R with z=a sqrt(2 epsilon_H), and mu_PCT=0.",
            "consequence": "A noncanonical kinetic prefactor or PCT mass term would change the transfer map.",
        },
        {
            "id": "A2",
            "status": "[assumed]",
            "statement": "The background has constant epsilon_H chosen so its alpha_PCT=0 baseline has n_s=0.965 and zero running.",
            "consequence": "The calculation isolates PCT-induced running; it does not derive the inflationary background.",
        },
        {
            "id": "A3",
            "status": "[assumed]",
            "statement": "The initial state is the zeroth-order adiabatic/Bunch-Davies vacuum at c_s k/(aH)=160.",
            "consequence": "Excited initial states are out of scope and could alter P_R(k).",
        },
        {
            "id": "A4",
            "status": "[assumed]",
            "statement": "The sampled d_s curve is reconstructed by monotone PCHIP in ln(t_flow), clamped to its measured endpoints.",
            "consequence": "This supplies an explicit smoothing/interpolation rule but PCT has not selected it physically.",
        },
        {
            "id": "A5",
            "status": "[assumed]",
            "statement": "Three positive flow-clock maps a proportional to t_flow^q with q in {1/2,1,2} are carried without weights.",
            "consequence": "Their envelope is a sensitivity set, not a probability or confidence interval.",
        },
        {
            "id": "A6",
            "status": "[assumed]",
            "statement": "For centered runs, the midpoint of the measured d_s range exits at k_pivot=0.05 Mpc^-1 and N_before_end=57.5.",
            "consequence": "The absolute scale map is not derived; offsets of +/-2 e-folds are scanned.",
        },
        {
            "id": "A7",
            "status": "[assumed]",
            "statement": "The baseline scalar amplitude at the pivot is normalized to A_s=2.10e-9.",
            "consequence": "Absolute P_R values are conditional normalization; n_s, alpha_s, and beta_s do not depend on it.",
        },
        {
            "id": "A8",
            "status": "[assumed]",
            "statement": "Linear perturbation theory, negligible backreaction, and the same constant-epsilon z throughout the emergence transition are valid.",
            "consequence": "Failure of adiabatic single-clock evolution would invalidate this mode equation even with the locked dispersion.",
        },
        {
            "id": "A9",
            "status": "[assumed]",
            "statement": "The +/-2 e-fold alignment grid and factor-10 spread ceiling are operational identifiability tests, not probability statements.",
            "consequence": "They can disprove robustness on this declared range but cannot turn the range into a confidence interval.",
        },
        {
            "id": "A10",
            "status": "[assumed]",
            "statement": "Natural units c=M_Pl=H_pivot(code)=1 are used during mode evolution; a common scale factor fixes the displayed baseline amplitude.",
            "consequence": "Only the assumed A_s normalization supplies the absolute vertical scale of P_R.",
        },
        {
            "id": "D1",
            "status": "[derived]",
            "statement": "With x=ln(a/a_pivot), H=H_pivot exp(-epsilon_H x), d/deta=aH d/dx, and z proportional to a, z''/[z(aH)^2]=2-epsilon_H.",
            "consequence": "The mode equation becomes v_xx+(1-epsilon_H)v_x+[(c_s k/aH)^2-(2-epsilon_H)]v=0.",
        },
        {
            "id": "D2",
            "status": "[derived]",
            "statement": "P_R=k^3|v/z|^2/(2 pi^2); n_s=1+d ln P_R/d ln k, alpha_s=d^2 ln P_R/d(ln k)^2, beta_s=d^3 ln P_R/d(ln k)^3.",
            "consequence": "All four requested observables follow from the same evolved mode functions.",
        },
        {
            "id": "D3",
            "status": "[derived]",
            "statement": "A local quintic fit in ln k differentiates the numerical spectrum; 0.4, 0.5, and 0.6 half-width fits audit derivative sensitivity.",
            "consequence": "Derivative-extraction uncertainty is reported separately from physical map sensitivity.",
        },
        {
            "id": "D4",
            "status": "[derived]",
            "statement": "For each candidate and alignment, k_code,pivot=1/c_s(x=0) follows from the sound-horizon condition c_s k=aH with aH(x=0)=1.",
            "consequence": "Differentiation with respect to ln(k/k_pivot) is identical to differentiation with respect to physical ln k.",
        },
    ]


def build_result(source: Path) -> dict[str, Any]:
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    rows = source_payload["flow_rows"]
    times = np.asarray([row["flow_time"] for row in rows], dtype=float)
    dimensions = np.asarray([row["d_s_at_ell1"] for row in rows], dtype=float)
    flow = DimensionFlow(times, dimensions)

    baseline = baseline_spectrum()
    amplitude_scale = float(baseline["amplitude_scale"])
    central_results = [
        calculate_spectrum(flow, flow_map, 0.0, LOG_K_GRID, amplitude_scale)
        for flow_map in FLOW_MAPS
    ]
    tight_results = [
        calculate_spectrum(
            flow,
            flow_map,
            0.0,
            LOCAL_LOG_K_GRID,
            amplitude_scale,
            rtol=TIGHT_RTOL,
            atol=TIGHT_ATOL,
        )
        for flow_map in FLOW_MAPS
    ]
    convergence_rows = []
    for central, tight in zip(central_results, tight_results):
        deltas = {
            key: abs(
                central["pivot_observables"][key]
                - tight["pivot_observables"][key]
            )
            for key in ("n_s", "alpha_s", "beta_s")
        }
        convergence_rows.append(
            {
                "map_id": central["map_id"],
                "tight_tolerances": {"rtol": TIGHT_RTOL, "atol": TIGHT_ATOL},
                "tight_pivot_observables": tight["pivot_observables"],
                "absolute_difference_from_production_run": deltas,
            }
        )

    alignment_scan: list[dict[str, Any]] = []
    for flow_map in FLOW_MAPS:
        for offset in ALIGNMENT_OFFSETS_EFOLDS:
            if offset == 0.0:
                central = next(item for item in central_results if item["map_id"] == flow_map.map_id)
                summary = {
                    "map_id": flow_map.map_id,
                    "center_offset_efolds": 0.0,
                    "pivot_state": central["pivot_state"],
                    "pivot_observables": central["pivot_observables"],
                    "derivative_method_check": central["derivative_method_check"],
                }
            else:
                local = calculate_spectrum(
                    flow, flow_map, offset, LOCAL_LOG_K_GRID, amplitude_scale
                )
                summary = {
                    "map_id": flow_map.map_id,
                    "center_offset_efolds": offset,
                    "pivot_state": local["pivot_state"],
                    "pivot_observables": local["pivot_observables"],
                    "derivative_method_check": local["derivative_method_check"],
                }
            alignment_scan.append(summary)

    central_alpha = [item["pivot_observables"]["alpha_s"] for item in central_results]
    central_beta = [item["pivot_observables"]["beta_s"] for item in central_results]
    scan_alpha = [item["pivot_observables"]["alpha_s"] for item in alignment_scan]
    scan_beta = [item["pivot_observables"]["beta_s"] for item in alignment_scan]
    derivative_differences = {
        key: max(
            abs(
                item["pivot_observables"][key]
                - candidate[key]
            )
            for item in central_results
            for candidate in (
                item["derivative_method_check"]["narrow_half_width_0p4"],
                item["derivative_method_check"]["wide_half_width_0p6"],
            )
        )
        for key in ("n_s", "alpha_s", "beta_s")
    }
    convergence_differences = {
        key: max(
            row["absolute_difference_from_production_run"][key]
            for row in convergence_rows
        )
        for key in ("n_s", "alpha_s", "beta_s")
    }
    baseline_errors = baseline["absolute_errors"]
    numerical_pass = bool(
        baseline_errors["n_s"] < 2.0e-6
        and baseline_errors["alpha_s"] < 2.0e-6
        and baseline_errors["beta_s"] < 2.0e-6
        and baseline["max_wronskian_abs_error_at_c_s_k_over_aH_1"] < 2.0e-6
        and max(
            item["numerics"]["max_wronskian_abs_error_at_c_s_k_over_aH_1"]
            for item in central_results
        )
        < 2.0e-6
        and convergence_differences["n_s"] < 2.0e-6
        and convergence_differences["alpha_s"] < 1.0e-5
        and convergence_differences["beta_s"] < 1.0e-4
    )
    derivative_pass = bool(
        derivative_differences["n_s"] < 5.0e-4
        and derivative_differences["alpha_s"] < 5.0e-4
        and derivative_differences["beta_s"] < 5.0e-3
    )

    def nonzero_sign(values: list[float], floor: float = 1.0e-5) -> int:
        signs = {1 if value > floor else -1 if value < -floor else 0 for value in values}
        return signs.pop() if len(signs) == 1 else 0

    central_sign = nonzero_sign(central_alpha)
    alignment_sign = nonzero_sign(scan_alpha)
    nonzero_abs = [abs(value) for value in scan_alpha if abs(value) > 1.0e-5]
    alignment_spread_ratio = max(nonzero_abs) / min(nonzero_abs) if nonzero_abs else math.inf
    meaningful_band = bool(
        numerical_pass
        and derivative_pass
        and central_sign != 0
        and alignment_sign == central_sign
        and alignment_spread_ratio <= 10.0
    )

    gap_status = {
        "transfer_map": {
            "status": "conditionally_instantiated_not_derived",
            "detail": "Canonical Mukhanov--Sasaki propagation supplies a transfer only after assumptions A1--A3.",
        },
        "scale_map": {
            "status": "not_closed",
            "detail": "No PCT relation fixes the P6b flow midpoint relative to the CMB pivot; the offset scan changes the result.",
        },
        "smoothing_rule": {
            "status": "explicit_but_assumed",
            "detail": "Monotone PCHIP in ln(t_flow) is reproducible but not selected by a PCT theorem.",
        },
        "coupling_amplitude_and_sign": {
            "status": "model_locked_for_this_attempt",
            "detail": "alpha_PCT=0.02 with positive sign is obeyed exactly; the canonical scalar embedding and mu_PCT=0 remain assumptions.",
        },
    }

    result: dict[str, Any] = {
        "schema_version": "a1-scalar-sector-attempt-v1",
        "purpose": "Phase A1 attempt to derive scalar tilt and runnings from the temporalized emergence flow through Mukhanov--Sasaki evolution.",
        "scope": "Exploratory conditional calculation in a toy epoch engine; not a v4 manuscript claim and not an observational fit.",
        "provenance": {
            "source": "outputs/p6b_emergence_cosmology_probe.json",
            "source_sha256": sha256(source),
            "source_field": "flow_rows[*].d_s_at_ell1",
            "flow_samples": [
                {"t_flow": time, "d_s": dimension}
                for time, dimension in zip(times, dimensions)
            ],
        },
        "assumption_and_derivation_ledger": labeled_ledger(),
        "gap_status": gap_status,
        "conventions": {
            "forward_efold_coordinate": "x=ln(a/a_pivot), increasing with time",
            "standard_efolds_before_end": "N_before_end=57.5-x",
            "pivot_k_mpc_inverse": K_PIVOT_MPC,
            "epsilon_H": EPSILON_H,
            "baseline_A_s": A_S_BASELINE,
            "alpha_PCT": ALPHA_PCT,
            "mu_PCT_squared": 0.0,
            "dimension_flow_center_definition": "midpoint of first and last executed d_s_at_ell1 values",
            "dimension_flow_center_d_s": flow.d_mid,
            "dimension_flow_center_t": flow.t_center,
        },
        "candidate_maps": [
            {
                "map_id": item.map_id,
                "q": item.q,
                "status": "[assumed]",
                "statement": item.statement,
                "motivation": item.motivation,
            }
            for item in FLOW_MAPS
        ],
        "numerical_configuration": {
            "integrator": "scipy.solve_ivp DOP853",
            "rtol": RTOL,
            "atol": ATOL,
            "initial_c_s_k_over_aH": INITIAL_SOUND_HORIZON_RATIO,
            "final_c_s_k_over_aH": FINAL_SOUND_HORIZON_RATIO,
            "central_log_k_over_pivot_grid": LOG_K_GRID.tolist(),
            "alignment_scan_offsets_efolds": list(ALIGNMENT_OFFSETS_EFOLDS),
            "alignment_local_log_k_grid": LOCAL_LOG_K_GRID.tolist(),
            "random_seed": None,
            "wall_clock_fields": "none",
        },
        "baseline_validation": baseline,
        "tight_tolerance_convergence": {
            "rows": convergence_rows,
            "maximum_absolute_differences": convergence_differences,
        },
        "centered_candidate_results": central_results,
        "identifiability_scan": {
            "purpose": "Move the unfixed transition center relative to the pivot while retaining the locked dispersion and each flow map.",
            "rows": alignment_scan,
            "centered_map_envelope": {
                "alpha_s_min": min(central_alpha),
                "alpha_s_max": max(central_alpha),
                "beta_s_min": min(central_beta),
                "beta_s_max": max(central_beta),
                "common_nonzero_alpha_sign": central_sign,
            },
            "map_plus_alignment_envelope": {
                "alpha_s_min": min(scan_alpha),
                "alpha_s_max": max(scan_alpha),
                "beta_s_min": min(scan_beta),
                "beta_s_max": max(scan_beta),
                "common_nonzero_alpha_sign": alignment_sign,
                "nonzero_abs_alpha_spread_ratio": alignment_spread_ratio,
            },
        },
        "closure_rule": {
            "status": "operational acceptance rule encoded in this executable",
            "requirements": [
                "constant-epsilon baseline reproduces analytic n_s, alpha_s=0, beta_s=0 within 2e-6 and Wronskian error is below 2e-6",
                "tight-tolerance reruns agree within (2e-6,1e-5,1e-4) for (n_s,alpha_s,beta_s)",
                "0.4/0.5/0.6 log-k derivative windows agree within (5e-4,5e-4,5e-3) for (n_s,alpha_s,beta_s)",
                "all three centered map candidates give the same nonzero sign for alpha_s",
                "the +/-2 e-fold pivot-placement scan for every map retains that same nonzero sign",
                "the nonzero absolute alpha_s spread over map plus placement scan is no more than a factor 10",
            ],
            "numerical_validation_pass": numerical_pass,
            "derivative_extraction_stability_pass": derivative_pass,
            "centered_common_sign_pass": central_sign != 0,
            "alignment_common_sign_pass": alignment_sign == central_sign and alignment_sign != 0,
            "alignment_spread_pass": alignment_spread_ratio <= 10.0,
            "meaningful_band_exists": meaningful_band,
            "max_centered_derivative_fit_differences": derivative_differences,
        },
        "derivation_status": "closed" if meaningful_band else "not_closed",
        "closed_alpha_s_band": [min(scan_alpha), max(scan_alpha)] if meaningful_band else None,
        "corresponding_beta_s_band": [min(scan_beta), max(scan_beta)] if meaningful_band else None,
        "conditional_centered_alpha_s_envelope": [min(central_alpha), max(central_alpha)],
        "conditional_centered_beta_s_envelope": [min(central_beta), max(central_beta)],
        "manuscript_draft_created": bool(meaningful_band),
        "failure_diagnosis": None
        if meaningful_band
        else {
            "style": "D1a identifiability update after explicit Mukhanov--Sasaki propagation",
            "numerical_pipeline_validated": numerical_pass,
            "what_was_gained": [
                "The locked dispersion now has an explicit transfer to a full conditional P_R(k).",
                "n_s(k), alpha_s(k), and beta_s(k) are obtained by differentiating the same evolved spectrum.",
                "The positive coupling amplitude and sign are no longer scanned in this attempt.",
            ],
            "why_identification_still_fails": [
                "The three centered flow clocks give materially different conditional alpha_s values and a beta_s envelope that crosses zero.",
                "Moving the otherwise unfixed transition center by only +/-2 e-folds makes alpha_s cross zero and span both signs.",
                "The map-plus-alignment set has no theory-defined weights, so its min/max cannot be called a predictive confidence band.",
                "The fastest map also exposes beta_s sensitivity to the declared derivative window near the clamped flow endpoint.",
            ],
            "historical_negative_running_band": [-0.02, -0.005],
            "historical_band_status": "not rederived; centered candidates are positive, while arbitrary scale offsets can enter or leave the historical band",
            "minimum_new_physics_needed": [
                "A derived or calibrated relation between emergence flow time and the cosmological scale factor/e-fold clock.",
                "A derived placement of the d_s transition relative to k_pivot (equivalently, a physical transition energy/epoch).",
                "A perturbation action fixing z, any mu_PCT term, and the vacuum prescription rather than assuming their canonical values.",
                "A physically selected smooth d_s(t_flow) reconstruction beyond the seven-point toy probe.",
            ],
        },
        "conclusion": (
            "The scalar-sector closure criterion is met under the fully enumerated assumptions."
            if meaningful_band
            else "The attempt does not close: the locked dispersion can be propagated, but the unfixed flow/e-fold and transition-to-pivot scale maps do not define a stable alpha_s band. Conditional centered values and beta_s are reported as identifiability diagnostics, not predictions."
        ),
    }
    return clean_json(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="outputs/p6b_emergence_cosmology_probe.json")
    parser.add_argument("--out", default="outputs/a1_scalar_sector_attempt.json")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.out)
    result = build_result(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "out": output.as_posix(),
        "derivation_status": result["derivation_status"],
        "conditional_centered_alpha_s_envelope": result[
            "conditional_centered_alpha_s_envelope"
        ],
        "conditional_centered_beta_s_envelope": result[
            "conditional_centered_beta_s_envelope"
        ],
        "meaningful_band_exists": result["closure_rule"]["meaningful_band_exists"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
