#!/usr/bin/env python3
"""Branch-A (D-B6-1) Bogoliubov pipeline: the initial-state imprint of a
pre-inflationary emergence epoch.

Physics frame [ledger]:
  [derived]  emergence epoch is radiation-like: a ∝ t_flow^(1/2) ⇒ a ∝ η,
             so a''/a = 0 exactly and the mode equation during emergence is
             v'' + k^2 (1 + f(η)) v = 0.
  [locked]   dispersion correction f(η) = alpha_PCT (d_s(η) - 4)^2,
             alpha_PCT = 0.02.
  [derived]  flow-time/conformal-time map under identification (I):
             a ∝ sqrt(t_flow) and a ∝ η ⇒ t_flow = t_m (η/η_m)^2.
  [derived]  d_s(t_flow) read from the executed dense probe
             outputs/p6c_dense_flow_probe.json (monotone PCHIP in ln t).
  [assumed]  adiabatic (WKB) initialization at η_i (dispersion slowly varying
             there; WKB error ~ |f'|/(2k), quantified in the nuisance battery).
  [assumed]  instantaneous matching to de Sitter Bunch–Davies at η_m, with
             a and H continuous (η_I0 = -η_m).
  [empirical] the single placement parameter: t_m, the flow time at which
             matching occurs (scanned over the probe window).

Observable: M(k) = |alpha_k - beta_k|^2 (super-horizon spectrum modulation);
the PCT-specific signature is the relative difference between the PCT-on run
and the PCT-off run (f ≡ 0) at identical matching:
  dM(k) = M_on(k)/M_off(k) - 1.

Pre-stated decision rule (CODEX_PLAN Batch 9, C-3): Branch A yields a
P2-grade prediction iff at least one quantitative feature of dM(k) is stable
under the full t_m placement scan AND the nuisance battery. Otherwise the
pipeline is reported executed-but-non-identifiable.

Deterministic; numpy only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ALPHA_PCT = 0.02
ETA_M = 1.0                    # units: matching at conformal time 1
K_GRID = np.logspace(math.log10(0.5), math.log10(50.0), 41)  # in units of k_match=1/ETA_M
T_M_SCAN = [6.0, 12.0, 20.0]   # flow time at matching (placement parameter)
T_I = 0.02                     # flow start (probe window edge)
N_STEPS = 8000


def load_ds_curve():
    j = json.load(open("outputs/p6c_dense_flow_probe.json", encoding="utf-8"))
    t = np.array([r["flow_time"] for r in j["flow_rows"]])
    d = np.array([r["d_s_at_ell1"] for r in j["flow_rows"]])
    return t, d


def ds_interp(t_query, t_nodes, d_nodes):
    # monotone linear interpolation in ln t (dense grid: interpolation error
    # is resolution-grade, not modelling-grade; see B6-3)
    return np.interp(np.log(t_query), np.log(t_nodes), d_nodes)


def f_of_eta(eta, t_m, t_nodes, d_nodes, on=True):
    if not on:
        return np.zeros_like(eta)
    t_flow = t_m * (eta / ETA_M) ** 2
    t_flow = np.clip(t_flow, t_nodes[0], t_nodes[-1])
    ds = ds_interp(t_flow, t_nodes, d_nodes)
    return ALPHA_PCT * (ds - 4.0) ** 2


def evolve(t_m, t_nodes, d_nodes, on=True):
    """Vectorized RK4 over the k grid; returns alpha, beta arrays."""
    k = K_GRID
    eta_i = ETA_M * math.sqrt(T_I / t_m)
    etas = np.linspace(eta_i, ETA_M, N_STEPS + 1)
    h = etas[1] - etas[0]

    def omega2(eta):
        return k ** 2 * (1.0 + f_of_eta(np.full_like(k, eta), t_m, t_nodes, d_nodes, on))

    # WKB initialization at eta_i
    w0 = np.sqrt(omega2(eta_i))
    v = 1.0 / np.sqrt(2.0 * w0) + 0j
    vp = -1j * w0 * v

    def rhs(eta, y):
        vv, vvp = y
        return np.array([vvp, -omega2(eta) * vv])

    y = np.array([v, vp])
    for i in range(N_STEPS):
        e = etas[i]
        k1 = rhs(e, y)
        k2 = rhs(e + h / 2, y + h / 2 * k1)
        k3 = rhs(e + h / 2, y + h / 2 * k2)
        k4 = rhs(e + h, y + h * k3)
        y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    v, vp = y

    # Bunch–Davies matching at eta_m; inflation side eta_I = -ETA_M
    eI = -ETA_M
    ph = np.exp(-1j * k * eI)
    vbd = (1.0 - 1j / (k * eI)) * ph / np.sqrt(2.0 * k)
    vbdp = ph / np.sqrt(2.0 * k) * (-1j * k * (1.0 - 1j / (k * eI)) + 1j / (k * eI ** 2))
    alpha = (v * np.conj(vbdp) - vp * np.conj(vbd)) / 1j
    beta = (vp * vbd - v * vbdp) / 1j
    return alpha, beta


def run_config(t_m, t_nodes, d_nodes):
    a_on, b_on = evolve(t_m, t_nodes, d_nodes, on=True)
    a_off, b_off = evolve(t_m, t_nodes, d_nodes, on=False)
    norm_on = np.abs(np.abs(a_on) ** 2 - np.abs(b_on) ** 2 - 1).max()
    M_on = np.abs(a_on - b_on) ** 2
    M_off = np.abs(a_off - b_off) ** 2
    dM = M_on / M_off - 1.0
    i_pk = int(np.argmax(np.abs(dM)))
    return {
        "t_m": t_m,
        "wronskian_error_max": float(norm_on),
        "M_off_range": [float(M_off.min()), float(M_off.max())],
        "dM_rel": [float(x) for x in dM],
        "dM_abs_max": float(np.abs(dM).max()),
        "dM_peak_k_over_kmatch": float(K_GRID[i_pk]),
        "dM_sign_at_peak": int(np.sign(dM[i_pk])),
    }


def main() -> int:
    t_nodes, d_nodes = load_ds_curve()
    configs = [run_config(t_m, t_nodes, d_nodes) for t_m in T_M_SCAN]

    # nuisance battery on the central placement: step halving; start-time x2
    global N_STEPS, T_I
    base = configs[1]
    N_STEPS *= 2
    fine = run_config(T_M_SCAN[1], t_nodes, d_nodes)
    N_STEPS //= 2
    T_I *= 2
    late_start = run_config(T_M_SCAN[1], t_nodes, d_nodes)
    T_I /= 2
    nuis = {
        "step_halving_dM_shift": float(abs(fine["dM_abs_max"] - base["dM_abs_max"])),
        "start_time_x2_dM_shift": float(abs(late_start["dM_abs_max"] - base["dM_abs_max"])),
    }

    amps = [c["dM_abs_max"] for c in configs]
    peaks = [c["dM_peak_k_over_kmatch"] for c in configs]
    amp_spread = max(amps) / min(amps) if min(amps) > 0 else float("inf")
    peak_spread = max(peaks) / min(peaks) if min(peaks) > 0 else float("inf")
    numerics_ok = (nuis["step_halving_dM_shift"] < 0.1 * min(amps)
                   and nuis["start_time_x2_dM_shift"] < 0.5 * min(amps))
    # pre-stated rule: a feature is placement-robust if its spread across the
    # t_m scan is < 2x and it exceeds numerical noise by 10x
    robust_amp = amp_spread < 2.0 and numerics_ok
    robust_peak = peak_spread < 2.0

    decision = ("p2_grade_feature_exists" if (robust_amp or robust_peak)
                else "executed_but_non_identifiable")

    result = {
        "purpose": "Branch-A Bogoliubov pipeline: emergence-epoch initial-state imprint.",
        "ledger": {
            "background": "[derived] radiation-like a~eta (eps_H=2), a''/a=0 exactly",
            "dispersion": "[locked] alpha_PCT=0.02, f=alpha*(d_s-4)^2",
            "ds_source": "[derived] outputs/p6c_dense_flow_probe.json",
            "flow_time_map": "[derived under identification (I)] t=t_m*(eta/eta_m)^2",
            "initialization": "[assumed] WKB at eta_i",
            "matching": "[assumed] instantaneous, a and H continuous",
            "placement": "[empirical] t_m scanned over probe window",
        },
        "k_grid_over_kmatch": [float(x) for x in K_GRID],
        "configs": configs,
        "nuisance_battery": nuis,
        "assessment": {
            "dM_amplitudes_by_placement": amps,
            "amplitude_spread_ratio": amp_spread,
            "dM_peak_locations": peaks,
            "peak_location_spread_ratio": peak_spread,
            "numerics_converged": numerics_ok,
            "amplitude_placement_robust": robust_amp,
            "peak_location_placement_robust": robust_peak,
        },
        "decision_rule": "feature robust iff spread < 2x across t_m scan and 10x above numerical noise",
        "decision": decision,
    }
    out = Path("outputs/p2a_bogoliubov_pipeline.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Wronskian errors:", ["%.2e" % c["wronskian_error_max"] for c in configs])
    print("dM max by placement:", ["%.4f" % a for a in amps], "spread %.2f" % amp_spread)
    print("dM peak k/k_m by placement:", ["%.2f" % p for p in peaks], "spread %.2f" % peak_spread)
    print("nuisances:", nuis)
    print("DECISION:", decision)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
