#!/usr/bin/env python3
"""P9 PiSR pilot: constrained selection over two admissible dynamics families.

Spec and decision rules frozen in p9_pisr_pilot_spec.md (see freeze manifest).
Families: porous medium (m in {1,1.5,2,3}) and stable/fractional (alpha in
{0.5,1,1.5,2}); m=1 and alpha=2 are both the heat flow (cross-check cells).

Tiers T1a/T1b/T1c/T2 are hard filters; C1 (Gaussian-fixed-point consistency)
and C2 (derived clock eps_H = 1/p) are unweighted soft criteria reported as a
Pareto analysis. Resurrection audit runs the T-campaign step-vs-smooth AIC
comparison on projected d_s(ell) for every survivor.

Deterministic; numpy only. Smoke mode: --smoke writes only the declared smoke
output.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

SIGMA_PI = 0.35  # locked projection width (audit tier)


# --------------------------------------------------------------- flow engines
def evolve_pme(kappa0, z, m, t_final, n_snap=30):
    """Explicit FD for d_t k = Lap(k^m), adaptive dt, snapshots log-spaced.
    Profile clipped to >=0 after each step (positivity-preserving scheme for
    compact-support fronts); pre-clip minimum tracked as the T1a diagnostic."""
    dz = z[1] - z[0]
    k = kappa0.copy()
    t = 0.0
    preclip_min = 0.0
    snaps_t = np.logspace(math.log10(t_final / 300), math.log10(t_final), n_snap)
    out, ti = [], 0
    while t < t_final and ti < n_snap:
        diff = m * np.clip(k, 1e-12, None) ** (m - 1)
        dt = min(0.2 * dz * dz / max(diff.max(), 1e-12), snaps_t[ti] - t + 1e-15)
        u = np.clip(k, 0.0, None) ** m
        lap = (np.roll(u, 1) + np.roll(u, -1) - 2 * u) / dz ** 2
        k = k + dt * lap
        preclip_min = min(preclip_min, float(k.min() / max(k.max(), 1e-30)))
        k = np.clip(k, 0.0, None)
        t += dt
        if t >= snaps_t[ti] - 1e-12:
            out.append((t, k.copy()))
            ti += 1
    return out, preclip_min


def evolve_stable(kappa0, z, alpha, t_final, n_snap=30):
    dz = z[1] - z[0]
    kk = 2 * math.pi * np.fft.fftfreq(len(z), d=dz)
    F0 = np.fft.fft(np.fft.ifftshift(kappa0))
    out = []
    for t in np.logspace(math.log10(t_final / 300), math.log10(t_final), n_snap):
        F = F0 * np.exp(-np.abs(kk) ** alpha * t)
        prof = np.real(np.fft.fftshift(np.fft.ifft(F)))
        out.append((float(t), prof))
    return out, 0.0


# ------------------------------------------------------------------ measures
def iqr_width(z, prof):
    p = np.clip(prof, 0, None)
    c = np.cumsum(p); c /= c[-1]
    lo = z[np.searchsorted(c, 0.25)]
    hi = z[np.searchsorted(c, 0.75)]
    return float(hi - lo)


def entropy(z, prof):
    p = np.clip(prof, 1e-300, None)
    p = p / np.trapezoid(p, z)
    return float(-np.trapezoid(p * np.log(p), z))


def gauss_shape_dist(z, prof):
    p = np.clip(prof, 0, None); p = p / p.max()
    w = iqr_width(z, p) / 1.349  # gaussian sigma from IQR
    return float(np.max(np.abs(p - np.exp(-(z ** 2) / (2 * w * w)))))


def spectral_min(prof):
    F = np.real(np.fft.fft(np.fft.ifftshift(prof)))
    return float(F.min()), float(F.max())


# ------------------------------------------------------ resurrection audit
def ds_curve_from_profile(z, prof, n_lab=80, span=10.0, lam=1e-6):
    xg = np.linspace(0, span, n_lab)
    # Gaussian projection of the kernel profile (mollified correlator row)
    dz = z[1] - z[0]
    kk = 2 * math.pi * np.fft.fftfreq(len(z), d=dz)
    F = np.fft.fft(np.fft.ifftshift(prof)) * np.exp(-SIGMA_PI ** 2 * kk ** 2)
    proj = np.real(np.fft.fftshift(np.fft.ifft(F)))
    diff = np.abs(xg[:, None] - xg[None, :])
    G = np.interp(diff.ravel(), z[z >= 0], proj[z >= 0]).reshape(diff.shape)
    G = 0.5 * (G + G.T)
    vals = np.linalg.eigvalsh(G)
    inv = 1.0 / (np.clip(vals, 0, None) + lam * max(vals.max(), 1e-30))
    ells = np.logspace(math.log10(0.3), math.log10(3.0), 24)
    P = np.array([np.exp(-(e ** 2) * inv).sum() for e in ells])
    x = np.log(ells ** 2); y = np.log(np.clip(P, 1e-300, None))
    return ells, -2.0 * np.gradient(y, x)


def step_vs_smooth_aic(ells, ds):
    x = np.log(ells)
    n = len(x)

    def sse_smooth():
        best = None
        for x0 in np.linspace(x[3], x[-4], 12):
            for w in (0.2, 0.4, 0.8):
                s = 0.5 * (1 + np.tanh((x - x0) / w))
                A = np.vstack([np.ones(n), s]).T
                c, res, *_ = np.linalg.lstsq(A, ds, rcond=None)
                r = float(((A @ c - ds) ** 2).sum())
                best = r if best is None or r < best else best
        return best

    def sse_step():
        best = None
        for i in range(3, n - 3):
            r = float(((ds[:i] - ds[:i].mean()) ** 2).sum()
                      + ((ds[i:] - ds[i:].mean()) ** 2).sum())
            best = r if best is None or r < best else best
        return best

    s_sm, s_st = sse_smooth(), sse_step()
    aic_sm = n * math.log(max(s_sm, 1e-300) / n) + 2 * 4
    aic_st = n * math.log(max(s_st, 1e-300) / n) + 2 * 3
    return float(aic_sm - aic_st)  # >= +10 would prefer the step


# ----------------------------------------------------------------- pilot run
def run_flow(label, family, param, z, kappa0, t_final):
    snaps, preclip = (evolve_pme(kappa0, z, param, t_final) if family == "P"
                      else evolve_stable(kappa0, z, param, t_final))
    mass0 = float(np.trapezoid(np.clip(kappa0, 0, None), z))
    t1a = t1b = t2 = True
    if preclip < -1e-6: t1a = False
    min_prof = preclip; min_spec_rel = 0.0; mass_err = 0.0
    S_prev = -1e18
    xs, ws = [], []
    for (t, prof) in snaps:
        mp = float(prof.min() / max(prof.max(), 1e-30))
        min_prof = min(min_prof, mp)
        if mp < -1e-6: t1a = False
        smin, smax = spectral_min(prof)
        msr = smin / max(smax, 1e-30)
        min_spec_rel = min(min_spec_rel, msr)
        if msr < -1e-8: t1b = False
        me = abs(np.trapezoid(np.clip(prof, 0, None), z) - mass0) / mass0
        mass_err = max(mass_err, me)
        S = entropy(z, prof)
        if S < S_prev - 1e-8: t2 = False
        S_prev = S
        xs.append(math.log(t)); ws.append(math.log(iqr_width(z, prof)))
    if mass_err > 1e-6: t1c = False
    else: t1c = True
    # clock fit over the late third (seed-width contamination decays)
    h = (2 * len(xs)) // 3
    p_fit = float(np.polyfit(xs[h:], ws[h:], 1)[0])
    eps_H = 1.0 / p_fit if p_fit > 0 else float("inf")
    c1 = gauss_shape_dist(z, snaps[-1][1])
    admissible = t1a and t1b and t1c and t2
    row = {
        "flow": label, "family": family, "param": param,
        "T1a_profile_positivity": t1a, "min_profile_rel": min_prof,
        "T1b_bochner_psd": t1b, "min_spectrum_rel": min_spec_rel,
        "T1c_mass_conservation": t1c, "mass_err_max": mass_err,
        "T2_entropy_monotone": t2,
        "admissible": admissible,
        "C2_clock_exponent_p": p_fit, "C2_eps_H": eps_H,
        "C1_gauss_shape_distance": c1,
    }
    if admissible:
        ells, ds = ds_curve_from_profile(z, snaps[-1][1])
        row["audit_delta_aic_smooth_minus_step"] = step_vs_smooth_aic(ells, ds)
        row["audit_step_preferred"] = bool(row["audit_delta_aic_smooth_minus_step"] >= 10)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    # family-specific grids: PME stays compact; stable flows spread as t^(1/alpha)
    zP = np.linspace(-120.0, 120.0, 4096, endpoint=False)
    zS = np.linspace(-800.0, 800.0, 32768, endpoint=False)
    kP = np.exp(-np.abs(zP))  # PSD, non-Gaussian seed
    kS = np.exp(-np.abs(zS))
    XI_TARGET = 80.0  # asymptotic regime: xi_final / xi_seed ~ 60

    if args.smoke:
        cells = [("P m=1.0", "P", 1.0, zP, kP, 400.0),
                 ("S a=2.0", "S", 2.0, zS, kS, XI_TARGET ** 2)]
        outname = "outputs/p9_pisr_pilot_smoke.json"
    else:
        cells = ([("P m=%.1f" % m, "P", m, zP, kP, 400.0) for m in (1.0, 1.5, 2.0, 3.0)]
                 + [("S a=%.1f" % a, "S", a, zS, kS, XI_TARGET ** a) for a in (0.5, 1.0, 1.5, 2.0)])
        outname = "outputs/p9_pisr_pilot.json"

    rows = [run_flow(lab, fam, prm, zz, kk0, tf) for (lab, fam, prm, zz, kk0, tf) in cells]

    # validity gates and Pareto analysis (pre-registered rules 2-4)
    for r in rows:
        fam, prm = r["family"], r["param"]
        expected = (prm + 1.0) if fam == "P" else prm
        ok = (r["C2_eps_H"] != float("inf")
              and abs(r["C2_eps_H"] - expected) / expected < 0.10)
        r["validity_eps_matches_analytic"] = bool(ok)
    heat_ok = all(abs(r["C2_eps_H"] - 2.0) < 0.10 for r in rows
                  if (r["family"], r["param"]) in (("P", 1.0), ("S", 2.0)))

    adm = [r for r in rows if r["admissible"] and r["validity_eps_matches_analytic"]]
    def dominated(a):
        return any(b is not a and b["C1_gauss_shape_distance"] < a["C1_gauss_shape_distance"]
                   and b["C2_eps_H"] < a["C2_eps_H"] for b in adm)
    pareto = [r["flow"] for r in adm if not dominated(r)]
    both = [r["flow"] for r in adm
            if r["C1_gauss_shape_distance"] < 0.05 and r["C2_eps_H"] < 0.1]
    winner = pareto[0] if len(pareto) == 1 else None
    resurrect = [r["flow"] for r in adm if r.get("audit_step_preferred")]

    result = {
        "purpose": "P9 PiSR pilot: constrained dynamics selection (spec frozen; see p9_freeze_manifest.json).",
        "seed_kernel": "Laplace profile (PSD, non-Gaussian)",
        "rows": rows,
        "heat_flow_crosscheck_eps2": bool(heat_ok),
        "admissible_flows": [r["flow"] for r in adm],
        "pareto_front": pareto,
        "dominant_winner": winner,
        "flows_satisfying_BOTH_register_and_inflation_clock": both,
        "resurrection_audit_triggers": resurrect,
        "headline": (
            "Within the scanned families, register-consistency (Gaussian fixed "
            "point) and an inflation-compatible derived clock (eps_H < 0.1) "
            + ("are jointly achievable by: " + ", ".join(both)
               if both else "are NOT jointly achievable: the two criteria pull "
               "in opposite directions (C1 selects alpha=2/m=1; C2 selects "
               "small alpha), and no scanned flow satisfies both.")
        ),
    }
    Path(outname).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for r in rows:
        print("%-9s adm=%s  epsH=%7.3f (expect %s)  C1=%.3f  auditAIC=%s" % (
            r["flow"], r["admissible"], r["C2_eps_H"],
            ("%.1f" % (r["param"] + 1)) if r["family"] == "P" else ("%.1f" % r["param"]),
            r["C1_gauss_shape_distance"],
            ("%.1f" % r["audit_delta_aic_smooth_minus_step"]) if "audit_delta_aic_smooth_minus_step" in r else "n/a"))
    print("Pareto front:", pareto, "| both criteria:", both, "| resurrection triggers:", resurrect)
    print("Wrote", outname)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
