#!/usr/bin/env python3
"""P6c: dense-sampling emergence-flow probe (assumption A4 elimination).

Reuses the P6b machinery unchanged (imported, not copied) and evaluates the
flow at 121 log-spaced times over the same window [0.02, 20.0], so that the
A1 pipeline's monotone-interpolation step (ledger item A4, PCHIP over 7
samples) becomes a pure numerical-resolution statement rather than a
modelling assumption. Output rows carry the same fields A1 consumes
(`flow_time`, `d_s_at_ell1`).

Deterministic; no wall-clock fields.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import p6b_emergence_cosmology_probe as p6b

N_TIMES = 121


def main() -> int:
    z = np.linspace(-p6b.HALFWIDTH, p6b.HALFWIDTH, p6b.GRID_N, endpoint=False)
    K0 = np.exp(-np.abs(z) / 0.05)
    times = np.logspace(math.log10(0.02), math.log10(20.0), N_TIMES)

    rows = []
    for t in times:
        Kt = p6b.evolve_profile(K0, z, float(t))
        ds = p6b.ds_at_fixed_window(z, Kt)
        rows.append({
            "flow_time": float(t),
            "d_s_at_ell1": ds["d_s_at_ell1"],
            "d_s_peak": ds["d_s_peak"],
            "correlation_length": round(p6b.correlation_length(z, Kt), 6),
        })

    ds_vals = [r["d_s_at_ell1"] for r in rows]
    monotone = all(b >= a for a, b in zip(ds_vals[:-1], ds_vals[1:]))

    result = {
        "purpose": "P6c dense flow probe: eliminates A1 ledger item A4 (7-sample PCHIP smoothing) by direct dense evaluation.",
        "n_times": N_TIMES,
        "time_window": [0.02, 20.0],
        "reuses": "p6b_emergence_cosmology_probe functions unchanged (imported)",
        "d_s_monotone_on_window": bool(monotone),
        "flow_rows": rows,
    }

    out = Path("outputs/p6c_dense_flow_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"rows: {N_TIMES}; d_s monotone: {monotone}; "
          f"d_s range [{min(ds_vals):.4f}, {max(ds_vals):.4f}]")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
