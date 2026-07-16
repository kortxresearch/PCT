"""Environment capture helpers (dependency-light).

Used by the capsule runner to record enough provenance to support reproducibility
claims without requiring heavyweight tooling.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional


def python_env_summary() -> Dict[str, Any]:
    return {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
    }


def try_pip_freeze(timeout_s: int = 30) -> Optional[List[str]]:
    """Best-effort `pip freeze` capture; returns None if unavailable."""

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception:
        return None

    if proc.returncode != 0:
        return None

    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    return lines