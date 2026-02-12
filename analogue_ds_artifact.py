"""Analogue downscaling (DS) artifact utilities.

This module provides a small, dependency-light way to persist and load the
outputs of an analogue downscaling pipeline (or any similar workflow) as an
"artifact" on disk.

Design goals
------------
- Minimal required dependencies (standard library only).
- Optional support for NumPy arrays if NumPy is installed.
- Stable, explicit on-disk layout (versioned).

On-disk format
--------------
An artifact is a directory containing:

- ``meta.json``: metadata (version, created timestamp, user-provided fields)
- ``data/``: payload files

Payload conventions used by :class:`DSArtifact`:

- ``data/tables.json``: mapping of table-name -> list-of-rows (JSON)
- ``data/arrays.npz``: mapping of array-name -> NumPy arrays (NPZ)

You can use these utilities as-is, or as a reference implementation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional


FORMAT_VERSION = 1


class ArtifactError(RuntimeError):
    """Raised when an artifact cannot be read or written."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Atomically write text to *path*.

    We write to a temporary file in the same directory and then replace.
    """

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ArtifactError(f"Missing required file: {path}") from e
    except json.JSONDecodeError as e:
        raise ArtifactError(f"Invalid JSON in file: {path}") from e


def _write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=indent, sort_keys=True) + "\n")


def _try_import_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except Exception:
        return None


@dataclass
class DSArtifact:
    """A simple container for analogue-DS outputs.

    Parameters
    ----------
    meta:
        Free-form metadata to store in ``meta.json``.
    tables:
        JSON-serializable mapping of name -> table data (list/dict primitives).
    arrays:
        Mapping of name -> NumPy arrays (requires NumPy to save/load).
    """

    meta: MutableMapping[str, Any] = field(default_factory=dict)
    tables: MutableMapping[str, Any] = field(default_factory=dict)
    arrays: MutableMapping[str, Any] = field(default_factory=dict)

    def add_meta(self, **kwargs: Any) -> None:
        self.meta.update(kwargs)

    def add_table(self, name: str, table: Any) -> None:
        self.tables[name] = table

    def add_array(self, name: str, array: Any) -> None:
        self.arrays[name] = array

    def save(self, artifact_dir: str | Path, *, overwrite: bool = False) -> Path:
        """Write this artifact to *artifact_dir*.

        If *overwrite* is False, the directory must not exist.
        """

        root = Path(artifact_dir)
        if root.exists() and not overwrite:
            raise ArtifactError(
                f"Artifact directory already exists: {root} (set overwrite=True)"
            )

        _ensure_dir(root)
        data_dir = root / "data"
        _ensure_dir(data_dir)

        meta_obj: Dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "created_utc": _utc_now_iso(),
            **dict(self.meta),
        }
        _write_json(root / "meta.json", meta_obj)

        # Tables (JSON)
        if self.tables:
            _write_json(data_dir / "tables.json", dict(self.tables))

        # Arrays (NPZ) - optional
        if self.arrays:
            np = _try_import_numpy()
            if np is None:
                raise ArtifactError(
                    "NumPy is required to save arrays, but it is not installed. "
                    "Either install NumPy or omit arrays."
                )
            np.savez_compressed(data_dir / "arrays.npz", **dict(self.arrays))

        return root

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "DSArtifact":
        """Load an artifact previously saved with :meth:`save`."""

        root = Path(artifact_dir)
        meta_obj = _read_json(root / "meta.json")

        version = meta_obj.get("format_version")
        if version != FORMAT_VERSION:
            raise ArtifactError(
                f"Unsupported artifact format_version={version!r}; "
                f"expected {FORMAT_VERSION}"
            )

        # Keep all fields except the reserved ones.
        reserved = {"format_version", "created_utc"}
        meta = {k: v for k, v in meta_obj.items() if k not in reserved}

        data_dir = root / "data"

        tables: Dict[str, Any] = {}
        tables_path = data_dir / "tables.json"
        if tables_path.exists():
            tables_obj = _read_json(tables_path)
            if not isinstance(tables_obj, dict):
                raise ArtifactError(f"Expected object in {tables_path}")
            tables = tables_obj

        arrays: Dict[str, Any] = {}
        arrays_path = data_dir / "arrays.npz"
        if arrays_path.exists():
            np = _try_import_numpy()
            if np is None:
                raise ArtifactError(
                    "NumPy is required to load arrays (arrays.npz is present), "
                    "but NumPy is not installed."
                )
            with np.load(arrays_path, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}

        return cls(meta=meta, tables=tables, arrays=arrays)


def save_artifact(
    artifact_dir: str | Path,
    *,
    meta: Optional[Mapping[str, Any]] = None,
    tables: Optional[Mapping[str, Any]] = None,
    arrays: Optional[Mapping[str, Any]] = None,
    overwrite: bool = False,
) -> Path:
    """Convenience wrapper to create and save a :class:`DSArtifact`."""

    art = DSArtifact(
        meta=dict(meta or {}),
        tables=dict(tables or {}),
        arrays=dict(arrays or {}),
    )
    return art.save(artifact_dir, overwrite=overwrite)


def load_artifact(artifact_dir: str | Path) -> DSArtifact:
    """Convenience wrapper for :meth:`DSArtifact.load`."""

    return DSArtifact.load(artifact_dir)


if __name__ == "__main__":
    # Minimal CLI for quick inspection.
    import argparse

    parser = argparse.ArgumentParser(description="Load and summarize a DS artifact")
    parser.add_argument("artifact_dir", help="Path to the artifact directory")
    args = parser.parse_args()

    art = DSArtifact.load(args.artifact_dir)

    print("meta:")
    for k in sorted(art.meta):
        print(f"  {k}: {art.meta[k]!r}")

    print("tables:")
    for k in sorted(art.tables):
        v = art.tables[k]
        n = len(v) if hasattr(v, "__len__") else "?"
        print(f"  {k}: {type(v).__name__} (len={n})")

    print("arrays:")
    for k in sorted(art.arrays):
        a = art.arrays[k]
        shape = getattr(a, "shape", None)
        dtype = getattr(a, "dtype", None)
        print(f"  {k}: shape={shape}, dtype={dtype}")