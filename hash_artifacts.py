#!/usr/bin/env python3
"""Refresh SHA256 checksums for repository artifacts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


DEFAULT_PATTERNS = (
    "*.py",
    "*.md",
    "*.txt",
    "*.tex",
    "*.pdf",
    "README/*.md",
    "README/*.txt",
    "configs/*.json",
    "data/*.csv",
    "outputs/*.json",
    "outputs/**/*.json",
    "outputs/*.txt",
    "chains/*.txt",
    "chains/*.yaml",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_artifacts(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    paths: set[Path] = set()
    manifest = root / "outputs" / "sha256SUMS.txt"
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file() and path.resolve() != manifest.resolve():
                paths.add(path)
    return sorted(paths, key=lambda p: p.as_posix().lower())


def main() -> int:
    parser = argparse.ArgumentParser(description="Write outputs/sha256SUMS.txt")
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        help="Glob pattern relative to root; may be repeated",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    patterns = tuple(args.patterns) if args.patterns else DEFAULT_PATTERNS
    artifacts = iter_artifacts(root, patterns)

    out = root / "outputs" / "sha256SUMS.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for path in artifacts:
        rel = path.relative_to(root).as_posix()
        lines.append(f"{sha256(path)}  {rel}")
    # Keep the manifest directly consumable by GNU sha256sum on every host.
    # ``newline="\n"`` prevents Windows text-mode CRLF translation.
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out} ({len(artifacts)} artifacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
