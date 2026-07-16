#!/usr/bin/env python3
"""Complete reference [40] in references.docx reproducibly.

The edit is made directly in ``word/document.xml`` so paragraph/run formatting
and every unrelated DOCX part are preserved byte-for-byte.  The archive is then
repacked with sorted members and fixed timestamps, making repeated runs stable.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "4. PAPERS" / "current" / "references.docx"
DOCUMENT_XML = "word/document.xml"
FIXED_ZIP_TIME = (2000, 1, 1, 0, 0, 0)

OLD = (
    '[40] “Dynamical topological phase realized in a trapped-ion quantum '
    'simulator,” Nature (2022).'
)
NEW = (
    '[40] P. T. Dumitrescu et al., “Dynamical topological phase realized in a '
    'trapped-ion quantum simulator,” Nature 607, 463–467 (2022). '
    'DOI: 10.1038/s41586-022-04853-4.'
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    before = sha256(TARGET)
    old_bytes = OLD.encode("utf-8")
    new_bytes = NEW.encode("utf-8")

    with zipfile.ZipFile(TARGET, "r") as reader:
        infos = sorted(reader.infolist(), key=lambda member: member.filename)
        archive_comment = reader.comment
        members = {info.filename: reader.read(info.filename) for info in infos}

    document = members[DOCUMENT_XML]
    old_count = document.count(old_bytes)
    new_count = document.count(new_bytes)
    if old_count == 1 and new_count == 0:
        members[DOCUMENT_XML] = document.replace(old_bytes, new_bytes, 1)
        state = "updated"
    elif old_count == 0 and new_count == 1:
        state = "already-current"
    else:
        raise RuntimeError(
            "Reference [40] did not match the single expected old or current text: "
            f"old_count={old_count}, new_count={new_count}"
        )

    with tempfile.TemporaryDirectory(prefix="pct-r4-reference40-") as tmpdir:
        stable = Path(tmpdir) / TARGET.name
        with zipfile.ZipFile(
            stable, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as writer:
            writer.comment = archive_comment
            for info in infos:
                out_info = zipfile.ZipInfo(info.filename, date_time=FIXED_ZIP_TIME)
                out_info.compress_type = zipfile.ZIP_DEFLATED
                out_info.comment = info.comment
                out_info.extra = b""
                out_info.create_system = 0
                out_info.external_attr = info.external_attr
                writer.writestr(out_info, members[info.filename])
        os.replace(stable, TARGET)

    print(
        json.dumps(
            {
                "artifact": str(TARGET.relative_to(ROOT)),
                "state": state,
                "reference_40": NEW,
                "sha256_before": before,
                "sha256_after": sha256(TARGET),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
