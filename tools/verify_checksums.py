#!/usr/bin/env python3
"""Verify a sha256sum-style manifest without external dependencies."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    root = manifest.parent
    seen: set[str] = set()
    checked = 0

    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = LINE.fullmatch(line)
        if not match:
            raise SystemExit(f"{manifest}:{line_number}: malformed checksum line")

        expected, relative_text = match.groups()
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(f"{manifest}:{line_number}: unsafe path {relative_text!r}")
        if relative_text in seen:
            raise SystemExit(f"{manifest}:{line_number}: duplicate path {relative_text!r}")
        seen.add(relative_text)

        target = root / relative
        if not target.is_file():
            raise SystemExit(f"{relative_text}: missing")
        actual = sha256(target)
        if actual != expected:
            raise SystemExit(
                f"{relative_text}: checksum mismatch\n"
                f"  expected {expected}\n"
                f"  actual   {actual}"
            )
        print(f"{relative_text}: OK")
        checked += 1

    if checked == 0:
        raise SystemExit("checksum manifest is empty")
    print(f"verified {checked} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
