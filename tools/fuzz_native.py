#!/usr/bin/env python3
"""Deterministic mutation testing for the authenticated native model loader."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def run(runtime: Path, model: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(runtime.resolve()), "model-info", "--model", str(model.resolve()), "--threads", "1"],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    original = args.model.read_bytes()
    baseline = run(args.runtime, args.model)
    if baseline.returncode != 0:
        raise SystemExit(f"baseline model failed to load: {baseline.stderr}")

    mutations: list[tuple[str, bytes]] = []
    for length in sorted({0, 1, 7, 8, 63, 64, len(original) // 2, len(original) - 1}):
        mutations.append((f"truncate_{length}", original[:length]))
    for offset in sorted({0, 7, 8, 20, 64, 72, 96, len(original) // 2, len(original) - 1}):
        changed = bytearray(original)
        changed[offset] ^= 0x5A
        mutations.append((f"bit_flip_{offset}", bytes(changed)))
    mutations.append(("trailing_byte", original + b"X"))

    cases = []
    with tempfile.TemporaryDirectory(prefix="pickle50-fuzz-") as directory:
        root = Path(directory)
        for index, (name, data) in enumerate(mutations):
            path = root / f"{index:03}-{name}.bin"
            path.write_bytes(data)
            completed = run(args.runtime, path)
            rejected = completed.returncode != 0
            cases.append(
                {
                    "name": name,
                    "bytes": len(data),
                    "rejected": rejected,
                    "returncode": completed.returncode,
                    "stderr": completed.stderr.strip()[:300],
                }
            )

    result = {
        "format": "pickle-native-mutation-test-v1",
        "baseline_valid": True,
        "mutations": len(cases),
        "rejected": sum(int(case["rejected"]) for case in cases),
        "all_rejected": all(case["rejected"] for case in cases),
        "cases": cases,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    return 0 if result["all_rejected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
