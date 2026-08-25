#!/usr/bin/env python3
"""Compare scalar, single-thread AVX2, and multicore AVX2 decode paths."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def benchmark(
    runtime: Path,
    model: Path,
    tokens: str,
    steps: int,
    iterations: int,
    threads: int,
    kernel: str,
) -> dict:
    completed = subprocess.run(
        [
            str(runtime.resolve()),
            "model-bench",
            "--model",
            str(model.resolve()),
            "--tokens",
            tokens,
            "--new-tokens",
            str(steps),
            "--iterations",
            str(iterations),
            "--threads",
            str(threads),
            "--kernel",
            kernel,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip())
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--tokens", default="4068,793,3064,728,1178,98,1334,885,2079"
    )
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    scalar = benchmark(
        args.runtime, args.model, args.tokens, args.new_tokens, args.iterations, 1, "scalar"
    )
    avx2 = benchmark(
        args.runtime, args.model, args.tokens, args.new_tokens, args.iterations, 1, "avx2"
    )
    multicore = benchmark(
        args.runtime,
        args.model,
        args.tokens,
        args.new_tokens,
        args.iterations,
        args.threads,
        "avx2",
    )
    scalar_rate = float(scalar["tokens_per_second"])
    result = {
        "format": "pickle-kernel-profile-v1",
        "scalar_1_thread": scalar,
        "avx2_1_thread": avx2,
        "avx2_multicore": multicore,
        "speedup": {
            "avx2_vs_scalar": float(avx2["tokens_per_second"]) / scalar_rate,
            "multicore_avx2_vs_scalar": float(multicore["tokens_per_second"]) / scalar_rate,
        },
        "generated_tokens_exact_match": (
            scalar["generated_token_ids_last_iteration"]
            == avx2["generated_token_ids_last_iteration"]
            == multicore["generated_token_ids_last_iteration"]
        ),
    }
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
