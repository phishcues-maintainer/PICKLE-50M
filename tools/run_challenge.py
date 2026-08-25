#!/usr/bin/env python3
"""One-command, machine-readable challenge runner for size, speed, parity, and retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, destination: Path, expected: str | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)
    if expected and sha256(destination) != expected.lower():
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch after downloading {url}")


def command(arguments: list[str], cwd: Path) -> tuple[str, float]:
    started = time.perf_counter()
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(arguments)}\n{completed.stderr}"
        )
    return completed.stdout, elapsed


def json_command(arguments: list[str], cwd: Path) -> tuple[dict[str, Any], float]:
    stdout, elapsed = command(arguments, cwd)
    return json.loads(stdout), elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prompt", default="The history of computing began.")
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--kernel", choices=["auto", "scalar", "avx2"], default="auto")
    parser.add_argument("--runtime-url")
    parser.add_argument("--runtime-sha256")
    parser.add_argument("--model-url")
    parser.add_argument("--model-sha256")
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--parity-tokens",
        default="4068,793,3064,728,1178,98,1334,885,2079",
    )
    parser.add_argument("--retrieval-tokens", type=int, default=100_000)
    parser.add_argument("--retrieval-questions", type=int, default=280)
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--hf-checkpoint", type=Path)
    parser.add_argument("--run-lm-eval", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    runtime = args.runtime if args.runtime.is_absolute() else root / args.runtime
    model = args.model if args.model.is_absolute() else root / args.model
    if args.runtime_url and not runtime.exists():
        fetch(args.runtime_url, runtime, args.runtime_sha256)
    if args.model_url and not model.exists():
        fetch(args.model_url, model, args.model_sha256)
    if not runtime.exists() or not model.exists():
        parser.error("runtime and model must exist or have download URLs")
    if args.runtime_sha256 and sha256(runtime) != args.runtime_sha256.lower():
        parser.error("runtime checksum mismatch")
    if args.model_sha256 and sha256(model) != args.model_sha256.lower():
        parser.error("model checksum mismatch")

    common = [
        str(runtime.resolve()),
        "--model",
        str(model.resolve()),
        "--threads",
        str(args.threads),
        "--kernel",
        args.kernel,
    ]
    info, info_wall = json_command([common[0], "model-info", *common[1:]], root)
    benchmark, benchmark_wall = json_command(
        [
            common[0],
            "model-bench",
            *common[1:],
            "--prompt",
            args.prompt,
            "--new-tokens",
            str(args.new_tokens),
            "--iterations",
            str(args.iterations),
        ],
        root,
    )

    fuzz_path = args.out.parent / "native-mutation-test.json"
    _, fuzz_wall = command(
        [
            sys.executable,
            str(root / "tools" / "fuzz_native.py"),
            "--runtime",
            str(runtime),
            "--model",
            str(model),
            "--out",
            str(fuzz_path),
        ],
        root,
    )
    fuzz = json.loads(fuzz_path.read_text(encoding="utf-8"))

    tokenizer = None
    if args.vocab or args.tokenizer:
        tokenizer_path = args.out.parent / "tokenizer-parity.json"
        _, tokenizer_wall = command(
            [
                sys.executable,
                str(root / "tools" / "validate_tokenizer.py"),
                "--runtime",
                str(runtime),
                "--model",
                str(model),
                "--vocab" if args.vocab else "--tokenizer",
                str(args.vocab or args.tokenizer),
                "--out",
                str(tokenizer_path),
            ],
            root,
        )
        tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        tokenizer["runner_wall_seconds"] = tokenizer_wall

    native_parity = None
    if args.reference:
        parity_path = args.out.parent / "native-parity.json"
        _, parity_wall = command(
            [
                sys.executable,
                str(root / "tools" / "validate_native.py"),
                "--runtime",
                str(runtime),
                "--model",
                str(model),
                "--reference",
                str(args.reference),
                "--tokens",
                args.parity_tokens,
                "--new-tokens",
                "16",
                "--json",
                str(parity_path),
            ],
            root,
        )
        native_parity = json.loads(parity_path.read_text(encoding="utf-8"))
        native_parity["runner_wall_seconds"] = parity_wall

    retrieval = None
    if not args.skip_retrieval:
        retrieval_root = args.out.parent / "challenge-retrieval"
        retrieval_root.mkdir(parents=True, exist_ok=True)
        command(
            [
                str(runtime.resolve()),
                "generate",
                "--out",
                str(retrieval_root),
                "--tokens",
                str(args.retrieval_tokens),
                "--questions",
                str(args.retrieval_questions),
                "--seed",
                "20260824",
            ],
            root,
        )
        index_json = retrieval_root / "index-result.json"
        index_result, _ = json_command(
            [
                str(runtime.resolve()),
                "index",
                "--archive",
                str(retrieval_root / "archive.txt"),
                "--out",
                str(retrieval_root / "archive.idx"),
                "--json",
                str(index_json),
            ],
            root,
        )
        bench_result, _ = json_command(
            [
                str(runtime.resolve()),
                "bench",
                "--archive",
                str(retrieval_root / "archive.txt"),
                "--index",
                str(retrieval_root / "archive.idx"),
                "--bank",
                str(retrieval_root / "bank.tsv"),
                "--iterations",
                str(args.iterations),
                "--threads",
                str(args.threads),
            ],
            root,
        )
        retrieval = {"index": index_result, "benchmark": bench_result}

    lm_eval = None
    if args.run_lm_eval:
        if not args.hf_checkpoint:
            parser.error("--run-lm-eval requires --hf-checkpoint")
        lm_output = args.out.parent / "lm-eval"
        _, lm_wall = command(
            [
                sys.executable,
                "-m",
                "lm_eval",
                "run",
                "--model",
                "hf",
                "--model_args",
                f"pretrained={args.hf_checkpoint}",
                "--tasks",
                "piqa,hellaswag,arc_easy,arc_challenge",
                "--num_fewshot",
                "0",
                "--batch_size",
                "auto",
                "--output_path",
                str(lm_output),
            ],
            root,
        )
        lm_eval = {"output_directory": str(lm_output), "runner_wall_seconds": lm_wall}

    result = {
        "format": "pickle-challenge-result-v1",
        "artifacts": {
            "model": {"path": str(model), "bytes": model.stat().st_size, "sha256": sha256(model)},
            "runtime": {
                "path": str(runtime),
                "bytes": runtime.stat().st_size,
                "sha256": sha256(runtime),
            },
            "combined_bytes": model.stat().st_size + runtime.stat().st_size,
            "combined_mib": (model.stat().st_size + runtime.stat().st_size) / (1024**2),
        },
        "model_info": info,
        "decode_benchmark": benchmark,
        "tokenizer_parity": tokenizer,
        "native_parity": native_parity,
        "loader_mutation_test": fuzz,
        "retrieval": retrieval,
        "lm_eval": lm_eval,
        "orchestration_wall_seconds": {
            "model_info": info_wall,
            "decode_benchmark": benchmark_wall,
            "loader_mutation_test": fuzz_wall,
        },
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
