#!/usr/bin/env python3
"""Compare native Rust text encoding with the TokenMonster reference."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
import time
import unicodedata
from pathlib import Path

import tokenmonster


CASES = [
    "",
    "Hello world!",
    "Explain photosynthesis in two sentences.",
    "don't stop",
    "café déjà vu",
    "Cafe\u0301",
    "你好，世界",
    "emoji 🤖🚀 test",
    "line one\nline two",
    "abc.Def-42",
    "  multiple   spaces  ",
    "Greek Αλφα",
    "Arabic مرحبا",
    "math Ⅻ ½",
]


def native_tokens(runtime: Path, model: Path, *, prompt: str | None = None,
                  prompt_file: Path | None = None) -> list[int]:
    command = [str(runtime), "model-tokenize", "--model", str(model)]
    if prompt_file is not None:
        command.extend(["--prompt-file", str(prompt_file)])
    else:
        command.extend(["--prompt", prompt or ""])
    process = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    first_line = process.stdout.splitlines()[0]
    encoded = first_line.removeprefix("token_ids=")
    return [int(value) for value in encoded.split(",")] if encoded else []


def reference_tokens(vocab: object, prompt: str) -> list[int]:
    encoded = vocab.tokenize(prompt)
    return [] if encoded is None else [int(token) for token in encoded]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--tokenmonster-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.tokenmonster_dir:
        args.tokenmonster_dir.mkdir(parents=True, exist_ok=True)
        tokenmonster.set_local_directory(str(args.tokenmonster_dir))
    reference = tokenmonster.load(str(args.vocab))

    failures: list[dict[str, object]] = []
    checked_tokens = 0
    started = time.perf_counter()
    for prompt in CASES:
        expected = reference_tokens(reference, prompt)
        actual = native_tokens(args.runtime, args.model, prompt=prompt)
        checked_tokens += len(expected)
        if actual != expected:
            failures.append(
                {"prompt": prompt, "reference": expected, "native": actual}
            )

    rng = random.Random(20260824)
    fragments = [
        "hello", "WORLD", "camelCase", "42", "3.14159", "isn't", "A.B-C",
        "\n", "\t", "   ", "é", "\u0301", "🤖", "你", "Ⅻ", ".", ",", "!?",
        "/path?q=x",
    ]
    large_prompt = "".join(
        rng.choice(fragments) + rng.choice([" ", "", "\n"])
        for _ in range(12_000)
    )
    expected = reference_tokens(reference, large_prompt)
    with tempfile.TemporaryDirectory(prefix="pickle50-tokenizer-") as temp_dir:
        temp_path = Path(temp_dir)
        prompt_path = Path(temp_dir) / "prompt.txt"
        prompt_path.write_text(large_prompt, encoding="utf-8", newline="")
        actual = native_tokens(args.runtime, args.model, prompt_file=prompt_path)

        checked_tokens += len(expected)
        if actual != expected:
            first_difference = next(
                (
                    index
                    for index, pair in enumerate(zip(expected, actual))
                    if pair[0] != pair[1]
                ),
                min(len(expected), len(actual)),
            )
            failures.append(
                {
                    "prompt": "generated mixed-script corpus",
                    "reference_token_count": len(expected),
                    "native_token_count": len(actual),
                    "first_difference": first_difference,
                }
            )

        unicode_scalars = [
            chr(codepoint)
            for codepoint in range(0x110000)
            if not 0xD800 <= codepoint <= 0xDFFF
            and unicodedata.category(chr(codepoint))[0] in "LNM"
        ]
        unicode_prompt = "." + ".".join(unicode_scalars)
        unicode_path = temp_path / "unicode.txt"
        unicode_path.write_text(unicode_prompt, encoding="utf-8", newline="")
        expected = reference_tokens(reference, unicode_prompt)
        actual = native_tokens(args.runtime, args.model, prompt_file=unicode_path)
        checked_tokens += len(expected)
        if actual != expected:
            first_difference = next(
                (
                    index
                    for index, pair in enumerate(zip(expected, actual))
                    if pair[0] != pair[1]
                ),
                min(len(expected), len(actual)),
            )
            failures.append(
                {
                    "prompt": "Unicode letter/number/mark scalar probe",
                    "reference_token_count": len(expected),
                    "native_token_count": len(actual),
                    "first_difference": first_difference,
                }
            )

    result = {
        "format": "pickle-native-tokenizer-validation-v1",
        "reference": "TokenMonster Python package",
        "fixed_cases": len(CASES),
        "generated_corpus_characters": len(large_prompt),
        "unicode_scalars_checked": len(unicode_scalars),
        "reference_tokens_checked": checked_tokens,
        "exact": not failures,
        "failures": failures,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8", newline="\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
