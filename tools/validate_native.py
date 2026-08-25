#!/usr/bin/env python3
"""Cross-check native low-bit inference against the expanded PyTorch adapter."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaForCausalLM


def parse_tokens(value: str) -> list[int]:
    tokens = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("at least one token ID is required")
    return tokens


def native_logits(runtime: Path, model: Path, tokens: list[int]) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="pickle-native-validation-") as temp:
        output = Path(temp) / "logits.bin"
        subprocess.run(
            [
                str(runtime.resolve()),
                "model-logits",
                "--model",
                str(model.resolve()),
                "--tokens",
                ",".join(map(str, tokens)),
                "--out",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return np.fromfile(output, dtype="<f4")


def native_generate(runtime: Path, model: Path, tokens: list[int], count: int) -> list[int]:
    process = subprocess.run(
        [
            str(runtime.resolve()),
            "model-generate",
            "--model",
            str(model.resolve()),
            "--tokens",
            ",".join(map(str, tokens)),
            "--new-tokens",
            str(count),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    line = next(
        (line for line in process.stdout.splitlines() if line.startswith("token_ids=")), None
    )
    if line is None:
        raise RuntimeError("native runtime did not print generated token IDs")
    return [int(value) for value in line.removeprefix("token_ids=").split(",") if value]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--tokens", type=parse_tokens, required=True)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    reference = LlamaForCausalLM.from_pretrained(
        args.reference,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).eval()
    input_ids = torch.tensor([args.tokens], dtype=torch.long)
    with torch.inference_mode():
        reference_logits = reference(input_ids=input_ids).logits[0, -1].float().cpu().numpy()
        generated = reference.generate(
            input_ids=input_ids,
            max_new_tokens=args.new_tokens,
            min_new_tokens=args.new_tokens,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=reference.config.pad_token_id,
            use_cache=True,
        )
    reference_tokens = generated[0, input_ids.shape[1] :].tolist()
    candidate_logits = native_logits(args.runtime, args.model, args.tokens)
    candidate_tokens = native_generate(
        args.runtime, args.model, args.tokens, args.new_tokens
    )

    if candidate_logits.shape != reference_logits.shape:
        raise SystemExit(
            f"logit shape mismatch: native={candidate_logits.shape}, "
            f"reference={reference_logits.shape}"
        )
    difference = candidate_logits - reference_logits
    denominator = float(np.linalg.norm(candidate_logits) * np.linalg.norm(reference_logits))
    cosine = float(np.dot(candidate_logits, reference_logits) / denominator)
    result = {
        "format": "pickle-native-validation-v1",
        "runtime": str(args.runtime),
        "native_model": str(args.model),
        "reference": str(args.reference),
        "prompt_token_ids": args.tokens,
        "logits": {
            "count": int(candidate_logits.size),
            "reference_argmax": int(reference_logits.argmax()),
            "native_argmax": int(candidate_logits.argmax()),
            "max_absolute_error": float(np.max(np.abs(difference))),
            "mean_absolute_error": float(np.mean(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(np.square(difference)))),
            "cosine_similarity": cosine,
        },
        "greedy_generation": {
            "tokens_checked": args.new_tokens,
            "exact_match": candidate_tokens == reference_tokens,
            "reference_token_ids": reference_tokens,
            "native_token_ids": candidate_tokens,
        },
    }
    result["valid"] = bool(
        result["logits"]["reference_argmax"] == result["logits"]["native_argmax"]
        and result["logits"]["max_absolute_error"] < 1e-3
        and result["logits"]["cosine_similarity"] > 0.99999
        and result["greedy_generation"]["exact_match"]
    )
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered, encoding="utf-8")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
