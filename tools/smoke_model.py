"""Functional comparison of the source checkpoint and reconstructed low-bit artifact.

This dequantizes into ordinary float32 PyTorch weights. It validates the artifact and
reports quality drift; it is deliberately not presented as the final low-bit CPU runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from pklm_format import decode_tensor, embedded_file, read_container


EVAL_TEXT = """
The history of computing includes mechanical calculators, electronic circuits, and programmable
machines. A computer follows encoded instructions, stores intermediate results in memory, and
communicates through input and output devices. Modern systems combine processors, memory, storage,
and networks, while software provides reusable abstractions over the hardware. Reliable experiments
state their assumptions, preserve raw measurements, and make the complete procedure reproducible.
Plants use sunlight to convert water and carbon dioxide into chemical energy. This process releases
oxygen and supports most food chains on Earth. Scientific explanations become more useful when they
distinguish observations, models, and conclusions.
""".strip()


def load_quantized(artifact: Path) -> tuple[LlamaForCausalLM, dict]:
    manifest, payload = read_container(artifact)
    config_dict = json.loads(embedded_file(manifest, payload, "config.json").decode("utf-8"))
    config = LlamaConfig.from_dict(config_dict)
    model = LlamaForCausalLM(config)
    state = {}
    for index, entry in enumerate(manifest["tensors"], 1):
        array = np.array(decode_tensor(entry, payload), dtype=np.float32, copy=True)
        state[entry["name"]] = torch.from_numpy(array)
        if index % 25 == 0 or index == len(manifest["tensors"]):
            print(f"reconstructed {index}/{len(manifest['tensors'])} tensors", flush=True)
    missing, unexpected = model.load_state_dict(state, strict=True, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"state mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, manifest


def timed_forward(model: LlamaForCausalLM, input_ids: torch.Tensor) -> tuple[float, torch.Tensor, float]:
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(input_ids=input_ids, labels=input_ids)
    elapsed = time.perf_counter() - started
    return float(output.loss), output.logits.detach(), elapsed


def timed_generate(model: LlamaForCausalLM, prompt: torch.Tensor, new_tokens: int) -> tuple[torch.Tensor, float]:
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids=prompt,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=model.config.pad_token_id,
            use_cache=True,
        )
    return output, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--eval-tokens", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(
        args.source, trust_remote_code=True, local_files_only=True
    )
    ids = tokenizer.encode(EVAL_TEXT, add_special_tokens=False)[: args.eval_tokens]
    input_ids = torch.tensor([ids], dtype=torch.long)
    prompt_text = "The history of computing began"
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt = torch.tensor([[tokenizer.bos_token_id, *prompt_ids]], dtype=torch.long)

    source_started = time.perf_counter()
    source_model = LlamaForCausalLM.from_pretrained(
        args.source, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    source_load_s = time.perf_counter() - source_started
    compressed_started = time.perf_counter()
    compressed_model, manifest = load_quantized(args.artifact)
    compressed_load_s = time.perf_counter() - compressed_started

    source_loss, source_logits, source_forward_s = timed_forward(source_model, input_ids)
    compressed_loss, compressed_logits, compressed_forward_s = timed_forward(
        compressed_model, input_ids
    )
    source_top = source_logits[:, :-1].argmax(dim=-1)
    compressed_top = compressed_logits[:, :-1].argmax(dim=-1)
    top1_agreement = float((source_top == compressed_top).float().mean())
    source_flat = source_logits[:, :-1].float().reshape(-1)
    compressed_flat = compressed_logits[:, :-1].float().reshape(-1)
    logit_cosine = float(
        torch.nn.functional.cosine_similarity(source_flat, compressed_flat, dim=0)
    )

    source_generated, source_generate_s = timed_generate(source_model, prompt, args.new_tokens)
    compressed_generated, compressed_generate_s = timed_generate(
        compressed_model, prompt, args.new_tokens
    )
    source_new = source_generated[0, prompt.shape[1] :].tolist()
    compressed_new = compressed_generated[0, prompt.shape[1] :].tolist()

    result = {
        "format": "pickle-model-smoke-v1",
        "timing_scope": (
            "expanded FP32 reference path; not a native low-bit runtime; "
            "wall time is sensitive to host contention"
        ),
        "artifact": str(args.artifact),
        "parameters": manifest["parameters"],
        "threads": args.threads,
        "eval_tokens": len(ids),
        "source": {
            "load_seconds": source_load_s,
            "loss": source_loss,
            "perplexity": math.exp(min(source_loss, 50)),
            "forward_seconds": source_forward_s,
            "generate_seconds": source_generate_s,
            "generated_tokens": len(source_new),
            "tokens_per_second": len(source_new) / source_generate_s,
            "completion": tokenizer.decode(source_new, skip_special_tokens=True),
        },
        "compressed": {
            "load_seconds": compressed_load_s,
            "loss": compressed_loss,
            "perplexity": math.exp(min(compressed_loss, 50)),
            "forward_seconds": compressed_forward_s,
            "generate_seconds": compressed_generate_s,
            "generated_tokens": len(compressed_new),
            "tokens_per_second": len(compressed_new) / compressed_generate_s,
            "completion": tokenizer.decode(compressed_new, skip_special_tokens=True),
        },
        "drift": {
            "loss_delta": compressed_loss - source_loss,
            "top1_logit_agreement": top1_agreement,
            "logit_cosine": logit_cosine,
        },
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    print(rendered, end="")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
