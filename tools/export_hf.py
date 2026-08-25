"""Export a PICKLE artifact as a temporary Hugging Face checkpoint for evaluation.

The expanded directory is an evaluation adapter, not the deployment artifact. Its
weights are exactly the values reconstructed from the compressed container.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from pklm_format import decode_tensor, embedded_file, read_container


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    manifest, payload = read_container(args.artifact)
    args.out.mkdir(parents=True, exist_ok=True)
    for entry in manifest["embedded_files"]:
        name = str(entry["name"])
        (args.out / name).write_bytes(embedded_file(manifest, payload, name))

    special = {"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>"}
    (args.out / "special_tokens_map.json").write_text(
        json.dumps(special, indent=2) + "\n", encoding="utf-8"
    )
    if not (args.out / "generation_config.json").exists():
        config = json.loads((args.out / "config.json").read_text(encoding="utf-8"))
        generation = {
            "bos_token_id": config["bos_token_id"],
            "eos_token_id": config["eos_token_id"],
            "pad_token_id": config["pad_token_id"],
        }
        (args.out / "generation_config.json").write_text(
            json.dumps(generation, indent=2) + "\n", encoding="utf-8"
        )

    state = {}
    for index, entry in enumerate(manifest["tensors"], 1):
        array = np.array(decode_tensor(entry, payload), dtype=np.float32, copy=True)
        state[entry["name"]] = torch.from_numpy(array).contiguous()
        if index % 25 == 0 or index == len(manifest["tensors"]):
            print(f"expanded {index}/{len(manifest['tensors'])} tensors", flush=True)
    save_file(
        state,
        args.out / "model.safetensors",
        metadata={
            "format": "pt",
            "base_model": str(manifest["base_model"]),
            "compressed_artifact": args.artifact.name,
        },
    )
    (args.out / "pickle_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.out),
                "expanded_weight_bytes": (args.out / "model.safetensors").stat().st_size,
                "parameters": manifest["parameters"],
                "warning": "expanded evaluation adapter; not the deployment size",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
