#!/usr/bin/env python3
"""Export the audited PKLM weights into the fixed-layout native runtime format.

This does not requantize or otherwise alter a tensor payload. It orders the
tensors as the native runtime expects and appends the original TokenMonster
vocabulary so text encoding and decoding are available inside the runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

from pklm_format import read_container


MAGIC = b"PKNATV3\0"
GROUP_SIZE = 256
TOKENIZER_MAGIC = b"TMC1"
MISSING_U24 = 0xFFFFFF
MISSING_U13 = 0x1FFF
MISSING_U16 = 0xFFFF
FLAG_VALUES = [1, 3, 4, 5, 16, 17, 128, 131, 132, 133, 136, 140, 152, 165]


def tensor_order(layers: int) -> list[str]:
    names = ["model.embed_tokens.weight"]
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        names.extend(
            [
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            ]
        )
    names.extend(["model.norm.weight", "lm_head.weight"])
    return names


def expected_layout(config: dict) -> list[tuple[str, str, tuple[int, ...]]]:
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    vocab = int(config["vocab_size"])
    layers = int(config["num_hidden_layers"])
    kv_width = int(config["num_key_value_heads"]) * int(config["head_dim"])
    layout: list[tuple[str, str, tuple[int, ...]]] = [
        ("model.embed_tokens.weight", "quantized", (vocab, hidden))
    ]
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        layout.extend(
            [
                (f"{prefix}.input_layernorm.weight", "fp16", (hidden,)),
                (f"{prefix}.self_attn.q_proj.weight", "quantized", (hidden, hidden)),
                (f"{prefix}.self_attn.k_proj.weight", "quantized", (kv_width, hidden)),
                (f"{prefix}.self_attn.v_proj.weight", "quantized", (kv_width, hidden)),
                (f"{prefix}.self_attn.o_proj.weight", "quantized", (hidden, hidden)),
                (f"{prefix}.post_attention_layernorm.weight", "fp16", (hidden,)),
                (f"{prefix}.mlp.gate_proj.weight", "quantized", (intermediate, hidden)),
                (f"{prefix}.mlp.up_proj.weight", "quantized", (intermediate, hidden)),
                (f"{prefix}.mlp.down_proj.weight", "quantized", (hidden, intermediate)),
            ]
        )
    layout.extend(
        [
            ("model.norm.weight", "fp16", (hidden,)),
            ("lm_head.weight", "quantized", (vocab, hidden)),
        ]
    )
    return layout


def read_u24(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 3], "little")


def compact_tokenmonster(data: bytes, expected_vocab_size: int) -> bytes:
    if len(data) < 24 or data[0] > 2 or data[1] > 2:
        raise ValueError("not a valid TokenMonster vocabulary")
    vocab_size = read_u24(data, 11)
    reverse_count = read_u24(data, 14)
    info_count = read_u24(data, 17)
    unk = read_u24(data, 8)
    delete_token = read_u24(data, 20)
    if vocab_size != expected_vocab_size or reverse_count != expected_vocab_size:
        raise ValueError("TokenMonster and model vocabulary sizes differ")
    if vocab_size > MISSING_U16 or info_count >= MISSING_U13:
        raise ValueError("TokenMonster vocabulary is too large for compact metadata")

    output = bytearray(TOKENIZER_MAGIC)
    output.extend(data[0:3])
    output.append(data[23])
    output.extend(
        struct.pack(
            "<5H",
            vocab_size,
            reverse_count,
            info_count,
            MISSING_U16 if unk == MISSING_U24 else unk,
            MISSING_U16 if delete_token == MISSING_U24 else delete_token,
        )
    )

    cursor = 24
    seen_ids: set[int] = set()
    previous_token = b""
    for index in range(info_count):
        length = data[cursor]
        cursor += 1
        token = data[cursor : cursor + length]
        cursor += length
        flag = data[cursor]
        words = data[cursor + 1]
        alt1 = read_u24(data, cursor + 2)
        alt2 = read_u24(data, cursor + 5)
        token_id = read_u24(data, cursor + 8)
        cursor += 15
        if flag not in FLAG_VALUES or words > 7:
            raise ValueError("TokenMonster flags cannot be compacted")
        if (alt1 != MISSING_U24 and alt1 >= index) or (
            alt2 != MISSING_U24 and alt2 >= index
        ):
            raise ValueError("invalid TokenMonster alternative index")
        is_new = token_id not in seen_ids
        if is_new:
            if token_id != len(seen_ids):
                raise ValueError("TokenMonster first-use IDs are not sequential")
            seen_ids.add(token_id)
        packed = (
            length
            | (FLAG_VALUES.index(flag) << 6)
            | (words << 10)
            | ((MISSING_U13 if alt1 == MISSING_U24 else alt1) << 13)
            | ((MISSING_U13 if alt2 == MISSING_U24 else alt2) << 26)
            | (int(is_new) << 39)
        )
        output.extend(packed.to_bytes(5, "little"))
        prefix = 0
        while (
            prefix < len(previous_token)
            and prefix < len(token)
            and previous_token[prefix] == token[prefix]
        ):
            prefix += 1
        output.append(prefix)
        if not is_new:
            output.extend(struct.pack("<H", token_id))
        output.extend(token[prefix:])
        previous_token = token

    begin_byte = data[cursor : cursor + 256]
    if len(begin_byte) != 256:
        raise ValueError("truncated TokenMonster begin-byte table")
    output.extend(begin_byte)
    return bytes(output)


def export(source: Path, config_path: Path, vocab_path: Path, output: Path) -> None:
    manifest, payload = read_container(source)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tokenmonster_vocab = compact_tokenmonster(
        vocab_path.read_bytes(), int(config["vocab_size"])
    )
    entries = {entry["name"]: entry for entry in manifest["tensors"]}
    layout = expected_layout(config)

    if set(entries) != {name for name, _, _ in layout}:
        missing = sorted({name for name, _, _ in layout} - set(entries))
        extra = sorted(set(entries) - {name for name, _, _ in layout})
        raise ValueError(f"tensor set mismatch; missing={missing}, extra={extra}")

    body = bytearray()
    tensor_bytes = 0
    kind_codes = {"fp16": 0, "q2_block": 2, "q4_block": 4}
    for name, expected_kind, shape in layout:
        entry = entries[name]
        actual_kind = str(entry["kind"])
        actual_shape = tuple(int(value) for value in entry["shape"])
        kind_valid = actual_kind == "fp16" if expected_kind == "fp16" else actual_kind in {
            "q2_block",
            "q4_block",
        }
        if not kind_valid or actual_shape != shape:
            raise ValueError(
                f"unexpected tensor layout for {name}: "
                f"{actual_kind} {actual_shape}, expected {expected_kind} {shape}"
            )
        if actual_kind != "fp16" and int(entry["group_size"]) != GROUP_SIZE:
            raise ValueError(f"unexpected group size for {name}")
        start = int(entry["offset"])
        end = start + int(entry["length"])
        raw = payload[start:end]
        if len(raw) != int(entry["length"]):
            raise ValueError(f"truncated payload for {name}")
        body.append(kind_codes[actual_kind])
        body.extend(raw)
        tensor_bytes += len(raw)

    body.extend(struct.pack("<I", len(tokenmonster_vocab)))
    body.extend(tokenmonster_vocab)
    body_digest = hashlib.sha256(body).digest()
    header_fields = MAGIC + struct.pack(
        "<12I2fQ",
        GROUP_SIZE,
        int(config["num_hidden_layers"]),
        int(config["hidden_size"]),
        int(config["intermediate_size"]),
        int(config["vocab_size"]),
        int(config["num_attention_heads"]),
        int(config["num_key_value_heads"]),
        int(config["head_dim"]),
        int(config["max_position_embeddings"]),
        int(config["bos_token_id"]),
        int(config["eos_token_id"]),
        int(config["pad_token_id"]),
        float(config["rms_norm_eps"]),
        float(config["rope_theta"]),
        len(body),
    )
    authenticated_digest = hashlib.sha256(header_fields + body_digest).digest()
    header = header_fields + authenticated_digest

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as writer:
        writer.write(header)
        writer.write(body)

    print(
        json.dumps(
            {
                "format": "pickle-native-model-v3",
                "source": str(source),
                "output": str(output),
                "tensor_count": len(layout),
                "tensor_bytes": tensor_bytes,
                "tokenmonster_vocab_bytes": len(tokenmonster_vocab),
                "authenticated_body_bytes": len(body),
                "authenticated_body_sha256": body_digest.hex(),
                "authenticated_header_and_body_sha256": authenticated_digest.hex(),
                "output_bytes": output.stat().st_size,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.config, args.vocab, args.out)


if __name__ == "__main__":
    main()
