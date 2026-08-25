#!/usr/bin/env python3
"""Export audited PKLM weights into the self-contained native runtime format.

This does not requantize or otherwise alter a tensor payload. It orders the
tensors as the native runtime expects and embeds either a compact TokenMonster
vocabulary or a compact byte-level BPE tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

from pklm_format import read_container


MAGIC = b"PKNATV4\0"
TOKENIZER_MAGIC = b"TMC1"
BPE_MAGIC = b"BPE1"
FLAG_TIED_EMBEDDINGS = 1 << 0
FLAG_DEFAULT_ADD_BOS = 1 << 1
MISSING_U24 = 0xFFFFFF
MISSING_U13 = 0x1FFF
MISSING_U16 = 0xFFFF
FLAG_VALUES = [1, 3, 4, 5, 16, 17, 128, 131, 132, 133, 136, 140, 152, 165]


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
    layout.append(("model.norm.weight", "fp16", (hidden,)))
    if not bool(config.get("tie_word_embeddings", False)):
        layout.append(("lm_head.weight", "quantized", (vocab, hidden)))
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


def gpt2_byte_encoder() -> dict[int, str]:
    values = list(range(ord("!"), ord("~") + 1))
    values += list(range(ord("¡"), ord("¬") + 1))
    values += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = values.copy()
    extra = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return dict(zip(values, map(chr, codepoints), strict=True))


def compact_byte_bpe(data: bytes, expected_vocab_size: int) -> bytes:
    document = json.loads(data.decode("utf-8"))
    model = document.get("model", {})
    pre = document.get("pre_tokenizer", {})
    decoder = document.get("decoder", {})
    if (
        model.get("type") != "BPE"
        or pre.get("type") != "ByteLevel"
        or decoder.get("type") != "ByteLevel"
        or bool(pre.get("add_prefix_space", False))
    ):
        raise ValueError("native BPE export requires GPT-2 byte-level BPE without prefix space")

    vocab_by_text = {str(key): int(value) for key, value in model["vocab"].items()}
    if len(vocab_by_text) != expected_vocab_size or set(vocab_by_text.values()) != set(
        range(expected_vocab_size)
    ):
        raise ValueError("BPE vocabulary IDs must be contiguous and match the model")
    vocab = [""] * expected_vocab_size
    for piece, token_id in vocab_by_text.items():
        vocab[token_id] = piece

    encoder = gpt2_byte_encoder()
    inverse = {character: byte for byte, character in encoder.items()}
    specials = {
        int(entry["id"]): str(entry["content"])
        for entry in document.get("added_tokens", [])
        if bool(entry.get("special", False))
    }
    if any(token_id >= expected_vocab_size for token_id in specials):
        raise ValueError("added BPE token is outside the model vocabulary")

    byte_ids = []
    unknown_id = vocab_by_text.get(str(model.get("unk_token", "")))
    for byte in range(256):
        piece = encoder[byte]
        if piece not in vocab_by_text and unknown_id is None:
            raise ValueError(f"BPE vocabulary lacks byte token {byte}")
        byte_ids.append(vocab_by_text.get(piece, unknown_id))

    reverse: list[bytes] = []
    for token_id, piece in enumerate(vocab):
        if token_id in specials:
            raw = specials[token_id].encode("utf-8")
        else:
            try:
                raw = bytes(inverse[character] for character in piece)
            except KeyError as error:
                raise ValueError(f"BPE token {token_id} is not byte-level") from error
        if len(raw) > 0xFFFF:
            raise ValueError("BPE token is too long for compact format")
        reverse.append(raw)

    merge_rows: list[tuple[int, int, int]] = []
    for rank, item in enumerate(model.get("merges", [])):
        if isinstance(item, list):
            left, right = map(str, item)
        else:
            left, right = str(item).split(" ", 1)
        merged = left + right
        try:
            merge_rows.append(
                (vocab_by_text[left], vocab_by_text[right], vocab_by_text[merged])
            )
        except KeyError as error:
            raise ValueError(f"invalid BPE merge at rank {rank}") from error

    output = bytearray(BPE_MAGIC)
    output.extend(struct.pack("<IIH", expected_vocab_size, len(merge_rows), len(specials)))
    output.extend(struct.pack("<256H", *byte_ids))
    for raw in reverse:
        output.extend(struct.pack("<H", len(raw)))
        output.extend(raw)
    for left, right, merged in merge_rows:
        output.extend(struct.pack("<3H", left, right, merged))
    for token_id, text in sorted(specials.items()):
        raw = text.encode("utf-8")
        output.extend(struct.pack("<HH", token_id, len(raw)))
        output.extend(raw)
    return bytes(output)


def export(source: Path, config_path: Path, tokenizer_path: Path, output: Path) -> None:
    manifest, payload = read_container(source)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer_source = tokenizer_path.read_bytes()
    token_ids = {
        "bos": int(config["bos_token_id"]),
        "eos": int(config["eos_token_id"]),
        "pad": int(config["pad_token_id"]),
    }
    if tokenizer_path.suffix.lower() == ".json" or tokenizer_source.lstrip().startswith(b"{"):
        tokenizer = compact_byte_bpe(tokenizer_source, int(config["vocab_size"]))
        tokenizer_name = "byte-level-bpe"
        document = json.loads(tokenizer_source.decode("utf-8"))
        vocab = {str(key): int(value) for key, value in document["model"]["vocab"].items()}
        tokenizer_config_path = tokenizer_path.with_name("tokenizer_config.json")
        if tokenizer_config_path.exists():
            tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
            for short, field in (("bos", "bos_token"), ("eos", "eos_token"), ("pad", "pad_token")):
                value = tokenizer_config.get(field)
                content = value.get("content") if isinstance(value, dict) else value
                if isinstance(content, str) and content in vocab:
                    token_ids[short] = vocab[content]
    else:
        tokenizer = compact_tokenmonster(tokenizer_source, int(config["vocab_size"]))
        tokenizer_name = "TokenMonster"
    entries = {entry["name"]: entry for entry in manifest["tensors"]}
    layout = expected_layout(config)
    group_sizes = {
        int(entry["group_size"])
        for entry in entries.values()
        if entry["kind"] != "fp16"
    }
    if len(group_sizes) != 1:
        raise ValueError(f"native format requires one quantization group size: {group_sizes}")
    group_size = group_sizes.pop()

    if set(entries) != {name for name, _, _ in layout}:
        missing = sorted({name for name, _, _ in layout} - set(entries))
        extra = sorted(set(entries) - {name for name, _, _ in layout})
        raise ValueError(f"tensor set mismatch; missing={missing}, extra={extra}")

    body = bytearray()
    tensor_bytes = 0
    kind_codes = {"fp16": 0, "q2_block": 2, "q2_symmetric": 3, "q3_block": 5, "q4_block": 4}
    for name, expected_kind, shape in layout:
        entry = entries[name]
        actual_kind = str(entry["kind"])
        actual_shape = tuple(int(value) for value in entry["shape"])
        kind_valid = actual_kind == "fp16" if expected_kind == "fp16" else actual_kind in {
            "q2_block",
            "q2_symmetric",
            "q3_block",
            "q4_block",
        }
        if not kind_valid or actual_shape != shape:
            raise ValueError(
                f"unexpected tensor layout for {name}: "
                f"{actual_kind} {actual_shape}, expected {expected_kind} {shape}"
            )
        if actual_kind != "fp16" and int(entry["group_size"]) != group_size:
            raise ValueError(f"unexpected group size for {name}")
        start = int(entry["offset"])
        end = start + int(entry["length"])
        raw = payload[start:end]
        if len(raw) != int(entry["length"]):
            raise ValueError(f"truncated payload for {name}")
        body.append(kind_codes[actual_kind])
        body.extend(raw)
        tensor_bytes += len(raw)

    body.extend(struct.pack("<I", len(tokenizer)))
    body.extend(tokenizer)
    body_digest = hashlib.sha256(body).digest()
    flags = 0
    if bool(config.get("tie_word_embeddings", False)):
        flags |= FLAG_TIED_EMBEDDINGS
    # TokenMonster models historically insert BOS; byte-level BPE follows the
    # upstream tokenizer, whose post-processor does not add one.
    if tokenizer_name == "TokenMonster":
        flags |= FLAG_DEFAULT_ADD_BOS
    rope_theta = float(
        config.get("rope_theta", config.get("rope_parameters", {}).get("rope_theta", 10000.0))
    )
    header_fields = MAGIC + struct.pack(
        "<12I2fIQ",
        group_size,
        int(config["num_hidden_layers"]),
        int(config["hidden_size"]),
        int(config["intermediate_size"]),
        int(config["vocab_size"]),
        int(config["num_attention_heads"]),
        int(config["num_key_value_heads"]),
        int(config["head_dim"]),
        int(config["max_position_embeddings"]),
        token_ids["bos"],
        token_ids["eos"],
        token_ids["pad"],
        float(config["rms_norm_eps"]),
        rope_theta,
        flags,
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
                "format": "pickle-native-model-v4",
                "source": str(source),
                "output": str(output),
                "tensor_count": len(layout),
                "tensor_bytes": tensor_bytes,
                "tokenizer": tokenizer_name,
                "tokenizer_bytes": len(tokenizer),
                "tied_embeddings": bool(flags & FLAG_TIED_EMBEDDINGS),
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
    parser.add_argument("--vocab", "--tokenizer", dest="tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.config, args.tokenizer, args.out)


if __name__ == "__main__":
    main()
