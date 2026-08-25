"""Reader/writer helpers for the deterministic PICKLE low-bit model container."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"PKLMQ2\x00\x01"
LEVELS = np.asarray([-1.510418, -0.452780, 0.452780, 1.510418], dtype=np.float32)
Q3_LEVELS = np.asarray(
    [-2.151933, -1.343899, -0.755999, -0.2450922, 0.2450922, 0.755999, 1.343899, 2.151933],
    dtype=np.float32,
)
Q4_LEVELS = np.asarray(
    [
        -1.0,
        -0.6961928,
        -0.52507305,
        -0.39491749,
        -0.28444138,
        -0.18477343,
        -0.09105,
        0.0,
        0.0795803,
        0.1609302,
        0.2461123,
        0.3379152,
        0.4407098,
        0.562617,
        0.7229568,
        1.0,
    ],
    dtype=np.float32,
)


def read_container(path: str | Path) -> tuple[dict[str, Any], bytes]:
    raw = Path(path).read_bytes()
    if raw[: len(MAGIC)] != MAGIC:
        raise ValueError("not a PICKLE Q2 container")
    manifest_length = struct.unpack_from("<I", raw, len(MAGIC))[0]
    manifest_start = len(MAGIC) + 4
    manifest_end = manifest_start + manifest_length
    manifest = json.loads(raw[manifest_start:manifest_end].decode("utf-8"))
    return manifest, raw[manifest_end:]


def unpack_codes(packed: bytes, count: int) -> np.ndarray:
    source = np.frombuffer(packed, dtype=np.uint8)
    codes = np.empty(source.size * 4, dtype=np.uint8)
    codes[0::4] = source & 0x03
    codes[1::4] = (source >> 2) & 0x03
    codes[2::4] = (source >> 4) & 0x03
    codes[3::4] = (source >> 6) & 0x03
    return codes[:count]


def unpack_nibbles(packed: bytes, count: int) -> np.ndarray:
    source = np.frombuffer(packed, dtype=np.uint8)
    codes = np.empty(source.size * 2, dtype=np.uint8)
    codes[0::2] = source & 0x0F
    codes[1::2] = (source >> 4) & 0x0F
    return codes[:count]


def unpack_q3(packed: bytes, count: int) -> np.ndarray:
    source = np.frombuffer(packed, dtype=np.uint8)
    groups = math.ceil(count / 8)
    triples = np.zeros((groups, 3), dtype=np.uint32)
    available = source[: groups * 3].reshape(-1, 3)
    triples[:, 0] = available[:, 0]
    triples[:, 1] = available[:, 1]
    triples[:, 2] = available[:, 2]
    words = triples[:, 0] | triples[:, 1] << 8 | triples[:, 2] << 16
    codes = np.empty((groups, 8), dtype=np.uint8)
    for index in range(8):
        codes[:, index] = (words >> (index * 3)) & 7
    return codes.reshape(-1)[:count]


def decode_tensor(entry: dict[str, Any], payload: bytes) -> np.ndarray:
    start = int(entry["offset"])
    end = start + int(entry["length"])
    data = payload[start:end]
    shape = tuple(int(value) for value in entry["shape"])
    count = int(np.prod(shape, dtype=np.int64))
    kind = entry["kind"]
    if kind == "fp16":
        return np.frombuffer(data, dtype="<f2", count=count).astype(np.float32).reshape(shape)
    if kind not in {"q2_block", "q2_symmetric", "q3_block", "q4_block"}:
        raise ValueError(f"unsupported tensor kind: {kind}")

    group_size = int(entry["group_size"])
    groups = int(entry["groups"])
    scale_bytes = groups * (4 if kind == "q2_symmetric" else 2)
    padded_count = groups * group_size
    if kind == "q2_block":
        scales = np.frombuffer(data[:scale_bytes], dtype="<f2", count=groups).astype(np.float32)
        codes = unpack_codes(data[scale_bytes:], padded_count)
        values = LEVELS[codes] * np.repeat(scales, group_size)
    elif kind == "q2_symmetric":
        magnitudes = np.frombuffer(data[:scale_bytes], dtype="<f2", count=groups * 2).astype(
            np.float32
        ).reshape(groups, 2)
        codes = unpack_codes(data[scale_bytes:], padded_count).reshape(groups, group_size)
        levels = np.stack(
            (-magnitudes[:, 1], -magnitudes[:, 0], magnitudes[:, 0], magnitudes[:, 1]),
            axis=1,
        )
        values = np.take_along_axis(levels[:, None, :], codes[:, :, None], axis=2)[
            :, :, 0
        ].reshape(-1)
    elif kind == "q3_block":
        scales = np.frombuffer(data[:scale_bytes], dtype="<f2", count=groups).astype(np.float32)
        codes = unpack_q3(data[scale_bytes:], padded_count)
        values = Q3_LEVELS[codes] * np.repeat(scales, group_size)
    else:
        scales = np.frombuffer(data[:scale_bytes], dtype="<f2", count=groups).astype(np.float32)
        codes = unpack_nibbles(data[scale_bytes:], padded_count)
        values = Q4_LEVELS[codes] * np.repeat(scales, group_size)
    return values[:count].reshape(shape)


def embedded_file(manifest: dict[str, Any], payload: bytes, name: str) -> bytes:
    for entry in manifest["embedded_files"]:
        if entry["name"] == name:
            start = int(entry["offset"])
            return payload[start : start + int(entry["length"])]
    raise KeyError(name)
