"""Compress a supported Llama checkpoint into a deterministic low-bit artifact.

This is post-training compression. It performs no gradient updates and consumes no
training or calibration examples.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

from pklm_format import LEVELS, MAGIC, Q3_LEVELS, Q4_LEVELS, read_container


EMBEDDED_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenmonster.vocab",
    "tokenmonster_hf.py",
    "generation_config.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pack_codes(codes: np.ndarray) -> bytes:
    padding = (-codes.size) % 4
    if padding:
        codes = np.pad(codes, (0, padding), constant_values=0)
    codes = codes.reshape(-1, 4).astype(np.uint8, copy=False)
    packed = codes[:, 0] | (codes[:, 1] << 2) | (codes[:, 2] << 4) | (codes[:, 3] << 6)
    return packed.tobytes()


def pack_nibbles(codes: np.ndarray) -> bytes:
    padding = (-codes.size) % 2
    if padding:
        codes = np.pad(codes, (0, padding), constant_values=0)
    codes = codes.reshape(-1, 2).astype(np.uint8, copy=False)
    return (codes[:, 0] | (codes[:, 1] << 4)).tobytes()


def pack_q3(codes: np.ndarray) -> bytes:
    padding = (-codes.size) % 8
    if padding:
        codes = np.pad(codes, (0, padding), constant_values=0)
    values = codes.reshape(-1, 8).astype(np.uint32, copy=False)
    words = np.zeros(values.shape[0], dtype=np.uint32)
    for index in range(8):
        words |= values[:, index] << (index * 3)
    packed = np.empty((values.shape[0], 3), dtype=np.uint8)
    packed[:, 0] = words & 0xFF
    packed[:, 1] = (words >> 8) & 0xFF
    packed[:, 2] = (words >> 16) & 0xFF
    return packed.tobytes()


def quantize_q2(
    array: np.ndarray, group_size: int, refinement_rounds: int = 2
) -> tuple[bytes, dict[str, float | int]]:
    original = np.asarray(array, dtype=np.float32).reshape(-1)
    count = original.size
    groups = math.ceil(count / group_size)
    padded = np.zeros(groups * group_size, dtype=np.float32)
    padded[:count] = original
    blocks = padded.reshape(groups, group_size)

    rms = np.sqrt(np.mean(blocks * blocks, axis=1, dtype=np.float32))
    scales = np.maximum(rms, np.float32(1e-12))
    codes = np.zeros_like(blocks, dtype=np.uint8)

    # Lloyd-style assignment with a fixed symmetric four-level codebook and a
    # least-squares scale per block. Two refinement rounds are deterministic.
    for _ in range(refinement_rounds):
        distances = np.abs(blocks[:, :, None] - scales[:, None, None] * LEVELS[None, None, :])
        codes = np.argmin(distances, axis=2).astype(np.uint8)
        selected = LEVELS[codes]
        numerator = np.sum(blocks * selected, axis=1, dtype=np.float32)
        denominator = np.sum(selected * selected, axis=1, dtype=np.float32)
        scales = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, 1e-12),
            where=denominator > 0,
        )

    selected = LEVELS[codes]
    reconstructed = (selected * scales[:, None]).reshape(-1)[:count]
    error = reconstructed - original
    squared_error = float(np.dot(error.astype(np.float64), error.astype(np.float64)))
    squared_original = float(np.dot(original.astype(np.float64), original.astype(np.float64)))
    dot = float(np.dot(original.astype(np.float64), reconstructed.astype(np.float64)))
    squared_reconstructed = float(
        np.dot(reconstructed.astype(np.float64), reconstructed.astype(np.float64))
    )

    payload = scales.astype("<f2").tobytes() + pack_codes(codes.reshape(-1))
    metrics: dict[str, float | int] = {
        "groups": groups,
        "squared_error": squared_error,
        "squared_original": squared_original,
        "dot": dot,
        "squared_reconstructed": squared_reconstructed,
    }
    return payload, metrics


def quantize_q2_symmetric(
    array: np.ndarray, group_size: int, refinement_rounds: int = 2
) -> tuple[bytes, dict[str, float | int]]:
    """Two-bit blocks with two learned symmetric magnitudes per group."""
    original = np.asarray(array, dtype=np.float32).reshape(-1)
    count = original.size
    groups = math.ceil(count / group_size)
    padded = np.zeros(groups * group_size, dtype=np.float32)
    padded[:count] = original
    blocks = padded.reshape(groups, group_size)
    absolute = np.abs(blocks)
    rms = np.maximum(
        np.sqrt(np.mean(blocks * blocks, axis=1, dtype=np.float32)),
        np.float32(1e-12),
    )
    inner = rms * np.float32(abs(LEVELS[1]))
    outer = rms * np.float32(abs(LEVELS[0]))
    outer_mask = np.zeros_like(blocks, dtype=bool)
    for _ in range(refinement_rounds):
        outer_mask = absolute >= ((inner + outer) * np.float32(0.5))[:, None]
        inner_count = np.sum(~outer_mask, axis=1)
        outer_count = np.sum(outer_mask, axis=1)
        inner = np.divide(
            np.sum(np.where(~outer_mask, absolute, 0.0), axis=1, dtype=np.float32),
            inner_count,
            out=inner,
            where=inner_count > 0,
        )
        outer = np.divide(
            np.sum(np.where(outer_mask, absolute, 0.0), axis=1, dtype=np.float32),
            outer_count,
            out=outer,
            where=outer_count > 0,
        )
    # Evaluate the exact FP16 codebook that is serialized.
    magnitudes = np.stack((inner, outer), axis=1).astype("<f2")
    exact = magnitudes.astype(np.float32)
    negative = blocks < 0
    codes = np.where(outer_mask, np.where(negative, 0, 3), np.where(negative, 1, 2)).astype(
        np.uint8
    )
    reconstructed_blocks = np.where(outer_mask, exact[:, 1, None], exact[:, 0, None])
    reconstructed_blocks = np.where(negative, -reconstructed_blocks, reconstructed_blocks)
    reconstructed = reconstructed_blocks.reshape(-1)[:count]
    error = reconstructed - original
    original64 = original.astype(np.float64)
    reconstructed64 = reconstructed.astype(np.float64)
    error64 = error.astype(np.float64)
    payload = magnitudes.tobytes() + pack_codes(codes.reshape(-1))
    return payload, {
        "groups": groups,
        "squared_error": float(np.dot(error64, error64)),
        "squared_original": float(np.dot(original64, original64)),
        "dot": float(np.dot(original64, reconstructed64)),
        "squared_reconstructed": float(np.dot(reconstructed64, reconstructed64)),
    }


def quantize_q3(
    array: np.ndarray, group_size: int, refinement_rounds: int = 2
) -> tuple[bytes, dict[str, float | int]]:
    original = np.asarray(array, dtype=np.float32).reshape(-1)
    count = original.size
    groups = math.ceil(count / group_size)
    padded = np.zeros(groups * group_size, dtype=np.float32)
    padded[:count] = original
    blocks = padded.reshape(groups, group_size)
    rms = np.maximum(np.sqrt(np.mean(blocks * blocks, axis=1)), np.float32(1e-12))
    scales = rms
    codes = np.zeros_like(blocks, dtype=np.uint8)
    for _ in range(refinement_rounds):
        distances = np.abs(
            blocks[:, :, None] - scales[:, None, None] * Q3_LEVELS[None, None, :]
        )
        codes = np.argmin(distances, axis=2).astype(np.uint8)
        selected = Q3_LEVELS[codes]
        numerator = np.sum(blocks * selected, axis=1, dtype=np.float32)
        denominator = np.sum(selected * selected, axis=1, dtype=np.float32)
        scales = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, 1e-12),
            where=denominator > 0,
        )
    exact_scales = scales.astype("<f2")
    reconstructed = (Q3_LEVELS[codes] * exact_scales.astype(np.float32)[:, None]).reshape(-1)[:count]
    original64 = original.astype(np.float64)
    reconstructed64 = reconstructed.astype(np.float64)
    error64 = reconstructed64 - original64
    return exact_scales.tobytes() + pack_q3(codes.reshape(-1)), {
        "groups": groups,
        "squared_error": float(np.dot(error64, error64)),
        "squared_original": float(np.dot(original64, original64)),
        "dot": float(np.dot(original64, reconstructed64)),
        "squared_reconstructed": float(np.dot(reconstructed64, reconstructed64)),
    }


def quantize_q4(
    array: np.ndarray, group_size: int, refinement_rounds: int = 2
) -> tuple[bytes, dict[str, float | int]]:
    original = np.asarray(array, dtype=np.float32).reshape(-1)
    count = original.size
    groups = math.ceil(count / group_size)
    padded = np.zeros(groups * group_size, dtype=np.float32)
    padded[:count] = original
    blocks = padded.reshape(groups, group_size)

    scales = np.maximum(np.max(np.abs(blocks), axis=1), np.float32(1e-12))
    codes = np.zeros_like(blocks, dtype=np.uint8)
    for _ in range(refinement_rounds):
        distances = np.abs(
            blocks[:, :, None] - scales[:, None, None] * Q4_LEVELS[None, None, :]
        )
        codes = np.argmin(distances, axis=2).astype(np.uint8)
        selected = Q4_LEVELS[codes]
        numerator = np.sum(blocks * selected, axis=1, dtype=np.float32)
        denominator = np.sum(selected * selected, axis=1, dtype=np.float32)
        scales = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, 1e-12),
            where=denominator > 0,
        )

    selected = Q4_LEVELS[codes]
    reconstructed = (selected * scales[:, None]).reshape(-1)[:count]
    error = reconstructed - original
    squared_error = float(np.dot(error.astype(np.float64), error.astype(np.float64)))
    squared_original = float(np.dot(original.astype(np.float64), original.astype(np.float64)))
    dot = float(np.dot(original.astype(np.float64), reconstructed.astype(np.float64)))
    squared_reconstructed = float(
        np.dot(reconstructed.astype(np.float64), reconstructed.astype(np.float64))
    )
    payload = scales.astype("<f2").tobytes() + pack_nibbles(codes.reshape(-1))
    return payload, {
        "groups": groups,
        "squared_error": squared_error,
        "squared_original": squared_original,
        "dot": dot,
        "squared_reconstructed": squared_reconstructed,
    }


def select_q4_by_error_budget(
    model_path: Path,
    group_size: int,
    budget_bytes: int,
    forced_patterns: list[str],
    refinement_rounds: int,
) -> tuple[set[str], dict[str, object]]:
    """Choose whole tensors for Q4 using exact knapsack over reconstruction-error reduction."""
    if budget_bytes < 0:
        raise ValueError("Q4 extra-byte budget cannot be negative")
    candidates: list[dict[str, object]] = []
    with safe_open(model_path, framework="pt", device="cpu") as model:
        names = sorted(model.keys())
        for index, name in enumerate(names, 1):
            array = model.get_tensor(name).float().numpy()
            if array.ndim < 2:
                continue
            q2_data, q2_metrics = quantize_q2(array, group_size, refinement_rounds)
            q4_data, q4_metrics = quantize_q4(array, group_size, refinement_rounds)
            extra = len(q4_data) - len(q2_data)
            gain = float(q2_metrics["squared_error"]) - float(q4_metrics["squared_error"])
            forced = any(fnmatch.fnmatchcase(name, pattern) for pattern in forced_patterns)
            candidates.append(
                {
                    "name": name,
                    "extra_bytes": extra,
                    "squared_error_reduction": gain,
                    "reduction_per_extra_byte": gain / max(extra, 1),
                    "forced": forced,
                }
            )
            print(
                f"[analysis {index:03}/{len(names):03}] {name}: "
                f"Q4 saves {gain:.6g} SSE for {extra:,} bytes",
                flush=True,
            )

    selected = {str(item["name"]) for item in candidates if bool(item["forced"])}
    forced_bytes = sum(
        int(item["extra_bytes"]) for item in candidates if bool(item["forced"])
    )
    if forced_bytes > budget_bytes:
        raise ValueError(
            f"forced Q4 tensors require {forced_bytes:,} bytes, over budget {budget_bytes:,}"
        )

    # Tensor sizes share coarse byte increments, so a sparse exact 0/1 knapsack is small.
    states: dict[int, tuple[float, tuple[str, ...]]] = {forced_bytes: (0.0, tuple())}
    for item in candidates:
        if bool(item["forced"]) or float(item["squared_error_reduction"]) <= 0:
            continue
        cost = int(item["extra_bytes"])
        gain = float(item["squared_error_reduction"])
        name = str(item["name"])
        updated = dict(states)
        for used, (score, names) in states.items():
            next_used = used + cost
            if next_used > budget_bytes:
                continue
            candidate = (score + gain, (*names, name))
            if next_used not in updated or candidate[0] > updated[next_used][0]:
                updated[next_used] = candidate
        states = updated
    used_bytes, (gain, chosen) = max(states.items(), key=lambda entry: entry[1][0])
    selected.update(chosen)
    ranked = sorted(
        candidates,
        key=lambda item: float(item["reduction_per_extra_byte"]),
        reverse=True,
    )
    report: dict[str, object] = {
        "strategy": "exact-whole-tensor-knapsack",
        "objective": "maximum weight reconstruction SSE reduction",
        "calibration_examples": 0,
        "training_performed": False,
        "q4_extra_byte_budget": budget_bytes,
        "q4_extra_bytes_used": used_bytes,
        "selected_q4_tensors": sorted(selected),
        "selected_q4_tensor_count": len(selected),
        "estimated_squared_error_reduction_vs_all_q2": gain
        + sum(
            float(item["squared_error_reduction"])
            for item in candidates
            if bool(item["forced"])
        ),
        "candidate_ranking": ranked,
    }
    return selected, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="directory containing model.safetensors")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--base-model",
        required=True,
        help="public source model identifier",
    )
    parser.add_argument(
        "--base-revision", required=True, help="immutable public source revision"
    )
    parser.add_argument("--license", required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--refinement-rounds", type=int, default=8)
    parser.add_argument("--quant-bits", type=int, choices=(2, 3), default=3)
    parser.add_argument(
        "--q2-method",
        choices=("fixed", "symmetric"),
        default="fixed",
        help="two-bit block codebook; symmetric learns two magnitudes per weight group",
    )
    parser.add_argument(
        "--q4-pattern",
        action="append",
        default=[],
        help="exact tensor name or prefix to store with four-bit NF4 levels; repeatable",
    )
    parser.add_argument(
        "--q4-list-from",
        type=Path,
        help="reuse the exact Q4 tensor selection recorded in another PKLM artifact",
    )
    parser.add_argument(
        "--q2-pattern",
        action="append",
        default=[],
        help="force an exact tensor or wildcard back to Q2 after Q4 selection; repeatable",
    )
    parser.add_argument(
        "--auto-q4-extra-bytes",
        type=int,
        help=(
            "select Q4 tensors without calibration by maximizing weight-error reduction "
            "within this many bytes above an all-Q2 payload; --q4-pattern entries become mandatory"
        ),
    )
    args = parser.parse_args()

    if args.group_size <= 0 or args.group_size % 4:
        parser.error("--group-size must be positive and divisible by four")
    if args.refinement_rounds <= 0:
        parser.error("--refinement-rounds must be positive")
    if args.quant_bits == 3 and args.auto_q4_extra_bytes is not None:
        parser.error("automatic Q4 selection currently requires a two-bit base")
    model_path = args.source / "model.safetensors"
    if not model_path.exists():
        parser.error(f"missing {model_path}")
    if args.q4_list_from:
        source_manifest, _ = read_container(args.q4_list_from)
        args.q4_pattern.extend(
            str(entry["name"])
            for entry in source_manifest["tensors"]
            if entry["kind"] == "q4_block"
        )

    auto_selection = None
    selected_q4: set[str] | None = None
    if args.auto_q4_extra_bytes is not None:
        try:
            selected_q4, auto_selection = select_q4_by_error_budget(
                model_path,
                args.group_size,
                args.auto_q4_extra_bytes,
                args.q4_pattern,
                args.refinement_rounds,
            )
        except ValueError as error:
            parser.error(str(error))

    payload = bytearray()
    tensors: list[dict[str, object]] = []
    parameters = 0
    total_squared_error = 0.0
    total_squared_original = 0.0
    total_dot = 0.0
    total_squared_reconstructed = 0.0

    with safe_open(model_path, framework="pt", device="cpu") as model:
        names = sorted(model.keys())
        for index, name in enumerate(names, 1):
            array = model.get_tensor(name).float().numpy()
            count = int(array.size)
            parameters += count
            offset = len(payload)
            if array.ndim >= 2:
                use_q4 = (
                    name in selected_q4
                    if selected_q4 is not None
                    else any(fnmatch.fnmatchcase(name, pattern) for pattern in args.q4_pattern)
                )
                if any(fnmatch.fnmatchcase(name, pattern) for pattern in args.q2_pattern):
                    use_q4 = False
                if use_q4:
                    data, metrics = quantize_q4(
                        array, args.group_size, args.refinement_rounds
                    )
                    kind = "q4_block"
                else:
                    if args.quant_bits == 3:
                        data, metrics = quantize_q3(
                            array, args.group_size, args.refinement_rounds
                        )
                        kind = "q3_block"
                    elif args.q2_method == "symmetric":
                        data, metrics = quantize_q2_symmetric(
                            array, args.group_size, args.refinement_rounds
                        )
                        kind = "q2_symmetric"
                    else:
                        data, metrics = quantize_q2(
                            array, args.group_size, args.refinement_rounds
                        )
                        kind = "q2_block"
                groups = int(metrics["groups"])
                total_squared_error += float(metrics["squared_error"])
                total_squared_original += float(metrics["squared_original"])
                total_dot += float(metrics["dot"])
                total_squared_reconstructed += float(metrics["squared_reconstructed"])
                extra = {"group_size": args.group_size, "groups": groups}
            else:
                data = array.astype("<f2").tobytes()
                kind = "fp16"
                reconstructed = array.astype(np.float16).astype(np.float32)
                error = reconstructed - array
                total_squared_error += float(np.dot(error.astype(np.float64), error.astype(np.float64)))
                total_squared_original += float(np.dot(array.astype(np.float64), array.astype(np.float64)))
                total_dot += float(np.dot(array.astype(np.float64), reconstructed.astype(np.float64)))
                total_squared_reconstructed += float(
                    np.dot(reconstructed.astype(np.float64), reconstructed.astype(np.float64))
                )
                extra = {}
            payload.extend(data)
            tensors.append(
                {
                    "name": name,
                    "shape": list(array.shape),
                    "kind": kind,
                    "offset": offset,
                    "length": len(data),
                    **extra,
                }
            )
            print(
                f"[{index:03}/{len(names):03}] {name}: {kind} {count:,} -> {len(data):,} bytes",
                flush=True,
            )

    blobs: list[dict[str, object]] = []
    for name in EMBEDDED_FILES:
        path = args.source / name
        if not path.exists():
            continue
        data = path.read_bytes()
        offset = len(payload)
        payload.extend(data)
        blobs.append(
            {
                "name": name,
                "offset": offset,
                "length": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )

    nrmse = math.sqrt(total_squared_error / max(total_squared_original, 1e-30))
    cosine = total_dot / math.sqrt(
        max(total_squared_original * total_squared_reconstructed, 1e-30)
    )
    manifest = {
        "format": "pickle-low-bit-model-v1",
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "license": args.license,
        "training_performed": False,
        "quantization": {
            "method": (
                "error-guided-whole-tensor-fixed-level-block"
                if auto_selection is not None
                else "mixed-fixed-level-block"
            ),
            "nominal_bits": args.quant_bits,
            "group_size": args.group_size,
            "q2_levels": LEVELS.tolist(),
            "q3_levels": Q3_LEVELS.tolist(),
            "q2_method": args.q2_method,
            "q4_levels": Q4_LEVELS.tolist(),
            "q4_patterns": sorted(selected_q4) if selected_q4 is not None else args.q4_pattern,
            "q2_overrides": args.q2_pattern,
            "scale_dtype": "float16",
            "refinement_rounds": args.refinement_rounds,
        },
        "parameters": parameters,
        "source_sha256": sha256_file(model_path),
        "quality": {"global_nrmse": nrmse, "global_cosine": cosine},
        "data_free_q4_selection": auto_selection,
        "tensors": tensors,
        "embedded_files": blobs,
        "payload_bytes": len(payload),
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        output.write(MAGIC)
        output.write(struct.pack("<I", len(manifest_bytes)))
        output.write(manifest_bytes)
        output.write(payload)

    artifact_bytes = args.output.stat().st_size
    print(
        json.dumps(
            {
                "output": str(args.output),
                "parameters": parameters,
                "artifact_bytes": artifact_bytes,
                "artifact_mib": artifact_bytes / (1024**2),
                "effective_artifact_bits_per_parameter": artifact_bytes * 8 / parameters,
                "global_nrmse": nrmse,
                "global_cosine": cosine,
                "sha256": sha256_file(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
