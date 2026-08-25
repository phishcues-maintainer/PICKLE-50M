"""Validate container integrity and optionally compare it with the source checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from safetensors import safe_open

from pklm_format import decode_tensor, read_container


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--source", type=Path, help="optional source model directory")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    manifest, payload = read_container(args.artifact)
    if len(payload) != int(manifest["payload_bytes"]):
        raise SystemExit("payload length does not match manifest")

    tensor_entries = manifest["tensors"]
    total_parameters = 0
    occupied: list[tuple[int, int, str]] = []
    for entry in tensor_entries:
        shape = tuple(int(value) for value in entry["shape"])
        total_parameters += int(np.prod(shape, dtype=np.int64))
        start = int(entry["offset"])
        end = start + int(entry["length"])
        if start < 0 or end > len(payload):
            raise SystemExit(f"tensor outside payload: {entry['name']}")
        occupied.append((start, end, str(entry["name"])))
        decoded = decode_tensor(entry, payload)
        if decoded.shape != shape or not np.isfinite(decoded).all():
            raise SystemExit(f"invalid decoded tensor: {entry['name']}")

    for blob in manifest["embedded_files"]:
        start = int(blob["offset"])
        end = start + int(blob["length"])
        data = payload[start:end]
        if hashlib.sha256(data).hexdigest() != blob["sha256"]:
            raise SystemExit(f"embedded file checksum failed: {blob['name']}")
        occupied.append((start, end, str(blob["name"])))

    occupied.sort()
    for previous, current in zip(occupied, occupied[1:]):
        if previous[1] > current[0]:
            raise SystemExit(f"overlapping payload entries: {previous[2]} and {current[2]}")

    measured = None
    if args.source:
        model_path = args.source / "model.safetensors"
        squared_error = 0.0
        squared_original = 0.0
        dot = 0.0
        squared_decoded = 0.0
        with safe_open(model_path, framework="np") as source:
            source_names = set(source.keys())
            artifact_names = {entry["name"] for entry in tensor_entries}
            if source_names != artifact_names:
                raise SystemExit("source and artifact tensor names differ")
            for entry in tensor_entries:
                original = np.asarray(source.get_tensor(entry["name"]), dtype=np.float32).reshape(-1)
                decoded = decode_tensor(entry, payload).reshape(-1)
                error = decoded - original
                original64 = original.astype(np.float64)
                decoded64 = decoded.astype(np.float64)
                error64 = error.astype(np.float64)
                squared_error += float(np.dot(error64, error64))
                squared_original += float(np.dot(original64, original64))
                dot += float(np.dot(original64, decoded64))
                squared_decoded += float(np.dot(decoded64, decoded64))
        measured = {
            "global_nrmse": math.sqrt(squared_error / max(squared_original, 1e-30)),
            "global_cosine": dot / math.sqrt(max(squared_original * squared_decoded, 1e-30)),
        }

    result = {
        "valid": True,
        "artifact": str(args.artifact),
        "sha256": sha256_file(args.artifact),
        "artifact_bytes": args.artifact.stat().st_size,
        "artifact_mib": args.artifact.stat().st_size / (1024**2),
        "parameters": total_parameters,
        "tensor_count": len(tensor_entries),
        "embedded_file_count": len(manifest["embedded_files"]),
        "effective_artifact_bits_per_parameter": args.artifact.stat().st_size * 8 / total_parameters,
        "manifest_quality": manifest["quality"],
        "measured_quality": measured,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
