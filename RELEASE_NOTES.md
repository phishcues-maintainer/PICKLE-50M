# PICKLE-50M v1.1.0

V1.1 replaces the original candidate with a smaller, faster, better-scoring no-training
deployment and removes the external arbitrary-text tokenizer dependency.

## Release contents

- Public Apache-2.0 `SEN-AGI/Sable-1.1-30M` source checkpoint pinned to immutable revision
  `1a845020ed104bd38b67b2a95472fb166cd4a99b`.
- Authenticated V4 model with Q3 weights and complete native byte-level BPE tokenizer:
  12,534,009 bytes.
- Windows inference pair: 12,918,521 bytes (12.3201 MiB).
- Linux inference pair: 13,115,097 bytes (12.5075 MiB).
- Audited PKLM artifact with source provenance, quality metrics, and
  `training_performed: false`.
- Self-contained Rust scalar/AVX2 CPU inference, persistent workers, tied embeddings, KV cache,
  native text encoding/decoding, and deterministic retrieval utilities.
- Source, Windows/Linux binaries, checksums, full benchmark JSON, matched FP32 baseline,
  retrieval evidence, parity validation, and loader-mutation tests.

## What improved

- Windows inference deployment: 16.63% smaller than v1.0.
- Four-worker AVX2 packed decode: 111.460 tok/s on the same i5-7500, 64.83% faster.
- Sampled peak working set: 18.102 MiB, 12.58% lower.
- PIQA normalized: +2.5572 points.
- ARC-Easy normalized: +4.9243 points.
- Native tokenizer parity: 663,409/663,409 token IDs exact.

HellaSwag is effectively unchanged, and ARC-Challenge regresses by 0.6826 point versus v1.0.
Against the matched uncompressed Sable checkpoint, Q3’s largest loss is 6.1448 points on
ARC-Easy. Both limitations are disclosed in `RESULTS.md`.

## Validation

- Cargo release tests: 11/11 passed on Windows and on a native Linux build.
- Native/reference logits: 24,000 checked, same argmax, 0.00001526 maximum absolute error.
- Greedy native/reference generation: 16/16 token IDs exact.
- Corruption rejection: 18/18 malformed native files rejected.
- Retrieval: 560/560 across all 12 task types.
- Standard tasks: all 15,428 validation examples and 58,032 likelihood requests, 0-shot,
  unfiltered.
- Real Linux smoke: public commit cloned, built, authenticated, tokenized arbitrary text,
  generated text, and completed a 15-iteration AVX2 benchmark.

This release does not claim a head-to-head victory until the other entry is public and both are
measured on identical idle hardware with the same commands and timing boundaries.
