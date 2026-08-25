# PICKLE-50M

PICKLE-50M is a transparent, no-training entry for the public 50M-model challenge:

- 49,430,016 public pretrained parameters compressed without training, fine-tuning,
  distillation, or calibration examples;
- native TokenMonster text encoding and packed Q2/Q4 CPU inference;
- a separate deterministic disk-retrieval and extraction benchmark;
- public commands and machine-readable evidence.

Retrieval is not described as model context. Model, runtime, archive, and index sizes are reported
separately.

## Current deployment

`artifacts/pickle-50m-native.bin` is the authenticated V3 deployment model. It is 15,128,816 bytes
(14.4280 MiB) and includes the complete compact TokenMonster vocabulary. The same model file is
used on Windows and Linux.

| Deployment pair | Model | Runtime | Combined | MiB | Below 15 MiB |
|---|---:|---:|---:|---:|---:|
| Windows inference-only | 15,128,816 | 366,592 | 15,495,408 | 14.7776 | 233,232 bytes |
| Linux inference-only | 15,128,816 | 586,848 | 15,715,664 | 14.9876 | 12,976 bytes |

The full Windows runtime, including retrieval generation/indexing, is also below 15 MiB when
paired with the model. The full Linux utility binary is reported separately and is not used for
the inference-only deployment-size claim.

The model format authenticates its dimensions, tensor-kind metadata, packed weights, scales, and
tokenizer data before parsing. Any changed header/body byte, truncation, or trailing byte is
rejected.

Published artifact hashes are collected in [`CHECKSUMS.sha256`](CHECKSUMS.sha256).

## Downloads

- Repository: <https://github.com/phishcues-maintainer/PICKLE-50M>
- Reproducible release assets: <https://github.com/phishcues-maintainer/PICKLE-50M/releases/tag/v1.0.0>
- Complete measurements and claim boundaries: [RESULTS.md](RESULTS.md)

## Headline validation

| Check | Result |
|---|---:|
| Native TokenMonster parity | 609,889 / 609,889 token IDs exact |
| Native vs reconstructed reference generation | 16 / 16 greedy token IDs exact |
| Loader corruption suite | 18 / 18 malformed models rejected |
| Linux release smoke | Hashes matched; load, text generation, and AVX2 benchmark passed |
| i5-7500 packed AVX2 decode, four workers | 67.622 tok/s |
| Warm arbitrary-text request, 32 generated tokens | 143.1 ms TTFT, 53.696 tok/s |
| Strengthened retrieval bank | 560 / 560 across 12 task types |

Throughput numbers are local measurements, not a same-hardware victory claim. Timing boundaries,
machine details, accuracy tasks, and raw-result paths are in [RESULTS.md](RESULTS.md).

## Build and run

```powershell
cargo build --release --bin pickle50
cargo build --release --bin pickle50-infer
cargo test --release

bin/pickle50-infer.exe model-generate `
  --model artifacts/pickle-50m-native.bin `
  --prompt "The history of computing began." `
  --new-tokens 32 --threads 4 --kernel avx2
```

`--prompt-file` accepts a UTF-8 text file. `--tokens` remains available for exact low-level tests.
The inference runtime contains NFD normalization, Unicode-13-compatible NoCapcode handling,
TokenMonster's ungreedy segmentation, and decoding; Python and the TokenMonster server are not
deployment dependencies.

Use `--kernel scalar` for the portable reference kernel, `--kernel avx2` to require AVX2, or
`--kernel auto` for runtime detection. `--threads N` controls the persistent worker pool.

## One-command challenge runner

The runner verifies artifact hashes/sizes, measures packed decode, mutation-tests the loader, and
optionally runs tokenizer parity, native/PyTorch parity, retrieval, and the four standard tasks:

```powershell
D:/CodexCache/pickle-50m/venv/Scripts/python.exe tools/run_challenge.py `
  --runtime bin/pickle50.exe `
  --model artifacts/pickle-50m-native.bin `
  --out results/challenge-v3.json `
  --threads 4 --kernel avx2 `
  --vocab D:/CodexCache/pickle-50m/stentor3-50m/tokenmonster.vocab `
  --reference D:/CodexCache/pickle-50m/eval-balanced
```

Add `--run-lm-eval --hf-checkpoint D:/CodexCache/pickle-50m/eval-balanced` to run PIQA,
HellaSwag, ARC-Easy, and ARC-Challenge. Optional URL and SHA-256 arguments let the runner download
and authenticate published artifacts before testing.

For a Linux cloud worker, `cloud/setup.sh` exports the selected artifact and installs the pinned
stack; `cloud/run-all.sh` records the machine and runs all four full tasks in one command.

## Reproduce the selected quantization

The selected artifact keeps the established attention-sensitive Q4 layout, performs six
deterministic scale/code refinement rounds, and returns ten lowest reconstruction-benefit
attention tensors to Q2 so the Linux deployment fits. Selection and refinement use weights only:
zero training or calibration examples.

```powershell
python tools/quantize_model.py `
  --source D:/CodexCache/pickle-50m/stentor3-50m `
  --output artifacts/pickle-50m-balanced-refined.pklm `
  --group-size 256 --refinement-rounds 6 `
  --q4-list-from artifacts/pickle-50m-mixed-attn.pklm `
  --q2-pattern model.layers.0.self_attn.v_proj.weight `
  --q2-pattern model.layers.1.self_attn.v_proj.weight `
  --q2-pattern model.layers.0.self_attn.k_proj.weight `
  --q2-pattern model.layers.5.self_attn.k_proj.weight `
  --q2-pattern model.layers.0.self_attn.q_proj.weight `
  --q2-pattern model.layers.2.self_attn.v_proj.weight `
  --q2-pattern model.layers.12.self_attn.k_proj.weight `
  --q2-pattern model.layers.1.self_attn.k_proj.weight `
  --q2-pattern model.layers.11.self_attn.k_proj.weight `
  --q2-pattern model.layers.13.self_attn.k_proj.weight

python tools/export_native.py `
  --source artifacts/pickle-50m-balanced-refined.pklm `
  --config D:/CodexCache/pickle-50m/stentor3-50m/config.json `
  --vocab D:/CodexCache/pickle-50m/stentor3-50m/tokenmonster.vocab `
  --out artifacts/pickle-50m-native.bin
```

`--auto-q4-extra-bytes N` is also available. It performs an exact whole-tensor knapsack search
that maximizes reconstruction-error reduction within a byte budget and records every candidate
and selection in the PKLM manifest.

## Validation commands

```powershell
python tools/profile_kernels.py `
  --runtime bin/pickle50.exe --model artifacts/pickle-50m-native.bin `
  --threads 4 --out results/kernel-profile-v3.json

python tools/fuzz_native.py `
  --runtime bin/pickle50.exe --model artifacts/pickle-50m-native.bin `
  --out results/native-mutation-test.json

python tools/validate_tokenizer.py `
  --runtime bin/pickle50.exe --model artifacts/pickle-50m-native.bin `
  --vocab D:/CodexCache/pickle-50m/stentor3-50m/tokenmonster.vocab `
  --out results/native-tokenizer-validation.json
```

See [RESULTS.md](RESULTS.md) for measurements and claim boundaries.

## Retrieval coverage

The generated bank now covers direct lookup, close lookalikes, latest-wins duplicates, two-hop
references, absent identifiers, Unicode identifiers, long values, malformed records, missing
pointers, punctuation, case-sensitive collisions, and adversarial abstention. Index build time,
index size, sequential latency, and concurrent throughput are recorded separately.

## Provenance and licensing

The base model is [`StentorLabs/Stentor3-50M`](https://huggingface.co/StentorLabs/Stentor3-50M),
Apache-2.0. PICKLE-50M does not claim authorship of its pretraining. Project code is Apache-2.0;
see [LICENSE](LICENSE) and [THIRD_PARTY.md](THIRD_PARTY.md) for dependency and data attribution.
