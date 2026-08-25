# PICKLE-50M

PICKLE-50M is a reproducible, **no-training-by-this-project** entry for the public small-model
challenge. The current v1.1 candidate improves the original release instead of hiding its weak
results:

- 31,171,072 public pretrained parameters;
- deterministic, data-free Q3 weight compression;
- a self-contained Rust CPU runtime with native byte-level BPE text tokenization;
- a separate disk lookup-and-extraction benchmark;
- complete commands, hashes, parity tests, corruption tests, and unfiltered standard scores.

The source checkpoint is
[`SEN-AGI/Sable-1.1-30M`](https://huggingface.co/SEN-AGI/Sable-1.1-30M) at immutable revision
`1a845020ed104bd38b67b2a95472fb166cd4a99b`, Apache-2.0. It was pretrained by its authors;
this project performed **zero** training, fine-tuning, distillation, or example-based calibration.
The upstream card describes it as a preview base model, so this repository does not present it as
a production chat model.

Retrieval is reported as a separate system, never as model context.

## Current deployment

`artifacts/pickle-31m-sable-q3.pkn` is the authenticated V4 deployment model. It contains the
packed Q3 weights, FP16 group scales, dimensions, special-token metadata, and complete compact
byte-level BPE tokenizer.

| Deployment | Model bytes | Runtime bytes | Combined bytes | Combined MiB |
|---|---:|---:|---:|---:|
| Windows inference-only | 12,534,009 | 384,512 | 12,918,521 | 12.3201 |
| Windows full utility | 12,534,009 | 471,552 | 13,005,561 | 12.4031 |
| Linux inference-only | 12,534,009 | 581,088 | 13,115,097 | 12.5075 |
| Linux full utility | 12,534,009 | 669,640 | 13,203,649 | 12.5920 |

The model itself is 11.9534 MiB. Both inference pairs have more than 2.6 MB of margin below
15 MiB. A native Linux build and real Linux smoke run are recorded in [RESULTS.md](RESULTS.md).

The V4 loader authenticates its header and body before parsing. Any tested truncation, bit
mutation, or trailing byte is rejected.

## What v1.1 fixes

| Measurement | v1.0 | v1.1 candidate | Change |
|---|---:|---:|---:|
| Windows inference pair | 14.7776 MiB | 12.3201 MiB | 16.63% smaller |
| i5-7500 packed decode, AVX2, 4 workers | 67.622 tok/s | 111.460 tok/s | 64.83% faster |
| Peak sampled working set | 20.707 MiB | 18.102 MiB | 12.58% lower |
| Native arbitrary-text tokenizer | TokenMonster | Byte-level BPE | exact reference parity |
| PIQA normalized, full validation set | 53.3732% | 55.9304% | +2.5572 points |
| ARC-Easy normalized, full validation set | 27.9040% | 32.8283% | +4.9243 points |

HellaSwag is essentially flat and ARC-Challenge regresses by 0.69 percentage point. Both are
published in [RESULTS.md](RESULTS.md); this is not presented as a universal quality win.

## Validation highlights

| Check | Result |
|---|---:|
| Native tokenizer vs Transformers fast tokenizer | 663,409 / 663,409 token IDs exact |
| Native vs expanded-Q3 logits | 24,000 checked; same argmax; max error 0.00001526 |
| Native vs reference greedy generation | 16 / 16 token IDs exact |
| Loader corruption suite | 18 / 18 malformed files rejected |
| Packed decode on i5-7500, AVX2, four workers, 15 iterations | 111.460 tok/s |
| Peak sampled Windows working set | 18.102 MiB |
| Strengthened retrieval bank | 560 / 560 across 12 task types |

Timing boundaries, machines, all four standard scores, raw evidence, and claim boundaries are in
[RESULTS.md](RESULTS.md).

## Downloads

- Repository: <https://github.com/phishcues-maintainer/PICKLE-50M>
- v1.1 release: <https://github.com/phishcues-maintainer/PICKLE-50M/releases/tag/v1.1.0>
- Checksums: [CHECKSUMS.sha256](CHECKSUMS.sha256)
- Copy-ready challenge response: [REDDIT_DRAFT.md](REDDIT_DRAFT.md)

## Build and run

```powershell
cargo build --release
cargo test --release

bin/pickle50-infer.exe model-generate `
  --model artifacts/pickle-31m-sable-q3.pkn `
  --prompt "The history of computing began." `
  --new-tokens 32 --threads 4 --kernel avx2
```

`--prompt` and `--prompt-file` are fully native. Python, Transformers, TokenMonster, and any
external tokenizer service are not deployment dependencies. `--tokens` remains available for
exact low-level tests.

Use `--kernel scalar` for the portable reference path, `--kernel avx2` to require AVX2, or
`--kernel auto` for runtime detection. `--threads N` controls the persistent worker pool.

## One-command challenge validation

```powershell
D:/CodexCache/pickle-50m/venv/Scripts/python.exe tools/run_challenge.py `
  --runtime bin/pickle50.exe `
  --model artifacts/pickle-31m-sable-q3.pkn `
  --out results/challenge-sable-q3.json `
  --threads 4 --kernel avx2 --iterations 15 `
  --tokenizer D:/CodexCache/pickle-50m/sable-30m `
  --reference D:/CodexCache/pickle-50m/sable-q3-eval `
  --parity-tokens 2,417,268,283,288,430,17 `
  --retrieval-tokens 1000000 --retrieval-questions 560
```

The runner records exact file sizes and hashes, model metadata, packed decode, tokenizer parity,
native/PyTorch parity, loader mutation rejection, and retrieval accuracy/speed in one JSON file.

## Reproduce the selected compression

The fixed eight-level Q3 codebook is derived once from an ideal normal distribution. Each
128-weight group receives an FP16 scale refined for eight deterministic rounds. This process reads
weights only: there are no prompts, labels, held-out examples, calibration text, gradients, or
optimizer steps.

```powershell
python tools/quantize_model.py `
  --source D:/CodexCache/pickle-50m/sable-30m `
  --output artifacts/pickle-31m-sable-q3.pklm `
  --base-model SEN-AGI/Sable-1.1-30M `
  --base-revision 1a845020ed104bd38b67b2a95472fb166cd4a99b `
  --license Apache-2.0 `
  --quant-bits 3 --group-size 128 --refinement-rounds 8

python tools/export_native.py `
  --source artifacts/pickle-31m-sable-q3.pklm `
  --config D:/CodexCache/pickle-50m/sable-30m/config.json `
  --tokenizer D:/CodexCache/pickle-50m/sable-30m/tokenizer.json `
  --out artifacts/pickle-31m-sable-q3.pkn
```

The audited PKLM manifest pins the upstream revision and source SHA-256, records
`training_performed: false`, and embeds the config, tokenizer, tokenizer config, and generation
config. Its measured global weight cosine is 0.986646 and normalized RMSE is 0.162881.

## Standard evaluation

Expand the exact values represented by the PKLM file, then use the unmodified harness:

```powershell
python tools/export_hf.py `
  --source artifacts/pickle-31m-sable-q3.pklm `
  --out D:/CodexCache/pickle-50m/sable-q3-eval

lm_eval --model hf `
  --model_args pretrained=D:/CodexCache/pickle-50m/sable-q3-eval,dtype=float32 `
  --tasks piqa,hellaswag,arc_easy,arc_challenge `
  --num_fewshot 0 --batch_size 64 `
  --output_path results/lm-eval-sable-q3
```

## Retrieval coverage

The generated bank covers direct lookup, close lookalikes, latest-wins duplicates, two-hop
references, absent identifiers, Unicode identifiers, long values, malformed records, missing
pointers, punctuation, case-sensitive collisions, and adversarial abstention. The index stores
identifiers and archive byte ranges, never answers; answers are read from the archive at query
time.

## Claim boundary

This repository makes the entry reproducible; it does not declare a head-to-head winner before the
other model, immutable commit, artifacts, and commands are public and both entries are run on the
same idle hardware with identical timing boundaries.

## Provenance and licensing

The current source model is
[`SEN-AGI/Sable-1.1-30M`](https://huggingface.co/SEN-AGI/Sable-1.1-30M), Apache-2.0.
PICKLE-50M does not claim authorship of its pretraining. Project code is Apache-2.0; see
[LICENSE](LICENSE) and [THIRD_PARTY.md](THIRD_PARTY.md).
