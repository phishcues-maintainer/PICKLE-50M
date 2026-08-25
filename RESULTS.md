# PICKLE-50M results

Measured on 2026-08-24 and 2026-08-25. These results make a future same-hardware comparison
possible; they do not declare a winner before the other 50M entry and its harness are public.

## Scope and claim boundary

- No training, fine-tuning, distillation, or calibration examples were used.
- The language model is the public pretrained `StentorLabs/Stentor3-50M` checkpoint. This project
  contributes deterministic post-training compression, native packaging, inference, and tests.
- Retrieval is a separate lookup-and-extraction system. It is not model context, and retrieval
  accuracy is not attributed to the language model.
- Standard model scores use weights reconstructed from the selected deployment artifact.
- Native decode numbers use the packed Q2/Q4 bytes directly. Decode-only and end-to-end timings
  are reported separately.
- Arbitrary-text TokenMonster encoding and decoding are native. The Python reference tokenizer is
  used only for parity validation.

## Test machines

Native Windows measurements:

- Windows 10 build 19045
- Intel Core i5-7500, 4 cores / 4 threads, 3.40 GHz
- 32 GiB RAM
- Rust/Cargo 1.97.0

The four standard accuracy tasks were accelerated on a temporary Google Cloud
`c4-highcpu-16` in `us-central1-b`: 16 vCPUs, 32 GiB RAM, Debian 12, Python 3.11,
PyTorch 2.8.0 CPU, Transformers 4.52.3, and lm-evaluation-harness 0.4.12. Hardware affects
evaluation time, not deterministic accuracy. It is not used for a same-hardware speed claim.

The Linux binaries were cross-compiled with the official Zig 0.14.1 toolchain. The inference-only
binary was then executed in Google Cloud Shell on Linux 6.6.143+ x86_64 with an Intel Xeon 2.20
GHz allocation (2 cores / 4 threads). Its uploaded hashes matched, the authenticated V3 model
loaded with the AVX2 kernel, arbitrary-text generation completed, and a 3-iteration benchmark
completed. This Cloud Shell timing is a portability smoke measurement, not the same-hardware
headline speed claim. Evidence is in `results/linux-smoke.json`.

## Deployment size and integrity

The same authenticated model file is used on both operating systems.

| Component or pair | Bytes | MiB | Margin below 15 MiB |
|---|---:|---:|---:|
| Native model plus compact TokenMonster vocabulary | 15,128,816 | 14.4280 | — |
| Windows inference-only runtime | 366,592 | 0.3496 | — |
| Windows inference-only pair | 15,495,408 | 14.7776 | 233,232 bytes |
| Linux inference-only runtime | 586,848 | 0.5597 | — |
| Linux inference-only pair | 15,715,664 | 14.9876 | 12,976 bytes |
| Windows full utility pair | 15,582,448 | 14.8606 | 146,192 bytes |
| Linux full utility pair | 15,804,544 | 15.0724 | over by 75,904 bytes |
| Audited PKLM source artifact | 15,236,303 | 14.5305 | — |

The model occupies 2.44852 artifact bits per source parameter, including FP16 scales, the V3
header, integrity metadata, and tokenizer data. The audited PKLM artifact occupies 2.46592 bits per
parameter and records the base model, source hash, selected tensor kinds, and
`training_performed: false`.

| Artifact | SHA-256 |
|---|---|
| Audited PKLM | `555f267830322d8fff600c04ff8648c2518b2d4c13e43730d2f9fcf7adea968a` |
| Native model | `5d0442b66bf39de8e4cd6bc12ce4ca998d9f10c3fe5d6234debb509774f1cc3c` |
| Windows inference-only runtime | `38093b8425585509a1f7d04f90d46ab9dab507c02f1382c389c8c21bde33769d` |
| Linux inference-only runtime | `eb9bab69f8112e9379feccb223f41f1c916648d88f3e3bfff807eaa9f7f61d88` |
| Windows full utility runtime | `6205612a9cc7d6e9d58e3b71a68a9f0e8aa118a5e657d9d0aef72a021704dadb` |
| Linux full utility runtime | `7bcee3b5e9c5a98a8d4943a4e98de7530ada41520a8bc34ff8368078aa582c0b` |

Native format V3 authenticates dimensions, tensor-kind metadata, all packed weights and scales,
and tokenizer data before parsing. A deterministic mutation suite tested empty and partial files,
eight truncation points, nine header/body bit flips, and a trailing byte. All 18/18 corruptions
were rejected. The authenticated header/body digest reported by the loader is
`45a78bcc2391b8792539adca39ae9895c17f5b355455dc30487ae03be437c99b`.

## Native tokenizer validation

The native encoder was compared token-for-token with TokenMonster 1.1.12. The test covers 14 fixed
cases, a deterministic 46,106-character mixed-script/adversarial corpus, and all 135,955 Unicode
letter, number, and mark scalars recognized by the validation environment.

| Check | Result |
|---|---:|
| Reference token IDs compared | 609,889 |
| Token-ID differences | 0 |
| Exact match | Yes |

`--prompt` and `--prompt-file` now use this native path. Python and the TokenMonster server are not
deployment dependencies. Raw evidence is in `results/native-tokenizer-validation.json`.

## Quantization quality

The selected artifact performs six deterministic code/scale refinement rounds, then fits the
Linux 15 MiB pair by returning the ten lowest reconstruction-benefit attention tensors to Q2.
Selection uses weights only, never text or calibration examples.

Across all 49,430,016 weights:

| Check | Selected artifact | Earlier mixed-attention artifact |
|---|---:|---:|
| Global cosine similarity | 0.969864 | 0.969548 |
| Global normalized RMSE | 0.243649 | 0.244902 |
| Fixed-text compressed loss | 3.693212 | 3.749761 |
| Fixed-text perplexity | 40.1737 | 42.5109 |
| Logit cosine similarity | 0.965928 | 0.964320 |
| Top-1 logit agreement | 61.42% | 63.78% |

The selected artifact improves reconstruction, loss, perplexity, and logit cosine while making the
native model 212,814 bytes smaller than the earlier version. Top-1 agreement on this small smoke
sample regressed, so it is disclosed rather than hidden. This fixed 128-token sample is a drift
test, not a standard benchmark. Evidence is in `results/balanced-refined-verify.json` and
`results/balanced-refined-smoke.json`.

## Native inference parity and performance

Against the exact quantized values expanded into the Hugging Face Llama reference:

| Native parity check | Result |
|---|---:|
| Last-token logits checked | 4,096 |
| Reference/native argmax | 704 / 704 |
| Maximum absolute logit error | 0.0000123978 |
| Mean absolute logit error | 0.0000038887 |
| Logit cosine similarity | 1.000000 |
| Greedy generation | Exact, 16/16 token IDs |

The AVX2 path was tested against the scalar path with identical generated IDs:

| Packed decode path | Threads | Speed | Relative to scalar |
|---|---:|---:|---:|
| Scalar | 1 | 26.475 tok/s | 1.000x |
| AVX2 | 1 | 29.625 tok/s | 1.119x |
| AVX2 plus persistent workers | 4 | 67.622 tok/s | 2.554x |

Decode-only timing excludes load, prompt tokenization, and prefill. The separate full request test
uses an arbitrary text prompt, 32 generated tokens, AVX2, and four workers:

| End-to-end stage | Cold | Warm mean |
|---|---:|---:|
| Model load | 82.015 ms | already loaded |
| Tokenization | 0.108 ms | 0.122 ms |
| State allocation | 0.292 ms | 0.256 ms |
| Prompt prefill | 166.196 ms | 142.675 ms |
| Time to first token | 248.612 ms | 143.053 ms |
| Total request | 695.017 ms | 595.948 ms |
| Generated-token rate | — | 53.696 tok/s |

A separately sampled inference-only run peaked at 20.707 MiB working set. Its speed sample was
taken while the host was busy and is not used as the headline throughput number. Raw evidence is
in `results/native-parity.json`, `results/kernel-profile-v3.json`,
`results/native-e2e-v3.json`, and `results/native-memory-v3.json`.

The published Linux inference pair also passed a real Linux smoke test. On Google Cloud Shell,
the runtime selected AVX2 with four workers, loaded the authenticated model, encoded and generated
from an arbitrary text prompt, and measured 23.960 tok/s over three 32-token iterations. That
shared-cloud figure is disclosed for portability only and is not compared with the dedicated
i5-7500 result. Raw evidence is in `results/linux-smoke.json`.

## Standard benchmark

The selected artifact was expanded into a temporary Hugging Face checkpoint without changing any
quantized value and evaluated with the unmodified EleutherAI harness. Conditions: full validation
sets, 0-shot, CPU, batch size 64, no sample limit, harness 0.4.12.

| Task | Examples | Accuracy | Normalized accuracy |
|---|---:|---:|---:|
| PIQA | 1,838 | 54.9510% ± 1.1608% | 53.3732% ± 1.1639% |
| HellaSwag | 10,042 | 26.4887% ± 0.4404% | 27.2356% ± 0.4443% |
| ARC-Easy | 2,376 | 26.6835% ± 0.9076% | 27.9040% ± 0.9204% |
| ARC-Challenge | 1,172 | 19.9659% ± 1.1682% | 23.9761% ± 1.2476% |

The scores are reported regardless of whether they improve over the earlier candidate. Hardware
was used only to shorten evaluation time. Full harness JSON is under `results/lm-eval-balanced/`;
its SHA-256 is `9c59d510f7d5f2250067fa6847170a27d81bb85f1b8d9b0030db3529c35ff303`.
The run processed 58,032 likelihood requests in 596.105 seconds.

## Retrieval benchmark

The strengthened public generator covers 12 task types: direct, close lookalike, latest-wins,
two-hop, absent identifier, Unicode identifier, long value, malformed record, missing pointer,
punctuation, case-sensitive collision, and adversarial abstention. The index stores identifiers and
archive byte ranges, never answers; answers are read from the archive at query time.

On the new adversarial 1M-whitespace-token tier:

| Check | Result |
|---|---:|
| Archive bytes | 6,787,177 |
| Questions | 560 |
| Correct | 560/560 (100.00%) |
| Task types perfect | 12/12 |
| Index bytes / entries | 42,741 / 1,740 |
| Index build | 90.596 ms, 71.446 MiB/s |
| Loaded-index query mean / p50 / p95 | 5.031 / 4 / 11 us |
| Reopen-per-query mean / p50 / p95 | 44.543 / 23 / 39 us |
| Four-thread throughput | 151,828.997 queries/s, 2,800/2,800 correct |

“Reopen” means the archive was reopened for every query; the OS page cache was not cleared. This
is explicitly not presented as cold-storage latency. The legacy 100M-token scale test remains
available separately: 678,129,025 archive bytes, 1,000/1,000 correct across the original five task
types, an 89,509-byte index, 9.699 us mean, 6 us p50, and 19 us p95.

Evidence is in `results/retrieval-adversarial-1m-index.json`,
`results/retrieval-adversarial-1m.json`, and `results/retrieval-100m.json`.

## What remains before declaring a winner

1. Obtain the other 50M entry, its exact commit, artifacts, and commands.
2. Run both entries on the same idle hardware with the same prompt IDs, context, task versions,
   thread count, and timing boundary.
3. Compare size, packed decode, end-to-end text latency, retrieval accuracy/speed, and standard
   accuracy as separate columns rather than collapsing them into one vague claim.
