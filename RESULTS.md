# PICKLE-50M v1.1 results

Measured on 2026-08-25. These results make a future same-hardware comparison possible; they do
not declare a winner before the other entry and its harness are public.

## Scope and claim boundary

- This project performed no training, fine-tuning, distillation, gradient updates, or
  example-based calibration.
- The model is the public pretrained `SEN-AGI/Sable-1.1-30M` checkpoint at immutable revision
  `1a845020ed104bd38b67b2a95472fb166cd4a99b`. The project contributes deterministic
  weight-only compression, native packaging, inference, and tests.
- The upstream checkpoint is a 31.2M-parameter preview base model, not an instruct model.
- Standard scores use the exact Q3 values reconstructed from the selected deployment artifact.
- Native decode uses those same packed Q3 bytes directly.
- Retrieval is a separate deterministic lookup-and-extraction system. It is not model context,
  and its accuracy is not attributed to the language model.
- Speed results below use stated timing boundaries and are not a same-hardware comparison with
  another entrant.

## Test machines

Windows native measurements:

- Windows 10 build 19045
- Intel Core i5-7500, 4 cores / 4 threads, 3.40 GHz
- 32 GiB RAM
- Rust/Cargo 1.97.0

Standard accuracy and Linux release validation:

- Google Cloud `c4-highcpu-16`, `us-central1-b`
- Intel Xeon Platinum 8581C, 8 cores / 16 threads, 2.30 GHz
- 32 GiB RAM, Debian 12, Linux 6.1.0-52-cloud-amd64
- PyTorch 2.8.0+cpu, Transformers 4.52.3, lm-evaluation-harness 0.4.12
- no CUDA build and no GPU attached

Hardware affects evaluation duration, not deterministic accuracy.

## Deployment size and integrity

The same authenticated model file is used on Windows and Linux.

| Component or pair | Bytes | MiB | Margin below 15 MiB |
|---|---:|---:|---:|
| Native V4 model including compact BPE tokenizer | 12,534,009 | 11.9534 | — |
| Windows inference-only runtime | 384,512 | 0.3667 | — |
| Windows inference-only pair | 12,918,521 | 12.3201 | 2,810,119 bytes |
| Windows full utility pair | 13,005,561 | 12.4031 | 2,723,079 bytes |
| Linux inference-only runtime | 581,088 | 0.5542 | — |
| Linux inference-only pair | 13,115,097 | 12.5075 | 2,613,543 bytes |
| Linux full utility pair | 13,203,617 | 12.5920 | 2,525,023 bytes |
| Audited PKLM source artifact | 13,892,464 | 13.2489 | — |

The deployment model occupies 3.21683 all-in bits per source parameter, including FP16 scales,
header, integrity metadata, and the 343,615-byte compact tokenizer. The audited PKLM occupies
3.56548 bits per parameter and records `training_performed: false`.

| Artifact | SHA-256 |
|---|---|
| Audited PKLM | `b27d636d5b8b725cbd6188d1cfeb9a71da6c73e03454f6ca6d8b233c73952fef` |
| Native model | `ef082637bd79dfa4d8e216003a71ce4fa11245f27d37e8c38bc0d1593f023140` |
| Windows inference runtime | `178eeb8b2a5b5f9d4b92dc16c68ca4e6caa8069b18d235d571baf49a09e2494d` |
| Windows full utility | `c117ad6e29740d0e30165a33c8a85a231aa73f4a71dc90136114062fa5331f9f` |
| Linux inference runtime | `6cc0d05a3586e3aea74bd41cc740396d48c492448a8fb96744f88b4a56a12b9c` |
| Linux full utility | `6a42e4059825870fb48a435ff26406f87b160bd9fe81c31482c778190470be4d` |

The authenticated header/body digest reported by the loader is
`18670c7927173f8ff5d1110b4e5c2b7083c92c5318160ce782a7ba26a1a93433`. A deterministic
suite tested empty and partial files, eight truncations, nine header/body bit flips, and a
trailing byte. All 18/18 malformed files were rejected.

## Native arbitrary-text tokenizer

V1.1 removes the external-tokenizer limitation. The native runtime embeds the complete byte-level
BPE vocabulary, byte mapping, merge ranks, added tokens, and special-token metadata. `--prompt`
and `--prompt-file` need no Python or tokenizer service.

The encoder was compared token-for-token with the Transformers fast tokenizer over 14 fixed
cases, a deterministic 46,106-character mixed/adversarial corpus, and all 135,955 Unicode letter,
number, and mark scalars recognized by the test environment.

| Check | Result |
|---|---:|
| Reference token IDs compared | 663,409 |
| Token-ID differences | 0 |
| Exact match | Yes |

Raw evidence: `results/sable-tokenizer-validation.json`.

## Quantization quality

The selected artifact uses a fixed symmetric eight-level Q3 codebook, 128-weight groups, FP16
scales, and eight deterministic scale/code refinement rounds. Selection reads weights only.

Across all 31,171,072 parameters:

| Check | Result |
|---|---:|
| Global cosine similarity | 0.9866457325 |
| Global normalized RMSE | 0.1628809338 |
| PKLM manifest/measured agreement | exact to displayed precision |

Raw evidence: `results/sable-q3-integrity.json`.

## Native inference parity

Against the exact Q3 values expanded into the Hugging Face Llama reference:

| Native parity check | Result |
|---|---:|
| Last-token logits checked | 24,000 |
| Reference/native argmax | 17,397 / 17,397 |
| Maximum absolute logit error | 0.0000152588 |
| Mean absolute logit error | 0.0000027708 |
| RMSE | 0.0000035093 |
| Logit cosine similarity | 1.000000085 |
| Greedy generation | exact, 16/16 token IDs |

Raw evidence: `results/sable-q3-native-parity.json`.

## Native CPU performance

Decode-only timing excludes load, tokenization, and prompt prefill. The stable profile uses a
fixed seven-token prompt, 32 timed decode tokens, and 15 iterations:

| Packed path | Threads | Speed | Relative to scalar |
|---|---:|---:|---:|
| Scalar | 1 | 25.686 tok/s | 1.000x |
| AVX2 | 1 | 44.550 tok/s | 1.734x |
| AVX2 plus persistent workers | 4 | 111.460 tok/s | 4.339x |

All paths generated identical IDs. The separate one-command challenge run uses an arbitrary text
prompt and five iterations; one slow iteration reduced its aggregate to 98.790 tok/s. Both values
are retained rather than choosing only the favorable run.

A separately sampled full-utility process peaked at 18.102 MiB working set while decoding at
94.496 tok/s. Raw evidence: `results/kernel-profile-sable-q3.json`,
`results/challenge-sable-q3.json`, and `results/sable-q3-native-memory.json`.

## Real Linux release validation

Commit `b8b1cb9d9c1b2394a6c1878aa587c1454113aee0` was cloned from the public repository and
built natively on the Google Cloud VM with Rust 1.97.0. All 11 release tests passed. Both runtime
hashes above were taken from that native build.

The inference-only binary authenticated the V4 model, selected AVX2 with four workers, encoded an
arbitrary text prompt natively, and generated text. A 15-iteration end-to-end-style benchmark on
that VM measured:

| Linux smoke stage | Result |
|---|---:|
| Cold model load | 55.973 ms |
| Tokenization | 0.006 ms |
| Prompt prefill | 33.053 ms |
| Time to first token | 89.031 ms |
| Decode | 180.178 tok/s |

This is a portability result on disclosed hardware, not a comparison against the other entry.
Raw evidence: `results/linux-sable-q3-smoke.json`.

## Full standard benchmark

The selected PKLM was expanded without changing any quantized value and evaluated alongside the
uncompressed source checkpoint with the same unmodified EleutherAI harness. Conditions: full
validation sets, 0-shot, batch size 64, no sample limit, CPU only, identical task versions.

| Task | N | Q3 accuracy | Q3 normalized | FP32 normalized | Q3 delta |
|---|---:|---:|---:|---:|---:|
| PIQA | 1,838 | 55.2775% ± 1.1601% | 55.9304% ± 1.1583% | 57.8890% | -1.9587 pt |
| HellaSwag | 10,042 | 26.4091% ± 0.4399% | 27.2456% ± 0.4443% | 28.0621% | -0.8166 pt |
| ARC-Easy | 2,376 | 33.4596% ± 0.9682% | 32.8283% ± 0.9636% | 38.9731% | -6.1448 pt |
| ARC-Challenge | 1,172 | 19.1126% ± 1.1490% | 23.2935% ± 1.2353% | 23.7201% | -0.4266 pt |

Q3 normalized retention versus the matched FP32 checkpoint is 96.62% PIQA, 97.09% HellaSwag,
84.23% ARC-Easy, and 98.20% ARC-Challenge. ARC-Easy is the clear compression weakness; it is not
hidden or averaged away.

The Q3 run processed all 58,032 likelihood requests in 319.564 seconds. Raw harness files:
`results/lm-eval-sable-q3/results.json` (SHA-256
`1cdbdb6886b0467a266ad2e5931ce10ae73b94c233903a4a77166b6afc9797aa`) and
`results/lm-eval-sable-fp32/results.json` (SHA-256
`883ea62248eb5c4e3e5b326ed786f06d406bda1dc8244adaba859f5c58c638e6`).

## V1.0 to v1.1

The comparison uses the previously published v1.0 files and the current v1.1 files.

| Measurement | v1.0 | v1.1 | Change |
|---|---:|---:|---:|
| Source parameters | 49,430,016 | 31,171,072 | smaller base |
| Windows inference pair | 14.7776 MiB | 12.3201 MiB | -16.63% |
| Windows AVX2, four workers | 67.622 tok/s | 111.460 tok/s | +64.83% |
| Peak sampled working set | 20.707 MiB | 18.102 MiB | -12.58% |
| PIQA normalized | 53.3732% | 55.9304% | +2.5572 pt |
| HellaSwag normalized | 27.2356% | 27.2456% | +0.0100 pt |
| ARC-Easy normalized | 27.9040% | 32.8283% | +4.9243 pt |
| ARC-Challenge normalized | 23.9761% | 23.2935% | -0.6826 pt |

This is a broad engineering improvement, not a quality sweep: ARC-Challenge regressed slightly,
and the smaller base remains a weak open-ended generator.

## Retrieval benchmark

The seeded 1M-whitespace-token tier covers 12 task types: direct lookup, close lookalike,
latest-wins duplicates, two-hop references, absent identifiers, Unicode identifiers, long
values, malformed records, missing pointers, punctuation, case-sensitive collisions, and
adversarial abstention. The index stores identifiers and archive byte ranges, never answers.

| Check | Result |
|---|---:|
| Archive bytes | 6,787,177 |
| Questions | 560 |
| Correct | 560/560 (100.00%) |
| Task types perfect | 12/12 |
| Index bytes / entries | 42,741 / 1,740 |
| Index build | 32.701 ms, 197.939 MiB/s |
| Loaded-index query mean / p50 / p95 | 4.723 / 4 / 10 us |
| Reopen-per-query mean / p50 / p95 | 29.355 / 28 / 38 us |
| Four-thread throughput | 262,910.798 queries/s, 2,800/2,800 correct |

“Reopen” means the archive was reopened for every query; the OS page cache was not cleared. It is
not cold-storage latency. Raw evidence is included inside
`results/challenge-sable-q3.json` and under `results/challenge-retrieval/`.

## What remains before declaring a winner

1. Obtain the other entry, exact public commit, artifacts, and commands.
2. Run both on the same idle hardware with the same prompt text/IDs, context, task versions,
   thread count, warmup, and timing boundary.
3. Compare deployment size, packed decode, end-to-end text latency, standard accuracy, retrieval
   accuracy, and retrieval speed as separate columns.
