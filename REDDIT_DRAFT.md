Deal. The Google approval came through, so I finished the no-training entry and rebuilt the weak
first candidate before replying.

GitHub: https://github.com/phishcues-maintainer/PICKLE-50M

V1.1 release and binaries:
https://github.com/phishcues-maintainer/PICKLE-50M/releases/tag/v1.1.0

Full results, hardware, commands, raw JSON, and claim boundaries:
https://github.com/phishcues-maintainer/PICKLE-50M/blob/main/RESULTS.md

The current source is the public Apache-2.0 `SEN-AGI/Sable-1.1-30M` checkpoint, pinned to
revision `1a845020ed104bd38b67b2a95472fb166cd4a99b`: 31,171,072 parameters. Its authors
pretrained it; I did zero training, fine-tuning, distillation, gradient updates, or
example-based calibration. My contribution is deterministic weight-only Q3 compression, a
self-contained Rust CPU runtime, native tokenizer, packaging, retrieval tests, and the
reproducibility harness.

Deployment size, including the complete tokenizer:

- model: 12,534,009 bytes (11.9534 MiB)
- model + Windows inference runtime: 12,918,521 bytes (12.3201 MiB)
- model + Linux inference runtime: 13,115,097 bytes (12.5075 MiB)

Both full utility pairs, including retrieval tools, are also below 15 MiB. The audited artifact
records the immutable source revision/hash, measured reconstruction quality, every tensor, and
`training_performed: false`.

The arbitrary-text tokenizer limitation is fixed. The deployment embeds and runs the byte-level
BPE encoder/decoder natively; `--prompt` and `--prompt-file` do not use Python, Transformers,
TokenMonster, or an external service. Against the Transformers fast tokenizer, it matched
663,409/663,409 checked token IDs across fixed cases, a 46,106-character mixed/adversarial
corpus, and all 135,955 Unicode letter/number/mark scalars in the test environment.

Native inference was checked against the exact Q3 values expanded into the Hugging Face
reference: all 24,000 last-token logits checked, same argmax, 0.00001526 maximum absolute error,
cosine 1.0, and greedy generation matched 16/16 token IDs. The authenticated loader rejected
18/18 truncation, bit-mutation, and trailing-byte cases.

On my Windows i5-7500, packed decode after prefill, fixed seven-token prompt, 32 decode tokens,
15 iterations:

- scalar, 1 thread: 25.686 tok/s
- AVX2, 1 thread: 44.550 tok/s
- AVX2, 4 persistent workers: 111.460 tok/s

All three generated identical token IDs. The separate one-command arbitrary-text run measured
92.033 tok/s over 15 iterations, so both timing shapes are published. The separately sampled
process peaked at 18.102 MiB working set.

I also cloned the exact public commit on a Debian 12 Google `c4-highcpu-16`, built it natively
with Rust 1.97.0, passed all 11 tests, verified both Linux hashes, loaded/authenticated the model,
tokenized and generated from arbitrary text, and ran a 15-iteration AVX2 benchmark. That VM
measured 180.884 tok/s with four workers. It is a portability result, not a comparison with your
model.

Official lm-evaluation-harness 0.4.12, full validation sets, 0-shot, CPU, no sample limit, exact
values represented by the deployed artifact:

- PIQA: 55.2775% accuracy, 55.9304% normalized (1,838)
- HellaSwag: 26.4091%, 27.2456% normalized (10,042)
- ARC-Easy: 33.4596%, 32.8283% normalized (2,376)
- ARC-Challenge: 19.1126%, 23.2935% normalized (1,172)

I ran the uncompressed checkpoint through the identical harness too. Q3 retains 96.62% of
normalized PIQA, 97.09% HellaSwag, 84.23% ARC-Easy, and 98.20% ARC-Challenge. ARC-Easy is the
clear compression weakness. Compared with my original public v1.0 candidate, v1.1 improves PIQA
by 2.5572 points and ARC-Easy by 4.9243 points, leaves HellaSwag flat, and loses 0.6826 point on
ARC-Challenge. I am publishing the regression instead of averaging or omitting it.

For retrieval, I still keep lookup-and-extraction separate and do not call it model context. The
seeded 1M-token archive covers 12 task types: direct, lookalike, latest-wins, two-hop, absent,
Unicode IDs, long values, malformed records, missing pointers, punctuation, case collisions, and
adversarial abstention. It answered 560/560 correctly. The index is 42,741 bytes; loaded-index
latency was 5.460 us mean / 4 us p50 / 11 us p95. Reopening the archive per query without clearing
the OS cache was 32.941 us mean. Four-thread throughput was 183,068 queries/s with 8,400/8,400
correct.

Everything above has raw machine-readable evidence and SHA-256 hashes in the repo. The selected
model hash is:

`ef082637bd79dfa4d8e216003a71ce4fa11245f27d37e8c38bc0d1593f023140`

This completes my side, but I am not calling it a win yet. Send your public model, immutable
commit, artifacts, and exact commands, and I will run both entries on the same idle hardware with
the same prompt, task versions, thread count, warmup, and timing boundary. Then we can compare
size, CPU tok/s, retrieval accuracy/speed, and each standard benchmark as separate columns.
