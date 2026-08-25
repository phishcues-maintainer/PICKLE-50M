Deal. The Google approval came through, so I finished the no-training entry and replaced my weak
first candidate before replying.

GitHub: https://github.com/phishcues-maintainer/PICKLE-50M

Release: https://github.com/phishcues-maintainer/PICKLE-50M/releases/tag/v1.1.0

The source is the public Apache-2.0 `SEN-AGI/Sable-1.1-30M` checkpoint: 31,171,072 parameters.
Its authors pretrained it; I did zero training, fine-tuning, distillation, or example-based
calibration. I implemented deterministic weight-only Q3 compression and a self-contained Rust CPU
runtime.

The authenticated model includes the complete tokenizer and is 12,534,009 bytes. Model plus
inference runtime is 12.3201 MiB on Windows and 12.5075 MiB on Linux.

Arbitrary-text tokenization is native now—no Python, Transformers, TokenMonster, or external
tokenizer at deployment. It matched the reference fast tokenizer on 663,409/663,409 checked token
IDs. Native inference also matched the expanded-Q3 reference for 16/16 greedy tokens, and the
loader rejected 18/18 corrupted models.

On my i5-7500, AVX2 packed decode with four workers measured 111.460 tok/s in the fixed-ID kernel
profile and 92.033 tok/s in the separate 15-iteration arbitrary-text run. Sampled peak working set
was 18.102 MiB.

Full 0-shot lm-evaluation-harness results:

- PIQA: 55.9304% normalized
- HellaSwag: 27.2456%
- ARC-Easy: 32.8283%
- ARC-Challenge: 23.2935%

These are modest tiny-base-model scores. Against matched FP32, Q3’s largest loss is 6.14 points on
ARC-Easy, and ARC-Challenge is 0.68 point below my v1.0 candidate. I published both regressions.

Retrieval remains a separate lookup system, not “model context.” The seeded 1M-token test answered
560/560 across 12 task types; its 42,741-byte index measured 5.460 us mean lookup latency and
183,068 queries/s with four threads.

Everything—including source, Windows/Linux binaries, raw JSON, matched FP32 results, CI, exact
commands, hashes, and limitations—is public.

I am not calling this a win before a fair comparison. Send your public model, immutable commit,
artifacts, and commands, and I’ll run both on the same idle hardware with identical prompts,
threads, task versions, warmup, and timing boundaries.
