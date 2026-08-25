Deal. I took the no-training route and built the reproducible entry.

GitHub: https://github.com/phishcues-maintainer/PICKLE-50M

Versioned release and binaries: https://github.com/phishcues-maintainer/PICKLE-50M/releases/tag/v1.0.0

Full results and claim boundaries: https://github.com/phishcues-maintainer/PICKLE-50M/blob/main/RESULTS.md

The base is the public Apache-2.0 `StentorLabs/Stentor3-50M` checkpoint: 49,430,016 parameters.
I did no training, fine-tuning, distillation, or calibration. I used deterministic mixed Q2/Q4
post-training quantization, then wrote a native Rust CPU path for the packed weights: RMSNorm,
RoPE, grouped-query attention, KV cache, SwiGLU, output projection, and greedy generation. PyTorch
and model frameworks are not runtime dependencies.

The authenticated model, including the compact TokenMonster vocabulary, is 15,128,816 bytes.
With the inference-only runtimes:

- Windows: 15,495,408 bytes total, 14.7776 MiB
- Linux: 15,715,664 bytes total, 14.9876 MiB — 12,976 bytes under 15 MiB

The same model file is used on both systems. I also kept the larger full utility binaries, which
include retrieval generation/indexing, separate from the inference deployment claim.

I uploaded the published Linux pair to Google Cloud Shell and ran it on x86_64 Linux rather than
claiming a cross-compiled binary was enough. Both uploaded hashes matched. The authenticated model
loaded with AVX2 and four workers, native arbitrary-text generation ran, and a three-iteration
benchmark completed at 23.960 tok/s. That shared-cloud number is a portability smoke result, not
the same-hardware speed comparison below; the machine and raw output are in `results/linux-smoke.json`.

Arbitrary-text tokenization is native now. NFD normalization, TokenMonster's Unicode-13-compatible
NoCapcode handling, ungreedy segmentation, and decoding are compiled into the executable.
`--prompt` and `--prompt-file` need no Python or reference tokenizer. Against TokenMonster 1.1.12,
the native encoder matched 609,889/609,889 checked token IDs across fixed cases, a 46,106-character
mixed-script/adversarial corpus, and 135,955 Unicode letter/number/mark scalars.

I checked native inference against the exact quantized values reconstructed in the Hugging Face
Llama reference. On the parity prompt, all 4,096 logits had 0.00001240 maximum absolute error;
argmax matched 704/704, cosine was 1.0, and greedy generation matched 16/16 token IDs.

On my i5-7500 Windows machine, packed decode after prefill:

- scalar, 1 thread: 26.475 tok/s
- AVX2, 1 thread: 29.625 tok/s
- AVX2, 4 persistent workers: 67.622 tok/s

Generated IDs were identical across all three paths. A separate arbitrary-text end-to-end test
with 32 generated tokens measured 143.1 ms warm mean time-to-first-token, 595.9 ms mean total
request time, and 53.696 generated tok/s. The inference-only process peaked at 20.707 MiB working
set in a separately sampled run.

The model format now authenticates dimensions, tensor kinds, packed data, scales, and tokenizer
bytes before parsing. A deterministic suite rejected all 18/18 truncation, mutation, and
trailing-byte cases.

Official lm-evaluation-harness 0.4.12, 0-shot, full validation sets, exact values represented by
the selected artifact:

- PIQA: 54.9510% accuracy, 53.3732% normalized (1,838 examples)
- HellaSwag: 26.4887%, 27.2356% normalized (10,042 examples)
- ARC-Easy: 26.6835%, 27.9040% normalized (2,376 examples)
- ARC-Challenge: 19.9659%, 23.9761% normalized (1,172 examples)

The scores are modest; this is a tiny base model under aggressive compression. I’m publishing all
four rather than selecting only favorable tasks.

For retrieval, I still keep lookup-and-extraction separate and do not call it model context. The
strengthened seeded 1M-token test now has 12 task types, including lookalikes, Unicode IDs,
latest-wins duplicates, missing/malformed pointers, case collisions, punctuation, long values,
two-hop lookups, and adversarial abstention. It answered 560/560 correctly. The 6,787,177-byte
archive used a 42,741-byte index; loaded-index latency was 5.031 us mean, 4 us p50, and 11 us p95.
Reopening the archive for every query — without clearing the OS page cache — measured 44.543 us
mean. Four-thread throughput was 151,829 queries/s with 2,800/2,800 correct. The older 100M-token
scale tier remains 1,000/1,000 correct and is reported separately.

Selected model SHA-256:
`5d0442b66bf39de8e4cd6bc12ce4ca998d9f10c3fe5d6234debb509774f1cc3c`

Windows inference runtime SHA-256:
`38093b8425585509a1f7d04f90d46ab9dab507c02f1382c389c8c21bde33769d`

Linux inference runtime SHA-256:
`eb9bab69f8112e9379feccb223f41f1c916648d88f3e3bfff807eaa9f7f61d88`

The package includes source, selected artifact, Windows/Linux inference binaries, a one-command
challenge runner, tokenizer/native parity tests, corruption tests, raw JSON, quantization audit,
retrieval generator, exact hashes, and license disclosures. Release downloads can be checked
against `CHECKSUMS.sha256`.

This completes my side, but it is not a “win” until your model, commit, and commands are public and
both entries run on the same idle hardware with the same prompts, context, task versions, thread
count, and timing boundaries. Send those and I’ll run the head-to-head.
