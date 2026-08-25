# PICKLE-50M v1.0.0

First public, reproducible release of the no-training PICKLE-50M challenge entry.

## Release contents

- Authenticated native V3 model with embedded compact TokenMonster vocabulary:
  `15,128,816` bytes.
- Windows inference-only pair: `15,495,408` bytes (`14.7776 MiB`).
- Linux inference-only pair: `15,715,664` bytes (`14.9876 MiB`).
- Audited PKLM quantization artifact with provenance and `training_performed: false`.
- Native arbitrary-text TokenMonster encoder/decoder, packed Q2/Q4 inference, AVX2 and persistent
  worker support.
- Full source, standard benchmark JSON, retrieval evidence, validation tools, checksums, and
  licensing disclosures.

## Validation highlights

- TokenMonster parity: `609,889 / 609,889` token IDs exact.
- Native/reference greedy parity: `16 / 16` generated token IDs exact.
- Corruption rejection: `18 / 18` malformed native models rejected.
- Linux release smoke: matching hashes; authenticated load, arbitrary-text generation, and AVX2
  benchmark passed in Google Cloud Shell.
- Retrieval: `560 / 560` correct across 12 seeded adversarial task types.
- Full, zero-shot PIQA, HellaSwag, ARC-Easy, and ARC-Challenge results are included unfiltered.

See `RESULTS.md` for timing boundaries, machine details, raw-result paths, limitations, and the
protocol required before making any same-hardware winner claim. Verify downloads with
`CHECKSUMS.sha256`.
