# Third-party components

PICKLE-50M code is licensed under Apache-2.0. It does not claim authorship of third-party
pretraining, tokenization, evaluation software, or datasets.

| Component | Role | License | Source |
|---|---|---|---|
| Stentor3-50M | Source language-model weights and tokenizer files | Apache-2.0 | https://huggingface.co/StentorLabs/Stentor3-50M |
| TokenMonster | Native Rust port of encoding/decoding behavior; reference parity package/server | MIT | https://github.com/alasdairforsythe/tokenmonster |
| unicode-normalization | NFD normalization compiled into the native runtime | MIT OR Apache-2.0 | https://github.com/unicode-rs/unicode-normalization |
| unicode-general-category | Build-time generation of the Unicode 13 compatibility table | Apache-2.0 | https://github.com/yeslogic/unicode-general-category |
| Unicode Character Database | Unicode 13 category compatibility and post-13 age ranges | Unicode License v3 | https://www.unicode.org/license.txt |
| Rayon and Rayon Core | Persistent multicore data-parallel inference workers | MIT OR Apache-2.0 | https://github.com/rayon-rs/rayon |
| Crossbeam utilities/epoch/deque | Transitive work-stealing support for Rayon | MIT OR Apache-2.0 | https://github.com/crossbeam-rs/crossbeam |
| either | Transitive iterator utility used by Rayon | MIT OR Apache-2.0 | https://github.com/rayon-rs/either |
| lm-evaluation-harness | Standard benchmark runner | MIT | https://github.com/EleutherAI/lm-evaluation-harness |
| PyTorch | Expanded-weight reference evaluator | BSD-style | https://github.com/pytorch/pytorch |
| Transformers | Model architecture/evaluation adapter | Apache-2.0 | https://github.com/huggingface/transformers |

PIQA, HellaSwag, and ARC are downloaded by the evaluation harness and retain their own dataset
terms. They are not redistributed in this repository.

Required notices for code/data incorporated into the native tokenizer are under
`THIRD_PARTY_LICENSES/`.
