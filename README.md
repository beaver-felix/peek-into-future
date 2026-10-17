# Peeking Into The Future For Contextual Biasing (arXiv:2512.17657)

Reimplementation of:
> **Peeking Into The Future For Contextual Biasing**  
> Ramaneswaran Selvakumar, Cindy Tseng, Eesung Kim, Vijendra Raj Apsingekar, Yun Tang  
> Samsung Research America, ICASSP 2026.

## Structure

```
peek-into-future/
├── peek_future/
│   ├── env.py                  # Environment & path detection (local vs Kaggle)
│   ├── configs/                # Hydra configs (base, local, kaggle, datasets)
│   ├── models/                 # Conformer, Decoder, MTPModule, EntityScorer, BiasingModel
│   ├── baselines/              # CLASModel baseline
│   ├── data_prep/              # Dataset download, spaCy NER, tokenizer, manifests
│   ├── training/               # Dataset loader, loss, training loop
│   ├── inference/              # Unified search space decoding
│   └── evaluation/             # WER, B-WER, U-WER metrics (Le et al., 2021)
├── scripts/
│   ├── setup.py                # Environment setup & model download
│   ├── run_data_prep.py        # Data preparation & tokenizer training
│   ├── run_train.py            # Model training entry point
│   ├── run_evaluate.py         # Evaluation & CSV logging
│   └── run_ablations.py        # Table 2 ablations & lambda sweep
├── tests/                      # Unit test suite
├── pyproject.toml              # Build & local dependencies
└── requirements.txt            # Kaggle pip requirements
```

---

## Quickstart

### 1. Local Machine (using `uv`)

```bash
# Setup venv & dependencies
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .
uv run python scripts/setup.py

# Run tests
uv run python tests/test_model.py
uv run python tests/test_metrics.py

# Prepare LibriSpeech data
uv run python scripts/run_data_prep.py --dataset librispeech

# Train proposed BiasingModel
uv run python scripts/run_train.py --model biasing --dataset librispeech --epochs 100

# Evaluate at N=100 with lambda=4.4
uv run python scripts/run_evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --manifest data/librispeech/test.clean_manifest.jsonl \
    --bias_size 100 \
    --lambda_val 4.4
```

---

### 2. Kaggle (Bash Interface)

```bash
# In Kaggle terminal:
git clone https://github.com/<your-username>/peek-into-future.git /kaggle/working/repo
cd /kaggle/working/repo

# Install dependencies and download spaCy models
PEEK_ENV=kaggle python scripts/setup.py

# Run tests
python tests/test_model.py
python tests/test_metrics.py

# Prepare dataset (LibriSpeech or VietMed)
PEEK_ENV=kaggle python scripts/run_data_prep.py --dataset librispeech

# Train BiasingModel on Kaggle GPU
PEEK_ENV=kaggle python scripts/run_train.py \
    --model biasing \
    --dataset librispeech \
    --epochs 100 \
    --batch_size 16 \
    --device cuda

# Evaluate
PEEK_ENV=kaggle python scripts/run_evaluate.py \
    --checkpoint /kaggle/working/checkpoints/best_model.pt \
    --manifest /kaggle/working/data/librispeech/test.clean_manifest.jsonl \
    --bias_size 100 \
    --lambda_val 4.4
```

---

## Downstream: VietMed-NER

```bash
# Data preparation with Vietnamese NER
python scripts/run_data_prep.py --dataset vietmed

# Train on VietMed
python scripts/run_train.py --model biasing --dataset vietmed --epochs 100
```
