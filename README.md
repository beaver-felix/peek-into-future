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

### 2. Kaggle (Bash Interface — 2x NVIDIA T4 GPUs)

Maximize the Kaggle environment (2x T4 GPUs, 32GB combined VRAM, 4 vCPUs):

```bash
# In Kaggle terminal:
git clone https://github.com/<your-username>/peek-into-future.git /kaggle/working/repo
cd /kaggle/working/repo

# Install dependencies and download spaCy models
PEEK_ENV=kaggle python scripts/setup.py

# Run tests
python -m pytest tests/ -v

# Prepare dataset (LibriSpeech or VietMed)
PEEK_ENV=kaggle python scripts/run_data_prep.py --dataset librispeech

# Train BiasingModel on 2x T4 GPUs using Distributed Data Parallel (DDP)
PEEK_ENV=kaggle torchrun --nproc_per_node=2 scripts/run_train.py \
    --model biasing \
    --dataset librispeech \
    --epochs 100 \
    --batch_size 16 \
    --accumulate_grad 2 \
    --num_workers 2

# Evaluate at N=100 with lambda=4.4
PEEK_ENV=kaggle python scripts/run_evaluate.py \
    --checkpoint /kaggle/working/checkpoints/best_model.pt \
    --manifest /kaggle/working/data/librispeech/test.clean_manifest.jsonl \
    --bias_size 100 \
    --lambda_val 4.4
```

#### 2x T4 Hardware Optimization Architecture:
- **Distributed Data Parallel (DDP)**: 2 processes via `torchrun --nproc_per_node=2` on NCCL backend across both T4 GPUs.
- **Micro-Batch Decoupling (`model.no_sync()`)**: Gradient all-reduce is skipped during gradient accumulation steps, eliminating PCIe bus bottlenecks.
- **VRAM Utilization**: Batch size 16 per GPU $\times$ 2 GPUs $\times$ 2 accumulation steps = global effective batch size 64.
- **CPU Saturation**: 2 DataLoader workers per GPU (4 workers total) fully utilize Kaggle's 4 vCPUs with `pin_memory=True`, `persistent_workers=True`, and `prefetch_factor=2`.
- **Turing Tensor Cores**: Autocast FP16 AMP (`dtype=torch.float16`) with dynamic `GradScaler` for maximum T4 tensor core throughput.
- **Fused Optimizer**: Fused Adam kernel reduces optimizer CUDA launch overhead.
- **Clean Model Unwrapping**: Checkpoints are automatically unwrapped from DDP (`model.module`) for single-GPU inference and evaluation.


---

## Downstream: VietMed-NER

```bash
# Data preparation with Vietnamese NER
python scripts/run_data_prep.py --dataset vietmed

# Train on VietMed
python scripts/run_train.py --model biasing --dataset vietmed --epochs 100
```
