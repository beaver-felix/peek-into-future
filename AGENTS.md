# AGENTS.md — Agent Onboarding Guide for `peek-into-future`

## What this repo is

Reimplementation of **"Peeking Into The Future For Contextual Biasing"** (arXiv:2512.17657, ICASSP 2026).

The model is an **Attention-Based Encoder-Decoder (AED) ASR system** extended with:
1. **Multi-Token Prediction (MTP)** — K parallel prediction heads, sharing the final LM projection weight W_o.
2. **Entity Scorer** — learns to score a dynamic bias list using the MTP logit vectors, without any cross-attention or external encoder.
3. **Unified Search Space** — merges the static vocab and dynamic entity list into a single decoding distribution Q.

---

## Execution environments

| | Local | Kaggle (2x NVIDIA T4) |
|---|---|---|
| Runner | `uv run python scripts/...` | `PEEK_ENV=kaggle torchrun --nproc_per_node=2 scripts/run_train.py ...` |
| Device | CPU or CUDA | 2x CUDA T4 (DDP with NCCL) |
| Paths | Resolved by `peek_future/env.py` | Same, auto-detected via `Path("/kaggle/working").exists()` |
| Install | `uv pip install -e .` | `pip install -r requirements.txt` |
| Parallelism | Single GPU / CPU | DistributedDataParallel (DDP), `model.no_sync()`, 4 vCPU workers |

**Critical rule**: all path resolution must go through `peek_future.env`. No other file may hardcode absolute paths.


---

## Directory map

```
peek_future/
  env.py              Central path resolver + orjson I/O helpers
  configs/            Hydra-compatible YAML configs (base, local, kaggle, per-dataset)
  models/
    encoder.py        ConformerEncoder: Conv2dSubsampling4 → 12 ConformerBlocks
    decoder.py        AttentionDecoder: 6-layer Pre-LN Transformer
    aed.py            AEDModel: combines encoder + decoder + shared W_o (baseline A0)
    mtp_heads.py      MTPModule: K FFN or Transformer heads sharing W_o (rows A1, A2)
    entity_scorer.py  EntityScorer: extracts p_n vectors from MTP logits, scores via f_θ (rows B0, B1)
    biasing_model.py  BiasingModel: full proposed model with L_MTP + L_entity + unified decode
  baselines/
    clas.py           CLASModel: LSTM bias encoder + cross-attention (Table 1 baseline)
  data_prep/
    download.py       HuggingFace dataset loader (LibriSpeech & VietMed)
    annotate_ner.py   English NER using spaCy en_core_web_trf (excludes CARDINAL, ORDINAL)
    annotate_ner_vi.py Vietnamese NER using underthesea
    tokenizer.py      SentencePiece BPE wrapper (1024 vocab, pad=0, unk=1, bos=2, eos=3)
    build_bias_lists.py Per-sample training bias list (κ=2) and fixed eval lists (N=100/200/500)
    build_manifests.py Convert HuggingFace splits → NeMo-format JSONL with entity annotations
  training/
    dataset.py        ASRBiasingDataset: fbank extraction, SpecAugment, τ_s label generation
    train.py          train_epoch + evaluate_validation (AMP, TF32, fused Adam, non-blocking I/O)
    visualize.py      4-panel SVG training dashboard (loss curves, WER/B-WER/U-WER, best epoch)
  evaluation/
    metrics.py        compute_wer (jiwer), compute_bwer_uwer (word-level Levenshtein partition)
  inference/
    decode.py         decode_utterance wrapper around BiasingModel.decode_unified
scripts/
  setup.py            One-shot environment setup (deps + spaCy model download)
  run_data_prep.py    CLI: download → NER annotate → build JSONL manifests → train tokenizer → entity vocab
  run_train.py        CLI: train AEDModel / CLASModel / BiasingModel with live SVG curves
  run_evaluate.py     CLI: evaluate at N={0,100,200,500}, log results to CSV via polars
  run_ablations.py    CLI: Table 2 ablation sweep (A0–A2, B0–B3)
tests/
  test_model.py       Shape/backward pass tests for all model classes
  test_metrics.py     WER / B-WER / U-WER correctness tests
  test_bias_lists.py  Bias list construction tests
  test_visualize.py   SVG output and CSV export tests
```

---

## Paper implementation mapping

| Paper element | Code location |
|---|---|
| Eq. 1 — AR factorization | `AEDModel.forward` in `models/aed.py` |
| Eq. 2–3 — MTP parallel prediction | `MTPModule.forward` in `models/mtp_heads.py` |
| Eq. 4 — Entity logit vector p_n | `EntityScorer.extract_entity_logits` in `models/entity_scorer.py` |
| Eq. 5 — P_e entity posterior | `EntityScorer.forward` in `models/entity_scorer.py` |
| Eq. 6 — L_MTP loss (weighted sum) | `BiasingModel.compute_mtp_loss` in `models/biasing_model.py` |
| Eq. 7 — L_entity loss | `BiasingModel.compute_entity_loss` in `models/biasing_model.py` |
| Eq. 8 — Unified search space Q | `BiasingModel.decode_unified` in `models/biasing_model.py` |
| Table 1 — Main results | Reproduced via `scripts/run_train.py` + `scripts/run_evaluate.py` |
| Table 2 — Ablations | Reproduced via `scripts/run_ablations.py` |
| Figure 2 — Lambda sweep | `scripts/run_evaluate.py` with varying `--lambda_val` |
| B-WER / U-WER (Le et al. 2021) | `evaluation/metrics.py::compute_bwer_uwer` |

---

## Key hyperparameters (all in `configs/base.yaml`)

| Param | Value | Location in code |
|---|---|---|
| K (MTP heads) | 4 | `BiasingModel(k_heads=4)` |
| α (MTP loss weights) | [1.0, 0.2, 0.1, 0.05] | `BiasingModel.alpha` |
| κ (neg entity ratio) | 2 | `create_training_bias_list(kappa=2)` |
| λ default / tuned | 1.0 / 4.4 | `decode_unified(lambda_param=...)` |
| γ (confidence threshold) | 0.3 | `decode_unified(gamma_param=0.3)` |
| Tokenizer vocab | 1024 BPE | `SPETokenizer.train(vocab_size=1024)` |
| pad / unk / bos / eos | 0 / 1 / 2 / 3 | `SPETokenizer` properties |

---

## Invariants that must never be broken

1. **Shared W_o**: `BiasingModel.lm_head` is the single Linear(d_model, V) shared by base decoder and ALL K MTP heads. Do not create separate lm_head instances per head.
2. **Null entity at index 0**: bias_list[0] is always `"<null>"` / all-pad tokens. `tau_labels=0` means no entity at that step. Do not reorder bias lists.
3. **Token padding**: entities with fewer than K tokens are right-padded with `tokenizer.pad_id`. Entities with more than K tokens are truncated to first K. This is done in `entity_to_token_ids`.
4. **Entity index 0 == null in τ_s**: during training, `tau_labels` of 0 means no entity. Positive entity n has `tau_labels == n` (1-indexed in bias list).
5. **B-WER target**: `val_b_wer` is the **primary checkpoint selection metric**. Checkpointing is only triggered when B-WER improves.
6. **Env isolation**: all file paths must come from `peek_future.env` constants. No hardcoded paths anywhere else.
7. **SVG output**: training plots are saved as `.svg`. The visualizer enforces this even if a caller passes a `.png` path.

---

## How to add a new dataset

1. Create a HuggingFace download function in `data_prep/download.py`.
2. Create or reuse a NER annotation function (`annotate_ner.py` for EN, `annotate_ner_vi.py` for VI, or add a new file).
3. Create a manifest processing function in `build_manifests.py`.
4. Add a new YAML config in `configs/train_<dataset>.yaml` mirroring `train_vietmed.yaml`.
5. Wire everything in `scripts/run_data_prep.py` under a new `--dataset` choice.

## How to add a new model variant (ablation)

1. Subclass `AEDModel` or `BiasingModel` in `models/`.
2. Add a new `--model` choice to `scripts/run_train.py`.
3. Add a new row in `scripts/run_ablations.py::ablations` list.

## Running tests

```bash
# Local
uv run pytest tests/ -v

# Kaggle
PEEK_ENV=kaggle python -m pytest tests/ -v
```

## Running the full pipeline (LibriSpeech)

```bash
# 1. Data
uv run python scripts/run_data_prep.py --dataset librispeech --splits train.clean.100 validation.clean test.clean

# 2. Train
uv run python scripts/run_train.py --model biasing --dataset librispeech --epochs 100 --compile

# 3. Evaluate (Table 1)
for N in 0 100 200 500; do
  for lam in 1.0 4.4; do
    uv run python scripts/run_evaluate.py \
      --checkpoint checkpoints/best_model.pt \
      --manifest data/librispeech/test.clean_manifest.jsonl \
      --bias_size $N --lambda_val $lam
  done
done

# 4. Ablations (Table 2)
uv run python scripts/run_ablations.py \
  --train_manifest data/librispeech/validation.clean_manifest.jsonl \
  --val_manifest data/librispeech/test.clean_manifest.jsonl
```
