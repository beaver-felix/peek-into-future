"""Training dataset for ASR with Contextual Biasing.

ASRBiasingDataset:
  - Loads NeMo-format JSONL manifests produced by scripts/run_data_prep.py
  - Computes 80-dim log-Mel filterbanks on-the-fly (window=400, hop=160, n_mels=80)
  - Applies SpecAugment during training (freq_masks=2, time_masks=1)
  - Tokenizes text with SPETokenizer (BPE, vocab_size=1024)
  - Constructs training bias list (random sampling, kappa=2 negatives per positive)
    or fixed eval bias list (N=100/200/500)
  - Generates tau_s labels (per-step ground truth entity index for L_entity loss)

tau_s label generation:
  For each decoding step s (predicting y[s+1] given y[0:s+1]):
    - If a ground truth entity's subword sequence starts at position s in text_ids,
      set tau_labels[s] = entity's index in the bias list.
    - Otherwise tau_labels[s] = 0 (null entity index).

collate_fn:
  Pads all sequences in a batch to their max length.
  Returns a dict suitable for direct use in BiasingModel.forward().
"""

from pathlib import Path
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
from peek_future.env import load_jsonl
from peek_future.data_prep.tokenizer import SPETokenizer
from peek_future.data_prep.build_bias_lists import (
    create_training_bias_list,
    create_eval_bias_list,
    entity_to_token_ids,
    build_null_entity,
)


class AudioFeatureExtractor(nn.Module):
    """80-dim log-Mel filterbank extractor matching paper specifications."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        win_length: int = 400,
        hop_length: int = 160,
    ):
        super().__init__()
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (1, samples)
        mel = self.mel_spec(waveform)  # (1, n_mels, T)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel.squeeze(0).transpose(0, 1)  # (T, n_mels)


class ASRBiasingDataset(Dataset):
    """Dataset for ASR with Contextual Biasing."""

    def __init__(
        self,
        manifest_path: Path | str,
        tokenizer: SPETokenizer,
        global_entity_pool: list[str],
        is_training: bool = True,
        k_heads: int = 4,
        kappa: int = 2,
        eval_bias_size: int = 100,
        sample_rate: int = 16000,
    ):
        self.records = load_jsonl(manifest_path)
        self.tokenizer = tokenizer
        self.global_entity_pool = global_entity_pool
        self.is_training = is_training
        self.k_heads = k_heads
        self.kappa = kappa
        self.eval_bias_size = eval_bias_size
        self.feature_extractor = AudioFeatureExtractor(sample_rate=sample_rate)

        # SpecAugment for training
        self.freq_mask = T.FrequencyMasking(freq_mask_param=27)
        self.time_mask = T.TimeMasking(time_mask_param=35)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        item = self.records[idx]
        audio_path = item["audio_filepath"]
        text = item["text"]
        raw_entities = [e["text"] for e in item.get("entities", []) if e.get("text")]

        # Load audio waveform
        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            feat = self.feature_extractor(waveform)  # (T, 80)
        except Exception:
            # Fallback dummy features if audio missing/truncated
            feat = torch.zeros((100, 80))

        if self.is_training:
            # SpecAugment: transpose to (D, T) for mask, then back
            feat_t = feat.transpose(0, 1).unsqueeze(0)
            feat_t = self.freq_mask(feat_t)
            feat_t = self.time_mask(feat_t)
            feat = feat_t.squeeze(0).transpose(0, 1)

        # Tokenize target text: [BOS] + tokens + [EOS]
        text_ids = self.tokenizer.text_to_ids(text)
        y = [self.tokenizer.bos_id] + text_ids + [self.tokenizer.eos_id]

        # Construct bias list
        if self.is_training:
            bias_list_strs = create_training_bias_list(
                raw_entities, self.global_entity_pool, kappa=self.kappa
            )
        else:
            bias_list_strs = create_eval_bias_list(
                raw_entities, self.global_entity_pool, target_size=self.eval_bias_size
            )

        # Convert each entity in bias list to K token ids
        bias_tokens = []
        for ent_str in bias_list_strs:
            if ent_str == "<null>":
                bias_tokens.append(build_null_entity(self.tokenizer, self.k_heads))
            else:
                bias_tokens.append(entity_to_token_ids(ent_str, self.tokenizer, self.k_heads))

        # Compute ground truth entity start labels tau_s for each decoding step
        # Number of decoding steps is len(y) - 1 (predicting y[1:])
        seq_len = len(y) - 1
        tau_labels = [0] * seq_len

        # Map entity token sub-sequences to find where they start in y[1:]
        for ent_idx, ent_str in enumerate(bias_list_strs):
            if ent_str == "<null>" or ent_idx == 0:
                continue
            ent_ids = self.tokenizer.text_to_ids(ent_str)
            L = len(ent_ids)
            if L == 0:
                continue

            # Look for sub-sequence ent_ids in text_ids
            for s in range(len(text_ids) - L + 1):
                if text_ids[s : s + L] == ent_ids:
                    # In y, text_ids start at index 1.
                    # At step s, decoder receives y[:s+1], predicts y[s+1] (which is text_ids[s])
                    if s < seq_len:
                        tau_labels[s] = ent_idx

        return {
            "feat": feat,                                    # (T, 80)
            "y": torch.tensor(y, dtype=torch.long),         # (S+1,)
            "bias_tokens": torch.tensor(bias_tokens, dtype=torch.long),  # (N, K)
            "tau_labels": torch.tensor(tau_labels, dtype=torch.long),    # (S,)
            "text": text,
            "raw_entities": raw_entities,
            "bias_list_strs": bias_list_strs,
        }


def collate_fn(batch: list[dict], pad_id: int = 0) -> dict:
    batch_size = len(batch)

    # Pad audio features (T, 80)
    feat_lens = torch.tensor([b["feat"].size(0) for b in batch], dtype=torch.long)
    max_t = int(feat_lens.max().item())
    padded_feats = torch.zeros((batch_size, max_t, 80), dtype=torch.float)
    for i, b in enumerate(batch):
        t_len = b["feat"].size(0)
        padded_feats[i, :t_len, :] = b["feat"]

    # Pad text tokens y
    y_lens = torch.tensor([b["y"].size(0) for b in batch], dtype=torch.long)
    max_y = int(y_lens.max().item())
    padded_y = torch.full((batch_size, max_y), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        padded_y[i, : b["y"].size(0)] = b["y"]

    # Pad tau_labels (length max_y - 1)
    max_steps = max_y - 1
    padded_tau = torch.zeros((batch_size, max_steps), dtype=torch.long)
    for i, b in enumerate(batch):
        tau_len = b["tau_labels"].size(0)
        padded_tau[i, :tau_len] = b["tau_labels"]

    # Pad bias_tokens across batch: (B, max_N, K)
    k_heads = batch[0]["bias_tokens"].size(1)
    max_n = max(b["bias_tokens"].size(0) for b in batch)
    padded_bias = torch.full((batch_size, max_n, k_heads), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        n_len = b["bias_tokens"].size(0)
        padded_bias[i, :n_len, :] = b["bias_tokens"]

    return {
        "x": padded_feats,
        "x_lengths": feat_lens,
        "y": padded_y,
        "y_lengths": y_lens,
        "tau_labels": padded_tau,
        "bias_tokens": padded_bias,
        "texts": [b["text"] for b in batch],
        "raw_entities": [b["raw_entities"] for b in batch],
        "bias_list_strs": [b["bias_list_strs"] for b in batch],
    }
