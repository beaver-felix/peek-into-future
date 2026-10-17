import math
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from peek_future.env import CKPT_ROOT, save_json
from peek_future.evaluation.metrics import compute_bwer_uwer
from peek_future.data_prep.tokenizer import SPETokenizer


def get_lr_scheduler(optimizer, warmup_steps: int = 5500, total_steps: int = 100000):
    """Linear warmup followed by cosine decay."""
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    grad_clip: float = 1.0,
    accumulate_grad: int = 1,
) -> dict[str, float]:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    total_mtp_loss = 0.0
    total_entity_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc="Training")

    for step, batch in enumerate(pbar):
        x = batch["x"].to(device)
        x_lengths = batch["x_lengths"].to(device)
        y = batch["y"].to(device)
        bias_tokens = batch["bias_tokens"].to(device)
        tau_labels = batch["tau_labels"].to(device)

        use_amp = scaler is not None and device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(
                x,
                x_lengths,
                y,
                bias_tokens=bias_tokens,
                tau_labels=tau_labels,
            )
            loss = outputs["loss_total"] / accumulate_grad

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accumulate_grad == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

        batch_loss = outputs["loss_total"].item()
        total_loss += batch_loss
        if "loss_mtp" in outputs:
            total_mtp_loss += outputs["loss_mtp"].item()
        if "loss_entity" in outputs:
            total_entity_loss += outputs["loss_entity"].item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{batch_loss:.3f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    return {
        "loss_total": total_loss / max(1, num_batches),
        "loss_mtp": total_mtp_loss / max(1, num_batches),
        "loss_entity": total_entity_loss / max(1, num_batches),
    }


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    dataloader: DataLoader,
    tokenizer: SPETokenizer,
    device: torch.device,
    lambda_param: float = 1.0,
    gamma_param: float = 0.3,
) -> dict[str, float]:
    """Evaluate model on validation split."""
    model.eval()
    hyps = []
    refs = []
    entity_lists = []

    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        x = batch["x"].to(device)
        x_lengths = batch["x_lengths"].to(device)
        bias_tokens = batch["bias_tokens"].to(device)
        batch_texts = batch["texts"]
        batch_entities = batch["raw_entities"]
        batch_bias_strs = batch["bias_list_strs"]

        for i in range(x.size(0)):
            x_i = x[i : i + 1]
            x_len_i = x_lengths[i : i + 1]
            bias_tok_i = bias_tokens[i : i + 1]
            bias_strs_i = batch_bias_strs[i]

            # Convert entity strings to token lists
            ent_tok_lists = []
            for e_str in bias_strs_i:
                if e_str == "<null>":
                    ent_tok_lists.append([tokenizer.pad_id])
                else:
                    ent_tok_lists.append(tokenizer.text_to_ids(e_str))

            if hasattr(model, "decode_unified"):
                pred_tokens = model.decode_unified(
                    x_i,
                    x_len_i,
                    bias_tok_i,
                    ent_tok_lists,
                    lambda_param=lambda_param,
                    gamma_param=gamma_param,
                )
            else:
                pred_tokens = model.greedy_decode(x_i, x_len_i)[0]

            pred_text = tokenizer.ids_to_text(pred_tokens)
            hyps.append(pred_text)
            refs.append(batch_texts[i])
            entity_lists.append(batch_entities[i])

    metrics = compute_bwer_uwer(hyps, refs, entity_lists)
    return metrics
