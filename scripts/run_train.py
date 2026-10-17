import argparse
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from peek_future.env import DATA_ROOT, CKPT_ROOT, load_json
from peek_future.data_prep.tokenizer import SPETokenizer
from peek_future.models.biasing_model import BiasingModel
from peek_future.models.aed import AEDModel
from peek_future.baselines.clas import CLASModel
from peek_future.training.dataset import ASRBiasingDataset, collate_fn
from peek_future.training.train import get_lr_scheduler, train_epoch, evaluate_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ASR Biasing Model")
    parser.add_argument("--model", type=str, default="biasing", choices=["biasing", "aed", "clas"])
    parser.add_argument("--dataset", type=str, default="librispeech", choices=["librispeech", "vietmed"])
    parser.add_argument("--train_manifest", type=str, default=None)
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulate_grad", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--warmup_steps", type=int, default=5500)
    parser.add_argument("--k_heads", type=int, default=4)
    parser.add_argument("--mtp_head_type", type=str, default="linear", choices=["linear", "transformer"])
    parser.add_argument("--scorer_type", type=str, default="learned", choices=["learned", "heuristic"])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_name", type=str, default="best_model.pt")
    args = parser.parse_args()

    data_dir = DATA_ROOT / args.dataset
    tokenizer_path = data_dir / "tokenizer.model"
    entity_vocab_path = data_dir / "entity_vocab.json"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run scripts/run_data_prep.py first.")
    tokenizer = SPETokenizer(tokenizer_path)

    global_entities = load_json(entity_vocab_path) if entity_vocab_path.exists() else []

    train_m = args.train_manifest or str(next(data_dir.glob("*validation*.jsonl"), None))
    val_m = args.val_manifest or str(next(data_dir.glob("*test*.jsonl"), None))

    if not train_m or not Path(train_m).exists():
        raise FileNotFoundError(f"Train manifest not found: {train_m}")

    print(f"Loading train dataset: {train_m}")
    train_dataset = ASRBiasingDataset(
        manifest_path=train_m,
        tokenizer=tokenizer,
        global_entity_pool=global_entities,
        is_training=True,
        k_heads=args.k_heads,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id),
    )

    val_loader = None
    if val_m and Path(val_m).exists():
        val_dataset = ASRBiasingDataset(
            manifest_path=val_m,
            tokenizer=tokenizer,
            global_entity_pool=global_entities,
            is_training=False,
            k_heads=args.k_heads,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id),
        )

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Initialize model
    if args.model == "biasing":
        model = BiasingModel(
            vocab_size=tokenizer.vocab_size,
            k_heads=args.k_heads,
            mtp_head_type=args.mtp_head_type,
            scorer_type=args.scorer_type,
            pad_id=tokenizer.pad_id,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
        )
    elif args.model == "clas":
        model = CLASModel(
            vocab_size=tokenizer.vocab_size,
            pad_id=tokenizer.pad_id,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
        )
    else:
        model = AEDModel(
            vocab_size=tokenizer.vocab_size,
            pad_id=tokenizer.pad_id,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
        )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_lr_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_bwer = float("inf")
    save_path = CKPT_ROOT / args.ckpt_name

    print(f"Starting training: {args.epochs} epochs, {total_steps} total steps")
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            accumulate_grad=args.accumulate_grad,
        )
        print(f"Epoch {epoch} Training: loss={metrics['loss_total']:.4f}")

        if val_loader is not None:
            val_metrics = evaluate_validation(model, val_loader, tokenizer, device)
            print(
                f"Epoch {epoch} Validation: WER={val_metrics['wer']:.2f}% | "
                f"B-WER={val_metrics['b_wer']:.2f}% | U-WER={val_metrics['u_wer']:.2f}%"
            )
            if val_metrics["b_wer"] < best_val_bwer:
                best_val_bwer = val_metrics["b_wer"]
                torch.save(model.state_dict(), save_path)
                print(f"New best B-WER: {best_val_bwer:.2f}%. Saved checkpoint to {save_path}")
        else:
            torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main()
