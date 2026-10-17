import argparse
import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from peek_future.env import DATA_ROOT, CKPT_ROOT, RESULTS_ROOT, load_json
from peek_future.data_prep.tokenizer import SPETokenizer
from peek_future.models.biasing_model import BiasingModel
from peek_future.models.aed import AEDModel
from peek_future.baselines.clas import CLASModel
from peek_future.training.dataset import ASRBiasingDataset, collate_fn
from peek_future.training.train import get_lr_scheduler, train_epoch, evaluate_validation
from peek_future.training.visualize import plot_training_progress, save_training_history_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ASR Biasing Model")
    parser.add_argument("--model", type=str, default="biasing", choices=["biasing", "aed", "clas"])
    parser.add_argument("--dataset", type=str, default="librispeech", choices=["librispeech", "vietmed"])
    parser.add_argument("--train_manifest", type=str, default=None)
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--accumulate_grad", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--warmup_steps", type=int, default=5500)
    parser.add_argument("--k_heads", type=int, default=4)
    parser.add_argument("--mtp_head_type", type=str, default="linear", choices=["linear", "transformer"])
    parser.add_argument("--scorer_type", type=str, default="learned", choices=["learned", "heuristic"])
    parser.add_argument("--num_workers", type=int, default=2, help="Workers per GPU process (e.g. 2 for 4 vCPUs on 2x T4)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true", help="Compile model using torch.compile")
    parser.add_argument("--ckpt_name", type=str, default="best_model.pt")
    args = parser.parse_args()

    # Detect distributed training (e.g. torchrun --nproc_per_node=2)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1

    if is_distributed:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        rank = 0
        device = torch.device(args.device)

    is_main_process = (rank == 0)

    def log(msg: str) -> None:
        if is_main_process:
            print(msg)

    data_dir = DATA_ROOT / args.dataset
    tokenizer_path = data_dir / "tokenizer.model"
    entity_vocab_path = data_dir / "entity_vocab.json"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run scripts/run_data_prep.py first.")
    tokenizer = SPETokenizer(tokenizer_path)

    global_entities = load_json(entity_vocab_path) if entity_vocab_path.exists() else []

    if args.train_manifest:
        train_m = args.train_manifest
    else:
        train_candidates = sorted(list(data_dir.glob("*train*.jsonl")))
        if train_candidates:
            train_m = str(train_candidates[0])
        else:
            val_fallback = sorted(list(data_dir.glob("*validation*.jsonl")))
            train_m = str(val_fallback[0]) if val_fallback else None

    if args.val_manifest:
        val_m = args.val_manifest
    else:
        val_candidates = sorted(list(data_dir.glob("*val*.jsonl")))
        if val_candidates:
            val_m = str(val_candidates[0])
        else:
            test_fallback = sorted(list(data_dir.glob("*test*.jsonl")))
            val_m = str(test_fallback[0]) if test_fallback else None

    if not train_m or not Path(train_m).exists():
        raise FileNotFoundError(f"Train manifest not found: {train_m}")

    log(f"Process rank: {rank}/{world_size} | Local rank: {local_rank} | Device: {device}")
    log(f"Global effective batch size: {args.batch_size * world_size * args.accumulate_grad}")

    # Performance optimization: TF32 and cuDNN benchmark for modern GPUs
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    log(f"Loading train dataset: {train_m}")
    train_dataset = ASRBiasingDataset(
        manifest_path=train_m,
        tokenizer=tokenizer,
        global_entity_pool=global_entities,
        is_training=True,
        k_heads=args.k_heads,
    )

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if is_distributed
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
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
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=2 if args.num_workers > 0 else None,
            collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id),
        )

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

    # Optional torch.compile (applied before DDP wrap)
    if args.compile:
        try:
            log("Compiling model via torch.compile...")
            model = torch.compile(model)
        except Exception as e:
            log(f"Warning: torch.compile failed ({e}), continuing uncompiled.")

    # Wrap in DistributedDataParallel if running multi-GPU
    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    try:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, fused=(device.type == "cuda")
        )
    except Exception as exc:
        log(f"Notice: Fused Adam unavailable ({exc}). Using standard Adam.")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_lr_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_bwer = float("inf")
    save_path = CKPT_ROOT / args.ckpt_name
    history = []
    plot_stem = Path(args.ckpt_name).stem
    plot_dir = RESULTS_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"{plot_stem}_curves.svg"
    history_csv = RESULTS_ROOT / f"{plot_stem}_history.csv"

    log(f"Starting training: {args.epochs} epochs, {total_steps} total steps per process")
    for epoch in range(1, args.epochs + 1):
        log(f"\n--- Epoch {epoch}/{args.epochs} ---")
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            accumulate_grad=args.accumulate_grad,
            is_main_process=is_main_process,
        )
        log(f"Epoch {epoch} Training: loss={metrics['loss_total']:.4f}")

        epoch_entry = {
            "epoch": epoch,
            "loss_total": round(metrics["loss_total"], 4),
            "loss_mtp": round(metrics.get("loss_mtp", 0.0), 4),
            "loss_entity": round(metrics.get("loss_entity", 0.0), 4),
            "lr": scheduler.get_last_lr()[0],
        }

        # Validation & Checkpointing evaluated on main process
        if is_main_process:
            # Unwrap model for clean checkpoint saving and inference
            raw_model = model.module if hasattr(model, "module") else model
            raw_model = getattr(raw_model, "_orig_mod", raw_model)

            if val_loader is not None:
                val_metrics = evaluate_validation(
                    raw_model,
                    val_loader,
                    tokenizer,
                    device,
                    is_main_process=True,
                )
                print(
                    f"Epoch {epoch} Validation: WER={val_metrics['wer']:.2f}% | "
                    f"B-WER={val_metrics['b_wer']:.2f}% | U-WER={val_metrics['u_wer']:.2f}%"
                )
                epoch_entry["val_wer"] = val_metrics["wer"]
                epoch_entry["val_b_wer"] = val_metrics["b_wer"]
                epoch_entry["val_u_wer"] = val_metrics["u_wer"]

                if val_metrics["b_wer"] < best_val_bwer:
                    best_val_bwer = val_metrics["b_wer"]
                    torch.save(raw_model.state_dict(), save_path)
                    print(f"New best B-WER: {best_val_bwer:.2f}%. Saved clean checkpoint to {save_path}")
            else:
                torch.save(raw_model.state_dict(), save_path)

            history.append(epoch_entry)
            save_training_history_csv(history, history_csv)
            plot_training_progress(
                history, plot_path, title_suffix=f"({args.model.upper()} on {args.dataset})"
            )
            print(f"Saved training visualization dashboard to: {plot_path}")

        if is_distributed:
            dist.barrier()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
