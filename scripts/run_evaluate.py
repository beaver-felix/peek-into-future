import argparse
from pathlib import Path
import sys
import polars as pl
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from peek_future.env import DATA_ROOT, RESULTS_ROOT, load_json
from peek_future.data_prep.tokenizer import SPETokenizer
from peek_future.models.biasing_model import BiasingModel
from peek_future.models.aed import AEDModel
from peek_future.baselines.clas import CLASModel
from peek_future.training.dataset import ASRBiasingDataset, collate_fn
from peek_future.training.train import evaluate_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ASR Biasing Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model", type=str, default="biasing", choices=["biasing", "aed", "clas"])
    parser.add_argument("--dataset", type=str, default="librispeech", choices=["librispeech", "vietmed"])
    parser.add_argument("--manifest", type=str, required=True, help="Evaluation split manifest path")
    parser.add_argument("--bias_size", type=int, default=100, choices=[0, 100, 200, 500])
    parser.add_argument("--lambda_val", type=float, default=1.0)
    parser.add_argument("--gamma_val", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_csv", type=str, default="eval_results.csv")
    args = parser.parse_args()

    data_dir = DATA_ROOT / args.dataset
    tokenizer = SPETokenizer(data_dir / "tokenizer.model")
    entity_vocab_path = data_dir / "entity_vocab.json"
    global_entities = load_json(entity_vocab_path) if entity_vocab_path.exists() else []

    eval_dataset = ASRBiasingDataset(
        manifest_path=args.manifest,
        tokenizer=tokenizer,
        global_entity_pool=global_entities,
        is_training=False,
        eval_bias_size=args.bias_size,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id),
    )

    device = torch.device(args.device)

    # Instantiate model
    if args.model == "biasing":
        model = BiasingModel(
            vocab_size=tokenizer.vocab_size,
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

    # Load weights
    print(f"Loading checkpoint from: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    print(f"Running evaluation with bias_size N={args.bias_size}, lambda={args.lambda_val}, gamma={args.gamma_val}...")
    metrics = evaluate_validation(
        model=model,
        dataloader=eval_loader,
        tokenizer=tokenizer,
        device=device,
        lambda_param=args.lambda_val,
        gamma_param=args.gamma_val,
    )

    print("\n" + "=" * 50)
    print("RESULTS:")
    print(f"WER:   {metrics['wer']:.2f}%")
    print(f"B-WER: {metrics['b_wer']:.2f}%")
    print(f"U-WER: {metrics['u_wer']:.2f}%")
    print("=" * 50)

    # Save to CSV using polars
    res_path = RESULTS_ROOT / args.out_csv
    row_data = {
        "model": [args.model],
        "dataset": [args.dataset],
        "manifest": [Path(args.manifest).stem],
        "bias_size": [args.bias_size],
        "lambda": [args.lambda_val],
        "gamma": [args.gamma_val],
        "wer": [metrics["wer"]],
        "b_wer": [metrics["b_wer"]],
        "u_wer": [metrics["u_wer"]],
        "total_words": [metrics["total_words"]],
        "b_words": [metrics["b_words"]],
        "u_words": [metrics["u_words"]],
    }
    df_new = pl.DataFrame(row_data)

    if res_path.exists():
        df_old = pl.read_csv(res_path)
        df_combined = pl.concat([df_old, df_new], how="diagonal")
        df_combined.write_csv(res_path)
    else:
        df_new.write_csv(res_path)
    print(f"Appended results to: {res_path}")


if __name__ == "__main__":
    main()
