import argparse
from pathlib import Path
import subprocess
import sys
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from peek_future.env import RESULTS_ROOT


def run_cmd(cmd: list[str]) -> None:
    print(f"\n[Ablation Command]: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Table 2 Ablations and Figure 2 Lambda Sweep")
    parser.add_argument("--dataset", type=str, default="librispeech")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    args = parser.parse_args()

    ablations = [
        # (ID, model, k_heads, mtp_type, scorer_type, ckpt_name)
        ("A0", "aed", 0, "linear", "learned", "ablation_a0_aed.pt"),
        ("A1", "biasing", 4, "linear", "learned", "ablation_a1_linear.pt"),
        ("A2", "biasing", 4, "transformer", "learned", "ablation_a2_transformer.pt"),
        ("B0", "biasing", 4, "linear", "learned", "ablation_b0_learned.pt"),
        ("B1", "biasing", 4, "linear", "heuristic", "ablation_b1_heuristic.pt"),
        ("B2", "biasing", 1, "linear", "learned", "ablation_b2_1mtp.pt"),
        ("B3", "biasing", 2, "linear", "learned", "ablation_b3_2mtp.pt"),
    ]

    print("=" * 60)
    print("RUNNING TABLE 2 ABLATIONS")
    print("=" * 60)

    for ab_id, model_type, k, mtp_t, scorer_t, ckpt in ablations:
        print(f"\n>>> Running Ablation {ab_id} ({model_type}, K={k}, head={mtp_t}, scorer={scorer_t})")
        train_cmd = [
            args.python_bin,
            str(REPO_ROOT / "scripts" / "run_train.py"),
            "--model", model_type,
            "--dataset", args.dataset,
            "--train_manifest", args.train_manifest,
            "--val_manifest", args.val_manifest,
            "--epochs", str(args.epochs),
            "--k_heads", str(max(1, k)),
            "--mtp_head_type", mtp_t,
            "--scorer_type", scorer_t,
            "--ckpt_name", ckpt,
        ]
        run_cmd(train_cmd)

        eval_cmd = [
            args.python_bin,
            str(REPO_ROOT / "scripts" / "run_evaluate.py"),
            "--model", model_type,
            "--dataset", args.dataset,
            "--manifest", args.val_manifest,
            "--checkpoint", str(REPO_ROOT / "checkpoints" / ckpt),
            "--bias_size", "200" if ab_id.startswith("B") else "0",
            "--out_csv", "table2_ablations.csv",
        ]
        run_cmd(eval_cmd)

    print("\nAblations completed. Results saved in results/table2_ablations.csv")


if __name__ == "__main__":
    main()
