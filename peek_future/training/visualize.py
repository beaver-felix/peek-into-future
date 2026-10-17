"""Training visualization: 4-panel SVG dashboard and CSV history export.

plot_training_curves:
  Generates a 4-panel matplotlib figure and saves as .svg (forced regardless of extension).
  Panels:
    1. Train loss vs epoch (with optional rolling mean overlay)
    2. Validation loss vs epoch
    3. WER / B-WER / U-WER vs epoch (three lines)
    4. Learning rate schedule vs epoch

  Best epoch is annotated on B-WER panel (primary checkpoint metric).
  Output: <output_path>.svg

export_history_csv:
  Dumps per-epoch training history dict to a polars DataFrame and writes to CSV.
  One row per epoch, columns: epoch, train_loss, val_loss, wer, b_wer, u_wer, lr.
  Used for offline analysis and LaTeX table generation.

Usage:
  history = {"train_loss": [...], "val_loss": [...], "wer": [...],
             "b_wer": [...], "u_wer": [...], "lr": [...]}
  plot_training_curves(history, "results/training_curves.svg")
  export_history_csv(history, "results/training_history.csv")
"""

from pathlib import Path
from typing import Sequence
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / Kaggle / bash
import matplotlib.pyplot as plt
import polars as pl


def save_training_history_csv(history: list[dict], output_csv: Path | str) -> None:
    """Save epoch history list to CSV using polars."""
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(history)
    df.write_csv(path)


def plot_training_progress(
    history: Sequence[dict],
    output_path: Path | str,
    title_suffix: str = "",
) -> Path:
    """
    Generate training visualization dashboard.
    Subplots:
      1. Training Loss (Total)
      2. Loss Breakdown (MTP Loss vs Entity Loss)
      3. Validation Metrics (WER, B-WER, U-WER)
      4. B-WER Trend & Best Mark
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not history:
        return out_file

    epochs = [h["epoch"] for h in history]
    total_losses = [h["loss_total"] for h in history]
    mtp_losses = [h.get("loss_mtp", 0.0) for h in history]
    entity_losses = [h.get("loss_entity", 0.0) for h in history]

    has_val = "val_b_wer" in history[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Progress Dashboard {title_suffix}".strip(), fontsize=16, fontweight="bold")

    # Subplot 1: Total Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, total_losses, "b-o", linewidth=2, label="Total Loss")
    ax1.set_title("Total Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Subplot 2: Multi-task Loss Breakdown
    ax2 = axes[0, 1]
    ax2.plot(epochs, mtp_losses, "g-s", linewidth=1.8, label="L_MTP")
    ax2.plot(epochs, entity_losses, "m-^", linewidth=1.8, label="L_entity")
    ax2.set_title("Loss Decomposition (MTP vs Entity)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    # Subplot 3: Validation Metrics (WER, B-WER, U-WER)
    ax3 = axes[1, 0]
    if has_val:
        wers = [h.get("val_wer", 0.0) for h in history]
        b_wers = [h.get("val_b_wer", 0.0) for h in history]
        u_wers = [h.get("val_u_wer", 0.0) for h in history]

        ax3.plot(epochs, wers, "k-o", label="Overall WER")
        ax3.plot(epochs, b_wers, "r-s", linewidth=2, label="Biased WER (B-WER)")
        ax3.plot(epochs, u_wers, "c-^", label="Unbiased WER (U-WER)")
        ax3.set_title("Validation Error Rates (%)")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Error Rate (%)")
        ax3.grid(True, linestyle="--", alpha=0.6)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "No Validation Metrics Available", ha="center", va="center")
        ax3.set_title("Validation Metrics")

    # Subplot 4: B-WER Detail with Best Epoch Highlight
    ax4 = axes[1, 1]
    if has_val:
        b_wers = [h.get("val_b_wer", 0.0) for h in history]
        ax4.plot(epochs, b_wers, color="tab:red", marker="s", linewidth=2, label="B-WER")

        best_idx = int(min(range(len(b_wers)), key=lambda i: b_wers[i]))
        best_epoch = epochs[best_idx]
        best_bwer = b_wers[best_idx]

        ax4.axvline(x=best_epoch, color="gray", linestyle=":", label=f"Best (Ep {best_epoch})")
        ax4.scatter([best_epoch], [best_bwer], color="gold", s=150, zorder=5, edgecolor="black", label=f"Min: {best_bwer:.2f}%")

        ax4.set_title("Biased Word Error Rate (Target Metric)")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("B-WER (%)")
        ax4.grid(True, linestyle="--", alpha=0.6)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, "No Validation Metrics Available", ha="center", va="center")
        ax4.set_title("Biased WER Detail")

    plt.tight_layout()
    # Force .svg output format
    if out_file.suffix.lower() != ".svg":
        out_file = out_file.with_suffix(".svg")
    plt.savefig(out_file, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_file
