from pathlib import Path
from peek_future.training.visualize import plot_training_progress, save_training_history_csv
import polars as pl


def test_visualization(tmp_path: Path):
    dummy_history = [
        {
            "epoch": 1,
            "loss_total": 4.52,
            "loss_mtp": 3.21,
            "loss_entity": 1.31,
            "lr": 1e-4,
            "val_wer": 15.2,
            "val_b_wer": 28.5,
            "val_u_wer": 12.1,
        },
        {
            "epoch": 2,
            "loss_total": 3.80,
            "loss_mtp": 2.70,
            "loss_entity": 1.10,
            "lr": 5e-4,
            "val_wer": 12.4,
            "val_b_wer": 21.0,
            "val_u_wer": 10.3,
        },
        {
            "epoch": 3,
            "loss_total": 3.10,
            "loss_mtp": 2.20,
            "loss_entity": 0.90,
            "lr": 1e-3,
            "val_wer": 9.8,
            "val_b_wer": 15.5,
            "val_u_wer": 8.5,
        },
    ]

    csv_file = tmp_path / "test_history.csv"
    save_training_history_csv(dummy_history, csv_file)
    assert csv_file.exists()
    df = pl.read_csv(csv_file)
    assert len(df) == 3

    plot_file = tmp_path / "test_curves.svg"
    out_plot = plot_training_progress(dummy_history, plot_file, title_suffix="(Test)")
    assert out_plot.exists()
    assert out_plot.suffix == ".svg"
    assert out_plot.stat().st_size > 0
    content = out_plot.read_text(encoding="utf-8")
    assert "<svg" in content


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        print("Running visualization tests...")
        test_visualization(Path(td))
        print("Visualization tests passed!")
