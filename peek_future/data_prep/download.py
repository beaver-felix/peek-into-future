from pathlib import Path
from typing import Any
from huggingface_hub import snapshot_download
from datasets import load_dataset
from peek_future.env import DATA_ROOT, HF_CACHE

CLEAN_SPLIT_MAP = {
    "validation.clean": "validation",
    "validation": "validation",
    "test.clean": "test",
    "test": "test",
    "train.clean.100": "train.100",
    "train.100": "train.100",
    "train.clean.360": "train.360",
    "train.360": "train.360",
}


def download_librispeech(splits: list[str] | None = None) -> dict[str, Any]:
    """Snapshot and load LibriSpeech clean subset via HuggingFace Hub.

    Bypasses whole-dataset download (>100GB) and non-working streaming by
    snapshotting only parquet files belonging to requested clean splits.
    """
    if splits is None:
        splits = [
            "train.clean.100",
            "validation.clean",
            "test.clean",
        ]

    # Map requested splits to clean subset folders & patterns
    patterns = []
    active_splits = []
    for s in splits:
        if s in CLEAN_SPLIT_MAP:
            folder = CLEAN_SPLIT_MAP[s]
            patterns.append(f"clean/{folder}/*")
            active_splits.append((s, folder))
        else:
            print(f"Skipping split '{s}': only clean subset supported on Kaggle.")

    if not patterns:
        patterns = ["clean/validation/*", "clean/test/*"]
        active_splits = [("validation.clean", "validation"), ("test.clean", "test")]

    snapshot_dir = DATA_ROOT / "librispeech" / "raw_parquet"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Snapshotting clean subset from openslr/librispeech_asr (patterns: {patterns})...")
    local_repo = snapshot_download(
        repo_id="openslr/librispeech_asr",
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(snapshot_dir),
        resume_download=True,
    )

    dataset_dict = {}
    for split_key, folder_name in active_splits:
        split_dir = Path(local_repo) / "clean" / folder_name
        parquet_files = sorted(str(p) for p in split_dir.glob("*.parquet"))
        if not parquet_files:
            print(f"Warning: No parquet files found for {split_key} in {split_dir}")
            continue

        print(f"Loading {split_key} ({len(parquet_files)} parquet files)...")
        ds = load_dataset("parquet", data_files=parquet_files, split="train")
        if "audio" in ds.column_names:
            import datasets
            ds = ds.cast_column("audio", datasets.Audio(decode=False))
        dataset_dict[split_key] = ds

    return dataset_dict


def download_vietmed(splits: list[str] | None = None) -> dict[str, Any]:
    """Download VietMed-NER splits via HuggingFace datasets."""
    if splits is None:
        splits = ["train", "validation", "test"]

    dataset_dict = {}
    for split in splits:
        print(f"Downloading VietMed split: {split}...")
        try:
            ds = load_dataset(
                "leduckhai/VietMed-NER",
                split=split,
                cache_dir=str(HF_CACHE),
                trust_remote_code=True,
            )
            if "audio" in ds.column_names:
                import datasets
                ds = ds.cast_column("audio", datasets.Audio(decode=False))
            dataset_dict[split] = ds
        except Exception as e:
            print(f"Note: Could not load split {split} directly: {e}")
    return dataset_dict
