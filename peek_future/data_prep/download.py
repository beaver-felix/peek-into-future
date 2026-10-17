from pathlib import Path
from typing import Any
from datasets import load_dataset
from peek_future.env import DATA_ROOT, HF_CACHE


def download_librispeech(splits: list[str] | None = None) -> dict[str, Any]:
    """Download LibriSpeech splits via HuggingFace datasets."""
    if splits is None:
        splits = [
            "validation.clean",
            "validation.other",
            "test.clean",
            "test.other",
        ]

    dataset_dict = {}
    for split in splits:
        print(f"Downloading LibriSpeech split: {split}...")
        ds = load_dataset(
            "openslr/librispeech_asr",
            split=split,
            cache_dir=str(HF_CACHE),
            trust_remote_code=True,
        )
        dataset_dict[split] = ds
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
            dataset_dict[split] = ds
        except Exception as e:
            print(f"Note: Could not load split {split} directly: {e}")
    return dataset_dict
