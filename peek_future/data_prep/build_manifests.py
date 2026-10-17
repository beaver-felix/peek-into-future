import io
from pathlib import Path
from typing import Any
import numpy as np
import soundfile as sf
import torch
import tqdm
from peek_future.env import DATA_ROOT, save_jsonl, save_json
from peek_future.data_prep.annotate_ner import load_spacy_model, extract_entities
from peek_future.data_prep.annotate_ner_vi import extract_entities_vi


def extract_audio_samples(audio_info: Any) -> tuple[np.ndarray, int]:
    """Extract (audio_array, sampling_rate) supporting AudioDecoder, dict, Tensor, and bytes."""
    if hasattr(audio_info, "get_all_samples") or "AudioDecoder" in type(audio_info).__name__:
        arr = audio_info["array"]
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        sr = int(audio_info["sampling_rate"])
        return np.asarray(arr, dtype=np.float32), sr

    if isinstance(audio_info, dict):
        if "array" in audio_info:
            arr = audio_info["array"]
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()
            sr = int(audio_info.get("sampling_rate", 16000))
            return np.asarray(arr, dtype=np.float32), sr
        if "bytes" in audio_info and audio_info["bytes"] is not None:
            arr, sr = sf.read(io.BytesIO(audio_info["bytes"]))
            return np.asarray(arr, dtype=np.float32), int(sr)

    if isinstance(audio_info, torch.Tensor):
        return audio_info.cpu().numpy().astype(np.float32), 16000

    if isinstance(audio_info, np.ndarray):
        return audio_info.astype(np.float32), 16000

    if isinstance(audio_info, (str, Path)) and Path(audio_info).exists():
        arr, sr = sf.read(str(audio_info))
        return np.asarray(arr, dtype=np.float32), int(sr)

    raise ValueError(f"Unrecognized audio format in item: {type(audio_info)} -> {audio_info}")


def process_librispeech_split(
    hf_split_dataset: Any,
    split_name: str,
    output_dir: Path | str,
    spacy_model: str = "en_core_web_trf",
) -> Path:
    """Process a LibriSpeech HuggingFace split into NeMo format JSONL manifest with NER annotations."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio" / split_name
    audio_dir.mkdir(parents=True, exist_ok=True)

    nlp = load_spacy_model(spacy_model)
    records = []

    print(f"Processing LibriSpeech {split_name} ({len(hf_split_dataset)} samples)...")
    for idx, item in enumerate(tqdm.tqdm(hf_split_dataset)):
        item_id = item.get("id", f"{split_name}_{idx}")
        audio_info = item["audio"]
        audio_array, sr = extract_audio_samples(audio_info)

        audio_path = audio_dir / f"{item_id}.wav"
        if not audio_path.exists():
            sf.write(str(audio_path), audio_array, sr)

        duration = float(len(audio_array)) / float(sr)
        text = item.get("text", "").strip()

        entities = extract_entities(text, nlp)

        record = {
            "id": item_id,
            "audio_filepath": str(audio_path),
            "duration": duration,
            "text": text,
            "entities": entities,
        }
        records.append(record)

    manifest_path = out_dir / f"{split_name}_manifest.jsonl"
    save_jsonl(records, manifest_path)
    print(f"Saved {len(records)} records to {manifest_path}")
    return manifest_path


def process_vietmed_split(
    hf_split_dataset: Any,
    split_name: str,
    output_dir: Path | str,
) -> Path:
    """Process VietMed HuggingFace split into NeMo format JSONL manifest with Vietnamese NER annotations."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio" / split_name
    audio_dir.mkdir(parents=True, exist_ok=True)

    records = []
    print(f"Processing VietMed {split_name} ({len(hf_split_dataset)} samples)...")
    for idx, item in enumerate(tqdm.tqdm(hf_split_dataset)):
        item_id = item.get("id", f"{split_name}_{idx}")
        audio_info = item.get("audio")
        if audio_info is not None:
            audio_array, sr = extract_audio_samples(audio_info)
            audio_path = audio_dir / f"{item_id}.wav"
            if not audio_path.exists():
                sf.write(str(audio_path), audio_array, sr)
            duration = float(len(audio_array)) / float(sr)
        else:
            audio_path = out_dir / "placeholder.wav"
            duration = 0.0

        # Transcript text column in VietMed can be 'transcription', 'sentence', or 'text'
        text = (
            item.get("transcription")
            or item.get("sentence")
            or item.get("text")
            or ""
        ).strip()

        entities = extract_entities_vi(text)

        record = {
            "id": item_id,
            "audio_filepath": str(audio_path),
            "duration": duration,
            "text": text,
            "entities": entities,
        }
        records.append(record)

    manifest_path = out_dir / f"{split_name}_manifest.jsonl"
    save_jsonl(records, manifest_path)
    print(f"Saved {len(records)} records to {manifest_path}")
    return manifest_path
