import argparse
from pathlib import Path
import sys

# Ensure package root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from peek_future.env import DATA_ROOT, save_json, load_jsonl
from peek_future.data_prep.download import download_librispeech, download_vietmed
from peek_future.data_prep.build_manifests import process_librispeech_split, process_vietmed_split
from peek_future.data_prep.tokenizer import SPETokenizer
from peek_future.data_prep.build_bias_lists import collect_unique_entities


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data preparation pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        default="librispeech",
        choices=["librispeech", "vietmed"],
        help="Target dataset",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Specific splits to process",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=1024,
        help="SentencePiece BPE vocab size",
    )
    parser.add_argument(
        "--spacy_model",
        type=str,
        default="en_core_web_trf",
        help="spaCy model for NER (LibriSpeech)",
    )
    args = parser.parse_args()

    out_dir = DATA_ROOT / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting data preparation for: {args.dataset}")
    print(f"Output directory: {out_dir}")

    manifest_paths = []
    if args.dataset == "librispeech":
        splits = args.splits or ["train.clean.100", "validation.clean", "test.clean"]
        datasets_dict = download_librispeech(splits)
        for split_name, ds in datasets_dict.items():
            m_path = process_librispeech_split(ds, split_name, out_dir, spacy_model=args.spacy_model)
            manifest_paths.append(m_path)
    else:
        splits = args.splits or ["validation", "test"]
        datasets_dict = download_vietmed(splits)
        for split_name, ds in datasets_dict.items():
            m_path = process_vietmed_split(ds, split_name, out_dir)
            manifest_paths.append(m_path)

    # Collect texts and entities across processed manifests
    all_records = []
    text_corpus_path = out_dir / "text_corpus.txt"
    with open(text_corpus_path, "w", encoding="utf-8") as f_text:
        for m_path in manifest_paths:
            records = load_jsonl(m_path)
            all_records.extend(records)
            for r in records:
                txt = r.get("text", "").strip()
                if txt:
                    f_text.write(txt + "\n")

    # Train SentencePiece BPE tokenizer
    tokenizer_prefix = out_dir / "tokenizer"
    print(f"Training SentencePiece BPE tokenizer (vocab_size={args.vocab_size})...")
    SPETokenizer.train(
        text_filepath=text_corpus_path,
        output_prefix=tokenizer_prefix,
        vocab_size=args.vocab_size,
        model_type="bpe",
    )
    print(f"Tokenizer saved at {tokenizer_prefix}.model")

    # Extract global entity pool
    global_entities = collect_unique_entities(all_records)
    entity_vocab_path = out_dir / "entity_vocab.json"
    save_json(global_entities, entity_vocab_path)
    print(f"Collected {len(global_entities)} unique entities. Saved to {entity_vocab_path}")

    print("Data preparation completed successfully.")


if __name__ == "__main__":
    main()
