from pathlib import Path
import sentencepiece as spm


class SPETokenizer:
    """SentencePiece BPE Tokenizer wrapper."""

    def __init__(self, model_path: Path | str):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Tokenizer model not found at {self.model_path}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(self.model_path))

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    def text_to_ids(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def ids_to_text(self, ids: list[int]) -> str:
        # filter out special tokens like pad, bos, eos
        clean_ids = [i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.sp.decode(clean_ids)

    def text_to_tokens(self, text: str) -> list[str]:
        return self.sp.encode(text, out_type=str)

    @classmethod
    def train(
        cls,
        text_filepath: Path | str,
        output_prefix: Path | str,
        vocab_size: int = 1024,
        model_type: str = "bpe",
    ) -> "SPETokenizer":
        output_path = Path(output_prefix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        spm.SentencePieceTrainer.train(
            input=str(text_filepath),
            model_prefix=str(output_path),
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )
        return cls(f"{output_prefix}.model")
