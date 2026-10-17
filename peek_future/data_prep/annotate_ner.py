from typing import Any
import spacy


def _patch_thinc_dlpack() -> None:
    """Fix deprecation warning: use cupy.from_dlpack array constructor instead of legacy to_dlpack."""
    try:
        import thinc.util
        import thinc.compat
        import torch.utils.dlpack

        def modern_torch2xp(torch_tensor, *, ops=None):
            from thinc.api import NumpyOps
            thinc.util.assert_pytorch_installed()
            if thinc.util.is_torch_cuda_array(torch_tensor):
                if isinstance(ops, NumpyOps):
                    return torch_tensor.detach().cpu().numpy()
                if thinc.compat.cupy is not None:
                    # Direct array constructor per DLPack standard
                    return thinc.compat.cupy.from_dlpack(torch_tensor)
                return thinc.compat.cupy_from_dlpack(torch.utils.dlpack.to_dlpack(torch_tensor))
            if isinstance(ops, NumpyOps) or ops is None:
                return torch_tensor.detach().cpu().numpy()
            return thinc.compat.cupy.asarray(torch_tensor)

        thinc.util.torch2xp = modern_torch2xp
    except (ImportError, AttributeError) as exc:
        print(f"Notice: thinc DLPack patch deferred ({exc})")


_patch_thinc_dlpack()

EXCLUDED_LABELS = {
    "CARDINAL",
    "ORDINAL",
    "PERCENT",
    "QUANTITY",
    "MONEY",
    "DATE",
    "TIME",
}


def load_spacy_model(model_name: str = "en_core_web_trf") -> spacy.language.Language:
    """Load spaCy model, falling back to en_core_web_sm if trf not found."""
    try:
        gpu_enabled = spacy.prefer_gpu()
        if gpu_enabled:
            print("spaCy initialized with GPU acceleration.")
    except Exception as exc:
        print(f"Notice: spaCy GPU activation failed ({exc}), running on CPU.")

    try:
        return spacy.load(model_name)
    except Exception as exc:
        print(f"Warning: could not load {model_name} ({exc}), attempting fallback to en_core_web_sm")
        try:
            return spacy.load("en_core_web_sm")
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Neither {model_name} nor en_core_web_sm found ({fallback_exc}). Run: python -m spacy download {model_name}"
            ) from fallback_exc


def extract_entities(
    text: str, nlp: spacy.language.Language, excluded_labels: set[str] | None = None
) -> list[dict[str, Any]]:
    """Extract named entities from text excluding numeral types."""
    if excluded_labels is None:
        excluded_labels = EXCLUDED_LABELS

    # If text is all-uppercase (standard LibriSpeech), convert to Title Case for spaCy NER accuracy
    analysis_text = text.title() if text.isupper() else text
    doc = nlp(analysis_text)

    entities = []
    for ent in doc.ents:
        if ent.label_ in excluded_labels:
            continue
        # Map span back to original text slice
        ent_text = text[ent.start_char : ent.end_char].strip()
        if not ent_text:
            continue
        entities.append(
            {
                "text": ent_text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            }
        )
    return entities
