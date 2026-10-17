from typing import Any
import spacy

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
        return spacy.load(model_name)
    except Exception as e:
        print(f"Warning: could not load {model_name} ({e}), attempting fallback to en_core_web_sm")
        try:
            return spacy.load("en_core_web_sm")
        except Exception:
            raise RuntimeError(
                f"Neither {model_name} nor en_core_web_sm found. Run: python -m spacy download {model_name}"
            )


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
