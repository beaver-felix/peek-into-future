from typing import Any

EXCLUDED_LABELS_VI = {"NUM", "CARDINAL", "QUANTITY", "DATE", "TIME"}


def extract_entities_vi(text: str) -> list[dict[str, Any]]:
    """Extract named entities from Vietnamese text using underthesea."""
    try:
        from underthesea import ner
    except ImportError:
        raise ImportError("underthesea not installed. Install via pip install underthesea")

    results = ner(text)
    # underthesea returns list of tuples: (token, pos, chunk, ner_tag)
    # ner_tags: 'O', 'B-PER', 'I-PER', 'B-LOC', etc.
    entities: list[dict[str, Any]] = []
    current_tokens: list[str] = []
    current_label = ""
    current_start = -1

    cursor = 0
    for item in results:
        token = item[0]
        ner_tag = item[3] if len(item) > 3 else "O"

        # Find token offset in text
        pos = text.find(token, cursor)
        if pos == -1:
            pos = cursor
        cursor = pos + len(token)

        if ner_tag.startswith("B-"):
            if current_tokens:
                ent_text = " ".join(current_tokens)
                entities.append(
                    {
                        "text": ent_text,
                        "label": current_label,
                        "start_char": current_start,
                        "end_char": current_start + len(ent_text),
                    }
                )
            label = ner_tag[2:]
            if label not in EXCLUDED_LABELS_VI:
                current_tokens = [token]
                current_label = label
                current_start = pos
            else:
                current_tokens = []
                current_label = ""
                current_start = -1

        elif ner_tag.startswith("I-") and current_tokens:
            label = ner_tag[2:]
            if label == current_label:
                current_tokens.append(token)
            else:
                ent_text = " ".join(current_tokens)
                entities.append(
                    {
                        "text": ent_text,
                        "label": current_label,
                        "start_char": current_start,
                        "end_char": current_start + len(ent_text),
                    }
                )
                current_tokens = []
                current_label = ""
                current_start = -1
        else:
            if current_tokens:
                ent_text = " ".join(current_tokens)
                entities.append(
                    {
                        "text": ent_text,
                        "label": current_label,
                        "start_char": current_start,
                        "end_char": current_start + len(ent_text),
                    }
                )
                current_tokens = []
                current_label = ""
                current_start = -1

    if current_tokens:
        ent_text = " ".join(current_tokens)
        entities.append(
            {
                "text": ent_text,
                "label": current_label,
                "start_char": current_start,
                "end_char": current_start + len(ent_text),
            }
        )

    return entities
