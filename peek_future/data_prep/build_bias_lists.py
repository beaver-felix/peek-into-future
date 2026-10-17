import random
from typing import Sequence
import polars as pl
from peek_future.data_prep.tokenizer import SPETokenizer


def collect_unique_entities(manifest_records: Sequence[dict]) -> list[str]:
    """Collect unique entity texts across manifest records."""
    entities = set()
    for item in manifest_records:
        for ent in item.get("entities", []):
            txt = ent.get("text", "").strip()
            if txt:
                entities.add(txt)
    return sorted(list(entities))


def entity_to_token_ids(
    entity_text: str, tokenizer: SPETokenizer, k_heads: int = 4
) -> list[int]:
    """Tokenize entity text, pad to k_heads with pad_id or truncate to k_heads."""
    tokens = tokenizer.text_to_ids(entity_text)
    if len(tokens) < k_heads:
        tokens = tokens + [tokenizer.pad_id] * (k_heads - len(tokens))
    else:
        tokens = tokens[:k_heads]
    return tokens


def build_null_entity(tokenizer: SPETokenizer, k_heads: int = 4) -> list[int]:
    """Return null entity ∅ composed of pad tokens."""
    return [tokenizer.pad_id] * k_heads


def create_training_bias_list(
    sample_entities: list[str],
    global_entity_pool: list[str],
    kappa: int = 2,
    min_entities: int = 1,
    max_entities: int = 4,
) -> tuple[list[str], list[int]]:
    """
    Construct training bias list B = [∅] + B+ + B_neg.
    Returns: (bias_list_strings, positive_indices_in_sample)
    Index 0 is always null entity ∅.
    """
    # Sample between min_entities and max_entities from sample_entities
    if sample_entities:
        num_pos = min(len(sample_entities), random.randint(min_entities, max_entities))
        pos_entities = random.sample(sample_entities, num_pos)
    else:
        pos_entities = []

    num_neg = kappa * max(1, len(pos_entities))
    # Sample negative entities from global pool that are not in pos_entities
    neg_candidates = [e for e in global_entity_pool if e not in pos_entities]
    if len(neg_candidates) >= num_neg:
        neg_entities = random.sample(neg_candidates, num_neg)
    else:
        neg_entities = neg_candidates

    # Form list with null entity at index 0
    bias_list = ["<null>"] + pos_entities + neg_entities
    return bias_list


def create_eval_bias_list(
    sample_entities: list[str],
    global_entity_pool: list[str],
    target_size: int,
) -> list[str]:
    """
    Construct evaluation bias list for target_size N (e.g. 100, 200, 500).
    Includes [∅] + all sample ground truth entities + random negatives to fill target_size.
    Index 0 is always null entity ∅.
    """
    if target_size == 0:
        return ["<null>"]

    # Deduplicate sample entities
    unique_gt = list(dict.fromkeys(sample_entities))
    bias_list = ["<null>"] + unique_gt

    remaining = target_size - len(unique_gt)
    if remaining > 0:
        neg_candidates = [e for e in global_entity_pool if e not in unique_gt]
        if len(neg_candidates) >= remaining:
            sampled_negs = random.sample(neg_candidates, remaining)
        else:
            sampled_negs = neg_candidates
        bias_list.extend(sampled_negs)

    return bias_list
