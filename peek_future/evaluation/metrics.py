"""Evaluation metrics: WER, Biased WER (B-WER), and Unbiased WER (U-WER).

Follows the evaluation protocol of Le et al. (2021):
  "Contextualized Streaming End-to-End Speech Recognition with Trie-Based Deep Biasing"

Definitions:
  B-WER: WER computed only over words that are part of named entities in the reference.
  U-WER: WER computed only over words that are NOT part of named entities.
  Overall WER: combined error rate over all words.

Implementation approach for B-WER / U-WER:
  1. Find every contiguous word span in the reference that matches a named entity string.
  2. Mark those reference word positions as 'biased'.
  3. Run Levenshtein alignment between reference and hypothesis.
  4. For each edit (substitution, deletion, insertion), attribute the error to the
     biased or unbiased partition based on the aligned reference word's label.
     Insertions are attributed to the preceding reference word's partition.

Note on entity text matching:
  Matching is case-insensitive and whitespace-normalized.
  Entities are matched as contiguous multi-word spans, not individual tokens.
"""

from typing import Sequence
import jiwer


def compute_wer(hyps: Sequence[str], refs: Sequence[str]) -> float:
    """Standard Word Error Rate across hypotheses and references."""
    if not hyps or not refs:
        return 0.0
    return float(jiwer.wer(list(refs), list(hyps))) * 100.0


def align_words_levenshtein(ref_words: list[str], hyp_words: list[str]) -> list[tuple[str | None, str | None, str]]:
    """
    Computes Levenshtein alignment between ref_words and hyp_words.
    Returns list of operations: (ref_word, hyp_word, op) where op is 'correct', 'sub', 'del', 'ins'.
    """
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    ops = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        ops[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        ops[0][j] = "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1].lower() == hyp_words[j - 1].lower():
                dp[i][j] = dp[i - 1][j - 1]
                ops[i][j] = "correct"
            else:
                cost_sub = dp[i - 1][j - 1] + 1
                cost_del = dp[i - 1][j] + 1
                cost_ins = dp[i][j - 1] + 1

                min_cost = min(cost_sub, cost_del, cost_ins)
                dp[i][j] = min_cost
                if min_cost == cost_sub:
                    ops[i][j] = "sub"
                elif min_cost == cost_del:
                    ops[i][j] = "del"
                else:
                    ops[i][j] = "ins"

    # Backtrace
    i, j = n, m
    alignment = []
    while i > 0 or j > 0:
        op = ops[i][j]
        if op == "correct" or op == "sub":
            alignment.append((ref_words[i - 1], hyp_words[j - 1], op))
            i -= 1
            j -= 1
        elif op == "del":
            alignment.append((ref_words[i - 1], None, "del"))
            i -= 1
        elif op == "ins":
            alignment.append((None, hyp_words[j - 1], "ins"))
            j -= 1
        else:
            break

    alignment.reverse()
    return alignment


def compute_bwer_uwer(
    hyps: Sequence[str],
    refs: Sequence[str],
    entity_lists: Sequence[Sequence[str]],
) -> dict[str, float]:
    """
    Compute overall WER, Biased WER (B-WER), and Unbiased WER (U-WER) following Le et al., 2021.
    Args:
        hyps: List of hypothesis transcript strings
        refs: List of reference transcript strings
        entity_lists: List of entity strings present in each reference
    Returns:
        dict with keys: 'wer', 'b_wer', 'u_wer', 'total_words', 'b_words', 'u_words'
    """
    total_sub_b = total_del_b = total_ins_b = total_ref_b = 0
    total_sub_u = total_del_u = total_ins_u = total_ref_u = 0

    for hyp_str, ref_str, ents in zip(hyps, refs, entity_lists):
        ref_words = ref_str.strip().split()
        hyp_words = hyp_str.strip().split()

        # Identify which reference words are part of entities
        is_entity_word = [False] * len(ref_words)
        normalized_ents = [e.strip().lower().split() for e in ents if e.strip()]

        for ent_words in normalized_ents:
            L = len(ent_words)
            if L == 0:
                continue
            for start in range(len(ref_words) - L + 1):
                window = [w.lower() for w in ref_words[start : start + L]]
                if window == ent_words:
                    for k in range(start, start + L):
                        is_entity_word[k] = True

        alignment = align_words_levenshtein(ref_words, hyp_words)

        ref_idx = 0
        for ref_w, hyp_w, op in alignment:
            if ref_w is not None:
                is_biased = is_entity_word[ref_idx]
                ref_idx += 1

                if is_biased:
                    total_ref_b += 1
                    if op == "sub":
                        total_sub_b += 1
                    elif op == "del":
                        total_del_b += 1
                else:
                    total_ref_u += 1
                    if op == "sub":
                        total_sub_u += 1
                    elif op == "del":
                        total_del_u += 1
            else:
                # Insertion error: attribute to previous word type if available
                prev_is_biased = is_entity_word[ref_idx - 1] if ref_idx > 0 else False
                if prev_is_biased:
                    total_ins_b += 1
                else:
                    total_ins_u += 1

    b_errors = total_sub_b + total_del_b + total_ins_b
    u_errors = total_sub_u + total_del_u + total_ins_u
    total_errors = b_errors + u_errors
    total_words = total_ref_b + total_ref_u

    wer = (total_errors / max(1, total_words)) * 100.0
    b_wer = (b_errors / max(1, total_ref_b)) * 100.0
    u_wer = (u_errors / max(1, total_ref_u)) * 100.0

    return {
        "wer": round(wer, 2),
        "b_wer": round(b_wer, 2),
        "u_wer": round(u_wer, 2),
        "total_words": total_words,
        "b_words": total_ref_b,
        "u_words": total_ref_u,
    }
