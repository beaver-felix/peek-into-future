import torch
from peek_future.models.biasing_model import BiasingModel
from peek_future.data_prep.tokenizer import SPETokenizer


def decode_utterance(
    model: BiasingModel,
    feat: torch.Tensor,
    feat_len: torch.Tensor,
    bias_tokens: torch.Tensor,
    bias_entity_strings: list[str],
    tokenizer: SPETokenizer,
    lambda_param: float = 1.0,
    gamma_param: float = 0.3,
    max_len: int = 200,
) -> str:
    """
    Decodes single utterance with unified search space.
    """
    model.eval()
    ent_tok_lists = []
    for e_str in bias_entity_strings:
        if e_str == "<null>":
            ent_tok_lists.append([tokenizer.pad_id])
        else:
            ent_tok_lists.append(tokenizer.text_to_ids(e_str))

    pred_ids = model.decode_unified(
        feat,
        feat_len,
        bias_tokens,
        ent_tok_lists,
        lambda_param=lambda_param,
        gamma_param=gamma_param,
        max_len=max_len,
    )
    return tokenizer.ids_to_text(pred_ids)
