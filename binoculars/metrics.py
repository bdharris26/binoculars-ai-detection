"""
This module contains functions to compute perplexity and entropy metrics for AI text detection.

The `perplexity` function calculates the perplexity of a given text based on the logits from a language model.
The `entropy` function calculates the entropy between the logits of two language models for a given text.
"""

import numpy as np
import torch
import transformers

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)


def perplexity(encoding: transformers.BatchEncoding,
               logits: torch.Tensor,
               median: bool = False,
               temperature: float = 1.0):
    """
    Calculates the perplexity of a given text based on the logits from a language model.

    Args:
        encoding (transformers.BatchEncoding): The tokenized batch of text.
        logits (torch.Tensor): The logits from the language model.
        median (bool): Whether to use the median of the cross-entropy loss.
        temperature (float): The temperature to use for scaling the logits.

    Returns:
        np.ndarray: The perplexity of the given text.
    """
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    if median:
        ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                  masked_fill(~shifted_attention_mask.bool(), float("nan")))
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)

    else:
        ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
               shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.to("cpu").float().numpy()

    return ppl


def entropy(p_logits: torch.Tensor,
            q_logits: torch.Tensor,
            encoding: transformers.BatchEncoding,
            pad_token_id: int,
            median: bool = False,
            sample_p: bool = False,
            temperature: float = 1.0):
    """
    Calculates the entropy between the logits of two language models for a given text.

    Args:
        p_logits (torch.Tensor): The logits from the first language model.
        q_logits (torch.Tensor): The logits from the second language model.
        encoding (transformers.BatchEncoding): The tokenized batch of text.
        pad_token_id (int): The token ID used for padding.
        median (bool): Whether to use the median of the cross-entropy loss.
        sample_p (bool): Whether to sample from the probability distribution of the first model.
        temperature (float): The temperature to use for scaling the logits.

    Returns:
        np.ndarray: The entropy between the logits of the two language models.
    """
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())

    return agg_ce
