"""
This module contains utility functions for the Binoculars package.

The `assert_tokenizer_consistency` function ensures that the tokenizers of two models are identical.
"""

from transformers import AutoTokenizer


def assert_tokenizer_consistency(model_id_1, model_id_2):
    """
    Ensures that the tokenizers of two models are identical.

    Args:
        model_id_1 (str): The name or path of the first model.
        model_id_2 (str): The name or path of the second model.

    Raises:
        ValueError: If the tokenizers are not identical.
    """
    identical_tokenizers = (
            AutoTokenizer.from_pretrained(model_id_1).vocab
            == AutoTokenizer.from_pretrained(model_id_2).vocab
    )
    if not identical_tokenizers:
        raise ValueError(f"Tokenizers are not identical for {model_id_1} and {model_id_2}.")
