"""
This module contains the main class for AI text detection using the Binoculars method.

The Binoculars class provides methods to compute a score indicating the likelihood that a given text is AI-generated
and to predict whether a text is AI-generated or human-generated based on this score.
"""

from typing import Union

import os
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import assert_tokenizer_consistency
from .metrics import perplexity, entropy

torch.set_grad_enabled(False)

huggingface_config = {
    # Only required for private models from Huggingface (e.g. LLaMA models)
    "TOKEN": os.environ.get("HF_TOKEN", None)
}

# selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527  # optimized for low-fpr [chosen at 0.01%]

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1


class Binoculars(object):
    """
    The Binoculars class provides methods to compute a score indicating the likelihood that a given text is AI-generated
    and to predict whether a text is AI-generated or human-generated based on this score.
    """

    def __init__(self,
                 observer_name_or_path: str = "tiiuae/falcon-7b",
                 performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
                 use_bfloat16: bool = True,
                 max_token_observed: int = 512,
                 mode: str = "low-fpr",
                 ) -> None:
        """
        Initializes the Binoculars class with the given parameters.

        Args:
            observer_name_or_path (str): The name or path of the observer model.
            performer_name_or_path (str): The name or path of the performer model.
            use_bfloat16 (bool): Whether to use bfloat16 precision.
            max_token_observed (int): The maximum number of tokens to observe.
            mode (str): The mode to use for detection ("low-fpr" or "accuracy").
        """
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        self.change_mode(mode)
        self.observer_model = AutoModelForCausalLM.from_pretrained(observer_name_or_path,
                                                                   device_map={"": DEVICE_1},
                                                                   trust_remote_code=True,
                                                                   torch_dtype=torch.bfloat16 if use_bfloat16
                                                                   else torch.float32,
                                                                   token=huggingface_config["TOKEN"]
                                                                   )
        self.performer_model = AutoModelForCausalLM.from_pretrained(performer_name_or_path,
                                                                    device_map={"": DEVICE_2},
                                                                    trust_remote_code=True,
                                                                    torch_dtype=torch.bfloat16 if use_bfloat16
                                                                    else torch.float32,
                                                                    token=huggingface_config["TOKEN"]
                                                                    )
        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def change_mode(self, mode: str) -> None:
        """
        Changes the mode of the Binoculars class.

        Args:
            mode (str): The mode to use for detection ("low-fpr" or "accuracy").
        """
        if mode == "low-fpr":
            self.threshold = BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        """
        Tokenizes the input batch of text.

        Args:
            batch (list[str]): The batch of text to tokenize.

        Returns:
            transformers.BatchEncoding: The tokenized batch of text.
        """
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        """
        Computes the logits for the given encodings using the observer and performer models.

        Args:
            encodings (transformers.BatchEncoding): The tokenized batch of text.

        Returns:
            torch.Tensor: The logits computed by the observer and performer models.
        """
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        """
        Computes the Binoculars score for the given input text.

        Args:
            input_text (Union[list[str], str]): The input text to compute the score for.

        Returns:
            Union[float, list[float]]: The Binoculars score for the input text.
        """
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores

    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        """
        Predicts whether the given input text is AI-generated or human-generated.

        Args:
            input_text (Union[list[str], str]): The input text to predict.

        Returns:
            Union[list[str], str]: The prediction for the input text.
        """
        binoculars_scores = np.array(self.compute_score(input_text))
        pred = np.where(binoculars_scores < self.threshold,
                        "Most likely AI-generated",
                        "Most likely human-generated"
                        ).tolist()
        return pred
