"""Tokenizers used by openpi.

NOTE (DeMiAn release): only the pi0/pi05 ``PaligemmaTokenizer`` (sentencepiece) is
kept. The original ``FASTTokenizer``, ``BinningTokenizer``, and ``FSQTokenizer``
(FAST / RoboArena baselines) have been removed because they depend on the JAX FSQ
tokenizer and FAST configs that are not part of this release.
"""

import logging

import numpy as np
import sentencepiece

import openpi.shared.download as download


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(
        self,
        prompt: str,
        state: np.ndarray | None = None,
        reasoning_text: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize prompt, optionally with reasoning text.

        When reasoning_text is provided, returns a third array: reasoning_mask
        (bool, same length as tokens) that is True for reasoning token positions.
        This allows the model to mask out reasoning tokens from the action expert's
        attention while still computing LM loss on them.
        """
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        reasoning_mask = None

        if state is not None:
            # Pi05 format with discrete state input.
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            if reasoning_text:
                cleaned_reasoning = reasoning_text.strip().replace("\n", " ")
                full_prompt = f"Task: {cleaned_text} Reasoning: {cleaned_reasoning}, State: {state_str};\nAction: "
                # Tokenize without reasoning to find boundary.
                base_prompt = f"Task: {cleaned_text}"
                base_tokens = self._tokenizer.encode(base_prompt, add_bos=True)
                all_tokens = self._tokenizer.encode(full_prompt, add_bos=True)
                reasoning_mask = [False] * len(base_tokens) + [True] * (len(all_tokens) - len(base_tokens))
            else:
                full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # Pi0 format — state is continuous input in the suffix.
            if reasoning_text:
                cleaned_reasoning = reasoning_text.strip().replace("\n", " ")
                base_tokens = self._tokenizer.encode(cleaned_text, add_bos=True)
                reasoning_tokens = self._tokenizer.encode(f" Reasoning: {cleaned_reasoning}")
                newline_tokens = self._tokenizer.encode("\n")
                tokens = base_tokens + reasoning_tokens + newline_tokens
                reasoning_mask = [False] * len(base_tokens) + [True] * len(reasoning_tokens) + [False] * len(newline_tokens)
            else:
                tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
            if reasoning_mask is not None:
                reasoning_mask = reasoning_mask + [False] * (self._max_len - len(reasoning_mask))
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len
            if reasoning_mask is not None:
                reasoning_mask = reasoning_mask[: self._max_len]

        if reasoning_mask is not None:
            return np.asarray(tokens), np.asarray(mask), np.asarray(reasoning_mask)
        return np.asarray(tokens), np.asarray(mask)


