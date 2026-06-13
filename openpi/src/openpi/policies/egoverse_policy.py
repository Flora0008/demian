"""Policy input/output transforms for EgoVerse pi05 LM-only pre-training.

Maps the 3 frames emitted by ``EgoVerseDataset`` (first / middle / last of the
segment) to pi05's 3 PaliGemma image slots as a pseudo-temporal context. No
wrist cameras exist in EgoVerse, so the model sees the ``base_0_rgb`` /
``left_wrist_0_rgb`` / ``right_wrist_0_rgb`` slots as (t=start, t=mid, t=end).

Since training runs with ``lm_only=True``, the action slot is a zero
placeholder — the action expert forward is skipped in ``PI0Pytorch._forward_lm_only``.
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class EgoVerseInputs(transforms.DataTransformFn):
    """Convert EgoVerseDataset records to the model's observation format.

    Uses pi05's 3 camera slots as temporal frames (first / middle / last).
    """

    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        frame_0 = _parse_image(data["observation/frame_0"])
        frame_1 = _parse_image(data["observation/frame_1"])
        frame_2 = _parse_image(data["observation/frame_2"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": frame_0,
                "left_wrist_0_rgb": frame_1,
                "right_wrist_0_rgb": frame_2,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # Always pass the reasoning key (even empty) so TokenizePrompt creates a
        # consistent reasoning_token_mask. For the "annotation" baseline variant
        # the string is empty; the tokenizer treats empty reasoning as None.
        if "reasoning_text" in data:
            inputs["reasoning_text"] = data["reasoning_text"]

        return inputs


@dataclasses.dataclass(frozen=True)
class EgoVerseOutputs(transforms.DataTransformFn):
    """No-op at inference: EgoVerse pre-training isn't meant to produce actions."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
