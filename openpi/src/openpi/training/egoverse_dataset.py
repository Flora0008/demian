"""EgoVerse dataset for pi05 LM-only pre-training.

Reads raw EgoVerse episodes (Zarr v3 stores) and per-segment captions from
``<episode_id>/captions_Qwen3-VL-235B-A22B-Instruct.jsonl``. Samples 3 RGB
frames per segment (first / middle / last of ``[start_idx, end_idx)``), which
the policy transform maps to the 3 PaliGemma image slots as a pseudo-temporal
context.

Text-variant sampling (set at construction):

* ``annotation``                  – prompt = annotation, empty reasoning_text.
* ``annotation_current_action``   – prompt = annotation, reasoning_text = ``captions["current_action"]``.
* ``annotation_environment``      – prompt = annotation, reasoning_text = ``captions["environment"]``.
* ``annotation_plus_dense``       – DreamZero-matching. prompt = annotation;
  reasoning_text = space-join of {current_action, environment, hand_start_state,
  hand_end_state} in that fixed order, empty fields skipped. This reproduces the
  string that DreamZero's ``preprocess.py._compose_caption`` feeds to its Cosmos
  Reason 1 text encoder (minus the annotation prefix, which stays in ``prompt``
  so it receives bidirectional attention in pi05's prefix-LM mask).

Segment boundaries come from the pre-computed stage1 manifest at
``egoverse_cosmos_stage1/manifest.jsonl`` (reused purely as an index – the
VAE-latent / text-emb paths in that file are ignored).
"""

from __future__ import annotations

import io
import json
import logging
import os
import pathlib
from typing import Literal

import numpy as np
import torch
import torch.utils.data

logger = logging.getLogger(__name__)

TextVariant = Literal[
    "annotation",
    "annotation_current_action",
    "annotation_environment",
    "annotation_plus_dense",
]

_CAPTION_FILENAME = "captions_Qwen3-VL-235B-A22B-Instruct.jsonl"
_IMAGE_ARRAY = "images.front_1"

# Dense-caption fields that ``annotation_plus_dense`` concatenates, in the exact
# order used by DreamZero's ``preprocess.py._compose_caption`` for stage 1.
_DENSE_CAPTION_KEYS: tuple[str, ...] = (
    "current_action",
    "environment",
    "hand_start_state",
    "hand_end_state",
)


def _unwrap_zarr_bytes(obj) -> bytes:
    """Zarr v3 variable-length-bytes payloads come back as nested object ndarrays."""
    while not isinstance(obj, (bytes, bytearray)):
        if not hasattr(obj, "item"):
            raise TypeError(f"unexpected zarr payload type: {type(obj)}")
        nxt = obj.item()
        if nxt is obj:
            raise TypeError(f"zarr payload unwrap stuck at {type(obj)}")
        obj = nxt
    return bytes(obj)


def _compose_text(annotation: str, captions: dict, variant: TextVariant) -> tuple[str, str]:
    """Return (prompt, reasoning_text) for a given text_variant."""
    annotation = (annotation or "").strip()
    if variant == "annotation":
        return annotation, ""
    if variant == "annotation_current_action":
        return annotation, (captions.get("current_action") or "").strip()
    if variant == "annotation_environment":
        return annotation, (captions.get("environment") or "").strip()
    if variant == "annotation_plus_dense":
        parts: list[str] = []
        for key in _DENSE_CAPTION_KEYS:
            value = captions.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        return annotation, " ".join(parts)
    raise ValueError(f"unknown text_variant: {variant}")


class EgoVerseDataset(torch.utils.data.Dataset):
    """Raw EgoVerse Zarr stores + segment captions.

    Args:
        episodes_root: directory containing ``<episode_id>/`` subdirs with
            ``images.front_1/`` and ``captions_*.jsonl``.
        manifest_path: JSON-lines file listing segments. Each line must have
            ``episode_id``, ``segment_index``, ``start_idx``, ``end_idx``,
            ``annotation``. Extra fields are ignored.
        text_variant: which (prompt, reasoning_text) pair to emit.
        action_horizon: length of the dummy action tensor.
        action_dim: padded action dim (matches the model's action_dim).
        state_dim: padded state dim (matches the model's action_dim).
        random_frames: if True, sample one random frame from each third of
            ``[start_idx, end_idx)`` instead of always picking first/middle/last.
            Each epoch sees different frames per segment — prevents the model
            from memorizing the fixed 3-frame visual context (critical for the
            short-label ``annotation`` variant, which otherwise converges to
            near-zero loss by step ~3k).
    """

    def __init__(
        self,
        episodes_root: str | os.PathLike,
        manifest_path: str | os.PathLike,
        *,
        text_variant: TextVariant = "annotation",
        action_horizon: int = 10,
        action_dim: int = 32,
        random_frames: bool = False,
        state_dim: int = 32,
    ) -> None:
        super().__init__()
        self.episodes_root = pathlib.Path(episodes_root)
        self.manifest_path = pathlib.Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        self.text_variant: TextVariant = text_variant
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.random_frames = random_frames

        with self.manifest_path.open("r") as f:
            self._entries: list[dict] = [json.loads(line) for line in f if line.strip()]
        if not self._entries:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")

        # Lazy per-episode caches (keyed by episode_id). Each worker process builds
        # its own cache; the DataLoader uses spawn so there's no shared state issue.
        self._zarr_cache: dict[str, object] = {}
        self._caption_cache: dict[str, dict[int, dict]] = {}

        logger.info(
            f"EgoVerseDataset: {len(self._entries)} segments from {self.manifest_path} "
            f"(text_variant={self.text_variant})"
        )

    def __len__(self) -> int:
        return len(self._entries)

    def _load_captions(self, episode_id: str) -> dict[int, dict]:
        if episode_id not in self._caption_cache:
            path = self.episodes_root / episode_id / _CAPTION_FILENAME
            if not path.is_file():
                raise FileNotFoundError(f"Missing caption file: {path}")
            by_seg: dict[int, dict] = {}
            with path.open("r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    seg_idx = int(rec.get("segment_index", -1))
                    if seg_idx < 0:
                        continue
                    by_seg[seg_idx] = rec
            self._caption_cache[episode_id] = by_seg
            # Keep the cache bounded; each file is small (a few hundred segments)
            # but we're iterating across 2k+ episodes.
            if len(self._caption_cache) > 256:
                oldest = next(iter(self._caption_cache))
                del self._caption_cache[oldest]
        return self._caption_cache[episode_id]

    def _open_zarr(self, episode_id: str):
        """Open and cache the per-episode image Zarr array."""
        if episode_id not in self._zarr_cache:
            import zarr

            arr_path = self.episodes_root / episode_id / _IMAGE_ARRAY
            self._zarr_cache[episode_id] = zarr.open_array(str(arr_path), mode="r")
            if len(self._zarr_cache) > 128:
                oldest = next(iter(self._zarr_cache))
                del self._zarr_cache[oldest]
        return self._zarr_cache[episode_id]

    def _load_frame(self, episode_id: str, frame_idx: int) -> np.ndarray:
        from PIL import Image

        arr = self._open_zarr(episode_id)
        total = int(arr.shape[0])
        frame_idx = max(0, min(frame_idx, total - 1))
        raw = _unwrap_zarr_bytes(arr[frame_idx])
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.asarray(img)  # (H, W, 3) uint8

    def _pick_frame_indices(self, start: int, end: int) -> tuple[int, int, int]:
        """Pick 3 frame indices from [start, end) preserving temporal order.

        Deterministic mode: returns (start, middle, last).
        Random mode (``self.random_frames``): splits the segment into thirds and
        samples one random frame from each third. Preserves temporal ordering
        while giving the model a different 3-frame context each epoch.
        """
        if end <= start:
            end = start + 1
        if not self.random_frames:
            last = end - 1
            mid = start + (end - start) // 2
            return start, mid, last

        import random as _random

        span = end - start
        third = max(1, span // 3)
        i0 = start + _random.randrange(third)
        i1 = start + third + _random.randrange(third)
        # Remaining span covers any rounding remainder.
        remainder = max(1, span - 2 * third)
        i2 = start + 2 * third + _random.randrange(remainder)
        # Clamp to segment bounds (span < 3 edge cases).
        last_idx = end - 1
        return min(i0, last_idx), min(i1, last_idx), min(i2, last_idx)

    def __getitem__(self, index: int) -> dict:
        entry = self._entries[index]
        episode_id = entry["episode_id"]
        seg_idx = int(entry["segment_index"])
        start_idx = int(entry["start_idx"])
        end_idx = int(entry["end_idx"])
        annotation = entry.get("annotation", "")

        # Captions (from raw jsonl — the manifest's "caption" field is a precomputed
        # concatenation; we want the individual fields).
        caps_by_seg = self._load_captions(episode_id)
        captions = caps_by_seg.get(seg_idx, {}).get("annotations", {}) or {}

        # Frames.
        i0, i1, i2 = self._pick_frame_indices(start_idx, end_idx)
        frame_0 = self._load_frame(episode_id, i0)
        frame_1 = self._load_frame(episode_id, i1)
        frame_2 = self._load_frame(episode_id, i2)

        prompt, reasoning_text = _compose_text(annotation, captions, self.text_variant)

        return {
            "observation/frame_0": frame_0,
            "observation/frame_1": frame_1,
            "observation/frame_2": frame_2,
            "observation/state": np.zeros(self.state_dim, dtype=np.float32),
            "actions": np.zeros((self.action_horizon, self.action_dim), dtype=np.float32),
            "prompt": prompt,
            "reasoning_text": reasoning_text,
            "task_index": 0,
            "episode_id": episode_id,
            "segment_index": seg_idx,
            "frame_indices": np.asarray([i0, i1, i2], dtype=np.int64),
        }
