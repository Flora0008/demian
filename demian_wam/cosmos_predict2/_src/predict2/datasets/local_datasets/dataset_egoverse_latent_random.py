# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EgoVerse dataset with precomputed VAE latents + random dense caption.

Uses upstream's precomputed Wan2.1 VAE latents (so we skip VAE encode in the
training step) but picks the text input freshly per sample per step:

    ai_caption = annotation + " " + <one of {current_action, environment, reasoning} chosen at random>

Because the text choice is random per step, the corresponding Reason 1
embedding cannot be precomputed — we rely on the online text encoder for
the embedding. That adds only ~0.2s/iter per profiling, so the net iter
time should match the existing `_precomputed` variants (~5.3s).
"""

from __future__ import annotations

import json
import random
import traceback
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_predict2._src.imaginaire.utils import log


LATENT_SHAPE = (16, 24, 60, 80)
RAW_HW = (480, 640)
NUM_RAW_FRAMES = 93

# Dense caption fields picked from at random per sample.
RANDOM_CAPTION_FIELDS: tuple[str, ...] = ("current_action", "environment", "reasoning")


class EgoVerseLatentRandomCaptionDataset(Dataset):
    """Precomputed-latent EgoVerse dataset with random dense-caption selection.

    Args:
        latent_dataset_dir: root with ``manifest.jsonl`` + ``latents/`` produced
            by upstream's preprocess (either ``egoverse_cosmos_stage1_annotation``
            or ``egoverse_cosmos_stage1`` — the latents are identical across
            text variants).
        raw_dataset_dir: root containing the original per-episode Zarr stores
            and ``captions_*.jsonl`` files. Needed to re-fetch the individual
            dense caption fields since the precomputed manifest only stores
            the already-composed caption.
        caption_glob: pattern for the VLM caption JSONL inside each episode.
        max_episodes: optional cap for debug.

    Returns per sample::

        {
            "precomputed_latent":    (16, 24, 60, 80) float,
            "ai_caption":            str  # annotation + " " + random dense field
            "fps":                   float,
            "image_size":            tensor [H, W, H, W],
            "num_frames":            int,
            "padding_mask":          (1, H, W) float, zeros,
        }
    """

    def __init__(
        self,
        latent_dataset_dir: str,
        raw_dataset_dir: str = "/path/to/data/egoverse_dataset",
        caption_glob: str = "captions_*.jsonl",
        max_episodes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.latent_dir = Path(latent_dataset_dir)
        self.raw_dir = Path(raw_dataset_dir)
        self.caption_glob = caption_glob

        manifest = self.latent_dir / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        with manifest.open("r") as f:
            self.entries: list[dict] = [json.loads(line) for line in f if line.strip()]
        if max_episodes is not None:
            self.entries = self.entries[:max_episodes]
        if not self.entries:
            raise ValueError(f"Empty manifest: {manifest}")

        # Lazy per-episode cache: episode_id → {segment_index: captions_dict}.
        self._episode_cache: dict[str, dict[int, dict]] = {}

        log.info(
            f"EgoVerseLatentRandomCaptionDataset: {len(self.entries)} segments, "
            f"random field picked from {RANDOM_CAPTION_FIELDS}"
        )
        self.num_failed_loads = 0

    def __len__(self) -> int:
        return len(self.entries)

    def __str__(self) -> str:
        return (
            f"EgoVerseLatentRandomCaptionDataset({len(self.entries)} segments, "
            f"fields={RANDOM_CAPTION_FIELDS})"
        )

    def _load_tensor(self, rel_path: str) -> torch.Tensor:
        obj = torch.load(self.latent_dir / rel_path, map_location="cpu", weights_only=True)
        if isinstance(obj, dict) and "tensor" in obj:
            obj = obj["tensor"]
        if not isinstance(obj, torch.Tensor):
            raise TypeError(f"Expected tensor at {rel_path}, got {type(obj)}")
        return obj

    def _load_episode_captions(self, episode_id: str) -> dict[int, dict]:
        """Parse the raw captions JSONL for one episode; cached per-episode."""
        if episode_id in self._episode_cache:
            return self._episode_cache[episode_id]

        ep_path = self.raw_dir / episode_id
        cap_files = sorted(glob(str(ep_path / self.caption_glob)))
        if not cap_files:
            self._episode_cache[episode_id] = {}
            return {}

        segments: dict[int, dict] = {}
        with open(cap_files[0]) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                segments[int(rec.get("segment_index", 0))] = rec
        self._episode_cache[episode_id] = segments
        return segments

    def _compose_caption(self, entry: dict) -> str:
        """Build ``annotation + " " + <random dense field>``.

        Falls back to any available field if the drawn one is empty, and
        finally to ``annotation`` alone if none of the three has content.
        """
        annotation = (entry.get("annotation", "") or "").strip()
        captions_dict = self._load_episode_captions(entry["episode_id"]).get(
            int(entry.get("segment_index", 0)), {}
        ).get("annotations", {})

        # Collect present/non-empty fields in our candidate set.
        available = [
            k for k in RANDOM_CAPTION_FIELDS
            if isinstance(captions_dict.get(k), str) and captions_dict[k].strip()
        ]
        if available:
            chosen = random.choice(available)
            dense = captions_dict[chosen].strip()
        else:
            dense = ""

        # Punctuate annotation so it reads as its own sentence before the dense part.
        if annotation and not annotation.endswith((".", "!", "?")):
            annotation = annotation + "."
        return f"{annotation} {dense}".strip() if dense else annotation

    def __getitem__(self, index: int) -> dict:
        try:
            entry = self.entries[index]
            latent = self._load_tensor(entry["latent_path"]).float()
            if tuple(latent.shape) != LATENT_SHAPE:
                raise ValueError(
                    f"Unexpected latent shape {tuple(latent.shape)} for "
                    f"{entry['latent_path']}; expected {LATENT_SHAPE}"
                )

            caption = self._compose_caption(entry)
            if not caption:
                # Degenerate sample: fall back to the precomputed composed
                # caption from the manifest (guaranteed non-empty by upstream).
                caption = entry.get("caption", "").strip() or "video"

            h, w = RAW_HW
            return {
                "precomputed_latent": latent,
                "ai_caption": caption,
                "fps": float(entry.get("fps", 30.0)),
                "image_size": torch.tensor([h, w, h, w]),
                "num_frames": NUM_RAW_FRAMES,
                "padding_mask": torch.zeros(1, h, w),
            }
        except Exception as e:  # noqa: BLE001
            self.num_failed_loads += 1
            if self.num_failed_loads <= 10:
                entry = self.entries[index]
                log.warning(
                    f"EgoVerseLatentRandomCaptionDataset failed at index={index} "
                    f"(ep={entry.get('episode_id')} seg={entry.get('segment_index')}): {e}\n"
                    f"{traceback.format_exc()}",
                    rank0_only=False,
                )
            return self[int(np.random.randint(len(self.entries)))]
