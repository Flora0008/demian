# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EgoVerse precomputed-latent dataset for Cosmos Predict 2.5.

Skips the online Wan2.1 VAE encode and the online Cosmos Reason 1 text
encode by reading the VAE-encoded latents and MEAN_POOLING text
embeddings that were precomputed by
``upstream/cosmos_dreamzero/src/cosmos_dreamzero/data/preprocess.py``.

Two dataset roots are supported, one per caption variant:

- ``/path/to/.../egoverse_cosmos_stage1_annotation/`` → text_variant="annotation"
- ``/path/to/.../egoverse_cosmos_stage1/``            → text_variant="annotation_plus_dense"

Each manifest line carries ``latent_path`` ((C=16, T=24, H=60, W=80) float32),
``text_emb_path`` ((N=512, D=3584) float32), and the caption bookkeeping
fields. Returned samples match the MECKA schema with two extras
``precomputed_latent`` / ``precomputed_text_emb`` that the training forward
uses as opt-in shortcuts.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from cosmos_predict2._src.imaginaire.utils import log


# Shape of the VAE-encoded latent, as produced by upstream's preprocess at
# --resolution 480 and --target_frames 93 on 480x640 EgoVerse videos.
LATENT_SHAPE = (16, 24, 60, 80)
# Corresponding raw-video resolution the latent represents (Wan2.1 VAE is
# 8x spatial). Reported back to the conditioner via ``image_size``.
RAW_HW = (480, 640)
# Number of raw frames per segment used by upstream's preprocess.
NUM_RAW_FRAMES = 93


class EgoVerseLatentDataset(Dataset):
    """Precomputed (latent, text_emb) EgoVerse dataset.

    Args:
        dataset_dir: root with ``manifest.jsonl`` + ``latents/`` + ``text_emb/``.
        max_episodes: optional cap for debug / smoke tests.

    Returns per sample::

        {
            "precomputed_latent":    (16, 24, 60, 80) float,   # VAE latent
            "ai_caption":            str,  # fed to online text encoder
            "fps":                   float,
            "image_size":            int tensor [H, W, H, W],
            "num_frames":            int,
            "padding_mask":          (1, H, W) float, zeros,
        }

    Note: upstream's precomputed text embeddings are MEAN_POOLING (N, 3584),
    while the cosmos-predict2.5 2B model is configured for FULL_CONCAT
    (N, 100352). We keep the online text encoder so we stay architecture-
    compatible with the current raw-video runs; profiling showed the text
    encoder is only 0.2s/iter anyway, while VAE encode is 6.6s/iter —
    that's where the win is.
    """

    def __init__(self, dataset_dir: str, max_episodes: Optional[int] = None) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        manifest = self.dataset_dir / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}")

        with manifest.open("r") as f:
            self.entries: list[dict] = [json.loads(line) for line in f if line.strip()]
        if max_episodes is not None:
            self.entries = self.entries[:max_episodes]
        if not self.entries:
            raise ValueError(f"Empty manifest: {manifest}")

        log.info(f"EgoVerseLatentDataset: {len(self.entries)} segments from {self.dataset_dir}")
        self.num_failed_loads = 0

    def __len__(self) -> int:
        return len(self.entries)

    def __str__(self) -> str:
        return f"EgoVerseLatentDataset({len(self.entries)} segments, root={self.dataset_dir})"

    def _load_tensor(self, rel_path: str) -> torch.Tensor:
        obj = torch.load(self.dataset_dir / rel_path, map_location="cpu", weights_only=True)
        if isinstance(obj, dict) and "tensor" in obj:
            obj = obj["tensor"]
        if not isinstance(obj, torch.Tensor):
            raise TypeError(f"Expected tensor at {rel_path}, got {type(obj)}")
        return obj

    def __getitem__(self, index: int) -> dict:
        try:
            entry = self.entries[index]
            latent = self._load_tensor(entry["latent_path"]).float()
            if tuple(latent.shape) != LATENT_SHAPE:
                raise ValueError(
                    f"Unexpected latent shape {tuple(latent.shape)} for {entry['latent_path']};"
                    f" expected {LATENT_SHAPE}"
                )

            h, w = RAW_HW
            return {
                "precomputed_latent": latent,
                "ai_caption": entry.get("caption", entry.get("annotation", "")),
                "fps": float(entry.get("fps", 30.0)),
                "image_size": torch.tensor([h, w, h, w]),
                "num_frames": NUM_RAW_FRAMES,
                "padding_mask": torch.zeros(1, h, w),
            }
        except Exception as e:  # noqa: BLE001
            self.num_failed_loads += 1
            if self.num_failed_loads <= 10:
                log.warning(
                    f"EgoVerseLatentDataset failed to load index={index} entry={self.entries[index].get('episode_id')} "
                    f"seg={self.entries[index].get('segment_index')}: {e}\n{traceback.format_exc()}",
                    rank0_only=False,
                )
            import numpy as np
            return self[int(np.random.randint(len(self.entries)))]
