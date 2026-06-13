# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EgoVerse dataset loader for Cosmos Predict 2.5 mid-training.

Each episode is a Zarr-v3 store at ``<root>/<episode_id>/`` containing
``images.front_1`` (variable-length JPEG bytes) plus proprioception arrays.
Captions live next to the frames in ``captions_*.jsonl``; each line defines
one (start_idx, end_idx) segment with a short ``annotation`` and a
``captions`` dict of dense per-segment VLM descriptions.

Caption variants mirror the upstream/cosmos_dreamzero preprocess script so
our results are comparable:

- ``variant='annotation'``  — just the EgoVerse label (baseline).
- ``variant='annotation_plus_dense'`` —
  ``{annotation}[.] {current_action} {environment} {hand_start_state} {hand_end_state}``,
  space-joined, trailing period appended only if the annotation isn't
  already punctuated, empty dense fields skipped.
- ``variant='annotation_plus_dense_only'`` —
  ``{annotation}[.] {captions["dense"]}``. Uses the single literal
  ``dense`` field. Segments lacking a non-empty ``dense`` field are skipped.
- ``variant='annotation_plus_dense_human'`` —
  ``{annotation}[.] {captions["dense_human"]}``. Uses the re-prompted
  human-perspective dense field. Segments lacking a non-empty ``dense_human``
  field are skipped.
"""

from __future__ import annotations

import io
import json
import logging
import traceback
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils import ResizePreprocess, ToTensorVideo


CAPTION_VARIANTS = (
    "annotation",
    "annotation_plus_dense",
    "annotation_plus_dense_only",
    "annotation_plus_dense_human",
)

# Dense-caption fields appended after the annotation, in this fixed order,
# when variant='annotation_plus_dense'. Matches upstream/cosmos_dreamzero.
DENSE_CAPTION_KEYS: tuple[str, ...] = (
    "current_action",
    "environment",
    "hand_start_state",
    "hand_end_state",
)

DENSE_ONLY_KEY: str = "dense"
DENSE_HUMAN_KEY: str = "dense_human"


def _unwrap_zarr_bytes(obj) -> bytes:
    """Zarr v3's variable-length-bytes codec returns nested 0-d object arrays
    wrapping the real ``bytes`` payload. Unwrap until we hit concrete bytes."""
    while not isinstance(obj, (bytes, bytearray)):
        if not hasattr(obj, "item"):
            raise TypeError(f"unexpected zarr payload type: {type(obj)}")
        nxt = obj.item()
        if nxt is obj:
            raise TypeError(f"zarr payload unwrap stuck at {type(obj)}")
        obj = nxt
    return bytes(obj)


def _compose_caption(annotation: str, captions: dict, variant: str) -> str:
    annotation = (annotation or "").strip()
    if variant == "annotation":
        return annotation
    if variant == "annotation_plus_dense":
        parts: list[str] = []
        if annotation:
            parts.append(annotation if annotation.endswith((".", "!", "?")) else f"{annotation}.")
        for key in DENSE_CAPTION_KEYS:
            value = captions.get(key) if isinstance(captions, dict) else None
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        return " ".join(parts)
    if variant == "annotation_plus_dense_only":
        dense = captions.get(DENSE_ONLY_KEY) if isinstance(captions, dict) else None
        dense = dense.strip() if isinstance(dense, str) else ""
        if not dense:
            return ""
        if annotation:
            head = annotation if annotation.endswith((".", "!", "?")) else f"{annotation}."
            return f"{head} {dense}"
        return dense
    if variant == "annotation_plus_dense_human":
        dense = captions.get(DENSE_HUMAN_KEY) if isinstance(captions, dict) else None
        dense = dense.strip() if isinstance(dense, str) else ""
        if not dense:
            return ""
        if annotation:
            head = annotation if annotation.endswith((".", "!", "?")) else f"{annotation}."
            return f"{head} {dense}"
        return dense
    raise ValueError(f"Unknown variant {variant!r}; expected one of {CAPTION_VARIANTS}")


class EgoVerseVideoDataset(Dataset):
    """EgoVerse (Zarr frames + per-episode caption JSONL) → Cosmos video sample.

    Output schema matches ``MeckaVideoDataset``:

        {
            "video":         uint8 tensor (C, T, H, W),
            "ai_caption":    str,
            "fps":           float,
            "image_size":    int tensor [H, W, H, W],
            "num_frames":    int (= T),
            "padding_mask":  float tensor (1, H, W),
        }

    Args:
        dataset_dir: root containing one subdir per episode.
        num_frames: frames per clip. Segments shorter than this are skipped.
        video_size: target (H, W) after resize.
        variant: caption composition — 'annotation' or 'annotation_plus_dense'.
        caption_glob: glob (within each episode dir) matching the VLM caption
            JSONL file. Default picks up captions_Qwen3-VL-*.jsonl.
        max_episodes: optional cap for debug / smoke tests.
    """

    def __init__(
        self,
        dataset_dir: str,
        num_frames: int = 93,
        video_size: tuple[int, int] = (480, 640),
        variant: str = "annotation",
        caption_glob: str = "captions_*.jsonl",
        max_episodes: Optional[int] = None,
    ) -> None:
        super().__init__()
        if variant not in CAPTION_VARIANTS:
            raise ValueError(f"variant must be one of {CAPTION_VARIANTS}, got {variant!r}")

        self.dataset_dir = Path(dataset_dir)
        self.sequence_length = num_frames
        self.variant = variant
        self.caption_glob = caption_glob

        # Build (episode_path, segment_record) index by scanning the caption
        # JSONL in each episode subdir. An episode without a caption file,
        # without an images.front_1 store, or with every segment shorter
        # than num_frames is skipped silently.
        self.segments: list[dict] = []
        ep_dirs = sorted(p for p in self.dataset_dir.iterdir() if p.is_dir())
        if max_episodes is not None:
            ep_dirs = ep_dirs[:max_episodes]

        skipped_no_caption = 0
        skipped_no_frames = 0
        skipped_short = 0
        for ep_path in ep_dirs:
            cap_files = sorted(glob(str(ep_path / caption_glob)))
            if not cap_files:
                skipped_no_caption += 1
                continue
            if not (ep_path / "images.front_1").is_dir():
                skipped_no_frames += 1
                continue
            with open(cap_files[0]) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    start = int(rec["start_idx"])
                    end = int(rec["end_idx"])
                    if end - start < num_frames:
                        skipped_short += 1
                        continue
                    caption = _compose_caption(
                        rec.get("annotation", ""),
                        rec.get("annotations", {}),
                        variant,
                    )
                    if not caption:
                        continue
                    self.segments.append({
                        "episode_path": ep_path,
                        "episode_id": ep_path.name,
                        "segment_index": int(rec.get("segment_index", 0)),
                        "start_idx": start,
                        "end_idx": end,
                        "caption": caption,
                    })

        log.info(
            f"EgoVerseVideoDataset({variant}): {len(self.segments)} valid segments "
            f"from {len(ep_dirs)} episodes "
            f"(skipped: {skipped_no_caption} no-caption, {skipped_no_frames} no-frames, "
            f"{skipped_short} too-short)"
        )
        if not self.segments:
            raise RuntimeError(f"No valid EgoVerse segments found in {dataset_dir}")

        self.preprocess = T.Compose([ToTensorVideo(), ResizePreprocess(video_size)])
        self.num_failed_loads = 0

    def __len__(self) -> int:
        return len(self.segments)

    def __str__(self) -> str:
        return (
            f"EgoVerseVideoDataset(variant={self.variant}): "
            f"{len(self.segments)} segments from {self.dataset_dir}"
        )

    def _load_frames(self, ep_path: Path, start: int, end: int) -> np.ndarray:
        import zarr

        images = zarr.open_array(str(ep_path / "images.front_1"), mode="r")
        total = int(images.shape[0])
        start = max(0, min(start, total - 1))
        end = max(start + self.sequence_length, min(end, total))
        # Pick a random contiguous window of exactly sequence_length frames.
        max_window_start = end - self.sequence_length
        if max_window_start <= start:
            window_start = start
        else:
            window_start = int(np.random.randint(start, max_window_start + 1))
        window_end = window_start + self.sequence_length

        frames: list[np.ndarray] = []
        for i in range(window_start, window_end):
            raw = _unwrap_zarr_bytes(images[i])
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            frames.append(np.asarray(img))
        return np.stack(frames)  # (T, H, W, 3) uint8

    def __getitem__(self, index: int) -> dict:
        try:
            seg = self.segments[index]
            frames = self._load_frames(seg["episode_path"], seg["start_idx"], seg["end_idx"])

            # (T, H, W, 3) uint8  →  (T, 3, H, W) uint8  →  preprocess → (3, T, H, W) float
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
            frames = self.preprocess(frames)                           # float in [0,1]
            frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
            video = frames.permute(1, 0, 2, 3)                         # (C, T, H, W)

            _, _, h, w = video.shape
            return {
                "video": video,
                "ai_caption": seg["caption"],
                "fps": 30.0,
                "image_size": torch.tensor([h, w, h, w]),
                "num_frames": self.sequence_length,
                "padding_mask": torch.zeros(1, h, w),
            }
        except Exception as e:  # noqa: BLE001
            self.num_failed_loads += 1
            if self.num_failed_loads <= 10:
                seg = self.segments[index]
                log.warning(
                    f"Failed to load EgoVerse segment "
                    f"(ep={seg['episode_id']} seg={seg['segment_index']}, "
                    f"total failures: {self.num_failed_loads}): {e}\n{traceback.format_exc()}",
                    rank0_only=False,
                )
            return self[int(np.random.randint(len(self.segments)))]
