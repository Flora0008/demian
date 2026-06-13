"""Adapter that wraps Pi0's MlSpacesDataset so Cosmos action-head training can consume it.

Molmobot data lives in an H5 + MP4 "MlSpaces" format, which is NOT LeRobot.
The Pi0 codebase (/path/to/.../molmo/MolmoBot/MolmoBot-Pi0) already implements a
working loader for this format, so we reuse its `MlSpacesDataset` class as-is
and translate its output dict into the shape that the Cosmos action-head
`ActionHeadVideoModel` expects.

Output contract (per sample):
    video        : (C=3, T=num_frames, H, W) uint8
    action       : (num_action_steps, action_dim) float32
    ai_caption   : str
    dataset, fps, num_frames, image_size, padding_mask : boilerplate metadata

State / LAM / lm_* keys are intentionally omitted — the action-head model
doesn't read them.
"""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


# Pi0 (DeMiAn-VLA) is a sibling project; the MolmoBot WAM dataset reuses its
# `MlSpacesDataset`. Point MOLMOBOT_PI0_PATH at the release `code/` directory
# (which contains the `molmobot_pi0` package). If unset, we assume `molmobot_pi0`
# is already importable via PYTHONPATH.
_MOLMOBOT_PI0_PATH = os.environ.get("MOLMOBOT_PI0_PATH", "").strip()
if _MOLMOBOT_PI0_PATH and os.path.isdir(_MOLMOBOT_PI0_PATH) and _MOLMOBOT_PI0_PATH not in sys.path:
    sys.path.insert(0, _MOLMOBOT_PI0_PATH)


class MlSpacesVideoActionDataset(torch.utils.data.Dataset):
    """Wraps Pi0's MlSpacesDataset and emits Cosmos action-head samples."""

    DEFAULT_EXO_CAMERA = "droid_shoulder_light_randomization"
    DEFAULT_STATES = ("qpos.arm", "qpos.gripper")
    DEFAULT_ACTIONS = ("joint_pos.arm", "joint_pos.gripper")

    def __init__(
        self,
        dataset_path,
        num_frames=13,
        action_horizon=16,
        action_dim=8,
        video_height=256,
        video_width=256,
        data_split="train",
        use_image_aug=True,
        exo_camera=None,
        caption_key="current_action",
        captions_dir=None,
        # Accepted for signature compatibility with VideoActionDataset call sites
        # inside MultiVideoActionDataset — deliberately ignored.
        time_division_factor=4,
        height=None,
        width=None,
        embodiment=None,
        agibot_pad_freq10=False,
        waist_concat=False,
        single_base_index=False,
        caption_mode=None,
        fg_stride=4,
        lm_tokenize=False,
        text_embeddings_dir=None,
        **_ignored,
    ):
        from molmobot_pi0.dataset import MlSpacesDataset  # lazy: only import when adapter is used

        self.dataset_path = dataset_path
        self.num_frames = num_frames
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.video_height = video_height
        self.video_width = video_width

        exo_camera = exo_camera or self.DEFAULT_EXO_CAMERA
        self._obs_camera_key = f"observation.image.{exo_camera}"

        # Optional dataset subset (for overfit sanity tests). Set the env var
        # MLSPACES_HOUSE_IDXS="0,1,2" to restrict to those house indices.
        house_env = os.environ.get("MLSPACES_HOUSE_IDXS", "").strip()
        house_idxs = [int(x) for x in house_env.split(",") if x.strip()] if house_env else None
        if house_idxs is not None:
            print(f"[MlSpacesVideoActionDataset] SUBSET: restricting to house_idxs={house_idxs}")

        # Optional dense-caption source. When MLSPACES_CAPTIONS_DIR is set, Pi0's
        # MlSpacesDataset loads per-trajectory captions from JSONLs and appends
        # <caption_key> (e.g. "current_action") to the task string. Unset → just
        # the raw task description (same as our baseline runs).
        captions_env = os.environ.get("MLSPACES_CAPTIONS_DIR", "").strip()
        caption_key_env = os.environ.get("MLSPACES_CAPTION_KEY", "").strip()
        if captions_env:
            captions_dir = captions_env
            if caption_key_env:
                caption_key = caption_key_env
            print(f"[MlSpacesVideoActionDataset] CAPTIONS: dir={captions_dir} key={caption_key}")

        self._pi0 = MlSpacesDataset(
            data_root=Path(dataset_path),
            house_idxs=house_idxs,
            selected_states=list(self.DEFAULT_STATES),
            selected_actions=list(self.DEFAULT_ACTIONS),
            selected_observations=[exo_camera],
            selected_env_states=None,
            img_size=(video_height, video_width),
            action_chunking=True,
            obs_horizon=num_frames,
            action_horizon=action_horizon,
            drop_n_last_frames=0,
            batch_image_proc=False,
            augment_images=use_image_aug,
            randomize_prompts=True,
            captions_dir=Path(captions_dir) if captions_dir else None,
            caption_key=caption_key,
        )

        # Action normalization: Pi0 returns raw joint_pos deltas (+ raw gripper 0–255);
        # unnormalized these produce huge MSE loss spikes (gripper std≈124).
        # Load stats from aggregated_stats.json and apply z-score in __getitem__.
        self._action_mean, self._action_std = self._load_action_stats(Path(dataset_path))

        # Cached per-sample constants (cloned at getitem to be safe)
        self._image_size = torch.tensor(
            [video_height, video_width, video_height, video_width], dtype=torch.float32
        )
        self._padding_mask = torch.zeros(1, video_height, video_width, dtype=torch.float32)
        self._dataset_name = "molmobot"

        print(
            f"[MlSpacesVideoActionDataset] root={dataset_path} | samples={len(self)} | "
            f"exo={exo_camera} | num_frames={num_frames} | action=({action_horizon},{action_dim}) | "
            f"aug={use_image_aug}"
        )

    def _load_action_stats(self, data_root: Path):
        """Build z-score mean/std tensors matching the action concat order (arm then gripper)."""
        with open(data_root / "aggregated_stats.json") as f:
            stats = json.load(f)
        means, stds = [], []
        for selected in self.DEFAULT_ACTIONS:
            # "joint_pos.arm" → "actions/joint_pos/arm"
            key = "actions/" + selected.replace(".", "/")
            means.append(np.asarray(stats[key]["mean"], dtype=np.float32))
            stds.append(np.asarray(stats[key]["std"], dtype=np.float32))
        mean = torch.from_numpy(np.concatenate(means))
        std = torch.from_numpy(np.concatenate(stds))
        std = torch.clamp(std, min=1e-3)  # guard against zero-variance dims
        return mean, std

    def __len__(self):
        return len(self._pi0)

    def __getitem__(self, idx):
        try:
            sample = self._pi0[idx]
        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))

        video = sample.get(self._obs_camera_key)
        if video is None:
            return self.__getitem__(random.randint(0, len(self) - 1))

        # Pi0 returns (T=obs_horizon, 3, H, W) uint8; we want (C, T, H, W)
        video = video.permute(1, 0, 2, 3).contiguous()
        if video.shape[1] != self.num_frames:
            return self.__getitem__(random.randint(0, len(self) - 1))

        action = sample["action"]
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        action = action.to(torch.float32)[: self.action_horizon, : self.action_dim]
        # Z-score normalize so MSE loss magnitudes are comparable to Pi0's transform-normalized setup
        action = ((action - self._action_mean[: self.action_dim]) / self._action_std[: self.action_dim]).contiguous()

        caption = sample.get("task", "") or ""

        return {
            "video": video,
            "action": action,
            "ai_caption": caption,
            "dataset": self._dataset_name,
            "fps": int(getattr(self._pi0, "fps", 4)),
            "num_frames": self.num_frames,
            "padding_mask": self._padding_mask.clone(),
            "image_size": self._image_size.clone(),
        }
