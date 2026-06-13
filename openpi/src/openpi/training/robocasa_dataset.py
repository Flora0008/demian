"""Fast RoboCasa dataset that reads pre-extracted JPEG images + parquet data.

Bypasses LeRobot's slow initialization (HF datasets, video symlink checks, timestamp
validation) by reading directly from pre-extracted images and parquet files.

Init time: ~2 seconds (vs 30+ minutes with LeRobot for 6,550 episodes).
"""

import json
import logging
import os
import pathlib
import random

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.utils.data

logger = logging.getLogger(__name__)

CAMERA_KEYS = [
    "observation.images.robot0_eye_in_hand",
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
]


class RoboCasaDataset(torch.utils.data.Dataset):
    """Fast dataset for pre-extracted RoboCasa data.

    Directory layout expected:
        root/
          data/chunk-{chunk:03d}/episode_{ep:06d}.parquet
          images/{camera_key}/episode_{ep:06d}_frame_{frame:06d}.jpg
          meta/info.json
          meta/episodes.jsonl
          meta/tasks.jsonl
    """

    def __init__(
        self,
        root: str,
        action_horizon: int = 10,
        action_key: str = "action",
        caption_key: str | None = None,
        fg_stride: int | None = None,
    ):
        self.root = pathlib.Path(root)
        self.action_horizon = action_horizon
        self.action_key = action_key
        self._caption_key = caption_key
        self._fg_stride = fg_stride

        # Load metadata.
        with open(self.root / "meta" / "info.json") as f:
            self.info = json.load(f)
        self.fps = self.info["fps"]
        self.chunks_size = self.info["chunks_size"]
        self.total_episodes = self.info["total_episodes"]

        # Load episodes metadata for lengths.
        self.episodes = []
        with open(self.root / "meta" / "episodes.jsonl") as f:
            for line in f:
                self.episodes.append(json.loads(line))

        # Build global index: map flat index → (episode_idx, frame_idx, task_text).
        # We skip the last (action_horizon - 1) frames of each episode since we can't
        # form a full action chunk from them.
        self._index = []  # (episode_idx, frame_idx)
        self._episode_tasks = []  # task text per episode
        self._episode_reasoning = []  # reasoning text per episode (empty string if absent)
        self._episode_to_indices: dict[int, list[int]] = {}  # ep_idx → [global_indices]
        for ep in self.episodes:
            ep_idx = ep["episode_index"]
            ep_len = ep["length"]
            task_text = ep["tasks"][0]
            usable_frames = max(1, ep_len - action_horizon + 1)
            ep_indices = []
            for frame_idx in range(usable_frames):
                ep_indices.append(len(self._index))
                self._index.append((ep_idx, frame_idx))
            self._episode_to_indices[ep_idx] = ep_indices
            self._episode_tasks.append(task_text)
            self._episode_reasoning.append(ep.get("reasoning_text", ""))

        # Build per-step fine-grained caption lookup (only when fg_stride is set).
        self._episode_fg_captions: dict[int, dict[int, dict]] = {}
        if caption_key and fg_stride:
            for ep in self.episodes:
                ep_idx = ep["episode_index"]
                cps = ep.get("annotations_per_step", {})
                if cps:
                    self._episode_fg_captions[ep_idx] = {int(k): v for k, v in cps.items()}

        # When caption_key is set without fg_stride, use episode-level captions.
        # These are stored in ep["episode_captions"][caption_key] (patched by patch_episode_captions.py).
        #
        # Modes:
        #   "reasoning-short"                              → single key (deterministic)
        #   "environment-simple,reasoning-short"           → concatenated (deterministic)
        #   "random:current_action,environment-simple"     → randomly sample one key per access
        #   "structured:environment-simple,current_action" → "Scene: <1st sentence of key1>. Action: <key2>"
        #   "rw:path/to/reward_table.json"                 → reward-weighted per-task sampling (softmax T=2, top-3)
        self._random_caption_pool: dict[int, list[str]] = {}
        self._rw_caption_weights: dict[int, list[float]] = {}
        if caption_key and not fg_stride:
            is_random = caption_key.startswith("random:")
            is_structured = caption_key.startswith("structured:")
            is_rw = caption_key.startswith("rw:")
            if is_rw:
                self._init_rw_caption_pool(caption_key)
            elif is_structured:
                struct_keys = [k.strip() for k in caption_key[len("structured:"):].split(",")]
                if len(struct_keys) != 2:
                    raise ValueError(f"structured: mode requires exactly 2 keys, got {struct_keys}")
                scene_key, action_key = struct_keys
                overrides = 0
                for ep in self.episodes:
                    ep_idx = ep["episode_index"]
                    ep_caps = ep.get("episode_captions", {})
                    scene_text = ep_caps.get(scene_key, "")
                    action_text = ep_caps.get(action_key, "")
                    if scene_text and action_text:
                        # Take first sentence of scene caption.
                        first_sentence = scene_text.split(".")[0].strip() + "."
                        self._episode_reasoning[ep_idx] = f"Scene: {first_sentence} Action: {action_text}"
                        overrides += 1
                logger.info(
                    f"caption_key={caption_key} (structured): "
                    f"overrode {overrides}/{len(self.episodes)} episodes"
                )
            elif is_random:
                sample_keys = [k.strip() for k in caption_key[len("random:"):].split(",")]
                overrides = 0
                for ep in self.episodes:
                    ep_idx = ep["episode_index"]
                    ep_caps = ep.get("episode_captions", {})
                    pool = []
                    for k in sample_keys:
                        if k == "none":
                            pool.append("")  # empty = task description only (no Reasoning: prefix)
                        else:
                            cap = ep_caps.get(k, "")
                            if cap:
                                pool.append(cap)
                    if pool:
                        self._random_caption_pool[ep_idx] = pool
                        overrides += 1
                logger.info(
                    f"caption_key={caption_key} (random sampling from {len(sample_keys)} keys): "
                    f"{overrides}/{len(self.episodes)} episodes with caption pools"
                )
            else:
                keys = [k.strip() for k in caption_key.split(",")]
                overrides = 0
                for ep in self.episodes:
                    ep_idx = ep["episode_index"]
                    ep_caps = ep.get("episode_captions", {})
                    parts = [ep_caps.get(k, "") for k in keys if ep_caps.get(k)]
                    if parts:
                        self._episode_reasoning[ep_idx] = " ".join(parts)
                        overrides += 1
                logger.info(
                    f"caption_key={'+'.join(keys)} (episode-level): "
                    f"overrode {overrides}/{len(self.episodes)} episodes"
                )

        # Check for packed npz format (Lustre-friendly).
        self._use_packed = (self.root / "packed").is_dir()

        # Cache episode data (lazy loaded).
        self._parquet_cache = {}
        self._image_cache = {}

        logger.info(
            f"RoboCasaDataset: using {'packed npz' if self._use_packed else 'individual JPEG'} images"
        )
        logger.info(
            f"RoboCasaDataset: {self.total_episodes} episodes, "
            f"{len(self._index)} usable frames, action_horizon={action_horizon}"
        )

    def __len__(self) -> int:
        return len(self._index)

    def _init_rw_caption_pool(self, caption_key: str) -> None:
        """Initialize reward-weighted caption pools from a reward table JSON.

        Format: "rw:path/to/reward_table.json"

        The reward table maps task_name → {caption_type: success_rate}.
        For each episode, we look up its source_task, compute softmax probabilities
        (T=2, top-3 truncation) over the 5 caption types, and build a weighted pool.

        Tasks not in the reward table fall back to uniform sampling.
        """
        import math

        rw_path = caption_key[len("rw:"):]
        # Resolve relative paths from project root.
        if not os.path.isabs(rw_path):
            rw_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))), rw_path)
        with open(rw_path) as f:
            reward_table = json.load(f)

        RW_CAPTION_TYPES = ["None", "current_action", "environment-simple", "reasoning-short", "agent_arm_start_state"]
        # Map caption type to episode_captions key ("None" → empty string).
        CAPTION_TYPE_TO_KEY = {
            "None": "none",
            "current_action": "current_action",
            "environment-simple": "environment-simple",
            "reasoning-short": "reasoning-short",
            "agent_arm_start_state": "agent_arm_start_state",
        }
        T = 2.0  # softmax temperature
        TOP_K = 3  # keep top-k after softmax

        # Build source_task → reward_table_key mapping (handles variants like CloseFridge-DoubleDoor → CloseFridge).
        rw_keys = sorted(reward_table.keys(), key=len, reverse=True)  # longest first for prefix match

        def _resolve_task(source_task: str) -> str | None:
            if source_task in reward_table:
                return source_task
            for rw_key in rw_keys:
                if source_task.startswith(rw_key):
                    return rw_key
            return None

        overrides = 0
        fallback_uniform = 0
        for ep in self.episodes:
            ep_idx = ep["episode_index"]
            ep_caps = ep.get("episode_captions", {})
            source_task = ep.get("source_task", "")

            rw_key = _resolve_task(source_task)
            if rw_key is not None:
                weights = reward_table[rw_key]
                # Softmax with temperature T, top-k truncation.
                raw = [weights.get(ct, 0.0) for ct in RW_CAPTION_TYPES]
                exp_w = [math.exp(w / T) for w in raw]
                # Top-k: zero out all but the top-k.
                indexed = sorted(enumerate(exp_w), key=lambda x: -x[1])
                mask = set(i for i, _ in indexed[:TOP_K])
                exp_w = [w if i in mask else 0.0 for i, w in enumerate(exp_w)]
                total = sum(exp_w)
                probs = [w / total for w in exp_w] if total > 0 else [1.0 / len(RW_CAPTION_TYPES)] * len(RW_CAPTION_TYPES)
            else:
                # Uniform fallback for tasks not in reward table.
                probs = [1.0 / len(RW_CAPTION_TYPES)] * len(RW_CAPTION_TYPES)
                fallback_uniform += 1

            # Build pool: list of caption texts and corresponding probabilities.
            pool = []
            pool_weights = []
            for ct, prob in zip(RW_CAPTION_TYPES, probs):
                if prob <= 0:
                    continue
                key = CAPTION_TYPE_TO_KEY[ct]
                if key == "none":
                    pool.append("")
                else:
                    cap = ep_caps.get(key, "")
                    if not cap:
                        continue  # skip if caption missing
                    pool.append(cap)
                pool_weights.append(prob)

            if pool:
                # Re-normalize after dropping missing captions.
                total = sum(pool_weights)
                pool_weights = [w / total for w in pool_weights]
                self._random_caption_pool[ep_idx] = pool
                self._rw_caption_weights[ep_idx] = pool_weights
                overrides += 1

        logger.info(
            f"caption_key={caption_key} (reward-weighted, T={T}, top-{TOP_K}): "
            f"{overrides}/{len(self.episodes)} episodes with weighted pools "
            f"({fallback_uniform} uniform fallback, {len(reward_table)} tasks in table)"
        )

    def _get_reasoning_text(self, ep_idx: int, frame_idx: int) -> str:
        """Get reasoning text for a frame, using per-step captions if available.

        When fg_stride is set, maps frame_idx to the caption step via floor-division:
            step = (frame_idx // fg_stride) * fg_stride
        e.g. fg_stride=4: frame 5 → step 4, frame 0 → step 0
             fg_stride=8: frame 5 → step 0, frame 9 → step 8

        When caption_key="all", randomly picks one non-empty caption from the step.
        When caption_key="random:...", randomly picks one caption type per access.
        When caption_key="rw:...", reward-weighted sampling per task.
        """
        # Episode-level random/rw sampling.
        if self._random_caption_pool and ep_idx in self._random_caption_pool:
            weights = self._rw_caption_weights.get(ep_idx)
            if weights is not None:
                return random.choices(self._random_caption_pool[ep_idx], weights=weights, k=1)[0]
            return random.choice(self._random_caption_pool[ep_idx])

        if self._caption_key and self._fg_stride and ep_idx in self._episode_fg_captions:
            step = (frame_idx // self._fg_stride) * self._fg_stride
            captions = self._episode_fg_captions[ep_idx].get(step, {})
            if captions:
                if self._caption_key == "all":
                    non_empty = [v for v in captions.values() if v]
                    if non_empty:
                        return random.choice(non_empty)
                else:
                    text = captions.get(self._caption_key, "")
                    if text:
                        return text
        return self._episode_reasoning[ep_idx]

    @staticmethod
    def _retry_io(fn, max_retries=5, base_delay=1.0):
        """Retry an I/O operation with exponential backoff for transient Lustre errors."""
        import time

        for attempt in range(max_retries):
            try:
                return fn()
            except (BrokenPipeError, OSError) as e:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Transient I/O error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)

    def _load_episode_data(self, ep_idx: int):
        """Load and cache parquet data for an episode."""
        if ep_idx not in self._parquet_cache:
            chunk_idx = ep_idx // self.chunks_size
            pf = self.root / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
            df = self._retry_io(lambda: pq.read_table(str(pf)).to_pandas())
            states = np.stack(df["observation.state"].values).astype(np.float32)
            actions = np.stack(df[self.action_key].values).astype(np.float32)
            task_indices = df["task_index"].values
            self._parquet_cache[ep_idx] = (states, actions, task_indices)
            # Keep cache bounded to avoid memory issues.
            if len(self._parquet_cache) > 200:
                oldest = next(iter(self._parquet_cache))
                del self._parquet_cache[oldest]
        return self._parquet_cache[ep_idx]

    def _load_episode_images(self, ep_idx: int):
        """Load and cache all images for an episode from packed npz."""
        if ep_idx not in self._image_cache:
            npz_path = self.root / "packed" / f"episode_{ep_idx:06d}.npz"
            data = self._retry_io(lambda: np.load(str(npz_path)))
            self._image_cache[ep_idx] = {
                CAMERA_KEYS[0]: data["eye_in_hand"],
                CAMERA_KEYS[1]: data["agentview_left"],
                CAMERA_KEYS[2]: data["agentview_right"],
            }
            # Keep cache bounded.
            if len(self._image_cache) > 100:
                oldest = next(iter(self._image_cache))
                del self._image_cache[oldest]
        return self._image_cache[ep_idx]

    def _load_image_jpeg(self, cam_key: str, ep_idx: int, frame_idx: int) -> np.ndarray:
        """Load a single JPEG image (fallback for non-packed datasets)."""
        import cv2

        path = self.root / "images" / cam_key / f"episode_{ep_idx:06d}_frame_{frame_idx:06d}.jpg"
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index: int) -> dict:
        ep_idx, frame_idx = self._index[index]
        states, actions, task_indices = self._load_episode_data(ep_idx)

        # Build action chunk: action_horizon steps starting from frame_idx.
        ep_len = len(actions)
        action_chunk = []
        for t in range(self.action_horizon):
            idx = min(frame_idx + t, ep_len - 1)
            action_chunk.append(actions[idx])
        action_chunk = np.stack(action_chunk, axis=0)  # (action_horizon, action_dim)

        # Load images — packed npz (1 file per episode) or individual JPEGs.
        if self._use_packed:
            ep_images = self._load_episode_images(ep_idx)
            images = {k: ep_images[k][frame_idx] for k in CAMERA_KEYS}
        else:
            images = {}
            for cam_key in CAMERA_KEYS:
                images[cam_key] = self._load_image_jpeg(cam_key, ep_idx, frame_idx)

        # Get task description and reasoning text.
        task_text = self._episode_tasks[ep_idx]
        reasoning_text = self._get_reasoning_text(ep_idx, frame_idx)

        result = {
            "observation.state": states[frame_idx],
            "observation.images.robot0_eye_in_hand": images[CAMERA_KEYS[0]],
            "observation.images.robot0_agentview_left": images[CAMERA_KEYS[1]],
            "observation.images.robot0_agentview_right": images[CAMERA_KEYS[2]],
            self.action_key: action_chunk,
            "task_index": int(task_indices[frame_idx]),
            "prompt": task_text,
            "reasoning_text": reasoning_text,
            "timestamp": np.float32(frame_idx / self.fps),
            "frame_index": frame_idx,
            "episode_index": ep_idx,
            "index": index,
        }
        return result


class EpisodeLocalBatchSampler(torch.utils.data.Sampler):
    """Batch sampler that groups frames from a small number of episodes per batch.

    Instead of sampling batch_size random indices (hitting many different episodes/npz files),
    this picks `episodes_per_batch` random episodes and samples `batch_size // episodes_per_batch`
    frames from each. This dramatically reduces the number of npz file loads on Lustre.

    For DDP: each rank gets its own disjoint subset of episodes.
    """

    def __init__(
        self,
        dataset: RoboCasaDataset,
        batch_size: int,
        episodes_per_batch: int = 4,
        *,
        shuffle: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.episodes_per_batch = episodes_per_batch
        self.frames_per_episode = batch_size // episodes_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.epoch = 0

        # Split episodes across ranks.
        all_ep_ids = sorted(dataset._episode_to_indices.keys())
        self.rank_episodes = all_ep_ids[rank::world_size]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_skip_batches(self, n: int) -> None:
        """Skip the first *n* batches on the next iteration (for resume)."""
        self._skip_batches = n

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)

        # Shuffle episodes for this epoch.
        ep_ids = self.rank_episodes.copy()
        if self.shuffle:
            rng.shuffle(ep_ids)

        skip = getattr(self, "_skip_batches", 0)
        batch_idx = 0

        # Walk through episodes in groups of episodes_per_batch.
        for start in range(0, len(ep_ids) - self.episodes_per_batch + 1, self.episodes_per_batch):
            group = ep_ids[start : start + self.episodes_per_batch]
            batch = []
            for ep_id in group:
                ep_indices = self.dataset._episode_to_indices[ep_id]
                if len(ep_indices) >= self.frames_per_episode:
                    chosen = rng.choice(ep_indices, size=self.frames_per_episode, replace=False)
                else:
                    chosen = rng.choice(ep_indices, size=self.frames_per_episode, replace=True)
                batch.extend(chosen.tolist())

            if self.drop_last and len(batch) < self.batch_size:
                continue

            # Skip already-consumed batches on resume (RNG state is still advanced above).
            if batch_idx < skip:
                batch_idx += 1
                continue
            batch_idx += 1
            yield batch

        # Clear skip after the first epoch so subsequent epochs start from batch 0.
        self._skip_batches = 0

    def __len__(self) -> int:
        num_groups = len(self.rank_episodes) // self.episodes_per_batch
        return num_groups
