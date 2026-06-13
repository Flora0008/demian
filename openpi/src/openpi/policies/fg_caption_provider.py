"""Fine-grained ground truth caption provider for evaluation.

Loads per-step captions from episodes.jsonl and injects them as reasoning_text
during inference, matching the training-time fine-grained caption inputs.

Usage (server-side, via serve_policy.py):
    python scripts/serve_policy.py ... \
        --fg-captions-path /path/to/episodes.jsonl \
        --fg-caption-key current_action

Usage (client-side):
    provider = FineGrainedCaptionProvider("/path/to/episodes.jsonl")
    provider.set_episode(episode_index)
    for step in range(max_steps):
        obs["reasoning_text"] = provider.get_caption(step)
        action = policy.infer(obs)
"""

import json
import logging
from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

logger = logging.getLogger(__name__)


class FineGrainedCaptionProvider:
    """Loads and provides per-step captions from episodes.jsonl."""

    def __init__(
        self,
        episodes_jsonl_path: str,
        caption_key: str = "current_action",
        fg_stride: int = 4,
    ):
        self._caption_key = caption_key
        self._fg_stride = fg_stride

        # episode_index -> {step_int -> {caption_key -> text}}
        self._captions: dict[int, dict[int, dict[str, str]]] = {}
        # task_name -> [episode_indices]
        self._task_to_episodes: dict[str, list[int]] = {}

        self._load(episodes_jsonl_path)

    def _load(self, path: str):
        logger.info(f"Loading fine-grained captions from {path}")
        with open(path) as f:
            for line in f:
                ep = json.loads(line)
                ep_idx = ep["episode_index"]

                for task in ep.get("tasks", []):
                    self._task_to_episodes.setdefault(task, []).append(ep_idx)

                cps = ep.get("annotations_per_step", {})
                if cps:
                    self._captions[ep_idx] = {int(k): v for k, v in cps.items()}

        logger.info(
            f"Loaded captions for {len(self._captions)} episodes, "
            f"{len(self._task_to_episodes)} unique tasks"
        )

    def get_caption(self, episode_index: int, step: int) -> str | None:
        """Get caption for a specific episode and step.

        Maps step to caption step via floor-division by fg_stride,
        matching the training-time behavior in RoboCasaDataset.
        """
        ep_captions = self._captions.get(episode_index)
        if not ep_captions:
            return None

        caption_step = (step // self._fg_stride) * self._fg_stride
        step_captions = ep_captions.get(caption_step)
        if not step_captions:
            return None

        if self._caption_key == "all":
            non_empty = [v for v in step_captions.values() if v]
            return non_empty[0] if non_empty else None

        return step_captions.get(self._caption_key)

    def get_episodes_for_task(self, task: str) -> list[int]:
        """Get episode indices that match a task name (exact match)."""
        return self._task_to_episodes.get(task, [])

    def find_episode_for_task(self, task: str) -> int | None:
        """Find the first episode index matching a task name."""
        episodes = self.get_episodes_for_task(task)
        return episodes[0] if episodes else None

    @property
    def episode_indices(self) -> list[int]:
        return sorted(self._captions.keys())

    @property
    def tasks(self) -> list[str]:
        return sorted(self._task_to_episodes.keys())


class FineGrainedCaptionPolicy(_base_policy.BasePolicy):
    """Wraps a policy to inject fine-grained GT captions during evaluation.

    At each infer() call, looks up the caption for the current step and
    injects it as 'reasoning_text' in the observation dict. The step counter
    auto-increments and resets when the task prompt changes.

    Client-side overrides (send these keys in the obs dict):
        fg_episode: int  — override which episode's captions to use
        fg_step: int     — override the step counter
        fg_reset: bool   — reset the step counter to 0
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        provider: FineGrainedCaptionProvider,
        default_episode_index: int | None = None,
    ):
        self._policy = policy
        self._provider = provider
        self._step = 0
        self._current_task: str | None = None
        self._current_episode: int | None = default_episode_index

    def set_episode(self, episode_index: int):
        """Explicitly set the episode to use for captions and reset step counter."""
        self._current_episode = episode_index
        self._step = 0

    @override
    def infer(self, obs: dict) -> dict:
        obs = dict(obs)  # shallow copy to avoid mutating caller's dict

        # Handle client-side overrides (pop them so they don't reach the model).
        if obs.pop("fg_reset", False):
            self._step = 0
        if (fg_ep := obs.pop("fg_episode", None)) is not None:
            self._current_episode = int(fg_ep)
            self._step = 0
        fg_step_override = obs.pop("fg_step", None)

        # Detect task changes to auto-select episode and reset step counter.
        prompt = obs.get("prompt")
        if prompt is not None:
            if isinstance(prompt, np.ndarray):
                prompt = prompt.item()
            prompt = str(prompt)

        if prompt and prompt != self._current_task:
            self._current_task = prompt
            self._step = 0
            if self._current_episode is None:
                ep = self._provider.find_episode_for_task(prompt)
                if ep is not None:
                    self._current_episode = ep
                    logger.info(f"Auto-selected episode {ep} for task: {prompt}")
                else:
                    logger.warning(f"No caption episode found for task: {prompt}")

        # Determine which step to use.
        step = int(fg_step_override) if fg_step_override is not None else self._step

        # Inject caption as reasoning_text (only if not already provided by client).
        if self._current_episode is not None and "reasoning_text" not in obs:
            caption = self._provider.get_caption(self._current_episode, step)
            if caption:
                obs["reasoning_text"] = caption

        result = self._policy.infer(obs)

        # Include caption metadata in the output for logging/debugging.
        result["fg_caption_info"] = {
            "episode": self._current_episode,
            "step": step,
            "caption_injected": "reasoning_text" in obs and obs["reasoning_text"] != "",
        }

        self._step += 1
        return result

    @override
    def reset(self):
        self._step = 0
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    @property
    def metadata(self) -> dict[str, Any]:
        if hasattr(self._policy, "metadata"):
            return self._policy.metadata
        return {}
