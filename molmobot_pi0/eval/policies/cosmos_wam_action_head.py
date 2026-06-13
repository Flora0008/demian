"""Cosmos-WAM ActionHead policy wrapper for MolmoSpaces eval.

Wraps an ``ActionHeadVideoModel`` from the sibling ``cosmos_wam`` repo (same
Cosmos Predict2-2B DiT backbone as ``cosmos_policy``, but with a separate
cross-attention ActionHead that lives outside the FSDP boundary).

Differs from ``CosmosPolicy`` in four places:
  1. Loads a consolidated ``.pt`` produced by ``cosmos_wam``'s
     ``scripts/convert_dcp_to_regular.py`` — that script already merges the
     ``action_head`` weights from ``aux_optim.pt`` into the single state dict.
  2. Instantiates the model via cosmos_predict2's Hydra config machinery
     (experiment name registered in ``cosmos_predict2.experiments.base.action``).
  3. Calls ``generate_samples_with_latents_from_batch`` directly; no separate
     ``get_action`` helper and no T5 embeddings cache (the model does online
     Qwen encoding from the ``ai_caption`` string).
  4. Action normalization is z-score (``(x - mean) / std``) using stats
     exported from the dataset's ``aggregated_stats.json``. No proprio fed to
     the model — our ActionHead reads only video hidden states.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from molmo_spaces.policy.base_policy import InferencePolicy, StatefulPolicy


log = logging.getLogger(__name__)


# Caption templates copied verbatim from molmobot_pi0/eval/policies/pi.py.
# We can't import them directly because pi.py also imports openpi at module
# level (for the Pi0 weights loader), and the cosmos-policy-eval venv this
# adapter runs in doesn't include openpi. Keep these in lockstep with pi.py
# when adding new caption keys.
CAPTION_TEMPLATES: dict[str, dict[str, str]] = {
    "current_action": {
        "pick": "The robotic arm moves its gripper towards the {pickup_obj}, then closes the gripper to grasp the {pickup_obj}, lifting it off the surface.",
        "pick_and_place": "The robot arm moves its gripper towards the {pickup_obj}, grasps it, and lifts it upwards. The arm then repositions the {pickup_obj} over the {place_obj} and places it inside, completing the task.",
        "pick_and_place_next_to": "The robot arm moves its gripper towards the {pickup_obj}, grasps it, and lifts it. The arm then moves the {pickup_obj} toward the {place_obj}, placing it next to the {place_obj} on the surface.",
        "pick_and_place_color": "The robot arm moves its gripper towards the {pickup_obj}, grasps it, and lifts it upwards. The arm then places the {pickup_obj} into a nearby receptacle, completing the task.",
    },
    "environment-simple": {
        "pick": "A robotic arm is positioned over a surface where a {pickup_obj} is placed. The arm is attempting to pick up the {pickup_obj}.",
        "pick_and_place": "A robotic arm is positioned over a surface with various objects including a {pickup_obj} and a {place_obj}. The arm is manipulating the {pickup_obj} and placing it into the {place_obj}.",
        "pick_and_place_next_to": "A robotic arm is positioned over a surface with various objects including a {pickup_obj} and a {place_obj}. The arm is manipulating the {pickup_obj} and placing it next to the {place_obj}.",
        "pick_and_place_color": "A robotic arm is positioned over a surface with various objects including a {pickup_obj}. The arm is manipulating the {pickup_obj} and placing it into a nearby bowl.",
    },
    "agent_arm_state": {
        "pick": "The robotic arm is positioned above the surface. The arm moves its gripper towards the {pickup_obj}, closes the gripper around the {pickup_obj}, and lifts it upward, away from the surface.",
        "pick_and_place": "The robotic arm is positioned above the surface, with its gripper closed around the {pickup_obj}. The arm moves towards the {place_obj}, lowers the {pickup_obj} onto the {place_obj}, then lifts and retracts upward, moving away from the objects.",
        "pick_and_place_next_to": "The robotic arm is positioned above the surface, with its gripper closed around the {pickup_obj}. The arm moves laterally towards the {place_obj}, lowers the {pickup_obj} next to the {place_obj}, then lifts and retracts upward.",
        "pick_and_place_color": "The robotic arm is positioned above the surface, with its gripper closed around the {pickup_obj}. The arm lowers the {pickup_obj} into a nearby receptacle, then lifts and retracts upward, moving away from the objects.",
    },
    "reasoning-short": {
        "pick": "The robot arm moves toward the {pickup_obj} on the surface. It positions its gripper above the {pickup_obj}, preparing to grasp it. The arm then lowers and securely picks up the {pickup_obj}, lifting it off the surface.",
        "pick_and_place": "The robot arm picks up the {pickup_obj} from the surface. The arm then moves the {pickup_obj} towards the {place_obj}, positioning it for placement. The {pickup_obj} is carefully placed into the {place_obj}, completing the task of picking and placing.",
        "pick_and_place_next_to": "The robot arm picks up the {pickup_obj} from the surface. It then moves the {pickup_obj} towards the {place_obj}, positioning it next to the {place_obj}. The task is completed with the {pickup_obj} placed beside the {place_obj}.",
        "pick_and_place_color": "The robot arm moves towards the {pickup_obj} on the surface, grasps it with its gripper, and lifts it. It then carefully places the {pickup_obj} into a nearby receptacle, completing the task of picking and placing the object.",
    },
}


def _parse_objects_from_prompt(prompt: str) -> tuple[str, str, str]:
    """Extract (task_type, pickup_obj, place_obj) from a MolmoSpaces task
    description. Verbatim from pi.py:_parse_objects_from_prompt — kept here
    only because pi.py drags in openpi at import time."""
    import re
    m = re.match(r"pick up (?:the|some|a) (.+?) and place it next to (?:the )?(.+)", prompt, re.IGNORECASE)
    if m:
        return "pick_and_place_next_to", m.group(1).strip(), m.group(2).strip()
    m = re.match(r"pick up (?:the|some|a) (.+?) and place it (?:in or on|in|on) (?:the )?(.+)", prompt, re.IGNORECASE)
    if m:
        return "pick_and_place", m.group(1).strip(), m.group(2).strip()
    m = re.match(r"pick up (?:the|some|a) (.+)", prompt, re.IGNORECASE)
    if m:
        return "pick", m.group(1).strip(), ""
    return "pick", "the object", ""


PRIMARY_IMAGE_KEY = "primary_image"
WRIST_IMAGE_KEY = "wrist_image"


@dataclass
class CosmosWAMActionHeadState:
    actions_buffer: np.ndarray | None
    current_buffer_index: int
    last_prompt: str | None


def _center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w:
        return img
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    return img[top : top + side, left : left + side]


def _resize_cv(img: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """OpenCV resize — matches training's VideoResize."""
    import cv2  # lazy — cv2 pulls in a lot

    h, w = target_hw
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


class CosmosWAMActionHeadPolicy(InferencePolicy, StatefulPolicy):
    """Evaluates a cosmos_wam ActionHeadVideoModel as a MolmoSpaces policy."""

    # Match the training dataloader's video shape.
    NUM_VIDEO_FRAMES = 13
    TARGET_H = 256
    TARGET_W = 256

    def __init__(
        self,
        checkpoint_path: str,
        experiment: str,
        dataset_stats_path: str,
        cameras: dict[str, str] | None = None,
        action_dim: int = 8,
        num_action_steps: int = 16,
        num_open_loop_steps: int = 16,
        num_denoising_steps: int = 35,
        shift: float = 5.0,
        guidance: float = 1.5,
        seed: int = 0,
        randomize_seed: bool = False,
        caption_key: str | None = None,
        color_task: bool = False,
    ) -> None:
        super().__init__(None, "pick_and_place")

        # caption_key=None → no injection (raw task description, like the
        # mix5+none condition). When set, must be a key in pi.CAPTION_TEMPLATES;
        # we parse the task description ("Pick up the X and place it … the Y")
        # to extract pickup_obj / place_obj and format the matching template.
        if caption_key is not None and caption_key not in CAPTION_TEMPLATES:
            raise ValueError(
                f"Unknown caption_key: {caption_key!r}. "
                f"Available: {sorted(CAPTION_TEMPLATES.keys())}"
            )
        self._caption_key = caption_key
        self._color_task = bool(color_task)

        if cameras is None:
            raise ValueError("CosmosWAMActionHeadPolicy requires an explicit `cameras` mapping")
        invalid = [v for v in cameras.values() if v not in (PRIMARY_IMAGE_KEY, WRIST_IMAGE_KEY)]
        if invalid:
            raise ValueError(
                f"Camera values must be one of {{{PRIMARY_IMAGE_KEY!r}, "
                f"{WRIST_IMAGE_KEY!r}}}; got {invalid}"
            )
        # Find the "primary" (exo) camera key — we only feed exo to the model for now,
        # mirroring our training adapter which only subscribed to exo.
        self._primary_obs_key = None
        for obs_key, input_key in cameras.items():
            if input_key == PRIMARY_IMAGE_KEY:
                self._primary_obs_key = obs_key
                break
        if self._primary_obs_key is None:
            raise ValueError(
                f"cameras mapping must include one {PRIMARY_IMAGE_KEY!r} value; got {cameras}"
            )

        self.checkpoint_path = str(checkpoint_path)
        self.experiment = experiment
        self.dataset_stats_path = str(dataset_stats_path)

        self.action_dim = int(action_dim)
        self.num_action_steps = int(num_action_steps)
        self.num_open_loop_steps = int(num_open_loop_steps)
        self.num_denoising_steps = int(num_denoising_steps)
        self.shift = float(shift)
        self.guidance = float(guidance)
        self.seed = int(seed)
        self.randomize_seed = bool(randomize_seed)

        self._prepared = False
        self.model = None
        self.action_mean: np.ndarray | None = None
        self.action_std: np.ndarray | None = None
        self._last_prompt: str | None = None
        self._last_raw_task: str | None = None
        self.actions_buffer: np.ndarray | None = None
        self.current_buffer_index: int = 0

    # ---------------------------------------------------------------- state
    def reset(self, *args, **kwargs):
        self.actions_buffer = None
        self.current_buffer_index = 0
        self._last_prompt = None
        self._last_raw_task = None

    def get_state(self):
        return CosmosWAMActionHeadState(
            actions_buffer=self.actions_buffer.copy() if self.actions_buffer is not None else None,
            current_buffer_index=self.current_buffer_index,
            last_prompt=self._last_prompt,
        )

    def set_state(self, state: CosmosWAMActionHeadState):
        self.actions_buffer = state.actions_buffer.copy() if state.actions_buffer is not None else None
        self.current_buffer_index = state.current_buffer_index
        self._last_prompt = state.last_prompt

    def has_queued_actions(self) -> bool:
        return (
            self.actions_buffer is not None
            and self.current_buffer_index < min(self.num_open_loop_steps, len(self.actions_buffer))
        )

    # ---------------------------------------------------------------- setup
    def prepare_model(self, *_, **__):
        if self._prepared:
            return
        import importlib
        import sys

        import torch
        import torch.distributed as dist

        # --- 1. Single-GPU distributed init (our model uses FSDP machinery) ---
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29600")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("LOCAL_RANK", "0")
            dist.init_process_group("nccl")
        torch.cuda.set_device(0)

        from megatron.core import parallel_state
        if not parallel_state.is_initialized():
            parallel_state.initialize_model_parallel()

        # --- 2. Instantiate model via Hydra ---
        from cosmos_predict2._src.imaginaire.lazy_config import instantiate
        from cosmos_predict2._src.imaginaire.utils.config_helper import (
            get_config_module, override,
        )
        # Experiment configs register under this module path
        importlib.import_module("cosmos_predict2.experiments.base.action")

        config_path = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
        config_module = get_config_module(config_path)
        config = importlib.import_module(config_module).make_config()
        # Local tokenizer path (avoids HF gated-repo download for the Wan2.1 VAE).
        tokenizer_pth = os.environ.get(
            "COSMOS_TOKENIZER_PTH",
            "/path/to/"
            "models/Cosmos-Predict2.5-2B/tokenizer.pth",
        )
        config = override(
            config,
            [
                "--",
                f"experiment={self.experiment}",
                "model.config.fsdp_shard_size=1",
                f"+model.config.tokenizer.vae_pth={tokenizer_pth}",
            ],
        )
        log.info(f"[cosmos_wam] Building model for experiment={self.experiment}")
        model = instantiate(config.model)

        # --- 3. Load consolidated .pt (backbone + merged action_head) ---
        log.info(f"[cosmos_wam] Loading checkpoint: {self.checkpoint_path}")
        sd = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)

        # Separate action_head keys from the rest (our ActionHeadVideoModel's custom
        # load_state_dict silently drops action_head.* keys; we load them by hand).
        ah_sd = {
            k[len("action_head."):]: v
            for k, v in sd.items()
            if k.startswith("action_head.")
        }
        backbone_sd = {k: v for k, v in sd.items() if not k.startswith("action_head.")}

        # Our ActionHeadVideoModel has a custom load_state_dict that returns None
        # and logs its own IncompatibleKeys summary — so don't unpack.
        result = model.load_state_dict(backbone_sd, strict=False)
        if result is None:
            log.info("[cosmos_wam] Backbone load complete (custom loader; see checkpointer log for diff)")
        else:
            missing, unexpected = result
            log.info(
                f"[cosmos_wam] Backbone load: missing={len(missing)} unexpected={len(unexpected)}"
            )
        if ah_sd and hasattr(model, "action_head"):
            model.action_head.load_state_dict(ah_sd, strict=True)
            log.info(f"[cosmos_wam] Restored {len(ah_sd)} action_head tensors from ckpt")
        else:
            log.warning(
                "[cosmos_wam] No action_head weights in ckpt — action head will be random!"
            )

        model.eval()
        # Training uses bfloat16 mixed precision (via FSDP). At single-GPU eval we
        # cast the entire model to bf16 to match what the training pipeline saw —
        # mirrors hsg's socket_test_cosmos_wam.py (`model.cuda().to(torch.bfloat16)`).
        model = model.cuda().to(torch.bfloat16)
        self.model = model

        # --- 4. Action normalization stats (z-score over training dataset) ---
        log.info(f"[cosmos_wam] Loading dataset stats: {self.dataset_stats_path}")
        with open(self.dataset_stats_path) as f:
            stats = json.load(f)
        self.action_mean = np.asarray(stats["actions_mean"], dtype=np.float32)
        self.action_std = np.asarray(stats["actions_std"], dtype=np.float32)
        # Guard against zero-variance dims (same clamp as in training adapter)
        self.action_std = np.clip(self.action_std, 1e-3, None).astype(np.float32)
        assert self.action_mean.shape == (self.action_dim,), (
            f"actions_mean shape {self.action_mean.shape} != ({self.action_dim},)"
        )

        self._prepared = True
        log.info("[cosmos_wam] Ready.")

    def _build_prompt(self, raw_prompt: str) -> str:
        """Apply caption-key injection (if any) to the raw task description.

        With caption_key=None we pass the raw description through unchanged
        (matches the existing Tier-3 'none' condition). Otherwise we parse the
        MolmoSpaces task description to extract task_type/pickup_obj/place_obj
        and format the matching template from pi.CAPTION_TEMPLATES.

        color_task=True is the convention from pi.py: it overrides the parsed
        task_type to 'pick_and_place_color' so the color-disambiguation
        benchmarks ('pnp_color') get the color-task template even though their
        task descriptions look like a plain pick_and_place.
        """
        if self._caption_key is None:
            return raw_prompt
        task_type, pickup_obj, place_obj = _parse_objects_from_prompt(raw_prompt)
        if self._color_task:
            task_type = "pick_and_place_color"
        template = CAPTION_TEMPLATES[self._caption_key].get(task_type)
        if template is None:
            # Defensive: fall back to the raw description if a (caption_key,
            # task_type) pair isn't in the template dict. Should not happen
            # with the 4 keys we sweep (ca/es/aas/rs) — those cover all 4
            # task types — but keeps the policy robust to template gaps.
            log.warning(
                f"[cosmos_wam] no template for caption_key={self._caption_key!r}"
                f" task_type={task_type!r}; using raw prompt"
            )
            return raw_prompt
        caption = template.format(pickup_obj=pickup_obj, place_obj=place_obj)
        log.info(
            f"[cosmos_wam] injected caption ({self._caption_key}, {task_type}):"
            f" {caption!r}"
        )
        return caption

    # ------------------------------------------------------------- inference
    def obs_to_model_input(self, obs):
        if isinstance(obs, list):
            obs = obs[0]

        # Prompt — compare raw task descriptions to detect mid-inference
        # task changes (the existing sanity check), then format the prompt
        # once via _build_prompt (caption injection if caption_key is set).
        if "task" in obs:
            raw_task = obs["task"]
            if self._last_raw_task is not None and self._last_raw_task != raw_task:
                raise AssertionError(
                    f"Task changed during inference! "
                    f"{self._last_raw_task!r} -> {raw_task!r}"
                )
            if self._last_prompt is None:
                prompt = self._build_prompt(raw_task)
                self._last_raw_task = raw_task
            else:
                prompt = self._last_prompt
        elif self._last_prompt is not None:
            prompt = self._last_prompt
        elif self.task is not None:
            raw_task = self.task.get_task_description()
            prompt = self._build_prompt(raw_task)
            self._last_raw_task = raw_task
        else:
            raise ValueError("No prompt passed and no task registered!")
        if self._last_prompt is None:
            self._last_prompt = prompt

        model_input: dict = {"task": prompt}
        if not self.has_queued_actions():
            img = obs[self._primary_obs_key]
            if img.ndim != 3 or img.shape[-1] != 3:
                raise ValueError(
                    f"Expected (H,W,3) uint8 for camera {self._primary_obs_key!r}, "
                    f"got {img.shape}"
                )
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            img = _center_crop_square(img)
            img = _resize_cv(img, (self.TARGET_H, self.TARGET_W))
            model_input["primary_image"] = img

        return model_input

    def _build_data_batch(self, model_input):
        """Pack obs into the data_batch dict expected by ``ActionHeadVideoModel.forward``."""
        import torch

        device = "cuda:0"
        img = model_input["primary_image"]  # (H, W, 3) uint8

        # Replicate the single conditioning frame to fill the 13-frame window.
        # Matches hsg socket server's convention — repeating the last frame
        # keeps VAE temporal convs clean (vs zero-padding which bleeds into
        # the conditioning latent).
        frames = np.repeat(img[None, ...], self.NUM_VIDEO_FRAMES, axis=0)  # (13, H, W, 3)
        video = (
            torch.from_numpy(frames)
            .permute(3, 0, 1, 2)   # (3, 13, H, W)
            .unsqueeze(0)           # (1, 3, 13, H, W)
            .to(torch.uint8)
            .to(device)
        )

        # Action placeholder — the denoise() path in ActionHeadVideoModel builds its
        # own zero dummy for the DiT's legacy action_embedder, so this one is largely
        # redundant; keep it bf16 to match the cast-to-bf16 model.
        action_placeholder = torch.zeros(
            1, self.num_action_steps, self.action_dim,
            device=device, dtype=torch.bfloat16,
        )

        data_batch = {
            "video": video,
            "fps": torch.tensor([4], device=device),
            "padding_mask": torch.zeros(
                1, 1, self.TARGET_H, self.TARGET_W,
                device=device, dtype=torch.bfloat16,
            ),
            "action": action_placeholder,
            "num_conditional_frames": 1,
            "ai_caption": [model_input["task"]],
        }
        return data_batch

    def inference_model(self, model_input):
        self.prepare_model()
        import torch

        if not self.has_queued_actions():
            t0 = time.perf_counter()
            data_batch = self._build_data_batch(model_input)

            # The training forward() populates t5_text_embeddings from ai_caption via
            # the online Qwen encoder, but generate_samples_with_latents_from_batch
            # skips that step and the conditioner then hits None on t5_text_embeddings.
            # Replicate the online-encoding block here.
            cfg = self.model.config
            if cfg.text_encoder_config is not None and cfg.text_encoder_config.compute_online \
                    and "t5_text_embeddings" not in data_batch:
                # vlmimg variants (text_encoder_config.image_conditioning=True with
                # use_vlm_action_expert=False) require the caller to splice the
                # conditioning frame in before the encoder runs — the model's
                # own auto-splice in generate_samples_with_latents_from_batch is
                # gated on use_vlm_action_expert. Mirror cosmos_wam socket_test
                # (cosmos_wam/socket_test_cosmos_wam.py:706-710) and the model
                # forward path (action_head_video2world_model.py:573-578), both
                # of which pass video[:,:,0] as-is — no extra normalization,
                # since _normalize_video_databatch_inplace runs after the
                # encoder in both training and inference.
                if getattr(cfg.text_encoder_config, "image_conditioning", False):
                    _vt = data_batch.get(self.model.input_data_key, None)
                    if _vt is not None and _vt.dim() == 5:
                        # The encoder's joint path expects [-1, 1] float.
                        # Dataloader emits uint8 [0, 255]; convert here, matching
                        # the same fix in action_head_video2world_model.py:578.
                        _cf = _vt[:, :, 0]
                        if _cf.dtype == torch.uint8:
                            _cf = _cf.to(torch.float32) / 127.5 - 1.0
                        data_batch["conditioning_frame"] = _cf.contiguous()
                with torch.no_grad():
                    t5_emb = self.model.text_encoder.compute_text_embeddings_online(
                        data_batch, self.model.input_caption_key,
                    )
                data_batch["t5_text_embeddings"] = t5_emb
                data_batch["t5_text_mask"] = torch.ones(
                    t5_emb.shape[0], t5_emb.shape[1], device=t5_emb.device,
                )

            with torch.no_grad():
                _, _, action_pred, _ = self.model.generate_samples_with_latents_from_batch(
                    data_batch,
                    guidance=self.guidance,
                    seed=self.seed,
                    num_steps=self.num_denoising_steps,
                    shift=self.shift,
                )
            # (1, num_action_steps, action_dim) bf16
            actions_norm = action_pred[0].float().cpu().numpy()  # (16, 8)
            # Denormalize via training z-score stats.
            actions_raw = actions_norm * self.action_std + self.action_mean
            self.actions_buffer = actions_raw.astype(np.float32)
            self.current_buffer_index = 0
            dt = time.perf_counter() - t0
            log.info(
                f"[cosmos_wam] Action chunk ready: shape={self.actions_buffer.shape}, "
                f"{dt:.2f}s ({dt / max(self.num_denoising_steps, 1):.2f}s/step)"
            )

        assert self.actions_buffer is not None
        action = self.actions_buffer[self.current_buffer_index]
        self.current_buffer_index += 1
        return action  # (action_dim,) float32

    def model_output_to_action(self, model_output: np.ndarray) -> dict[str, np.ndarray]:
        arm = np.asarray(model_output[:7], dtype=np.float32).reshape(7)
        gripper_raw = float(np.asarray(model_output[7]).reshape(()))
        gripper = np.array([np.clip(gripper_raw, 0.0, 255.0)], dtype=np.float32)
        return {"arm": arm, "gripper": gripper}

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "cosmos_wam_action_head"
        info["policy_checkpoint"] = self.checkpoint_path
        info["policy_experiment"] = self.experiment
        info["num_action_steps"] = self.num_action_steps
        info["num_open_loop_steps"] = self.num_open_loop_steps
        info["num_denoising_steps"] = self.num_denoising_steps
        return info
