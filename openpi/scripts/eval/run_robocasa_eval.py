"""
Evaluate OpenPI (pi05) on RoboCasa 365 atomic and composite tasks.

Usage
-----
1. Start the policy server in a separate terminal:

       python scripts/serve_policy.py --policy:config pi05_robocasa \
           --policy:dir <checkpoint> --port 8000

2. Run the evaluator (all tasks):

       python scripts/eval/run_robocasa_eval.py

   Or a specific task / subset:

       python scripts/eval/run_robocasa_eval.py --tasks OpenDrawer PrepareCoffee
       python scripts/eval/run_robocasa_eval.py --split atomic
       python scripts/eval/run_robocasa_eval.py --split composite

   For sharded parallel evaluation across multiple workers:

       python scripts/eval/run_robocasa_eval.py --task-shard 0 --num-shards 8
       python scripts/eval/run_robocasa_eval.py --task-shard 1 --num-shards 8
       ...

3. Merge sharded results:

       python scripts/eval/merge_results.py --results-dir runs/robocasa/<date>/

Notes
-----
- Environment: gym.make("robocasa/<task>", split="target") via PandaOmron + HybridMobileBase
- Observation sent to server: 3 cameras (256x256) + 16-dim state + prompt
  State layout: [base_pos(3), base_quat(4), eef_pos_rel(3), eef_quat_rel(4), gripper_qpos(2)]
- Action received from server (7-dim):
  [eef_pos(3), eef_rot(3), gripper(1)]
  Client pads with base_motion(4)=0 + control_mode(1)=0 for the gym action dict.
"""

from __future__ import annotations

# DFW-eval: container ships numpy 1.26.4 and pip refuses to install 2.2.5
# (constraints conflict). robocasa's __init__.py asserts numpy.__version__
# == "2.2.5" — bypass by overriding the version string before the lazy
# robocasa import happens. robocasa's actual usage is array ops compatible
# across 1.26 / 2.x.
import os as _os
if _os.environ.get("DFW_EVAL_PATCH_NUMPY", "0") == "1":
    import numpy as _np_patch
    _np_patch.__version__ = "2.2.5"

import json
import logging
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
import mediapy
import numpy as np
import tyro
from tqdm import tqdm

from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from robocasa_task_list import (
    ATOMIC_TASKS,
    COMPOSITE_TASKS,
    ALL_TASKS,
    ATOMIC_TASK_NAMES,
    COMPOSITE_TASK_NAMES,
    ALL_TASK_NAMES,
)
from composite_subgoals import get_subgoal_checker, get_subgoal_plan, resolve_subgoal_runtime


# ---------------------------------------------------------------------------
# Inference client
# ---------------------------------------------------------------------------

class RoboCasaClient:
    """
    Wraps openpi_client.WebsocketClientPolicy with chunked open-loop inference
    for RoboCasa.

    Observation format expected by the server (pi05_robocasa / RoboCasaInputs):
      observation/agentview_left   — left agentview  (H, W, 3)
      observation/eye_in_hand      — wrist cam       (H, W, 3)
      observation/agentview_right  — right agentview (H, W, 3)
      observation/state            — 16-dim flat proprio
      prompt                       — language instruction

    Action returned by server (7-dim via RoboCasaOutputs):
      [eef_pos(3), eef_rot(3), gripper(1)]

    The action is padded with zeros for base_motion + control_mode and
    converted to a gym action dict before being passed to env.step().
    """

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        open_loop_horizon: int = 8,
        img_h: int = 256,
        img_w: int = 256,
        occlude_camera: str | None = None,
        occlude_fill: int = 0,
    ) -> None:
        self.client = WebsocketClientPolicy(remote_host, remote_port)
        self.open_loop_horizon = open_loop_horizon
        self.img_h = img_h
        self.img_w = img_w
        valid_occ = {None, "agentview_right", "agentview_left", "eye_in_hand"}
        if occlude_camera not in valid_occ:
            raise ValueError(f"occlude_camera must be one of {valid_occ}, got {occlude_camera!r}")
        self.occlude_camera = occlude_camera
        self.occlude_fill = int(occlude_fill)
        self._reset_state()

    def _reset_state(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self._last_generated_text = ""
        self._last_policy_timing = None

    def reset(self):
        self._reset_state()

    def act(self, obs: dict, instruction: str, reasoning_text: str | None = None) -> dict:
        """
        obs: gymnasium obs dict from RoboCasaGymEnv
        reasoning_text: optional fine-grained caption to inject for this step.
        Consumes one action from the cached chunk. The remote VLA policy is
        contacted only when that chunk is empty or exhausted; ordinary action
        steps never send a new policy request.
        Returns {"action_dict": gym action dict, "viz": np.ndarray (H, W*3, 3)}
        """
        extracted = self._extract_observation(obs)

        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            self.actions_from_chunk_completed = 0
            request_data = {
                "observation/agentview_left": image_tools.resize_with_pad(
                    extracted["left_image"], self.img_h, self.img_w
                ),
                "observation/eye_in_hand": image_tools.resize_with_pad(
                    extracted["wrist_image"], self.img_h, self.img_w
                ),
                "observation/agentview_right": image_tools.resize_with_pad(
                    extracted["right_image"], self.img_h, self.img_w
                ),
                "observation/state": extracted["state"].astype(np.float64),
                "prompt": instruction,
            }
            if reasoning_text:
                request_data["reasoning_text"] = reasoning_text
            result = self.client.infer(request_data)
            actions = np.asarray(result["actions"])
            assert actions.ndim == 2, f"Expected 2-D action chunk, got {actions.shape}"
            assert actions.shape[-1] == 7, f"Expected 7 action dims, got {actions.shape[-1]}"
            self.pred_action_chunk = actions
            self._last_policy_timing = result.get("policy_timing")
            # Capture generated text from LM-trained policy server (if present).
            self._last_generated_text = result.get("generated_text", "")

        action = self.pred_action_chunk[self.actions_from_chunk_completed].copy()
        new_action_chunk = self.actions_from_chunk_completed == 0
        self.actions_from_chunk_completed += 1

        viz = np.concatenate(
            [
                image_tools.resize_with_pad(extracted["right_image"], 224, 224),
                image_tools.resize_with_pad(extracted["wrist_image"], 224, 224),
                image_tools.resize_with_pad(extracted["left_image"], 224, 224),
            ],
            axis=1,
        )
        ret = {"action_dict": _to_gym_action_dict(action), "raw_action": action, "viz": viz}
        ret["new_action_chunk"] = new_action_chunk
        ret["policy_rpc"] = new_action_chunk
        if new_action_chunk and self._last_policy_timing is not None:
            ret["policy_timing"] = self._last_policy_timing
        if self._last_generated_text:
            ret["generated_text"] = self._last_generated_text
        return ret

    # Backward-compatible name for callers outside this evaluation harness.
    # It has the same chunked semantics as act(); it is not a per-step policy RPC.
    def infer(self, obs: dict, instruction: str, reasoning_text: str | None = None) -> dict:
        return self.act(obs, instruction, reasoning_text)

    def _extract_observation(self, obs: dict) -> dict:
        """
        Extract images and 16-dim state from gym wrapper obs dict.

        Gym wrapper obs keys:
          video.robot0_agentview_right / left / robot0_eye_in_hand  (H, W, 3) uint8
          state.base_position          (3,)
          state.base_rotation          (4,) quaternion
          state.end_effector_position_relative  (3,)
          state.end_effector_rotation_relative  (4,) quaternion
          state.gripper_qpos           (2,)

        State layout (matches training data / server expectation):
          [base_pos(3), base_quat(4), eef_pos_rel(3), eef_quat_rel(4), gripper_qpos(2)] = 16
        """
        right_image = obs["video.robot0_agentview_right"]
        left_image  = obs["video.robot0_agentview_left"]
        wrist_image = obs["video.robot0_eye_in_hand"]

        # Camera-occlusion causal test: blank one camera with a constant fill so the
        # policy gets no spatial signal from it (positional encodings still differ).
        if self.occlude_camera is not None:
            _occ = {
                "agentview_right": "right_image",
                "agentview_left": "left_image",
                "eye_in_hand": "wrist_image",
            }[self.occlude_camera]
            _img = {"right_image": right_image, "left_image": left_image, "wrist_image": wrist_image}[_occ]
            _blank = np.full_like(np.asarray(_img), self.occlude_fill)
            if _occ == "right_image":
                right_image = _blank
            elif _occ == "left_image":
                left_image = _blank
            else:
                wrist_image = _blank

        state = np.concatenate([
            obs["state.base_position"],                    # (3,)
            obs["state.base_rotation"],                    # (4,)
            obs["state.end_effector_position_relative"],   # (3,)
            obs["state.end_effector_rotation_relative"],   # (4,)
            obs["state.gripper_qpos"],                     # (2,)
        ], axis=-1).astype(np.float64)                     # (16,)

        return {
            "right_image": right_image,
            "left_image":  left_image,
            "wrist_image": wrist_image,
            "state":       state,
        }


def _subtitle_max_chars(frame_w: int, frame_h: int) -> int:
    """Chars that fit on one subtitle line at the subtitle font size."""
    font_size = max(10, frame_h // 18)
    return int(frame_w / (font_size * 0.6))


def _wrap_subtitle_lines(lines: list[str], max_chars: int, step: int) -> list[str]:
    """Word-wrap subtitle lines. Prepends a `Step N` header."""
    out: list[str] = [f"Step {step}"]
    for line in lines:
        if not line:
            continue
        while len(line) > max_chars:
            brk = line.rfind(" ", 0, max_chars)
            if brk <= 0:
                brk = max_chars  # Hard break on no space.
            out.append(line[:brk])
            line = line[brk:].lstrip()
        if line:
            out.append(line)
    return out


def _add_subtitle(
    frame: np.ndarray,
    lines: list[str],
    step: int,
    fixed_bar_lines: int = 4,
) -> np.ndarray:
    """Add a word-wrapped text subtitle to the bottom of a video frame.

    Uses a fixed bar height of ``fixed_bar_lines`` rows so every frame in an
    episode has identical dimensions (required by video writers). Callers
    should precompute the maximum wrapped line count across all text
    variants that will appear during the episode and pass it here.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame  # Fallback: return frame without subtitle

    h, w, _ = frame.shape
    font_size = max(10, h // 18)
    line_height = font_size + 4
    max_chars = _subtitle_max_chars(w, h)

    wrapped = _wrap_subtitle_lines(lines, max_chars, step)
    # Pad or truncate to match fixed bar height.
    if len(wrapped) > fixed_bar_lines:
        wrapped = wrapped[:fixed_bar_lines]
    bar_height = line_height * fixed_bar_lines + 4

    new_h = h + bar_height
    canvas = np.zeros((new_h, w, 3), dtype=np.uint8)
    canvas[:h] = frame
    canvas[h:] = 40  # dark gray bar

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    y = h + 2
    for i, dline in enumerate(wrapped):
        color = (180, 180, 180) if i == 0 else (255, 255, 255)
        draw.text((4, y), dline, fill=color, font=font)
        y += line_height

    return np.array(img)


def _to_gym_action_dict(action: np.ndarray) -> dict:
    """Convert 7-dim OpenPI action to gym wrapper dict format.

    OpenPI pi05_robocasa output: [eef_pos(3), eef_rot(3), gripper(1)]
    Gym dict keys match PandaOmronKeyConverter.deduce_action_space().

    base_motion and control_mode are zero-padded (robot stays stationary).
    """
    return {
        "action.base_motion":               np.zeros(4, dtype=np.float32),
        "action.control_mode":              np.zeros(1, dtype=np.float32),
        "action.end_effector_position":     action[0:3].astype(np.float32),
        "action.end_effector_rotation":     action[3:6].astype(np.float32),
        "action.gripper_close":             action[6:7].astype(np.float32),
    }


def _numeric_summary(value: np.ndarray) -> dict:
    arr = np.asarray(value)
    arr64 = arr.astype(np.float64, copy=False)
    finite = np.isfinite(arr64)
    out = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite": bool(finite.all()),
        "nan_count": int(np.isnan(arr64).sum()),
        "inf_count": int(np.isinf(arr64).sum()),
    }
    if finite.any():
        values = arr64[finite]
        out.update(
            {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "abs_max": float(np.abs(values).max()),
            }
        )
    return out


def _observation_diagnostics(obs: dict) -> dict:
    state = np.concatenate(
        [
            obs["state.base_position"],
            obs["state.base_rotation"],
            obs["state.end_effector_position_relative"],
            obs["state.end_effector_rotation_relative"],
            obs["state.gripper_qpos"],
        ],
        axis=-1,
    ).astype(np.float64)
    image_keys = [
        "video.robot0_agentview_left",
        "video.robot0_eye_in_hand",
        "video.robot0_agentview_right",
    ]
    return {
        "state": state.tolist(),
        "state_summary": _numeric_summary(state),
        "state_fields": {
            "base_position": np.asarray(obs["state.base_position"]).tolist(),
            "base_rotation": np.asarray(obs["state.base_rotation"]).tolist(),
            "eef_position_relative": np.asarray(obs["state.end_effector_position_relative"]).tolist(),
            "eef_rotation_relative": np.asarray(obs["state.end_effector_rotation_relative"]).tolist(),
            "gripper_qpos": np.asarray(obs["state.gripper_qpos"]).tolist(),
        },
        "images": {key: _numeric_summary(obs[key]) for key in image_keys if key in obs},
    }


def _step_info_diagnostics(info: dict) -> dict:
    out = {
        "keys": sorted(str(k) for k in info.keys()),
        "success": bool(info.get("success", False)),
    }
    if "subtask_successes" in info:
        out["subtask_successes"] = [bool(x) for x in info["subtask_successes"]]
    return out


# ---------------------------------------------------------------------------
# Async online caption generation
# ---------------------------------------------------------------------------

def _build_instructor_request(
    obs: dict,
    instruction: str,
    img_h: int = 256,
    img_w: int = 256,
) -> dict:
    """Build a policy-compatible observation dict from a gym observation.

    Same extraction + resize logic as ``RoboCasaClient.infer()`` so the
    instructor model sees identical inputs to the action model.
    """
    request = {
        "observation/agentview_left": image_tools.resize_with_pad(
            obs["video.robot0_agentview_left"], img_h, img_w,
        ),
        "observation/eye_in_hand": image_tools.resize_with_pad(
            obs["video.robot0_eye_in_hand"], img_h, img_w,
        ),
        "observation/agentview_right": image_tools.resize_with_pad(
            obs["video.robot0_agentview_right"], img_h, img_w,
        ),
        "observation/state": np.concatenate([
            obs["state.base_position"],
            obs["state.base_rotation"],
            obs["state.end_effector_position_relative"],
            obs["state.end_effector_rotation_relative"],
            obs["state.gripper_qpos"],
        ], axis=-1).astype(np.float64),
        "prompt": instruction,
    }
    return request


def _is_valid_caption(text: str) -> bool:
    """Sanity-check generated caption text.

    Rejects degenerate AR outputs: empty text, mostly non-ASCII / non-printable
    characters, or extremely short strings that are unlikely to be useful.
    """
    if not text or len(text.strip()) < 5:
        return False
    printable = sum(1 for c in text if c.isascii() and c.isprintable())
    return printable / len(text) >= 0.8


class QwenInstructor:
    """Client for the Qwen caption generation HTTP server.

    The server (``qwen_instructor_server.py``) runs in an isolated venv with
    the latest transformers.  This client sends 3 camera images (base64
    JPEG) + task description and receives the generated caption.
    """

    def __init__(self, server_url: str = "http://localhost:8001"):
        import urllib.request
        self._url = server_url
        logging.info("Waiting for Qwen caption server at %s ...", server_url)
        for _ in range(120):
            try:
                urllib.request.urlopen(f"{server_url}/healthz", timeout=2)
                logging.info("Qwen caption server is ready.")
                return
            except Exception:
                import time
                time.sleep(5)
        raise RuntimeError(f"Qwen caption server not reachable at {server_url}")

    @staticmethod
    def _encode_image(img_array: np.ndarray) -> str:
        """Encode a uint8 RGB numpy array as base64 JPEG string."""
        import base64
        import io
        from PIL import Image as PILImage
        buf = io.BytesIO()
        PILImage.fromarray(img_array).save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def generate(self, images: list[np.ndarray], task_description: str) -> str:
        """Generate a caption from 3 camera images + task description."""
        import urllib.request
        payload = {
            "task_description": task_description,
            "images": [self._encode_image(img) for img in images],
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self._url, data=data, headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
        return result.get("caption", "")


class AsyncInstructor:
    """Generates captions asynchronously using either a Pi0.5 AR model or a
    fine-tuned Qwen text model.

    A single-threaded ``ThreadPoolExecutor`` runs the generation in the
    background so the main eval loop can continue predicting actions.
    """

    def __init__(
        self,
        *,
        # Pi0.5 AR instructor args (instructor_type="pi0")
        config_name: str | None = None,
        checkpoint_dir: str | None = None,
        device: str = "cuda:1",
        # Qwen instructor args (instructor_type="qwen")
        model_path: str | None = None,
        # Common
        instructor_type: str = "pi0",
    ):
        self._type = instructor_type
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future | None = None
        self._gen_start_time: float | None = None
        self._last_gen_time_sec: float | None = None

        if instructor_type == "qwen":
            server_url = model_path  # For qwen, model_path is the server URL (e.g., http://localhost:8001)
            self._instructor = QwenInstructor(server_url)
        else:
            from openpi.training import config as _config
            from openpi.policies import policy_config as _policy_config
            logging.info("Loading Pi0.5 instructor: config=%s, ckpt=%s, device=%s", config_name, checkpoint_dir, device)
            self._instructor = _policy_config.create_trained_policy(
                _config.get_config(config_name), checkpoint_dir, pytorch_device=device,
            )

    def _generate(self, obs: dict, instruction: str) -> str:
        """Synchronous generation dispatch."""
        if self._type == "qwen":
            images = [
                obs["video.robot0_agentview_left"],
                obs["video.robot0_agentview_right"],
                obs["video.robot0_eye_in_hand"],
            ]
            return self._instructor.generate(images, instruction)
        else:
            request = _build_instructor_request(obs, instruction)
            return self._instructor.generate_text(request)

    def warmup(self, obs: dict, instruction: str) -> None:
        """Run one synchronous generation to warm up CUDA kernels."""
        import time
        t0 = time.monotonic()
        text = self._generate(obs, instruction)
        dt = time.monotonic() - t0
        valid = _is_valid_caption(text)
        preview = text[:80] if text else "(empty)"
        print(f"[instructor-warmup] {dt:.1f}s, valid={valid}, text=\"{preview}\"")

    def start_generation(self, obs: dict, instruction: str) -> None:
        """Kick off async caption generation from the initial gym observation."""
        import time
        self._gen_start_time = time.monotonic()
        self._last_gen_time_sec = None
        self._future = self._executor.submit(self._generate, obs, instruction)

    def get_result(self) -> str | None:
        """Return generated text if ready, else ``None``.  Non-blocking.

        Returns ``None`` if generation is still running or if the generated
        text fails the sanity check (degenerate / garbage output).
        """
        if self._future is not None and self._future.done():
            import time
            text = self._future.result()
            self._future = None
            if self._gen_start_time is not None:
                self._last_gen_time_sec = time.monotonic() - self._gen_start_time
            if _is_valid_caption(text):
                return text
            logging.warning("[async-caption] Rejected invalid caption: %s", repr(text[:60]))
            return None
        return None

    def last_gen_time_sec(self) -> float | None:
        return self._last_gen_time_sec

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Per-episode rollout
# ---------------------------------------------------------------------------

def run_episode(
    env,
    client: RoboCasaClient,
    instruction: str,
    max_steps: int,
    headless: bool,
    init_state: dict | None = None,
    seed: int | None = None,
    fg_caption_fn=None,
    reasoning_text: str | None = None,
    lock_instruction: bool = False,
    log_state: bool = False,
    instructor: AsyncInstructor | None = None,
    initial_caption: str | None = None,
    sync_instructor: bool = False,
    subgoal_fn=None,
    subgoal_plan: list[dict] | None = None,
    subgoal_switch_buffer: int = 20,
    record_video: bool = True,
    save_step_diagnostics: bool = False,
    hierarchy=None,
) -> dict:
    """
    Runs one episode. Returns a result dict with keys:
      success (bool), steps (int), subtask_successes (list[bool]),
      instruction (str), generated_texts (list[str])

    Args:
        init_state: If provided, dict with 'ep_meta' and 'states' for deterministic
            episode initialization from a saved dataset.
        seed: If provided, passed to env.reset(seed=seed) for reproducible initialization.
        fg_caption_fn: Optional callable(step) -> str|None that returns a fine-grained
            GT caption for the given env step.
        reasoning_text: Optional episode-level caption injected as reasoning_text on every
            policy server call. Overridden by fg_caption_fn if both are set.
        lock_instruction: If True, never override instruction from env annotation.
            Used for novel vocabulary eval where the prompt must stay as given.
        log_state: If True, record EE position and gripper state at each step.
        instructor: Optional async instructor. When provided, generates a scene-specific
            caption from the initial observation in a background thread.
        initial_caption: Caption to use before the async instructor finishes (None = empty,
            or a GT template string as fallback).
    """
    import time
    episode_start_time = time.monotonic()
    # Planner owns the phase / instruction, while the original asynchronous
    # Instructor owns captions and is restarted when a new phase begins.
    if hierarchy is not None and (subgoal_plan is not None or fg_caption_fn is not None):
        raise ValueError("Hierarchy mode cannot share phase ownership with legacy modes")
    if init_state is not None:
        # NOTE: eval-dataset mode has rendering issues. Prefer seed-based eval.
        # See evaluation_protocol.md for details.
        raw_env = env.unwrapped
        raw_env.set_ep_meta(init_state["ep_meta"])
        obs, _info = env.reset()
        if not lock_instruction:
            instruction = init_state["ep_meta"].get("lang", instruction)
    else:
        obs, _info = env.reset(seed=seed)
        # Prefer per-episode language from the env (varies with randomized layout/objects)
        if not lock_instruction:
            instruction = obs.get("annotation.human.task_description", instruction) or instruction

    # Resolve {object} placeholder in reasoning_text using the episode's task description.
    # E.g., instruction="Pick the egg from the drawer" → replaces {object} with "egg".
    if reasoning_text and "{object}" in reasoning_text:
        import re
        m = re.search(r'\b(?:pick up|pick|place|move|grab|get)\s+(?:the\s+)?(.+?)\s+(?:from|on|in|to|into|onto)\b', instruction, re.IGNORECASE)
        if m:
            obj_name = m.group(1)
            reasoning_text = reasoning_text.replace("{object}", obj_name)

    # Optional online hierarchy uses observation-only planning, not task-name plans.
    if hierarchy is not None:
        hierarchy.start(obs, instruction)
        if hierarchy.stopped:
            return {"success": False, "steps": 0, "subtask_successes": [], "video": [],
                    "action_log": [], "instruction": instruction, "hierarchy": hierarchy.summary(),
                    "episode_wall_seconds": time.monotonic() - episode_start_time}
        instruction = hierarchy.instruction
        reasoning_text = hierarchy.caption

    # --- Async caption generation ---
    # Start AR model caption generation in background thread immediately after
    # env.reset() so it runs concurrently with the first action chunks.
    caption_switch_step: int | None = None
    caption_gen_time_sec: float | None = None
    generated_caption: str | None = None
    pending_caption: str | None = None
    pending_caption_phase: int | None = None
    if instructor is not None:
        instructor.start_generation(obs, instruction)
        pending_caption_phase = hierarchy.index if hierarchy is not None else None
        if sync_instructor:
            # 2-stage eval: block-wait for caption before first action step.
            print("  [sync-caption] Waiting for caption...")
            while True:
                result_text = instructor.get_result()
                if result_text:
                    active_caption = result_text
                    generated_caption = result_text
                    caption_switch_step = 0
                    pending_caption = None
                    caption_gen_time_sec = instructor.last_gen_time_sec()
                    print(f"  [sync-caption] Ready in {caption_gen_time_sec:.2f}s: \"{result_text[:80]}\"")
                    break
                import time; time.sleep(0.1)
        else:
            # Async: use initial_caption (None or GT fallback) until generation completes.
            active_caption = initial_caption
    else:
        active_caption = reasoning_text

    video: list[np.ndarray] = []

    # Build subtitle text for video overlay — shows exactly what the model receives.
    # Two lines kept constant so frame dimensions don't change mid-episode (mediapy requirement).
    subtitle_lines = [f"Task: {instruction}"]
    if active_caption:
        subtitle_lines.append(f"Reasoning: {active_caption}")
    elif instructor is not None:
        subtitle_lines.append("Reasoning: (generating...)")
    else:
        subtitle_lines.append("")

    success = False
    subtask_successes: list[bool] = []
    action_log: list[list[float]] = []
    generated_texts: list[str] = []
    state_log: list[dict] = []
    step_diagnostics: list[dict] = []
    subgoal_latched: dict[str, bool] = {}
    subgoal_first_step: dict[str, int] = {}
    subgoal_switches: list[dict] = []
    current_subgoal_idx = 0
    pending_switch_step: int | None = None  # step when current phase's trigger first fired (latched)

    raw_env = env.unwrapped if (subgoal_fn is not None or subgoal_plan is not None) else None

    # Subgoal-driven mode: override instruction / reasoning from plan[0].
    if subgoal_plan is not None and len(subgoal_plan) > 0:
        phase = subgoal_plan[0]
        instruction = phase["instruction"]
        active_caption = phase.get("reasoning") or None
        # Rebuild initial subtitle lines to reflect the phase-0 text.
        subtitle_lines = [f"Task: {instruction}"]
        subtitle_lines.append(f"Reasoning: {active_caption}" if active_caption else "")
        print(f"  [subgoal-plan] phase 0: {phase['name']} — instruction=\"{instruction}\"")

    # Precompute fixed subtitle bar height:
    # count wrapped lines for every possible subtitle variant (initial + each
    # subgoal phase) so the bar stays the same height across mid-episode switches.
    _probe_w, _probe_h = 672, 224  # _extract_observation returns 3x 224x224 concat = 672x224
    _max_chars = _subtitle_max_chars(_probe_w, _probe_h)

    def _variant_line_count(task_line: str, reasoning_line: str) -> int:
        return len(_wrap_subtitle_lines([task_line, reasoning_line], _max_chars, 0))

    _line_counts = [_variant_line_count(subtitle_lines[0], subtitle_lines[1])]
    if subgoal_plan is not None:
        for ph in subgoal_plan:
            t = f"Task: {ph['instruction']}"
            r = f"Reasoning: {ph['reasoning']}" if ph.get("reasoning") else ""
            _line_counts.append(_variant_line_count(t, r))
    fixed_bar_lines = max(_line_counts) + 1  # +1 for safety padding

    for step in range(max_steps):
        step_diag = None
        if save_step_diagnostics:
            step_diag = {
                "step": step,
                "instruction": instruction,
                "reasoning_text": active_caption,
                "open_loop_index_before": int(client.actions_from_chunk_completed),
                "pre_observation": _observation_diagnostics(obs),
            }

        if log_state:
            state_log.append({
                "step": step,
                "ee_pos_rel": obs["state.end_effector_position_relative"].tolist(),
                "ee_rot_rel": obs["state.end_effector_rotation_relative"].tolist(),
                "gripper_qpos": obs["state.gripper_qpos"].tolist(),
                "base_pos": obs["state.base_position"].tolist(),
            })

        # Check if async caption generation has completed.
        if instructor is not None and (hierarchy is not None or caption_switch_step is None):
            result_text = instructor.get_result()
            if result_text:
                pending_caption = result_text
                generated_caption = result_text
                if hierarchy is None:
                    caption_switch_step = step
                caption_gen_time_sec = instructor.last_gen_time_sec()
                # Force the next policy request to start a fresh chunk. The
                # caption is applied only at that chunk boundary.
                client.actions_from_chunk_completed = client.open_loop_horizon
                _gt_str = f"{caption_gen_time_sec:.2f}s" if caption_gen_time_sec is not None else "?"
                print(f"  [async-caption] Ready at step {step} ({_gt_str}), pending chunk boundary: \"{result_text[:80]}\"")
                logging.info("[async-caption] Ready at step %d (%s): %s", step, _gt_str, result_text[:100])

        if pending_caption is not None and client.actions_from_chunk_completed >= client.open_loop_horizon:
            active_caption = pending_caption
            pending_caption = None
            r = active_caption if len(active_caption) <= 100 else active_caption[:97] + "..."
            subtitle_lines[1] = f"Generated@{step}: {r}"
            _gt_str = f"{caption_gen_time_sec:.2f}s" if caption_gen_time_sec is not None else "?"
            phase_label = f" phase {pending_caption_phase}" if pending_caption_phase is not None else ""
            print(f"  [async-caption] Injected{phase_label} at chunk boundary step {step} ({_gt_str}): \"{active_caption[:80]}\"")
            logging.info("[async-caption] Injected%s at chunk boundary step %d (%s): %s", phase_label, step, _gt_str, active_caption[:100])

        caption = fg_caption_fn(step) if fg_caption_fn else active_caption
        import time
        _infer_t0 = time.monotonic()
        ret = client.act(obs, instruction, reasoning_text=caption)
        _infer_wall_time_sec = time.monotonic() - _infer_t0
        if record_video:
            video.append(_add_subtitle(ret["viz"], subtitle_lines, step, fixed_bar_lines=fixed_bar_lines))

        # Capture generated text from LM-trained models (if server returns it).
        if "generated_text" in ret and ret["generated_text"]:
            generated_texts.append({"step": step, "text": ret["generated_text"]})

        raw = ret["raw_action"]
        action_log.append(raw.tolist())
        if step_diag is not None:
            step_diag["reasoning_text"] = caption
            step_diag["open_loop_index_after"] = int(client.actions_from_chunk_completed)
            step_diag["raw_action"] = raw.tolist()
            step_diag["raw_action_summary"] = _numeric_summary(raw)
            step_diag["infer_wall_time_sec"] = float(_infer_wall_time_sec)
            step_diag["new_action_chunk"] = bool(ret.get("new_action_chunk", False))
            if "policy_timing" in ret:
                step_diag["policy_timing"] = ret["policy_timing"]
            step_diag["gym_action"] = {
                key: np.asarray(value).tolist()
                for key, value in ret["action_dict"].items()
            }
        print(f"  [step {step:3d}] action: {np.array2string(raw, precision=4, suppress_small=True, separator=', ')}")

        if not headless and _HAS_CV2:
            cv2.imshow("OpenPI \u2013 RoboCasa", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        _env_step_t0 = time.monotonic()
        obs, _reward, done, _truncated, info = env.step(ret["action_dict"])
        _env_step_wall_time_sec = time.monotonic() - _env_step_t0
        if step_diag is not None:
            step_diag["env_step_wall_time_sec"] = float(_env_step_wall_time_sec)
            step_diag["reward"] = float(_reward)
            step_diag["done"] = bool(done)
            step_diag["truncated"] = bool(_truncated)
            step_diag["info"] = _step_info_diagnostics(info)
            step_diag["post_observation"] = _observation_diagnostics(obs)
            step_diagnostics.append(step_diag)

        if info.get("success", False):
            success = True
            done = True

        if "subtask_successes" in info:
            subtask_successes = info["subtask_successes"]

        # Evaluate per-subgoal milestones and latch "ever reached" state.
        if subgoal_fn is not None:
            try:
                sg = subgoal_fn(raw_env)
            except Exception as e:
                logging.warning("[subgoal] eval failed at step %d: %s", step, e)
                sg = {}
            for name, ok in sg.items():
                if ok and not subgoal_latched.get(name, False):
                    subgoal_latched[name] = True
                    subgoal_first_step[name] = step
                elif name not in subgoal_latched:
                    subgoal_latched[name] = False

        # Subgoal-driven instruction switching with buffer.
        # Once the trigger fires we latch it and delay the actual switch by
        # `subgoal_switch_buffer` steps so the robot can finish the current
        # motion (e.g., release the mug) before seeing the new instruction.
        if (
            subgoal_plan is not None
            and current_subgoal_idx < len(subgoal_plan) - 1
        ):
            cur = subgoal_plan[current_subgoal_idx]
            if pending_switch_step is None:
                try:
                    triggered = bool(cur["trigger_fn"](raw_env))
                except Exception as e:
                    logging.warning("[subgoal-plan] trigger failed at step %d: %s", step, e)
                    triggered = False
                if triggered:
                    pending_switch_step = step
                    print(f"  [subgoal-plan] trigger for {cur['name']} at step {step}; switching in {subgoal_switch_buffer} steps")
                    logging.info("[subgoal-plan] trigger %s at step %d; switching +%d", cur["name"], step, subgoal_switch_buffer)

            if pending_switch_step is not None and (step - pending_switch_step) >= subgoal_switch_buffer:
                prev_name = cur["name"]
                current_subgoal_idx += 1
                nxt = subgoal_plan[current_subgoal_idx]
                instruction = nxt["instruction"]
                active_caption = nxt.get("reasoning") or None
                subtitle_lines = [
                    f"Task: {instruction}",
                    f"Reasoning: {active_caption}" if active_caption else "",
                ]
                client.actions_from_chunk_completed = client.open_loop_horizon
                subgoal_switches.append({
                    "step": step,
                    "trigger_step": pending_switch_step,
                    "buffer": subgoal_switch_buffer,
                    "from": prev_name,
                    "to": nxt["name"],
                    "instruction": instruction,
                    "reasoning": active_caption,
                })
                print(f"  [subgoal-plan] @ step {step}: {prev_name} → {nxt['name']} (trigger fired @{pending_switch_step}, +{subgoal_switch_buffer} buffer)")
                logging.info("[subgoal-plan] @ step %d: %s -> %s", step, prev_name, nxt["name"])
                pending_switch_step = None

        if hierarchy is not None and not done and not _truncated:
            # The action just consumed may be the last action in the cached
            # chunk.  Verification is scheduled only at this boundary, while
            # any request already in flight is polled without waiting.
            changed = hierarchy.observe(
                obs,
                step + 1,
                at_chunk_boundary=(client.actions_from_chunk_completed >= client.open_loop_horizon),
            )
            if changed:
                instruction = hierarchy.instruction
                # Keep the previous caption while the new phase caption is
                # generated asynchronously; inject it at the next chunk.
                if instructor is None:
                    active_caption = hierarchy.caption
                if instructor is not None:
                    pending_caption = None
                    pending_caption_phase = hierarchy.index
                    caption_switch_step = None
                    instructor.start_generation(obs, instruction)
                    print(f"  [async-caption] Started phase {hierarchy.index} generation at step {step + 1}")
                    logging.info("[async-caption] Started phase %d generation at step %d", hierarchy.index, step + 1)
                client.actions_from_chunk_completed = client.open_loop_horizon
                subtitle_lines = [f"Task: {instruction}", f"Caption: {active_caption or ''}"]
            # Visual completion never sets success=True. Only env info does that.
            if hierarchy.stopped:
                done = True
        if hierarchy is not None and _truncated:
            done = True

        if done:
            break

    result = {"success": success, "steps": step + 1, "subtask_successes": subtask_successes, "video": video, "action_log": action_log, "instruction": instruction}
    if hierarchy is not None:
        result["hierarchy"] = hierarchy.summary()
        hierarchy.close()
        result["episode_wall_seconds"] = time.monotonic() - episode_start_time
    if generated_texts:
        result["generated_texts"] = generated_texts
    if state_log:
        result["state_log"] = state_log
    if step_diagnostics:
        result["step_diagnostics"] = step_diagnostics
    if generated_caption is not None:
        result["async_caption"] = generated_caption
        result["caption_switch_step"] = caption_switch_step
        result["caption_gen_time_sec"] = caption_gen_time_sec
    if subgoal_fn is not None:
        result["subgoals"] = subgoal_latched
        result["subgoal_first_step"] = subgoal_first_step
    if subgoal_plan is not None:
        result["subgoal_switches"] = subgoal_switches
        result["final_subgoal_idx"] = current_subgoal_idx
        result["subgoal_plan_names"] = [ph["name"] for ph in subgoal_plan]
    return result


def run_episode_dynamic(
    env,
    client: RoboCasaClient,
    task_instructions: list[str],
    instruction_steps: list[int],
    max_steps: int,
    headless: bool,
    record_video: bool = True,
) -> dict:
    """
    Runs one episode with dynamic per-step instructions.

    task_instructions[i] is active from instruction_steps[i] up to (but not
    including) instruction_steps[i+1].  The last instruction stays active
    until the episode ends.

    Returns the same result dict as run_episode, plus:
      active_instructions (list[str]) -- instruction used at each step
    """
    obs, _info = env.reset()
    client.reset()

    video: list[np.ndarray] = []
    active_instructions: list[str] = []

    success = False
    subtask_successes: list[bool] = []
    action_log: list[list[float]] = []
    instr_idx = 0

    for step in range(max_steps):
        # Advance to the next instruction if we've reached its start step
        while (
            instr_idx + 1 < len(instruction_steps)
            and step >= instruction_steps[instr_idx + 1]
        ):
            instr_idx += 1

        current_instruction = task_instructions[instr_idx]
        active_instructions.append(current_instruction)

        ret = client.act(obs, current_instruction)
        if record_video:
            video.append(ret["viz"])

        raw = ret["raw_action"]
        action_log.append(raw.tolist())
        print(f"  [step {step:3d}] action: {np.array2string(raw, precision=4, suppress_small=True, separator=', ')}")

        if not headless and _HAS_CV2:
            cv2.imshow("OpenPI \u2013 RoboCasa", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        obs, _reward, done, _truncated, info = env.step(ret["action_dict"])

        if info.get("success", False):
            success = True
            done = True

        if "subtask_successes" in info:
            subtask_successes = info["subtask_successes"]

        if done:
            break

    return {
        "success": success,
        "steps": step + 1,
        "subtask_successes": subtask_successes,
        "video": video,
        "action_log": action_log,
        "active_instructions": active_instructions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclass
class Args:
    # Filtering
    tasks: list[str] = field(default_factory=list)
    """Explicit list of task names to evaluate. Overrides --split."""
    split: str = "all"
    """Which split to evaluate: 'all', 'atomic', or 'composite'."""

    # Sharding (for parallel workers)
    task_shard: int = 0
    num_shards: int = 1

    # Rollout settings
    episodes: int = 20
    episode_start: int = 0
    """Start episode index (for resuming/extending evals). Episodes [episode_start, episodes) are run."""
    max_steps: int = 400
    open_loop_horizon: int = 8

    # Policy server
    host: str = "localhost"
    port: int = 8000

    # Model
    checkpoint: str = ""
    """Path to the model checkpoint (recorded in args.json for provenance)."""

    # Camera-occlusion causal test (physics-aux attention analysis Stage 2)
    occlude_camera: Optional[str] = None
    """If set, replace one camera with a constant fill before it reaches the policy.
    One of: agentview_right | agentview_left | eye_in_hand. Tests camera-reliance —
    e.g. whether phys_g0p3 depends on agentview_right more than baseline."""
    occlude_fill: int = 0
    """Constant uint8 value used to fill the occluded camera (0=black)."""

    # Output
    results_dir: str = "runs/robocasa"
    save_video: bool = True
    headless: bool = True
    save_step_diagnostics: bool = False
    """Record lightweight per-step observation/action diagnostics in results JSON.
    This stores proprioception vectors, camera image statistics, raw/gym actions,
    rewards, and done/info flags, but not raw image frames."""

    # Dynamic instructions
    dynamic_instructions_file: Optional[str] = None
    """Path to JSON file with per-task dynamic instructions (task_instructions + instruction_steps).
    Tasks present in the file use dynamic instruction switching; others fall back to fixed instruction."""

    # RoboCasa env
    camera_height: int = 256
    camera_width: int = 256

    # Reproducible seeding
    seed: Optional[int] = None
    """Base seed for env.reset(). Episode i of task j uses seed = base_seed + j * 10000 + i.
    When set, ensures identical initial conditions across different model checkpoints."""

    # GT reasoning injection
    gt_reasoning_file: Optional[str] = None
    """Path to JSON file mapping task_name → reasoning text. When set, appends
    'Reasoning: <text>' to the instruction for each task. Use with MAX_REASONING_TOKENS=0
    on the server to disable two-step AR generation."""

    # Deterministic eval from saved dataset
    eval_dataset: Optional[str] = None
    """Path to target dataset (e.g., .../target/atomic) with saved episode init states.
    When set, episodes are initialized deterministically from saved states, making
    results reproducible across different model checkpoints."""

    # Per-episode caption injection from target split captions
    caption_dir: Optional[str] = None
    """Path to target dataset root (e.g., .../target/atomic). Loads episode-level captions
    from <caption_dir>/<task>/<date>/lerobot/meta/captions_Qwen3-VL-30B-A3B-Instruct.jsonl
    and injects as reasoning_text per episode. Requires --eval-dataset to align episode indices."""
    caption_key: str = "reasoning-short"
    """Which caption key to inject (e.g., 'reasoning-short', 'current_action', 'environment-simple')."""

    # VLM-generated per-episode captions (replaces GT captions with scene-specific ones)
    vlm_captions_file: Optional[str] = None
    """Path to per-episode VLM captions JSON (from generate_robocasa_eval_captions.py).
    Format: {task_name: {"seed_ep": caption_text}}. Overrides gt_reasoning_file when both set."""

    # Fine-grained GT caption injection (per-step reasoning_text)
    fg_captions_path: Optional[str] = None
    """Path to episodes.jsonl with per-step captions. When set, injects the GT caption
    for the current env step as reasoning_text on each policy server call."""
    fg_caption_key: str = "current_action"
    """Which caption key to use (e.g., 'current_action', 'reasoning-short', 'all')."""
    fg_stride: int = 4
    """Caption step stride (must match the stride in the training data)."""

    # Novel vocabulary evaluation
    prompt_override: Optional[str] = None
    """Path to novel_vocabulary_tasks.json. When set, evaluates novel tasks defined in
    the file instead of the standard task list. Each novel task specifies its own
    env_name and novel_prompt."""
    log_state: bool = False
    """Record EE position and gripper state at each step. Enables post-hoc metric
    computation for Category B novel tasks (ee_proximity, gripper_closure)."""

    # Async online caption generation
    instructor_type: str = ""
    """Instructor backend: 'pi0' (Pi0.5 AR model) or 'qwen' (fine-tuned Qwen3.5-2B).
    Leave empty to disable async captioning."""
    instructor_config: Optional[str] = None
    """[pi0 only] AR model config name (e.g., 'pi05_robocasa_lm_ar_fg')."""
    instructor_checkpoint: Optional[str] = None
    """[pi0 only] AR model checkpoint path (e.g., 'checkpoints/.../30000')."""
    instructor_model_path: Optional[str] = None
    """[qwen only] Path to fine-tuned Qwen model (merged weights or LoRA adapter dir)."""
    instructor_device: str = "cuda:1"
    """GPU device for the instructor model (default: cuda:1, separate from action server on cuda:0)."""
    initial_caption: str = "none"
    """Caption to use before async generation completes: 'none' (empty reasoning) or
    'gt' (use GT per-task template from --gt-reasoning-file as fallback)."""
    sync_instructor: bool = False
    """If True, wait for instructor to finish before starting the episode (2-stage eval).
    This measures the upper bound of instructor performance without async latency."""

    instruction_override: Optional[str] = None
    """Override the env-provided instruction for all tasks/episodes. Useful for
    composite task eval where you want to test a custom concatenated instruction."""

    subgoal_driven: bool = False
    """Enable per-subgoal instruction/reasoning switching. Requires the task to have
    a TASK_SUBGOAL_PLAN entry in composite_subgoals.py. At each step the current
    subgoal's trigger_fn is evaluated; when it fires, the instruction/reasoning
    swaps to the next phase's text."""

    subgoal_swap_mode: str = "prompt"
    """What to swap per subgoal phase: 'prompt' | 'reasoning' | 'both'."""

    subgoal_switch_buffer: int = 20
    """Steps to wait after a subgoal trigger fires before actually switching
    the instruction. Lets the robot complete the current motion (e.g., release
    the mug) before seeing the next phase's instruction."""

    hierarchy_url: Optional[str] = None
    """Opt-in observation-only planner/verifier URL, e.g. http://localhost:8002."""
    hierarchy_caption_mode: str = "planner"
    """none | planner | instructor. Instructor is called afresh for EVERY phase."""
    hierarchy_caption_url: Optional[str] = None
    """Existing Qwen Instructor server URL, required for caption_mode=instructor."""
    hierarchy_skills_file: Optional[str] = None
    """Optional JSON list of supported skill names; shared across tasks, not a phase plan."""
    hierarchy_check_every: int = 16
    hierarchy_max_phases: int = 5
    hierarchy_confirmations: int = 2
    hierarchy_confidence: float = 0.85
    hierarchy_phase_timeout: int = 240
    hierarchy_max_replans: int = 2


def build_env(task_name: str, args: Args):
    """Create a RoboCasa gymnasium environment for the given task.

    Uses gym.make() -> RoboCasaGymEnv -> create_env(PandaOmron, HybridMobileBase).
    split="target" selects the held-out evaluation layouts/styles.
    """
    import gymnasium as gym
    import robocasa  # registers all envs  # noqa: F401

    env = gym.make(
        f"robocasa/{task_name}",
        split="target",
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
    )
    return env


def load_eval_episodes(eval_dataset: str, task_name: str, num_episodes: int) -> list[dict]:
    """Load deterministic episode init states from saved target dataset.

    Returns list of dicts with 'ep_meta' and 'states' keys.
    """
    import gzip

    task_dir = Path(eval_dataset) / task_name
    if not task_dir.exists():
        raise FileNotFoundError(f"No eval dataset for task {task_name} at {task_dir}")

    # Find the lerobot/extras dir (under a date subdir).
    extras_dir = None
    for child in task_dir.iterdir():
        candidate = child / "lerobot" / "extras"
        if candidate.is_dir():
            extras_dir = candidate
            break
    if extras_dir is None:
        raise FileNotFoundError(f"No extras dir found under {task_dir}")

    # List available episodes sorted by index (filter to episode_* dirs only).
    ep_dirs = sorted([d for d in extras_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")], key=lambda p: p.name)
    if len(ep_dirs) < num_episodes:
        print(f"  WARNING: only {len(ep_dirs)} saved episodes for {task_name}, requested {num_episodes}")

    init_states = []
    for ep_dir in ep_dirs[:num_episodes]:
        meta_path = ep_dir / "ep_meta.json"
        states_path = ep_dir / "states.npz"
        if not meta_path.exists() or not states_path.exists():
            continue
        with open(meta_path) as f:
            ep_meta = json.load(f)
        states = np.load(str(states_path))["states"]
        init_states.append({"ep_meta": ep_meta, "states": states, "episode_dir": str(ep_dir)})

    return init_states


def load_episode_captions(caption_dir: str, task_name: str) -> dict[int, dict[str, str]]:
    """Load episode-level captions for a task from the target split.

    Returns {episode_index: {caption_key: text}}.
    """
    import glob as _glob
    pattern = str(Path(caption_dir) / task_name / "*" / "lerobot" / "meta" / "captions_Qwen3-VL-30B-A3B-Instruct.jsonl")
    files = _glob.glob(pattern)
    if not files:
        return {}
    captions = {}
    with open(files[0]) as f:
        for line in f:
            d = json.loads(line)
            captions[d["episode_index"]] = d["annotations"]
    return captions


def select_tasks(args: Args) -> list[str]:
    if args.tasks:
        # Handle both ["A", "B"] and ["A,B,C"] (comma-separated single string from shell).
        task_pool = []
        for t in args.tasks:
            task_pool.extend(t.split(","))
    else:
        task_pool = {"atomic": ATOMIC_TASK_NAMES, "composite": COMPOSITE_TASK_NAMES}.get(args.split, ALL_TASK_NAMES)
    return [t for i, t in enumerate(task_pool) if i % args.num_shards == args.task_shard]


def load_novel_tasks(path: str, num_shards: int = 1, task_shard: int = 0) -> list[dict]:
    """Load novel vocabulary tasks from JSON file with sharding support."""
    data = json.loads(Path(path).read_text())
    tasks = data["tasks"]
    return [t for i, t in enumerate(tasks) if i % num_shards == task_shard]


def main(args: Args):
    hierarchy_skills = []
    hierarchy_config = None
    if args.hierarchy_url:
        from hierarchical_controller import HierarchicalController, HierarchyConfig, HTTPBackend
        if any((args.subgoal_driven, args.dynamic_instructions_file,
                args.gt_reasoning_file, args.caption_dir, args.vlm_captions_file,
                args.fg_captions_path)):
            raise ValueError("--hierarchy-url requires legacy subgoal/caption modes to be disabled")
        if args.instructor_type and args.hierarchy_caption_mode == "instructor":
            raise ValueError("Use either --hierarchy-caption-mode instructor or the legacy AsyncInstructor, not both")
        if args.hierarchy_caption_mode == "instructor" and not args.hierarchy_caption_url:
            raise ValueError("--hierarchy-caption-mode instructor needs --hierarchy-caption-url")
        if args.hierarchy_skills_file:
            hierarchy_skills = json.loads(Path(args.hierarchy_skills_file).read_text())
            if not isinstance(hierarchy_skills, list) or not hierarchy_skills or not all(
                isinstance(skill, str) and skill.strip() for skill in hierarchy_skills
            ):
                raise ValueError("Skills file must be a nonempty JSON list of skill names")
        hierarchy_config = HierarchyConfig(check_every=args.hierarchy_check_every,
            max_phases=args.hierarchy_max_phases,
            confirmations=args.hierarchy_confirmations, confidence_threshold=args.hierarchy_confidence,
            phase_timeout=args.hierarchy_phase_timeout, max_replans=args.hierarchy_max_replans,
            caption_mode=args.hierarchy_caption_mode)
    # --- Novel vocabulary override mode ---
    novel_tasks: list[dict] | None = None
    if args.prompt_override:
        novel_tasks = load_novel_tasks(args.prompt_override, args.num_shards, args.task_shard)
        task_names = [f"novel_{t['id']:02d}_{t['env_name']}" for t in novel_tasks]
        print(f"Novel vocab eval: {len(novel_tasks)} tasks (shard {args.task_shard}/{args.num_shards})")
        for nt in novel_tasks:
            print(f"  #{nt['id']:2d} [{nt['category']}] {nt['env_name']}: \"{nt['novel_prompt']}\"")
    else:
        task_names = select_tasks(args)
        print(f"Evaluating {len(task_names)} tasks (shard {args.task_shard}/{args.num_shards}): {task_names}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.results_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(asdict(args), indent=2))

    # Load dynamic instructions if provided
    dynamic_instructions: dict[str, dict] = {}
    if args.dynamic_instructions_file is not None:
        dynamic_instructions = json.loads(Path(args.dynamic_instructions_file).read_text())
        print(f"Loaded dynamic instructions for {len(dynamic_instructions)} tasks from {args.dynamic_instructions_file}")

    # Load GT reasoning texts if provided.
    # Supports two formats:
    #   Old: {"task": "caption text"}
    #   New: {"task": {"reasoning-short": "...", "current_action": "...", ...}}
    # When new format is detected, uses --caption-key to select which caption(s).
    # Multiple keys can be comma-separated (e.g., "environment-simple,reasoning-short")
    # and will be concatenated with " " separator.
    gt_reasoning: dict[str, str] = {}
    if args.gt_reasoning_file is not None:
        raw_gt = json.loads(Path(args.gt_reasoning_file).read_text())
        first_val = next(iter(raw_gt.values()), "")
        if isinstance(first_val, dict):
            # New format: extract caption by key(s)
            caption_keys = [k.strip() for k in args.caption_key.split(",")]
            for task, caps in raw_gt.items():
                parts = [caps.get(k, "") for k in caption_keys if caps.get(k)]
                gt_reasoning[task] = " ".join(parts)
            key_str = "+".join(caption_keys)
            print(f"Loaded GT captions for {len(gt_reasoning)} tasks, keys=[{key_str}] from {args.gt_reasoning_file}")
        else:
            # Old format: plain string
            gt_reasoning = raw_gt
            print(f"Loaded GT reasoning for {len(gt_reasoning)} tasks from {args.gt_reasoning_file}")

    # Load VLM per-episode captions if requested
    vlm_captions: dict[str, dict[str, str]] = {}
    if args.vlm_captions_file is not None:
        vlm_captions = json.loads(Path(args.vlm_captions_file).read_text())
        total = sum(len(v) for v in vlm_captions.values())
        print(f"Loaded VLM per-episode captions: {len(vlm_captions)} tasks, {total} episodes from {args.vlm_captions_file}")

    # Load fine-grained per-step caption provider if requested
    fg_provider = None
    if args.fg_captions_path is not None:
        from openpi.policies.fg_caption_provider import FineGrainedCaptionProvider
        fg_provider = FineGrainedCaptionProvider(
            args.fg_captions_path,
            caption_key=args.fg_caption_key,
            fg_stride=args.fg_stride,
        )
        print(f"FG captions enabled: key={args.fg_caption_key}, stride={args.fg_stride}")

    # Load async instructor if requested
    instructor = None
    _instructor_warmed_up = False
    if args.instructor_type == "pi0" and args.instructor_config and args.instructor_checkpoint:
        instructor = AsyncInstructor(
            instructor_type="pi0",
            config_name=args.instructor_config,
            checkpoint_dir=args.instructor_checkpoint,
            device=args.instructor_device,
        )
        print(f"Async instructor [pi0]: config={args.instructor_config}, device={args.instructor_device}")
        print(f"  Initial caption mode: {args.initial_caption}")
    elif args.instructor_type == "qwen" and args.instructor_model_path:
        instructor = AsyncInstructor(
            instructor_type="qwen",
            model_path=args.instructor_model_path,
            device=args.instructor_device,
        )
        print(f"Async instructor [qwen]: model={args.instructor_model_path}, device={args.instructor_device}")
        print(f"  Initial caption mode: {args.initial_caption}")

    client = RoboCasaClient(
        remote_host=args.host,
        remote_port=args.port,
        open_loop_horizon=args.open_loop_horizon,
        occlude_camera=args.occlude_camera,
        occlude_fill=args.occlude_fill,
    )
    if args.occlude_camera:
        print(f"[OCCLUSION] camera={args.occlude_camera} fill={args.occlude_fill}")

    all_results: dict[str, dict] = {}

    for task_idx, task_name in enumerate(task_names):
        # Resolve novel task metadata (if in prompt-override mode).
        novel = novel_tasks[task_idx] if novel_tasks else None
        env_name = novel["env_name"] if novel else task_name
        lock_instruction = novel is not None

        print(f"\n{'='*60}")
        if novel:
            print(f"Novel #{novel['id']} [{novel['category']}]: \"{novel['novel_prompt']}\"")
            print(f"  Env: {env_name} (original: \"{novel.get('original_prompt', '')}\")")
        else:
            print(f"Task: {task_name}")
        print(f"{'='*60}")

        try:
            env = build_env(env_name, args)
        except Exception as e:
            print(f"  [SKIP] Could not build env for {env_name}: {e}")
            traceback.print_exc()
            all_results[task_name] = {"error": str(e)}
            continue

        # Warm up the instructor once using the first env's observation.
        # This pays the CUDA kernel compilation cost before real episodes start.
        if instructor is not None and not _instructor_warmed_up:
            _warmup_instruction = novel["novel_prompt"] if novel else ALL_TASKS.get(env_name, env_name)
            _warmup_obs, _ = env.reset()
            instructor.warmup(_warmup_obs, _warmup_instruction)
            _instructor_warmed_up = True

        # Fallback instruction (overridden per-episode from obs unless lock_instruction).
        instruction = novel["novel_prompt"] if novel else ALL_TASKS.get(task_name, task_name)
        if args.instruction_override:
            instruction = args.instruction_override
            lock_instruction = True

        # GT reasoning template (may contain {object} placeholder, resolved per-episode).
        gt_reasoning_template = gt_reasoning.get(task_name, "") or ""
        if gt_reasoning_template:
            print(f"  GT reasoning template ({len(gt_reasoning_template)} chars, key={args.caption_key})")

        task_dir = out_dir / task_name
        if args.save_video:
            task_dir.mkdir(parents=True, exist_ok=True)

        dyn = dynamic_instructions.get(task_name)
        if dyn:
            print(f"  Using dynamic instructions ({len(dyn['task_instructions'])} steps): {dyn['task_instructions']}")

        # Load deterministic episode inits if eval_dataset is provided.
        eval_init_states = None
        if args.eval_dataset:
            try:
                eval_init_states = load_eval_episodes(args.eval_dataset, task_name, args.episodes)
                print(f"  Using {len(eval_init_states)} deterministic episodes from {args.eval_dataset}")
            except FileNotFoundError as e:
                print(f"  WARNING: {e} — falling back to random init")

        # Resolve per-subgoal checker for this composite task (if registered).
        subgoal_fn = get_subgoal_checker(task_name if not novel else env_name)
        if subgoal_fn is not None:
            print(f"  Subgoal tracking enabled for {task_name}")

        # Resolve subgoal-driven plan (only if user enabled the mode).
        task_subgoal_plan = None
        if args.subgoal_driven:
            _plan = get_subgoal_plan(task_name if not novel else env_name)
            if _plan is None:
                print(f"  WARNING: --subgoal-driven set but no plan registered for {task_name}; skipping")
            else:
                print(f"  Subgoal-driven mode: {len(_plan['subgoals'])} phases, swap_mode={args.subgoal_swap_mode}")
                task_subgoal_plan = _plan  # resolved per-episode below (needs original reasoning)

        # Build per-step caption function for this task (if fg captions enabled).
        fg_caption_fn = None
        fg_caption_episode = None
        if fg_provider is not None:
            # Try gym task name first, then instruction string
            fg_caption_episode = fg_provider.find_episode_for_task(task_name)
            if fg_caption_episode is None:
                fg_caption_episode = fg_provider.find_episode_for_task(instruction)
            if fg_caption_episode is not None:
                fg_caption_fn = lambda step, _ep=fg_caption_episode: fg_provider.get_caption(_ep, step)
                print(f"  FG captions: using episode {fg_caption_episode}")
            else:
                print(f"  FG captions: no matching episode for '{task_name}'")

        # Load per-episode captions if caption_dir is provided.
        ep_captions: dict[int, dict[str, str]] = {}
        if args.caption_dir:
            ep_captions = load_episode_captions(args.caption_dir, task_name)
            if ep_captions:
                print(f"  Loaded {len(ep_captions)} episode captions (key={args.caption_key})")
            else:
                print(f"  WARNING: No captions found for {task_name} in {args.caption_dir}")

        num_episodes = len(eval_init_states) if eval_init_states else args.episodes
        ep_start = args.episode_start
        episode_results = []
        successes = 0

        if ep_start > 0:
            print(f"  Skipping episodes 0-{ep_start - 1} (--episode-start={ep_start})")

        for ep in tqdm(range(ep_start, num_episodes), desc=task_name):
            init_state = eval_init_states[ep] if eval_init_states else None
            # Compute per-episode seed: unique per (task, episode) but reproducible.
            # Use canonical alphabetical index so seeds are independent of --tasks subset.
            _CANONICAL_TASKS = sorted([
                "CloseBlenderLid", "CloseFridge", "CloseToasterOvenDoor", "CoffeeSetupMug",
                "OpenCabinet", "OpenDrawer", "OpenStandMixerHead",
                "PickPlaceCounterToCabinet", "PickPlaceCounterToStove",
                "PickPlaceDrawerToCounter", "PickPlaceSinkToCounter", "PickPlaceToasterToCounter",
                "SlideDishwasherRack", "TurnOffStove", "TurnOnElectricKettle",
                "TurnOnMicrowave", "TurnOnSinkFaucet",
            ])
            # For novel tasks, use env_name for seed (same envs as standard eval).
            _seed_key = env_name if novel else task_name
            _canon_idx = _CANONICAL_TASKS.index(_seed_key) if _seed_key in _CANONICAL_TASKS else task_names.index(task_name)
            ep_seed = (args.seed + _canon_idx * 10000 + ep) if args.seed is not None else None

            # Per-episode caption priority: vlm_captions > caption_dir > gt_reasoning_file
            ep_reasoning = None
            if vlm_captions and task_name in vlm_captions:
                vlm_key = f"{args.seed}_{ep}"
                ep_reasoning = vlm_captions[task_name].get(vlm_key, "")
            if not ep_reasoning and ep_captions and ep in ep_captions:
                ep_reasoning = ep_captions[ep].get(args.caption_key, "")
            if not ep_reasoning and gt_reasoning_template:
                ep_reasoning = gt_reasoning_template

            if dyn:
                result = run_episode_dynamic(
                    env=env,
                    client=client,
                    task_instructions=dyn["task_instructions"],
                    instruction_steps=dyn["instruction_steps"],
                    max_steps=args.max_steps,
                    headless=args.headless,
                    record_video=args.save_video,
                )
            else:
                # Determine initial caption for async captioning mode.
                ep_initial_caption = None
                if instructor is not None and args.initial_caption == "gt":
                    ep_initial_caption = ep_reasoning  # GT template as fallback

                # Resolve the subgoal-driven runtime plan for this episode.
                ep_subgoal_plan = None
                if task_subgoal_plan is not None:
                    ep_subgoal_plan = resolve_subgoal_runtime(
                        task_subgoal_plan,
                        caption_key=args.caption_key,
                        original_instruction=instruction,
                        original_reasoning=ep_reasoning,
                        swap_mode=args.subgoal_swap_mode,
                    )

                result = run_episode(
                    env=env,
                    client=client,
                    instruction=instruction,
                    max_steps=args.max_steps,
                    headless=args.headless,
                    init_state=init_state,
                    seed=ep_seed,
                    fg_caption_fn=fg_caption_fn,
                    reasoning_text=ep_reasoning if instructor is None else None,
                    lock_instruction=lock_instruction,
                    log_state=args.log_state,
                    instructor=instructor,
                    initial_caption=ep_initial_caption,
                    sync_instructor=args.sync_instructor,
                    subgoal_fn=subgoal_fn,
                    subgoal_plan=ep_subgoal_plan,
                    subgoal_switch_buffer=args.subgoal_switch_buffer,
                    record_video=args.save_video,
                    save_step_diagnostics=args.save_step_diagnostics,
                    hierarchy=HierarchicalController(
                        HTTPBackend(args.hierarchy_url, args.hierarchy_caption_url),
                        hierarchy_config, skills=hierarchy_skills,
                    ) if args.hierarchy_url else None,
                )
            if result["success"]:
                successes += 1

            if args.save_video and result["video"]:
                mediapy.write_video(
                    task_dir / f"ep{ep:03d}_{'success' if result['success'] else 'fail'}.mp4",
                    result["video"],
                    fps=15,
                )
            episode_results.append({k: v for k, v in result.items() if k != "video"})

        actual_episodes = num_episodes - ep_start
        sr = successes / actual_episodes if actual_episodes > 0 else 0
        task_result = {
            "success_rate": sr,
            "successes": successes,
            "episodes": actual_episodes,
            "episode_start": ep_start,
            "episode_end": num_episodes,
            "episode_results": episode_results,
            "instruction": instruction,
            "split": "atomic" if env_name in ATOMIC_TASKS else ("novel" if novel else "composite"),
        }
        if novel:
            task_result["novel_task"] = {
                "id": novel["id"],
                "category": novel["category"],
                "novel_prompt": novel["novel_prompt"],
                "env_name": novel["env_name"],
                "original_prompt": novel.get("original_prompt", ""),
                "new_content_words": novel.get("new_content_words", []),
                "metric": novel.get("metric", ""),
            }
        if dyn:
            task_result["task_instructions"] = dyn["task_instructions"]
            task_result["instruction_steps"] = dyn["instruction_steps"]

        # Aggregate per-subgoal SR across episodes (only for tasks with a checker).
        if subgoal_fn is not None and episode_results:
            all_subgoal_keys: list[str] = []
            for er in episode_results:
                for k in (er.get("subgoals") or {}).keys():
                    if k not in all_subgoal_keys:
                        all_subgoal_keys.append(k)
            subgoal_sr: dict[str, dict] = {}
            for k in all_subgoal_keys:
                hits = sum(1 for er in episode_results if (er.get("subgoals") or {}).get(k))
                subgoal_sr[k] = {
                    "success_rate": hits / len(episode_results),
                    "successes": hits,
                    "episodes": len(episode_results),
                }
            task_result["subgoal_success_rate"] = subgoal_sr

        all_results[task_name] = task_result
        print(f"  Success rate: {successes}/{actual_episodes} = {sr:.1%}")
        if subgoal_fn is not None and "subgoal_success_rate" in task_result:
            for name, info in task_result["subgoal_success_rate"].items():
                print(f"    subgoal[{name}]: {info['successes']}/{info['episodes']} = {info['success_rate']:.0%}")

        # Incremental save after each task so a later MuJoCo SIGABRT (core dump,
        # uncatchable in Python) does not discard already-completed tasks.
        _incremental_path = out_dir / f"results_shard{args.task_shard}.json"
        _incremental_path.write_text(json.dumps(all_results, indent=2))

        env.close()

    if instructor is not None:
        instructor.shutdown()

    # Save results for this shard
    results_path = out_dir / f"results_shard{args.task_shard}.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {results_path}")

    # Print summary
    if novel_tasks:
        cat_a = [v for v in all_results.values() if v.get("novel_task", {}).get("category") == "A"]
        cat_b = [v for v in all_results.values() if v.get("novel_task", {}).get("category") == "B"]
        if cat_a:
            print(f"Category A mean SR: {np.mean([v['success_rate'] for v in cat_a]):.1%} over {len(cat_a)} tasks")
        if cat_b:
            print(f"Category B mean SR: {np.mean([v['success_rate'] for v in cat_b]):.1%} over {len(cat_b)} tasks")
        all_srs = [v["success_rate"] for v in all_results.values() if "error" not in v]
        if all_srs:
            print(f"Overall   mean SR: {np.mean(all_srs):.1%} over {len(all_srs)} tasks")
    else:
        atomic_srs    = [v["success_rate"] for k, v in all_results.items() if v.get("split") == "atomic"]
        composite_srs = [v["success_rate"] for k, v in all_results.items() if v.get("split") == "composite"]
        if atomic_srs:
            print(f"Atomic    mean SR: {np.mean(atomic_srs):.1%} over {len(atomic_srs)} tasks")
        if composite_srs:
            print(f"Composite mean SR: {np.mean(composite_srs):.1%} over {len(composite_srs)} tasks")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
