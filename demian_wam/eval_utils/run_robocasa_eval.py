"""
Evaluate Cosmos WAM on RoboCasa atomic and composite tasks.

Adapted from OpenPI's eval runner. Uses the same WebSocket protocol.

Usage
-----
1. Start the WAM policy server:

       python socket_test_cosmos_wam.py --port 6000 --checkpoint-path <path>

2. Run the evaluator:

       python eval_utils/run_robocasa_eval.py --port 6000
       python eval_utils/run_robocasa_eval.py --port 6000 --tasks OpenDrawer
       python eval_utils/run_robocasa_eval.py --port 6000 --split atomic

   For sharded parallel evaluation:

       python eval_utils/run_robocasa_eval.py --task-shard 0 --num-shards 8
       ...

3. Merge sharded results:

       python eval_utils/merge_results.py --results-dir runs/robocasa/<date>/

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

import json
import os
import traceback
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
from eval_utils.robocasa_task_list import (
    ATOMIC_TASKS,
    COMPOSITE_TASKS,
    ALL_TASKS,
    ATOMIC_TASK_NAMES,
    COMPOSITE_TASK_NAMES,
    ALL_TASK_NAMES,
)


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
    ) -> None:
        self.client = WebsocketClientPolicy(remote_host, remote_port)
        self.open_loop_horizon = open_loop_horizon
        self.img_h = img_h
        self.img_w = img_w
        self._reset_state()

    def _reset_state(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self._last_generated_text = ""

    def reset(self):
        self._reset_state()

    def infer(self, obs: dict, instruction: str, reasoning_text: str | None = None) -> dict:
        """
        obs: gymnasium obs dict from RoboCasaGymEnv
        reasoning_text: optional fine-grained caption to inject for this step.
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
            # Capture generated text from LM-trained policy server (if present).
            self._last_generated_text = result.get("generated_text", "")

        action = self.pred_action_chunk[self.actions_from_chunk_completed].copy()
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
        if self._last_generated_text:
            ret["generated_text"] = self._last_generated_text
        return ret

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


def _add_subtitle(frame: np.ndarray, lines: list[str], step: int) -> np.ndarray:
    """Add text subtitle to the bottom of a video frame."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame  # Fallback: return frame without subtitle

    h, w, _ = frame.shape
    font_size = max(10, h // 18)
    line_height = font_size + 4
    bar_height = line_height * (len(lines) + 1) + 4  # +1 for step counter

    # Create frame with subtitle bar
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
    draw.text((4, y), f"Step {step}", fill=(180, 180, 180), font=font)
    y += line_height
    for line in lines:
        # Truncate to fit width (~1.6 chars per pixel at this font size)
        max_chars = int(w / (font_size * 0.6))
        display = line[:max_chars]
        draw.text((4, y), display, fill=(255, 255, 255), font=font)
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


# ---------------------------------------------------------------------------
# Loose success check — task-specific "core goal achieved" predicate that DROPS
# the gripper_obj_far / gripper_fxtr_far retract requirement from the env's
# strict _check_success. Used to track `place_anytime`: did the policy ever
# achieve the manipulation goal during the episode, regardless of arm retract.
# Many policies place an object correctly but don't retract → strict success
# returns False even though the visual outcome was a success.
#
# Only PickPlaceCounterToStove, CloseBlenderLid, and CoffeeSetupMug differ
# strict-vs-loose. PickPlaceCounterToStove: strict uses th=0.07m + check_contact,
# loose uses XY < 0.12m (no contact). CloseBlenderLid: strict uses 3D dist < 0.04m
# + upright(th=7°), loose drops upright + widens dist to 0.10m. CoffeeSetupMug:
# strict adds gripper_obj_far on top of the receptacle-placement check, loose
# returns just the placement.
# OpenDrawer and TurnOnSinkFaucet already use only the core goal — loose
# == strict for those.
# ---------------------------------------------------------------------------

def _check_loose_success(env) -> bool:
    """Return True if the task's core manipulation goal is met THIS step,
    ignoring the strict env's gripper-retract requirement. Best-effort:
    swallow any exception (returns False) so eval continues unaffected if
    the raw env structure differs from the recognized task classes."""
    try:
        raw = env.unwrapped
        name = type(raw).__name__
        if name == "PickPlaceCounterToStove":
            # Loose check: XY distance from carrot to pan center < 12 cm, no
            # contact-pair requirement. The strict env uses th=0.07 + check_contact,
            # which fails when the carrot lands near the pan rim or just next to
            # the pan (visually "on the pan/stove" but ~9-10 cm from the pan's
            # body CoM). 12 cm covers a typical 20 cm-wide pan's full footprint
            # plus a small slop margin.
            obj_pos = raw.sim.data.body_xpos[raw.obj_body_id["obj"]]
            pan_pos = raw.sim.data.body_xpos[raw.obj_body_id["container"]]
            return bool(np.linalg.norm(np.asarray(obj_pos[:2]) - np.asarray(pan_pos[:2])) < 0.12)
        if name == "CloseBlenderLid":
            # Strict env: dist(curr_lid, closed_pos) < 0.04m AND upright (th=7°).
            # Loose drops the upright check and widens distance to 0.10m, so a
            # lid that lands approximately on top of the blender but is tilted
            # (>7°) or sits ~5-9cm off-target still counts. Same spirit as the
            # PickPlaceCounterToStove loose check (0.07m+contact → 0.12m XY-only).
            curr = raw.blender.get_curr_lid_pos(raw)
            if curr is None:
                return False
            closed = raw.blender.get_lid_closed_pos(raw)
            return bool(np.linalg.norm(np.asarray(curr) - np.asarray(closed)) < 0.10)
        if name == "CoffeeSetupMug":
            return bool(raw.coffee_machine.check_receptacle_placement_for_pouring(raw, "obj"))
        if name == "OpenDrawer":
            door_state = raw.drawer.get_door_state(env=raw)
            return all(p > 0.95 for p in door_state.values())
        if name == "TurnOnSinkFaucet":
            return bool(raw.sink.get_handle_state(env=raw)["water_on"])
    except Exception:
        pass
    return False


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
    caption_suffix: str | None = None,
    caption_format: str = "reasoning",
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
    """
    if init_state is not None:
        # Deterministic reset: set ep_meta then reset.
        # DO NOT use sim.set_state_from_flattened() — causes gray backgrounds
        # in rendered observations (see evaluation_protocol.md Rendering Note).
        raw_env = env.unwrapped
        raw_env.set_ep_meta(init_state["ep_meta"])
        obs, _info = env.reset()
        instruction = init_state["ep_meta"].get("lang", instruction)
    else:
        obs, _info = env.reset(seed=seed)
        # Prefer per-episode language from the env (varies with randomized layout/objects).
        # Reapply caption suffix on top of the env-provided task description so the
        # policy server receives the same string format the model was trained on.
        env_instruction = (obs.get("annotation.human.task_description", instruction) or instruction).strip()
        if caption_suffix:
            if caption_format == "concat":
                # training format: "<task_desc>. <suffix>"
                base = env_instruction.rstrip(".")
                instruction = f"{base}. {caption_suffix}"
            else:
                # training format: "Task: <task_desc> Reasoning: <suffix>"
                instruction = f"Task: {env_instruction} Reasoning: {caption_suffix}"
        else:
            instruction = env_instruction
    video: list[np.ndarray] = []

    # Build subtitle text for video overlay.
    subtitle_lines = [f"Task: {instruction}"]

    success = False
    place_anytime = False
    place_first_step: int | None = None
    subtask_successes: list[bool] = []
    action_log: list[list[float]] = []
    generated_texts: list[str] = []

    for step in range(max_steps):
        caption = fg_caption_fn(step) if fg_caption_fn else None
        ret = client.infer(obs, instruction, reasoning_text=caption)
        video.append(_add_subtitle(ret["viz"], subtitle_lines, step))

        # Capture generated text from LM-trained models (if server returns it).
        if "generated_text" in ret and ret["generated_text"]:
            generated_texts.append({"step": step, "text": ret["generated_text"]})

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

        # Loose success: task-core goal achieved at any step regardless of
        # gripper-retract. Sticky once True. See _check_loose_success above.
        if not place_anytime and _check_loose_success(env):
            place_anytime = True
            place_first_step = step

        if "subtask_successes" in info:
            subtask_successes = info["subtask_successes"]

        if done:
            break

    result = {
        "success": success,
        "place_anytime": place_anytime,
        "place_first_step": place_first_step,
        "steps": step + 1,
        "subtask_successes": subtask_successes,
        "video": video,
        "action_log": action_log,
        "instruction": instruction,
    }
    if generated_texts:
        result["generated_texts"] = generated_texts
    return result


def run_episode_dynamic(
    env,
    client: RoboCasaClient,
    task_instructions: list[str],
    instruction_steps: list[int],
    max_steps: int,
    headless: bool,
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

        ret = client.infer(obs, current_instruction)
        video.append(_add_subtitle(ret["viz"], [f"Task: {current_instruction}"], step))

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
    """Run episodes [episodes_start, episodes); default is 0..episodes-1."""
    episodes_start: int = 0
    """First episode index to run; lets you extend a 5-ep eval to 20 eps by setting
    episodes=20 and episodes_start=5. Existing per-task episode_results (0..start-1)
    are reused from the cell's prior results_shard{task_shard}.json if it exists."""
    max_steps: int = 400
    open_loop_horizon: int = 12  # WAM: 3 latent frames × 4 temporal compression

    # Policy server
    host: str = "localhost"
    port: int = 6000  # WAM default port

    # Model
    checkpoint: str = ""
    """Path to the model checkpoint (recorded in args.json for provenance)."""

    # Output
    results_dir: str = "runs/robocasa"
    run_name: Optional[str] = None
    """Optional stable subdirectory under results_dir. Useful for sharded jobs."""
    save_video: bool = True
    headless: bool = True

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
    on the server to disable two-step AR generation.

    The file may be either:
      - flat: {task_name: "caption text"}
      - nested: {task_name: {caption_key: "caption text", ...}}
    For nested files, pass --caption-key to select which key to use."""
    caption_key: Optional[str] = None
    """For nested gt_reasoning_file (e.g., gt_captions.json with multiple caption types
    per task), select which caption key to inject (e.g., 'current_action',
    'reasoning-short', 'environment-simple', 'agent_arm_start_state')."""
    caption_keys_concat: Optional[str] = None
    """Comma-separated list of caption keys whose values are concatenated to the
    task description with '. ' as separator. Bypasses the 'Reasoning:' prefix.
    Used when the training-time caption template is something like
    '<task_description>. <current_action>. <environment-simple>' (Egoverse-match)
    rather than '<task> Reasoning: <text>'.
    Mutually exclusive with --caption-key."""

    # Per-episode VLM-generated captions (overrides gt_reasoning per (task, seed_episode))
    vlm_captions_file: Optional[str] = None
    """Path to per-episode VLM captions JSON (e.g., vlm_captions_reward.json).
    Format: {task_name: {f"{seed}_{episode_index}": caption_text, ...}}.
    Looked up by f'{seed}_{episode}' inside each episode; falls back to
    gt_reasoning_file value when no VLM caption is available."""

    # Deterministic eval from saved dataset
    eval_dataset: Optional[str] = None
    """Path to target dataset (e.g., .../target/atomic) with saved episode init states.
    When set, episodes are initialized deterministically from saved states, making
    results reproducible across different model checkpoints."""

    # Fine-grained GT caption injection (per-step reasoning_text)
    fg_captions_path: Optional[str] = None
    """Path to episodes.jsonl with per-step captions. When set, injects the GT caption
    for the current env step as reasoning_text on each policy server call."""
    fg_caption_key: str = "current_action"
    """Which caption key to use (e.g., 'current_action', 'reasoning-short', 'all')."""
    fg_stride: int = 4
    """Caption step stride (must match the stride in the training data)."""


def build_env(task_name: str, args: Args):
    """Create a RoboCasa gymnasium environment for the given task.

    Uses gym.make() -> RoboCasaGymEnv -> create_env(PandaOmron, HybridMobileBase).
    split="target" selects the held-out evaluation layouts/styles.
    """
    import gymnasium as gym
    import numpy as _np
    import robosuite  # noqa: F401

    # The RoboCasa checkout asserts numpy==2.2.5 at import time, while the
    # deployed PyTorch build needs NumPy 1.x for torch.from_numpy. The assertion
    # is version-gating only; keep the runtime NumPy intact and spoof that check.
    _numpy_version = _np.__version__
    _np.__version__ = os.environ.get("ROBOCASA_NUMPY_VERSION_OVERRIDE", "2.2.5")
    try:
        import robocasa  # registers all envs  # noqa: F401
    finally:
        _np.__version__ = _numpy_version

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


def select_tasks(args: Args) -> list[str]:
    if args.tasks:
        # Handle both ["A", "B"] and ["A,B,C"] (comma-separated single string from shell).
        task_pool = []
        for t in args.tasks:
            task_pool.extend(t.split(","))
    else:
        task_pool = {"atomic": ATOMIC_TASK_NAMES, "composite": COMPOSITE_TASK_NAMES}.get(args.split, ALL_TASK_NAMES)
    return [t for i, t in enumerate(task_pool) if i % args.num_shards == args.task_shard]


def main(args: Args):
    task_names = select_tasks(args)
    print(f"Evaluating {len(task_names)} tasks (shard {args.task_shard}/{args.num_shards}): {task_names}")

    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.results_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    args_name = f"args_shard{args.task_shard}.json" if args.num_shards > 1 else "args.json"
    (out_dir / args_name).write_text(json.dumps(asdict(args), indent=2))

    # Load dynamic instructions if provided
    dynamic_instructions: dict[str, dict] = {}
    if args.dynamic_instructions_file is not None:
        dynamic_instructions = json.loads(Path(args.dynamic_instructions_file).read_text())
        print(f"Loaded dynamic instructions for {len(dynamic_instructions)} tasks from {args.dynamic_instructions_file}")

    # Load GT reasoning texts if provided. Supports flat (task→str) and nested
    # (task→{caption_key: str}) formats. Two output formats:
    #   - default ("Reasoning:" prefix): task→str of single caption text
    #   - concat (--caption-keys-concat): task→str of concatenated multi-key caption
    # We flatten to task→str + a flag for which join-style to use downstream.
    gt_reasoning: dict[str, str] = {}
    gt_reasoning_concat = False  # True → join with ". " at episode time, no "Reasoning:" prefix
    if args.gt_reasoning_file is not None:
        if args.caption_key and args.caption_keys_concat:
            raise ValueError("--caption-key and --caption-keys-concat are mutually exclusive")
        raw = json.loads(Path(args.gt_reasoning_file).read_text())
        sample = next(iter(raw.values())) if raw else None
        if args.caption_keys_concat:
            if not isinstance(sample, dict):
                raise ValueError(
                    f"--caption-keys-concat requires nested gt_reasoning_file; "
                    f"got flat: {args.gt_reasoning_file}"
                )
            keys = [k.strip() for k in args.caption_keys_concat.split(",") if k.strip()]
            missing_any = []
            for task_name, cap_dict in raw.items():
                parts = [cap_dict.get(k, "").strip().rstrip(".") for k in keys]
                parts = [p for p in parts if p]
                if not parts:
                    missing_any.append(task_name)
                    continue
                gt_reasoning[task_name] = ". ".join(parts) + "."
            gt_reasoning_concat = True
            print(f"Loaded GT reasoning (concat) for {len(gt_reasoning)}/{len(raw)} tasks "
                  f"(keys={keys}) from {args.gt_reasoning_file}")
            if missing_any:
                print(f"  WARNING: {len(missing_any)} tasks had no matching keys: {missing_any[:5]}")
        elif isinstance(sample, dict):
            if not args.caption_key:
                raise ValueError(
                    f"--gt-reasoning-file {args.gt_reasoning_file} is nested "
                    f"(task→{{caption_key: text}}); pass --caption-key or --caption-keys-concat. "
                    f"Available keys: {sorted(sample.keys())}"
                )
            for task_name, cap_dict in raw.items():
                if args.caption_key in cap_dict:
                    gt_reasoning[task_name] = cap_dict[args.caption_key]
            print(f"Loaded GT reasoning for {len(gt_reasoning)}/{len(raw)} tasks "
                  f"(key='{args.caption_key}') from {args.gt_reasoning_file}")
        else:
            gt_reasoning = raw
            print(f"Loaded GT reasoning for {len(gt_reasoning)} tasks from {args.gt_reasoning_file}")

    # Per-episode VLM captions (override per (task, seed_episode))
    vlm_captions: dict[str, dict[str, str]] = {}
    if args.vlm_captions_file is not None:
        vlm_captions = json.loads(Path(args.vlm_captions_file).read_text())
        n_total = sum(len(v) for v in vlm_captions.values())
        print(f"Loaded VLM captions for {len(vlm_captions)} tasks "
              f"({n_total} per-episode captions) from {args.vlm_captions_file}")

    # Load fine-grained per-step caption provider if requested
    fg_provider = None
    if args.fg_captions_path is not None:
        try:
            from openpi.policies.fg_caption_provider import FineGrainedCaptionProvider
        except ImportError:
            from eval_utils.fg_caption_provider import FineGrainedCaptionProvider
        fg_provider = FineGrainedCaptionProvider(
            args.fg_captions_path,
            caption_key=args.fg_caption_key,
            fg_stride=args.fg_stride,
        )
        print(f"FG captions enabled: key={args.fg_caption_key}, stride={args.fg_stride}")

    client = RoboCasaClient(
        remote_host=args.host,
        remote_port=args.port,
        open_loop_horizon=args.open_loop_horizon,
    )

    all_results: dict[str, dict] = {}

    # Resume preservation: when extending an existing eval (--episodes-start > 0),
    # pre-load the entire prior results_shard{S}.json so tasks NOT processed in
    # this run keep their existing entries. Without this, the first incremental
    # save (after processing task 0) would overwrite the shard JSON with only
    # task 0's entry, dropping all other tasks. The per-task resume logic below
    # then overlays this shard's task entries with reused prior episodes.
    if args.episodes_start > 0:
        _prior_shard_path = out_dir / f"results_shard{args.task_shard}.json"
        if _prior_shard_path.exists():
            try:
                _prior_all = json.loads(_prior_shard_path.read_text())
                if isinstance(_prior_all, dict):
                    all_results.update(_prior_all)
                    print(f"[resume] pre-loaded {len(_prior_all)} prior tasks from {_prior_shard_path.name}")
            except Exception as _e:
                print(f"[resume] could not pre-load shard JSON ({_e}); starting fresh")

    for task_name in task_names:
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")

        try:
            env = build_env(task_name, args)
        except Exception as e:
            print(f"  [SKIP] Could not build env for {task_name}: {e}")
            traceback.print_exc()
            all_results[task_name] = {"error": str(e)}
            continue

        # Fallback instruction (overridden per-episode from obs["annotation.human.task_description"])
        instruction = ALL_TASKS.get(task_name, task_name)

        # Compute caption suffix (without the task description) — appended to
        # env-provided task description inside run_episode after env.reset.
        # Two formats: "reasoning" → "<task> Reasoning: <suffix>"
        #              "concat"    → "<task>. <suffix>"
        caption_suffix = gt_reasoning.get(task_name, "") or None
        caption_format = "concat" if gt_reasoning_concat else "reasoning"
        if caption_suffix:
            print(f"  GT caption ({caption_format}): {len(caption_suffix)} chars")

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

        # Build per-step caption function for this task (if fg captions enabled).
        fg_caption_fn = None
        fg_caption_episode = None
        if fg_provider is not None:
            fg_caption_episode = fg_provider.find_episode_for_task(instruction)
            if fg_caption_episode is not None:
                fg_caption_fn = lambda step, _ep=fg_caption_episode: fg_provider.get_caption(_ep, step)
                print(f"  FG captions: using episode {fg_caption_episode}")
            else:
                print(f"  FG captions: no matching episode for '{instruction}'")

        num_episodes = len(eval_init_states) if eval_init_states else args.episodes
        episode_results = []
        successes = 0
        place_anytime_count = 0

        # Incremental save: pre-register the task in all_results before episodes run, so
        # a mid-task SIGABRT (MUJOCO step-383 crash) leaves the completed episodes on
        # disk. Updated after every episode. `complete=False` marks an in-progress task.
        results_path = out_dir / f"results_shard{args.task_shard}.json"

        # Resume support: when --episodes-start > 0, reuse ALL prior episode_results
        # for this task from the prior results_shard*.json and continue from
        # `len(prior_eps)`. This handles both the "task fully done" case (reuse 20
        # of 20, run 0 more) and the "task partially done" case (reuse 9 of 20,
        # run 11 more). Without this, a preempted retry would discard the 9 prior
        # eps and restart from 0 — wasting compute. Effectively `episodes_start`
        # is just a sentinel to opt into resume mode; the actual resume point is
        # set per-task from what's on disk.
        if args.episodes_start > 0 and results_path.exists():
            try:
                _prior = json.loads(results_path.read_text())
                _prior_eps = _prior.get(task_name, {}).get("episode_results", []) or []
                # Cap at num_episodes (in case prior file has more than we want).
                _prior_eps = _prior_eps[:num_episodes]
                if _prior_eps:
                    episode_results = list(_prior_eps)
                    successes = sum(1 for r in episode_results if r.get("success"))
                    place_anytime_count = sum(1 for r in episode_results if r.get("place_anytime"))
                    print(f"  [resume] reused {len(episode_results)} prior eps for {task_name}; will run eps {len(episode_results)}..{num_episodes-1}")
                else:
                    print(f"  [resume] no prior eps for {task_name}; running 0..{num_episodes-1}")
            except Exception as _e:
                print(f"  [resume] could not load prior episodes ({_e}); starting from ep 0")

        _seeded = len(episode_results)
        all_results[task_name] = {
            "success_rate": (successes / _seeded) if _seeded else 0.0,
            "successes": successes,
            "place_anytime_rate": (place_anytime_count / _seeded) if _seeded else 0.0,
            "place_anytime_count": place_anytime_count,
            "episodes": num_episodes,
            "episode_results": episode_results,
            "instruction": instruction,
            "split": "atomic" if task_name in ATOMIC_TASKS else "composite",
            "complete": False,
        }

        # Start from `episodes_start` if we successfully resumed; else from 0.
        _ep_start = len(episode_results)
        for ep in tqdm(range(_ep_start, num_episodes), desc=task_name, initial=_ep_start, total=num_episodes):
            init_state = eval_init_states[ep] if eval_init_states else None
            # Fresh env per episode. Two reasons:
            #  (1) When using eval_dataset, ep_meta differs per episode and reusing
            #      the env across episodes can cause objects to disappear (see
            #      evaluation_protocol.md).
            #  (2) MUJOCO with MUJOCO_GL=egl leaks offscreen render buffers across
            #      env.reset() — after ~one full episode's worth of renders, the EGL
            #      allocator hits a C-level SIGABRT (exit code 134), killing the
            #      shard mid-run.
            #
            # robosuite's MjSim.free() does `del self._render_context_offscreen`,
            # which RELIES on Python's __del__ firing synchronously — but with
            # reference cycles in CPython, __del__ can be deferred indefinitely,
            # leaving the GL context allocated. Force explicit cleanup:
            #   (a) call con.free() / gl_ctx.free() directly on the render context
            #   (b) env.close() to free the sim
            #   (c) gc.collect() to drop any remaining cycles
            if ep > 0:
                try:
                    sim = env.unwrapped.sim if hasattr(env, 'unwrapped') else None
                    rctx = getattr(sim, '_render_context_offscreen', None) if sim else None
                    if rctx is not None:
                        try:
                            rctx.con.free()
                        except Exception:
                            pass
                        try:
                            rctx.gl_ctx.free()
                        except Exception:
                            pass
                        sim._render_context_offscreen = None
                except Exception as e:
                    print(f"  [WARN] explicit GL cleanup failed: {e}")
                env.close()
                import gc
                gc.collect()
                env = build_env(task_name, args)
            # Compute per-episode seed: unique per (task, episode) but reproducible.
            ep_seed = (args.seed + task_names.index(task_name) * 10000 + ep) if args.seed is not None else None

            # Per-episode VLM caption override: replaces gt_reasoning's caption_suffix
            # for this (task, seed_episode) when available. Format: {task: {"{seed}_{ep}": text}}.
            ep_caption_suffix = caption_suffix
            ep_caption_format = caption_format
            if vlm_captions and task_name in vlm_captions and args.seed is not None:
                ep_key = f"{args.seed}_{ep}"
                vlm_text = vlm_captions[task_name].get(ep_key, "")
                if vlm_text:
                    ep_caption_suffix = vlm_text
                    ep_caption_format = "reasoning"  # VLM captions use the standard reasoning prefix

            if dyn:
                result = run_episode_dynamic(
                    env=env,
                    client=client,
                    task_instructions=dyn["task_instructions"],
                    instruction_steps=dyn["instruction_steps"],
                    max_steps=args.max_steps,
                    headless=args.headless,
                )
            else:
                result = run_episode(
                    env=env,
                    client=client,
                    instruction=instruction,
                    max_steps=args.max_steps,
                    headless=args.headless,
                    init_state=init_state,
                    seed=ep_seed,
                    fg_caption_fn=fg_caption_fn,
                    caption_suffix=ep_caption_suffix,
                    caption_format=ep_caption_format,
                )
            if result["success"]:
                successes += 1
            if result.get("place_anytime", False):
                place_anytime_count += 1

            if args.save_video and result["video"]:
                mediapy.write_video(
                    task_dir / f"ep{ep:03d}_{'success' if result['success'] else 'fail'}.mp4",
                    result["video"],
                    fps=15,
                )
            episode_results.append({k: v for k, v in result.items() if k != "video"})

            # Incremental save after every episode — survives mid-task SIGABRT.
            all_results[task_name]["successes"] = successes
            all_results[task_name]["success_rate"] = successes / (ep + 1)
            all_results[task_name]["place_anytime_count"] = place_anytime_count
            all_results[task_name]["place_anytime_rate"] = place_anytime_count / (ep + 1)
            results_path.write_text(json.dumps(all_results, indent=2))

        # Task finished: finalize.
        sr = successes / num_episodes
        all_results[task_name]["success_rate"] = sr
        all_results[task_name]["place_anytime_rate"] = place_anytime_count / num_episodes
        all_results[task_name]["complete"] = True
        if dyn:
            all_results[task_name]["task_instructions"] = dyn["task_instructions"]
            all_results[task_name]["instruction_steps"] = dyn["instruction_steps"]
        results_path.write_text(json.dumps(all_results, indent=2))
        print(f"  Success rate: {successes}/{args.episodes} = {sr:.1%}  (place_anytime: {place_anytime_count}/{num_episodes} = {place_anytime_count/num_episodes:.1%})")

        env.close()

    # Final write (no-op if all tasks completed and incrementally saved).
    results_path = out_dir / f"results_shard{args.task_shard}.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {results_path}")

    # Print summary
    atomic_srs    = [v["success_rate"] for k, v in all_results.items() if v.get("split") == "atomic"]
    composite_srs = [v["success_rate"] for k, v in all_results.items() if v.get("split") == "composite"]
    if atomic_srs:
        print(f"Atomic    mean SR: {np.mean(atomic_srs):.1%} over {len(atomic_srs)} tasks")
    if composite_srs:
        print(f"Composite mean SR: {np.mean(composite_srs):.1%} over {len(composite_srs)} tasks")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
