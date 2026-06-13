import base64
import io
import json
import os
from pathlib import Path
from dataclasses import dataclass
import pickle
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from typing import cast
import logging
import dataclasses

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.shared import download
from openpi.models import pi0_config

from molmo_spaces.policy.base_policy import StatefulPolicy, InferencePolicy
from molmo_spaces.policy.learned_policy.utils import resize_with_pad

from molmobot_pi0.eval.utils import resize_with_crop
import molmobot_pi0.config_openpi as _


log = logging.getLogger(__name__)


class QwenInstructorClient:
    """HTTP client for the Qwen3.5-2B instructor served by
    ``openpi/scripts/eval/qwen_instructor_server.py``. Sends the initial camera
    frames (base64 JPEG) + task description and returns the generated annotation.
    Used for the asynchronous-instructor MolmoSpaces eval (paper RQ2).
    """

    def __init__(self, server_url: str = "http://localhost:8001"):
        self._url = server_url.rstrip("/")
        log.info("Waiting for Qwen instructor server at %s ...", self._url)
        for _ in range(60):
            try:
                urllib.request.urlopen(f"{self._url}/healthz", timeout=2)
                log.info("Instructor server is up.")
                return
            except Exception:
                time.sleep(2)
        raise RuntimeError(f"Qwen instructor server not reachable at {self._url}")

    @staticmethod
    def _encode_image(img_array: np.ndarray) -> str:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(np.asarray(img_array).astype(np.uint8)).convert("RGB").save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def generate(self, images: list[np.ndarray], task_description: str) -> str:
        """Generate an annotation from N camera frames + task description."""
        payload = {
            "task_description": task_description,
            "images": [self._encode_image(img) for img in images],
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(self._url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
        return result.get("caption", "")


EXO_CAM_INPUT_KEY = "observation/exterior_image_1_left"
WRIST_CAM_INPUT_KEY = "observation/wrist_image_left"

# Caption templates for eval-time injection. Object names are extracted from the
# task description at runtime and substituted into {pickup_obj} / {place_obj}.
# These match the style of Qwen3-VL-30B captions used during training.
#
# NOTE on pick_and_place_color: The Color benchmark uses COLOR ALIASES in the
# task description (e.g., "orange bowl") that differ from the actual object name
# ("tan bowl with dark patterns"). Template captions must NOT echo the place
# target name, because it would reinforce the wrong visual grounding. Instead,
# Color templates omit the place target and use generic references ("a nearby
# receptacle"). The model must rely on visual matching to find the correct target.
CAPTION_TEMPLATES: dict[str, dict[str, str]] = {
    "current_action": {
        "pick": "The robotic arm moves its gripper towards the {pickup_obj}, then closes the gripper to grasp the {pickup_obj}, lifting it off the surface.",
        "pick_and_place": "The robot arm moves its gripper towards the {pickup_obj}, grasps it, and lifts it upwards. The arm then repositions the {pickup_obj} over the {place_obj} and places it inside, completing the task.",
        "pick_and_place_next_to": "The robot arm moves its gripper towards the {pickup_obj}, grasps it, and lifts it. The arm then moves the {pickup_obj} toward the {place_obj}, placing it next to the {place_obj} on the surface.",
        "pick_and_place_color": "The robot arm moves its gripper towards the {pickup_obj}, grasps it, and lifts it upwards. The arm then places the {pickup_obj} into a nearby receptacle, completing the task.",
    },
    "action_for_policy": {
        "pick": "Reach down from above toward the {pickup_obj}. Close the gripper around it. Lift up 10cm.",
        "pick_and_place": "Reach down from above toward the {pickup_obj} and close the gripper around it. Lift up 10cm, then move toward the {place_obj}. Lower the {pickup_obj} down onto the {place_obj} and release it when it is resting inside.",
        "pick_and_place_next_to": "Reach down from above and close the gripper around the {pickup_obj}. Lift up 10cm, then move toward the {place_obj}. Release the {pickup_obj} when it is positioned next to the {place_obj} on the surface.",
        "pick_and_place_color": "Reach down from above toward the {pickup_obj} and close the gripper around it. Lift up 15cm, then move toward a nearby receptacle. Lower and release when it is resting inside.",
    },
    "environment-simple": {
        "pick": "A robotic arm is positioned over a surface where a {pickup_obj} is placed. The arm is attempting to pick up the {pickup_obj}.",
        "pick_and_place": "A robotic arm is positioned over a surface with various objects including a {pickup_obj} and a {place_obj}. The arm is manipulating the {pickup_obj} and placing it into the {place_obj}.",
        "pick_and_place_next_to": "A robotic arm is positioned over a surface with various objects including a {pickup_obj} and a {place_obj}. The arm is manipulating the {pickup_obj} and placing it next to the {place_obj}.",
        "pick_and_place_color": "A robotic arm is positioned over a surface with various objects including a {pickup_obj}. The arm is manipulating the {pickup_obj} and placing it into a nearby bowl.",
    },
    "reasoning-short": {
        "pick": "The robot arm moves toward the {pickup_obj} on the surface. It positions its gripper above the {pickup_obj}, preparing to grasp it. The arm then lowers and securely picks up the {pickup_obj}, lifting it off the surface.",
        "pick_and_place": "The robot arm picks up the {pickup_obj} from the surface. The arm then moves the {pickup_obj} towards the {place_obj}, positioning it for placement. The {pickup_obj} is carefully placed into the {place_obj}, completing the task of picking and placing.",
        "pick_and_place_next_to": "The robot arm picks up the {pickup_obj} from the surface. It then moves the {pickup_obj} towards the {place_obj}, positioning it next to the {place_obj}. The task is completed with the {pickup_obj} placed beside the {place_obj}.",
        "pick_and_place_color": "The robot arm moves towards the {pickup_obj} on the surface, grasps it with its gripper, and lifts it. It then carefully places the {pickup_obj} into a nearby receptacle, completing the task of picking and placing the object.",
    },
    "agent_arm_state": {
        "pick": "The robotic arm is positioned above the surface. The arm moves its gripper towards the {pickup_obj}, closes the gripper around the {pickup_obj}, and lifts it upward, away from the surface.",
        "pick_and_place": "The robotic arm is positioned above the surface, with its gripper closed around the {pickup_obj}. The arm moves towards the {place_obj}, lowers the {pickup_obj} onto the {place_obj}, then lifts and retracts upward, moving away from the objects.",
        "pick_and_place_next_to": "The robotic arm is positioned above the surface, with its gripper closed around the {pickup_obj}. The arm moves laterally towards the {place_obj}, lowers the {pickup_obj} next to the {place_obj}, then lifts and retracts upward.",
        "pick_and_place_color": "The robotic arm is positioned above the surface, with its gripper closed around the {pickup_obj}. The arm lowers the {pickup_obj} into a nearby receptacle, then lifts and retracts upward, moving away from the objects.",
    },
    "dense": {
        "pick": "The gripper lowers toward the {pickup_obj} on the surface. The gripper grasps the {pickup_obj}. The gripper lifts the {pickup_obj} from the surface. The gripper moves upward with the {pickup_obj}.",
        "pick_and_place": "The gripper reaches downward toward the {pickup_obj} on the surface, then grasps the {pickup_obj}. The gripper lifts the {pickup_obj} upward and moves it forward toward the {place_obj}. The gripper lowers the {pickup_obj} onto the {place_obj}, placing it inside. The gripper releases the {pickup_obj}.",
        "pick_and_place_next_to": "The gripper reaches down to the {pickup_obj} on the surface, grasps the {pickup_obj}, lifts it upward, moves it forward toward the {place_obj}, and releases the {pickup_obj} next to the {place_obj}.",
        "pick_and_place_color": "The gripper reaches downward toward the {pickup_obj} on the surface, then grasps the {pickup_obj}. The arm lifts the {pickup_obj} upward and moves it forward over a nearby receptacle. The gripper lowers the {pickup_obj} onto the receptacle. The arm retracts upward.",
    },
    "summary": {
        "pick": "On the surface, the robot arm moves forward and lowers its gripper to grasp the {pickup_obj}. The gripper closes around the {pickup_obj} and lifts it upward. The immediate goal is to pick up the {pickup_obj}. The action is in progress.",
        "pick_and_place": "On the surface, the robotic arm reaches downward, grasps the {pickup_obj}, lifts it upward, and moves it forward toward the {place_obj}. The arm lowers the {pickup_obj} into the {place_obj}. The action is in progress.",
        "pick_and_place_next_to": "On the surface, the robotic arm reaches down to grasp the {pickup_obj}, lifts it, and moves it laterally to place it next to the {place_obj}. The action is in progress.",
        "pick_and_place_color": "On the surface, the robotic arm reaches downward, grasps the {pickup_obj}, lifts it, and places it into a nearby receptacle. The action is in progress.",
    },
}


def _parse_objects_from_prompt(prompt: str) -> tuple[str, str, str]:
    """Extract object names and task type from a task prompt.

    Returns (task_type, pickup_obj, place_obj).
    """
    import re

    # Pick and place next to: "Pick up the X and place it next to the Y"
    m = re.match(r"pick up (?:the|some|a) (.+?) and place it next to (?:the )?(.+)", prompt, re.IGNORECASE)
    if m:
        return "pick_and_place_next_to", m.group(1).strip(), m.group(2).strip()

    # Pick and place in/on: "Pick up the X and place it in or on/in/on the Y"
    # Try longest alternation first to avoid partial match on "in"
    m = re.match(r"pick up (?:the|some|a) (.+?) and place it (?:in or on|in|on) (?:the )?(.+)", prompt, re.IGNORECASE)
    if m:
        return "pick_and_place", m.group(1).strip(), m.group(2).strip()

    # Pick only: "Pick up the/some X"
    m = re.match(r"pick up (?:the|some|a) (.+)", prompt, re.IGNORECASE)
    if m:
        return "pick", m.group(1).strip(), ""

    # Fallback
    return "pick", "the object", ""


@dataclass
class PiPolicyState:
    actions_buffer: np.ndarray | None
    current_buffer_index: int
    last_prompt: str


class PiPolicy(InferencePolicy, StatefulPolicy):
    def __init__(
        self,
        model_name: str | None = None,
        checkpoint_dir: str | None = None,
        use_torch: bool = True,
        buffer_length: int | None = None,
        cameras: dict[str, str] | None = None,
        device_id: int | None = None,
        compile_mode: str | None = None,
        max_joint_delta: float | list[float] | None = None,
        caption_key: str | None = None,
        color_task: bool = False,
        caption_file: str | None = None,
        caption_strict_file: bool = False,
        instructor_url: str | None = None,
        instructor_sync: bool = False,
    ) -> None:
        super().__init__(None, "pick_and_place")
        self._caption_key = caption_key
        self._color_task = color_task
        # Asynchronous learned-instructor eval (paper RQ2): when instructor_url is
        # set, the policy queries a live Qwen3.5-2B instructor on the initial frames
        # and (async) splices the returned annotation at the next action-chunk
        # boundary, acting on the task description alone until then. instructor_sync
        # blocks the first action until the annotation is ready (the sync baseline).
        self._instructor = QwenInstructorClient(instructor_url) if instructor_url else None
        self._instructor_sync = instructor_sync
        self._pool = ThreadPoolExecutor(max_workers=1) if (self._instructor and not instructor_sync) else None
        # When True: use caption_file as the SOLE source of caption text
        #   - empty string in file → inject no caption (matches Mix5 "none" training)
        #   - missing key → raise (no template fallback)
        # When False (default): preserve existing template-fallback behavior.
        self._caption_strict_file = caption_strict_file
        # Per-episode caption lookup from JSON file (e.g., for Color benchmarks
        # where referral expressions differ from task description aliases).
        # Format: {"pickup_obj_lower": {"caption_key": "caption text"}}
        self._caption_lookup: dict | None = None
        if caption_file and os.path.exists(caption_file):
            with open(caption_file) as f:
                self._caption_lookup = json.load(f)
            log.info(f"Loaded {len(self._caption_lookup)} per-episode captions from {caption_file}")
        if caption_key and caption_key not in CAPTION_TEMPLATES:
            raise ValueError(f"Unknown caption_key: {caption_key}. Available: {list(CAPTION_TEMPLATES.keys())}")

        self.buffer_length = buffer_length if buffer_length and buffer_length > 0 else 1000
        self._prepared = False
        self._last_prompt: str | None = None
        self._use_torch = use_torch
        self._device_id = device_id
        if max_joint_delta is None or isinstance(max_joint_delta, float):
            self.max_joint_delta = max_joint_delta
        else:
            self.max_joint_delta = np.array(max_joint_delta)

        if not checkpoint_dir:
            assert model_name is not None, "model_name is required when checkpoint_dir is not provided"
            self._default_config = True
            self.pi_config = _config.get_config(model_name)
            if use_torch:
                log.info(f"Loading PyTorch weights from {self.pi_config.pytorch_weight_path}")
                checkpoint_dir = self.pi_config.pytorch_weight_path
            elif isinstance(self.pi_config.weight_loader, _config.weight_loaders.CheckpointWeightLoader):
                ckpt_path = os.path.dirname(self.pi_config.weight_loader.params_path)
                log.info(f"Loading jax weights from {ckpt_path}")
                checkpoint_dir = str(download.maybe_download(ckpt_path))
            else:
                weight_url = f"gs://openpi-assets/checkpoints/{model_name}"
                checkpoint_dir = str(download.maybe_download(weight_url))
        else:
            self._default_config = False
            config_file = Path(checkpoint_dir) / "assets/train_config.pkl"
            if config_file.exists():
                with open(config_file, "rb") as f:
                    self.pi_config = pickle.load(f)
                assert isinstance(self.pi_config, _config.TrainConfig)
                log.info(f"Loaded train config from {config_file}")
                if model_name is not None:
                    assert self.pi_config.name == model_name, f"Model name mismatch: {self.pi_config.name} != {model_name}"
                else:
                    model_name = self.pi_config.name
            else:
                raise ValueError(f"No train config found for checkpoint {checkpoint_dir}")

        self.model_name = model_name

        assert isinstance(self.pi_config.model, pi0_config.Pi0Config)
        self.pi_config = dataclasses.replace(
            self.pi_config,
            model=dataclasses.replace(
                self.pi_config.model,
                pytorch_compile_mode=compile_mode,
            ),
        )

        if cameras is not None:
            self._cameras = cameras
        else:
            log.warning("No cameras specified and could not infer from config, using default cameras")
            self._cameras = {
                "exo_camera_1": EXO_CAM_INPUT_KEY,
                "wrist_camera": WRIST_CAM_INPUT_KEY,
            }

        assert checkpoint_dir and os.path.exists(checkpoint_dir), f"Checkpoint directory {checkpoint_dir} does not exist"
        self.checkpoint_dir = Path(checkpoint_dir)

        self.reset()

    def reset(self, *args, **kwargs):
        self.actions_buffer: np.ndarray | None = None
        self.current_buffer_index: int = 0
        self._last_prompt = None
        # Instructor async state (only used when self._instructor is set).
        self._instr_state = "init"      # init -> pending (async) -> injected
        self._instr_future: Future | None = None
        self._instr_prompt: str | None = None
        self._instr_task: str | None = None

    def get_state(self):
        state = PiPolicyState(
            actions_buffer=self.actions_buffer.copy() if self.actions_buffer is not None else None,
            current_buffer_index=self.current_buffer_index,
            last_prompt=self._last_prompt,
        )
        return state

    def set_state(self, state: PiPolicyState):
        self.actions_buffer = state.actions_buffer.copy() if state.actions_buffer is not None else None
        self.current_buffer_index = state.current_buffer_index
        self._last_prompt = state.last_prompt
        # On resume, treat any already-built prompt as the resolved instructor
        # annotation so we don't re-query the instructor mid-episode.
        if self._instructor is not None and state.last_prompt is not None:
            self._instr_prompt = state.last_prompt
            self._instr_task = state.last_prompt.split("\n", 1)[0]
            self._instr_state = "injected"

    def prepare_model(self):
        if self._prepared:
            return
        self.model = _policy_config.create_trained_policy(
            self.pi_config,
            self.checkpoint_dir,
            pytorch_device=f"cuda:{self._device_id}" if self._device_id is not None else None,
        )
        self.reset()
        self._prepared = True

        log.info("Warming up model...")
        dummy_input = {
            EXO_CAM_INPUT_KEY: np.zeros((224, 224, 3), dtype=np.uint8),
            WRIST_CAM_INPUT_KEY: np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/joint_position": np.zeros((7,), dtype=np.float32),
            "observation/gripper_position": np.zeros((1,), dtype=np.float32),
            "prompt": "place the red block on the green block",
        }
        self.model.infer(dummy_input)
        log.info("Model warmup complete.")

    def has_queued_actions(self) -> bool:
        return self.actions_buffer is not None and self.current_buffer_index < min(self.buffer_length, len(self.actions_buffer))

    def _safe_caption(self, images, task):
        """Query the instructor; return '' on failure (acts as no-caption)."""
        try:
            cap = self._instructor.generate(images, task)
            return cap.strip() if cap else ""
        except Exception as e:  # noqa: BLE001
            log.warning("[instructor] caption generation failed: %s", e)
            return ""

    def _instructor_obs_to_model_input(self, obs):
        """obs_to_model_input variant for the learned-instructor eval. Unlike the
        base path, the prompt legitimately changes mid-episode (task-only ->
        task+annotation) when the async annotation is spliced in."""
        task = obs["task"] if "task" in obs else (self._instr_task or self.task.get_task_description())
        instr_images = [np.asarray(obs[cam]) for cam in self._cameras]

        if self._instr_state == "init":
            self._instr_task = task
            if self._instructor_sync:
                cap = self._safe_caption(instr_images, task)
                self._instr_prompt = f"{task}\n{cap}" if cap else task
                self._instr_state = "injected"
                log.info(f"[instructor sync] prompt: {self._instr_prompt!r}")
            else:
                self._instr_future = self._pool.submit(self._safe_caption, instr_images, task)
                self._instr_prompt = task  # act on task description alone until ready
                self._instr_state = "pending"
                log.info("[instructor async] acting on task-only; annotation pending")
        elif self._instr_state == "pending" and self._instr_future is not None and self._instr_future.done():
            cap = self._instr_future.result()
            if cap:
                self._instr_prompt = f"{self._instr_task}\n{cap}"
                self.actions_buffer = None  # splice now: recompute the next chunk with the new prompt
                log.info(f"[instructor async] spliced annotation: {self._instr_prompt!r}")
            self._instr_state = "injected"

        self._last_prompt = self._instr_prompt
        images = {}
        if not self.has_queued_actions():
            for cam_obs_key, cam_input_key in self._cameras.items():
                img = resize_with_crop(obs[cam_obs_key], 360, 640)
                images[cam_input_key] = resize_with_pad(img, 224, 224)
        return {
            "observation/joint_position": np.array(obs["qpos"]["arm"]),
            "observation/gripper_position": np.array(obs["qpos"]["gripper"]),
            "prompt": self._instr_prompt,
            **images,
        }

    def obs_to_model_input(self, obs):
        if isinstance(obs, list):
            # TODO: obs shouldn't be a list, this is a bug in MolmoSpaces
            obs = obs[0]
        if self._instructor is not None:
            return self._instructor_obs_to_model_input(obs)
        if "task" in obs:
            prompt = obs["task"]
            assert self._last_prompt is None or self._last_prompt == prompt, f"Prompt changed during inference! {self._last_prompt} -> {prompt}"
        elif self._last_prompt is not None:
            prompt = self._last_prompt
        elif self.task is not None:
            prompt = self.task.get_task_description()
        else:
            raise ValueError("No prompt passed and no task registered!")

        # Inject caption for caption-trained models (must happen after task
        # sampler constructs the prompt, which strips benchmark JSON captions).
        if self._caption_key and self._last_prompt is None:
            task_type, pickup_obj, place_obj = _parse_objects_from_prompt(prompt)
            caption = None
            lookup_key = pickup_obj.lower().strip()
            entry_present = False

            # Per-episode lookup (e.g., Color benchmarks, or learned-instructor
            # output). Keyed by lowercase pickup object name.
            if self._caption_lookup:
                entry = self._caption_lookup.get(lookup_key)
                if entry is not None:
                    entry_present = True
                    caption = entry.get(self._caption_key, None)
                    if caption:
                        log.info(f"Using per-episode caption (key='{lookup_key}')")

            if self._caption_strict_file:
                # Strict mode: caption_file is the SOLE source. No template fallback.
                if not entry_present:
                    raise KeyError(
                        f"caption_strict_file=True but no entry for '{lookup_key}' "
                        f"in caption_file. All episodes must be pre-captioned."
                    )
                if caption:
                    prompt = f"{prompt}\n{caption}"
                    log.info(f"Injected file-caption ({self._caption_key}): {prompt!r}")
                else:
                    # Empty/missing caption text → no append (matches "none" training)
                    log.info(f"Empty caption for '{lookup_key}' — leaving prompt unmodified")
            else:
                # Default behavior: fall back to template-based caption
                if caption is None:
                    if self._color_task and "place" in task_type:
                        task_type = "pick_and_place_color"
                    templates = CAPTION_TEMPLATES[self._caption_key]
                    template = templates.get(task_type, templates["pick"])
                    caption = template.format(pickup_obj=pickup_obj, place_obj=place_obj)
                prompt = f"{prompt}\n{caption}"
                log.info(f"Injected {self._caption_key} caption: {prompt!r}")

        if self._last_prompt is None:
            self._last_prompt = prompt

        # skip image processing if we're not going to run inference
        images = {}
        if not self.has_queued_actions():
            for cam_obs_key, cam_input_key in self._cameras.items():
                # resize to 360p to match data
                img = resize_with_crop(obs[cam_obs_key], 360, 640)
                images[cam_input_key] = resize_with_pad(img, 224, 224)

        model_input = {
            "observation/joint_position": np.array(obs["qpos"]["arm"]),
            "observation/gripper_position": np.array(obs["qpos"]["gripper"]),
            "prompt": prompt,
            **images
        }
        return model_input

    def model_output_to_action(self, model_output: np.ndarray) -> dict[str, np.ndarray]:
        gripper_pos = np.clip(model_output[7], 0.0, 1.0) * np.array([255.0])
        arm_output = model_output[:7].reshape(7)
        action = {
            "arm": arm_output,
            "gripper": gripper_pos,
        }
        return action

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "openpi"
        info["policy_checkpoint"] = self.model_name
        info["policy_buffer_length"] = self.buffer_length
        info["prompt"] = self._last_prompt
        info["timestamp"] = time.time()
        return info


class PiJointVelPolicy(PiPolicy):
    def inference_model(self, model_input):
        self.prepare_model()
        if not self.has_queued_actions():
            self.actions_buffer = cast(np.ndarray, self.model.infer(model_input)["actions"])
            self.current_buffer_index = 0

            max_vels = np.max(np.abs(self.actions_buffer), axis=-1)
            scales = np.maximum(max_vels, np.ones_like(max_vels))
            self.actions_buffer[:, :7] /= scales.reshape(-1, 1)

        assert self.actions_buffer is not None
        model_output = self.actions_buffer[self.current_buffer_index]
        max_delta = 0.2 if self.max_joint_delta is None else self.max_joint_delta
        model_output[:7] = model_input["observation/joint_position"] + model_output[:7] * max_delta
        self.current_buffer_index += 1
        return model_output


class PiJointPosPolicy(PiPolicy):
    def inference_model(self, model_input):
        self.prepare_model()
        if not self.has_queued_actions():
            self.actions_buffer = cast(np.ndarray, self.model.infer(model_input)["actions"])
            self.current_buffer_index = 0

        assert self.actions_buffer is not None
        model_output = self.actions_buffer[self.current_buffer_index]

        if self.max_joint_delta is not None:
            predicted_deltas = model_output[:7] - model_input["observation/joint_position"]
            relative_scale = np.abs(predicted_deltas) / self.max_joint_delta

            if np.max(relative_scale) > 1:
                predicted_deltas /= np.max(relative_scale)
                model_output[:7] = model_input["observation/joint_position"] + predicted_deltas
        
        self.current_buffer_index += 1
        return model_output
