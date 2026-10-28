"""Optional observation-only planner/verifier. No task-name registry or env access.

Planning, captions and verification run in the background.
Caption updates and phase transitions are applied only at chunk boundaries.
Confidence is an uncalibrated VLM score, never benchmark ground truth.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
import base64
import io
import json
import math
import threading
import time
import urllib.request


MAX_PHASES = 5
CAPTION_TYPES = {"current_action", "environment-simple", "agent_arm_state", "reasoning-short", "none"}
CAMERAS = ("video.robot0_agentview_left", "video.robot0_agentview_right", "video.robot0_eye_in_hand")


def parse_object(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() != "```":
            raise ValueError("Incomplete JSON fence")
        text = "\n".join(lines[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def short_text(value, name, maximum=600, allow_empty=False):
    if not isinstance(value, str) or len(value) > maximum or (not value.strip() and not allow_empty):
        raise ValueError(f"Invalid {name}: expected text of at most {maximum} characters")
    return value.strip()


def parse_plan(value, skills=(), max_phases=MAX_PHASES):
    if not isinstance(max_phases, int) or not 1 <= max_phases <= MAX_PHASES:
        raise ValueError(f"max_phases must be an integer in [1, {MAX_PHASES}]")
    phases = value.get("phases")
    if not isinstance(phases, list) or not 1 <= len(phases) <= max_phases:
        raise ValueError(f"Planner must return 1..{max_phases} phases")
    result = []
    for index, phase in enumerate(phases):
        if not isinstance(phase, dict):
            raise ValueError("Each phase must be an object")
        skill = short_text(phase.get("skill"), "skill", 160)
        if skills and skill not in skills:
            raise ValueError(f"Unsupported skill: {skill}")
        kind = phase.get("caption_type")
        if kind not in CAPTION_TYPES:
            raise ValueError(f"Unsupported caption_type: {kind}")
        caption = short_text(phase.get("caption"), "caption", 400, allow_empty=kind == "none")
        if kind == "none" and caption:
            raise ValueError("caption_type=none requires an empty caption")
        result.append({
            "id": index,
            "skill": skill,
            "instruction": short_text(phase.get("instruction"), "instruction", 300),
            "success_criteria": short_text(phase.get("success_criteria"), "success_criteria"),
            "caption_type": kind,
            "caption": caption,
        })
    return result


def parse_verdict(value):
    status = value.get("status")
    if status not in {"ongoing", "success", "failure", "uncertain"}:
        raise ValueError("Invalid verifier status")
    confidence = value.get("confidence")
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be a finite number in [0, 1]")
    return {"status": status, "confidence": confidence,
            "evidence": short_text(value.get("evidence"), "evidence")}


def encode_observation(obs):
    # Imports stay local: controller logic and mocked tests need no torch/MuJoCo.
    from PIL import Image
    encoded = []
    for camera in CAMERAS:
        buf = io.BytesIO()
        image = Image.fromarray(obs[camera]).convert("RGB")
        image.thumbnail((256, 256))
        image.save(buf, format="JPEG", quality=90)
        encoded.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return encoded


class HTTPBackend:
    def __init__(self, url, caption_url=None, timeout=60):
        self.url = url.rstrip("/")
        self.caption_url = caption_url
        self.timeout = timeout

    def __call__(self, operation, payload):
        if operation == "caption":
            if not self.caption_url:
                raise ValueError("Instructor caption mode requires --hierarchy-caption-url")
            url = self.caption_url
        else:
            url = self.url + "/" + operation
        request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            value = json.loads(response.read())
        if not isinstance(value, dict):
            raise ValueError("Server returned a non-object response")
        return value


@dataclass
class HierarchyConfig:
    check_every: int = 16
    min_phase_steps: int = 16
    confirmations: int = 2
    confidence_threshold: float = 0.85
    phase_timeout: int = 240
    max_replans: int = 2
    max_phases: int = MAX_PHASES
    caption_mode: str = "planner"  # none | planner | instructor
    async_verifier: bool = True
    async_planner: bool = True

    def __post_init__(self):
        if min(self.check_every, self.min_phase_steps, self.confirmations, self.phase_timeout) < 1:
            raise ValueError("Step intervals and confirmations must be positive")
        if self.max_replans < 0 or self.phase_timeout < self.min_phase_steps:
            raise ValueError("Invalid replan budget / phase timeout")
        if not 1 <= self.max_phases <= MAX_PHASES:
            raise ValueError(f"max_phases must be in [1, {MAX_PHASES}]")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("Invalid confidence threshold")
        if self.caption_mode not in {"none", "planner", "instructor"}:
            raise ValueError("Invalid caption mode")


class HierarchicalController:
    def __init__(self, backend, config=None, skills=(), encoder=encode_observation):
        self.backend = backend
        self.config = config or HierarchyConfig()
        self.skills = list(skills)
        self.encoder = encoder
        self.events = []
        self.history = []
        self.replans = 0
        self.generation = 0
        self.finished = False
        self.error = None
        self.inference_seconds = 0.0
        self._event_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="demian-verifier")
        self._caption_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="demian-caption")
        self._plan_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="demian-planner")
        self._plan_future: Future | None = None
        self._verify_future: Future | None = None
        self._caption_future: Future | None = None
        self._verify_request = None
        self._pending_transition = None
        self._closed = False
        self.caption = None

    @property
    def stopped(self):
        return self.finished or self.error is not None

    def _call(self, operation, payload, step):
        started = time.monotonic()
        generation = self.generation
        try:
            response = self.backend(operation, payload)
        except Exception as exc:
            with self._event_lock:
                self.events.append({"event": "http_error", "operation": operation, "step": step,
                                    "error": str(exc)})
            raise
        finally:
            with self._event_lock:
                self.inference_seconds += time.monotonic() - started
        with self._event_lock:
            self.events.append({"event": operation, "step": step, "generation": generation,
                                "response": response})
        return response

    def _abort(self, exc, step):
        self.error = f"{type(exc).__name__}: {exc}"
        with self._event_lock:
            self.events.append({"event": "aborted", "step": step, "error": self.error})

    def start(self, obs, task):
        self.task = task
        try:
            self._plan(obs, 0, "Initial planning")
        except Exception as exc:
            self._abort(exc, 0)

    def _plan(self, obs, step, feedback):
        self.generation += 1
        for future in (self._caption_future, self._verify_future):
            if future is not None:
                future.cancel()
        self._caption_future = self._verify_future = None
        self._verify_request = None
        self.instruction = self.task
        self.caption, self.caption_type = None, "none"
        self.index = -1
        payload = {
            "task": self.task, "images": self.encoder(obs), "skills": self.skills,
            "history": self.history[-12:], "feedback": feedback,
            "caption_mode": self.config.caption_mode,
            "max_phases": self.config.max_phases,
        }
        if self.config.async_planner:
            self._plan_future = self._plan_executor.submit(self._call, "plan", payload, step)
            self.events.append({"event": "plan_pending", "step": step, "generation": self.generation})
        else:
            self._install_plan(self._call("plan", payload, step), obs, step)

    def _install_plan(self, value, obs, step):
        self.phases = parse_plan(value, self.skills, self.config.max_phases)
        self.index = 0
        self._enter(obs, step)

    def _enter(self, obs, step):
        if self._caption_future is not None:
            self._caption_future.cancel()
            self._caption_future = None
        self.phase = self.phases[self.index]
        self.phase_start_images = self.encoder(obs)
        self.phase_start_step = step
        self.last_check = step
        self.success_streak = 0
        self.instruction = self.phase["instruction"]
        self.caption, self.caption_type = None, "none"
        if self.config.caption_mode == "planner":
            self.caption = self.phase["caption"] or None
            self.caption_type = self.phase["caption_type"]
        elif self.config.caption_mode == "instructor":
            self._caption_future = self._caption_executor.submit(
                self._call, "caption", {"images": self.phase_start_images,
                                        "task_description": self.instruction}, step)
        self.events.append({"event": "phase_enter", "step": step, "generation": self.generation,
                            "phase": self.index, "instruction": self.instruction,
                            "caption": self.caption, "caption_type": self.caption_type,
                            "success_criteria": self.phase["success_criteria"]})

    def _replan(self, obs, step, feedback):
        self.history.append({"instruction": self.instruction, "outcome": "not_confirmed",
                             "evidence": feedback})
        if self.replans >= self.config.max_replans:
            raise RuntimeError("Replan budget exhausted: " + feedback)
        self.replans += 1
        self._plan(obs, step, feedback)

    def _apply_pending(self, obs, step):
        if self._pending_transition is None:
            return False
        kind, payload = self._pending_transition
        self._pending_transition = None
        try:
            if kind == "advance":
                self.history.append({"instruction": self.instruction, "outcome": "visually_confirmed",
                                     "evidence": payload["evidence"]})
                if self.index + 1 == len(self.phases):
                    self.finished = True
                    with self._event_lock:
                        self.events.append({"event": "plan_finished", "step": step})
                    return False
                self.index += 1
                self._enter(obs, step)
                return True
            feedback = payload
            self._replan(obs, step, feedback)
            return True
        except Exception as exc:
            self._abort(exc, step)
        return False

    def _poll_verifier(self, obs, step):
        future = self._verify_future
        if future is None or not future.done():
            return
        self._verify_future = None
        request = self._verify_request
        # Never apply a response on the same simulation step on which it was
        # submitted. This prevents a fast local server from causing a second
        # verdict/transition in the same action observation.
        if request is not None and step <= request["step"]:
            self._verify_future = future
            return
        self._verify_request = None
        try:
            verdict = parse_verdict(future.result())
            # A late response from a previous phase must never alter the new phase.
            if request["phase"] != self.index or request["generation"] != self.generation:
                with self._event_lock:
                    self.events.append({"event": "stale_verify_discarded", "step": step})
                return
            confident = verdict["confidence"] >= self.config.confidence_threshold
            self.success_streak = self.success_streak + 1 if confident and verdict["status"] == "success" else 0
            if self.success_streak >= self.config.confirmations:
                self._pending_transition = ("advance", verdict)
            else:
                failed = confident and verdict["status"] == "failure"
                timed_out = step - self.phase_start_step >= self.config.phase_timeout
                if failed or timed_out:
                    self._pending_transition = (
                        "replan",
                        ("Failure: " if failed else "Phase timeout: ") + verdict["evidence"],
                    )
        except Exception as exc:
            self._abort(exc, step)

    def observe(self, obs, step, at_chunk_boundary=False):
        """Poll/schedule verification without blocking action-chunk execution.

        The verifier request runs on a dedicated background thread. The main
        rollout only applies a completed verdict at an action-chunk boundary;
        therefore a slow 8002 response cannot interrupt a chunk already sent to
        the robot. The VLA is still queried only when its cached chunk is empty.
        """
        if self.stopped:
            return False
        if self._plan_future is not None:
            if not at_chunk_boundary or not self._plan_future.done():
                return False
            future, self._plan_future = self._plan_future, None
            try:
                self._install_plan(future.result(), obs, step)
                return True
            except Exception as exc:
                self._abort(exc, step)
                return False
        self._poll_verifier(obs, step)
        if self.stopped:
            return False
        if (at_chunk_boundary and self._pending_transition is None
                and step - self.phase_start_step >= self.config.phase_timeout):
            self._pending_transition = ("replan", "Phase timeout: no confirmed completion")
            if self._verify_future is not None:
                self._verify_future.cancel()
                self._verify_future = None
                self._verify_request = None
        if at_chunk_boundary and self._pending_transition is not None:
            return self._apply_pending(obs, step)
        if at_chunk_boundary and self._caption_future is not None and self._caption_future.done():
            future, self._caption_future = self._caption_future, None
            try:
                response = future.result()
                self.caption = short_text(response.get("caption"), "Instructor caption", 1200, allow_empty=True) or None
                self.caption_type = "unspecified_by_instructor" if self.caption else "none"
                self.events.append({"event": "caption_applied", "step": step,
                                    "generation": self.generation, "phase": self.index,
                                    "caption": self.caption})
            except Exception as exc:
                self._abort(exc, step)
                return False
        if not at_chunk_boundary or step - self.last_check < self.config.check_every:
            return False
        if step - self.phase_start_step < self.config.min_phase_steps:
            return False
        if self._verify_future is not None:
            return False
        self.last_check = step
        payload = {
            "task": self.task, "phase": self.phase,
            "start_step": self.phase_start_step, "current_step": step,
            "start_images": self.phase_start_images, "images": self.encoder(obs),
        }
        self._verify_request = {"phase": self.index, "generation": self.generation, "step": step}
        if self.config.async_verifier:
            self._verify_future = self._executor.submit(self._call, "verify", payload, step)
        else:
            self._verify_future = _ImmediateFuture(self._call("verify", payload, step))
        return False

    def close(self):
        if not self._closed:
            self._closed = True
            if self._verify_future is not None:
                self._verify_future.cancel()
            if self._caption_future is not None:
                self._caption_future.cancel()
            if self._plan_future is not None:
                self._plan_future.cancel()
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._caption_executor.shutdown(wait=False, cancel_futures=True)
            self._plan_executor.shutdown(wait=False, cancel_futures=True)

    def summary(self):
        with self._event_lock:
            events = list(self.events)
            inference_seconds = self.inference_seconds
        return {"config": asdict(self.config), "status": "error" if self.error else
                ("plan_finished" if self.finished else ("planning" if self._plan_future is not None else "running")), "error": self.error,
                "task": self.task, "replans": self.replans, "events": events,
                "inference_seconds": inference_seconds,
                "note": "Visual completion is not benchmark success. Verifier runs asynchronously and transitions apply at chunk boundaries."}


class _ImmediateFuture:
    """Small Future-compatible wrapper used by deterministic unit tests."""
    def __init__(self, value):
        self._value = value

    def done(self):
        return True

    def result(self):
        return self._value

    def cancel(self):
        return False
