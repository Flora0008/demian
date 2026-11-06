"""Optional observation-only planner/verifier. No task-name registry or env access.

Planning, captions and verification run in the background.
Caption updates and phase transitions are applied only at chunk boundaries.
Confidence is an uncalibrated VLM score, never benchmark ground truth.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
import base64
import io
import http.client
import json
import math
import threading
import time
import urllib.error
import urllib.request
import uuid
from hierarchy_trace import EventLog


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
    if not isinstance(value, str):
        raise ValueError(f"Invalid {name}: expected text")
    text = value.strip()
    if not text and not allow_empty:
        raise ValueError(f"Empty {name}: expected non-empty text")
    if len(text) > maximum:
        raise ValueError(f"Oversized {name}: maximum {maximum} characters")
    return text


def parse_plan(value, skills=()):
    phases = value.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError(f"Planner must return a non-empty complete phase list")
    result = []
    for index, phase in enumerate(phases):
        if not isinstance(phase, dict):
            raise ValueError("Each phase must be an object")
        skill = short_text(phase.get("skill"), "skill", 160)
        if skills and skill not in skills:
            raise ValueError(f"Unsupported skill: {skill}")
        # Captions belong to the optional Instructor; legacy Planner fields are deliberately ignored.
        result.append({
            "id": index,
            "skill": skill,
            "instruction": short_text(phase.get("instruction"), "instruction", 300),
            "success_criteria": short_text(phase.get("success_criteria"), "success_criteria"),
        })
    return result


def parse_verdict(value):
    status = value.get("status")
    if not isinstance(status, str) or status not in {"ongoing", "success", "failure", "uncertain"}:
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
        # Match the existing Instructor's resize-with-padding, including non-square input.
        from openpi_client.image_tools import resize_with_pad
        image = Image.fromarray(resize_with_pad(obs[camera], 256, 256)).convert("RGB")
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
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            # Keep the bounded server diagnostic (e.g. invalid phase count), not image payloads.
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{operation} HTTP {exc.code}: {detail}") from exc
        except (TimeoutError, ConnectionError, http.client.HTTPException, urllib.error.URLError) as exc:
            raise RuntimeError(f"{operation} transport error; configured_timeout={self.timeout}s; "
                               f"{type(exc).__name__}: {exc}; server cause unconfirmed") from exc
        if not isinstance(value, dict):
            raise ValueError("Server returned a non-object response")
        return value


@dataclass
class HierarchyConfig:
    check_every: int = 16
    min_phase_steps: int = 16
    confirmations: int = 2
    confidence_threshold: float = 0.85
    max_replans: int = 2
    caption_mode: str = "none"  # none | instructor
    async_verifier: bool = True
    async_planner: bool = True

    def __post_init__(self):
        if min(self.check_every, self.min_phase_steps, self.confirmations) < 1:
            raise ValueError("Step intervals and confirmations must be positive")
        if self.max_replans < 0:
            raise ValueError("Invalid replan budget / phase timeout")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("Invalid confidence threshold")
        if self.caption_mode not in {"none", "instructor"}:
            raise ValueError("Invalid caption mode")


class HierarchicalController:
    def __init__(self, backend, config=None, skills=(), encoder=encode_observation, trace_dir=None, metadata=None):
        self.session_id = str(uuid.uuid4())
        self.backend = backend
        self.config = config or HierarchyConfig()
        self.skills = list(skills)
        self.encoder = encoder
        self.events = EventLog(trace_dir, {"config": asdict(self.config), "session_id": self.session_id,
                                           **(metadata or {})})
        self.history = []
        self.replans = 0
        self.total_replans = 0
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

    def _images(self, obs):
        images = self.encoder(obs)
        self.events.append({"event": "observation_encoded", "generation": self.generation,
                            "source_shapes": {key: list(value.shape) for key, value in obs.items()
                                              if key in CAMERAS and hasattr(value, "shape")},
                            "encoding": "resize_with_pad_256x256_JPEG90"})
        return images

    @property
    def stopped(self):
        """Terminal control state: the evaluator ends this episode without fallback."""
        return self.finished or self.error is not None

    def terminate(self, step, reason):
        """Stop without changing the task/caption into an unapproved fallback."""
        self.error = reason
        self.events.append({"event": "terminated", "step": step, "reason": reason})
        self.generation += 1
        for name in ("_plan_future", "_verify_future", "_caption_future"):
            future = getattr(self, name)
            if future is not None:
                future.cancel()
            setattr(self, name, None)
        self._verify_request = self._pending_transition = None
        return True

    def _submit(self, executor, operation, payload, step):
        # Capture identity on the rollout thread, not when a queued worker happens to start.
        return executor.submit(self._call, operation, payload, step, self.generation)

    def _call(self, operation, payload, step, generation=None):
        started = time.monotonic()
        generation = self.generation if generation is None else generation
        request_id = str(uuid.uuid4())
        payload = {**payload, "request_id": request_id, "generation": generation, "session_id": self.session_id}
        wire_attempt = 1
        try:
            # Two total attempts, shared with the server (which never silently retries).
            for attempt in range(1, 3):
                if self._closed or generation != self.generation:
                    raise RuntimeError("request superseded before attempt")
                payload = {**payload, "request_id": f"{request_id}:{wire_attempt}",
                           "deadline_unix": time.time() + getattr(self.backend, "timeout", 60)}
                self.events.append({"event": "request", "operation": operation, "step": step,
                                    "generation": generation, "attempt": attempt, "request_id": request_id, "payload": payload})
                attempt_start = time.monotonic()
                try:
                    response = self.backend(operation, payload)
                    self.events.append({"event": "response", "operation": operation, "step": step,
                                        "generation": generation, "attempt": attempt, "request_id": request_id,
                                        "response": response, "request_seconds": time.monotonic() - attempt_start})
                    if operation == "plan":
                        parse_plan(response, self.skills)
                    elif operation == "verify":
                        parse_verdict(response)
                    elif operation == "caption":
                        short_text(response.get("caption"), "Instructor caption", 1200)
                    response = {**response, "request_attempts": attempt}
                    break
                except Exception as exc:
                    self.events.append({"event": "request_error", "operation": operation,
                                        "step": step, "attempt": attempt, "request_id": request_id, "error_type": type(exc).__name__,
                                        "error": str(exc)})
                    # Transport timeout may leave the original GPU request running.
                    # Reuse its identity so the service can return the completed result.
                    transport = isinstance(exc, (TimeoutError, ConnectionError, http.client.HTTPException, urllib.error.URLError)) or "transport error;" in str(exc)
                    if not transport:
                        wire_attempt += 1
                    payload = {**payload, "retry_feedback": str(exc)}
                    if operation == "caption" or attempt == 2:
                        raise
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
        kind = "model_output_error" if isinstance(exc, ValueError) or "HTTP 422" in str(exc) else "service_error"
        self.error = f"{kind}: {type(exc).__name__}: {exc}"
        with self._event_lock:
            self.events.append({"event": "aborted", "step": step, "error": self.error})

    def start(self, obs, task):
        self.task = task
        self.instruction = task
        self.index = -1
        self.caption_type = "none"
        try:
            self._plan(obs, 0, "Initial planning")
        except Exception as exc:
            self._abort(exc, 0)

    def _plan(self, obs, step, feedback):
        self.generation += 1
        for operation, future in (("caption", self._caption_future), ("verify", self._verify_future)):
            if future is not None:
                self.events.append({"event": "request_discarded", "operation": operation,
                                    "step": step, "reason": "plan_replaced", "cancelled_before_start": future.cancel()})
        self._caption_future = self._verify_future = None
        self._verify_request = None
        self.events.append({"event": "caption_cleared", "step": step, "reason": feedback,
                            "previous_caption": self.caption, "generation": self.generation})
        # Keep the current subtask while planning; only initial planning uses the full task.
        self.caption, self.caption_type = None, "none"
        payload = {
            "task": self.task, "images": self._images(obs), "skills": self.skills,
            "history": self.history, "feedback": feedback,
            "previous_plan": getattr(self, "phases", []),
            "current_phase": self.index,
        }
        if self.config.async_planner:
            self._plan_future = self._submit(self._plan_executor, "plan", payload, step)
            self.events.append({"event": "plan_pending", "step": step, "generation": self.generation})
        else:
            self._install_plan(self._call("plan", payload, step), obs, step)

    def _install_plan(self, value, obs, step):
        self.phases = parse_plan(value, self.skills)
        self.index = 0
        self._enter(obs, step)

    def _enter(self, obs, step):
        if self._caption_future is not None:
            self.events.append({"event": "request_discarded", "operation": "caption", "step": step,
                                "reason": "phase_changed", "cancelled_before_start": self._caption_future.cancel()})
            self._caption_future = None
        self.generation += 1  # New phase invalidates queued work from its predecessor.
        self.phase = self.phases[self.index]
        self.phase_start_images = self._images(obs)
        self.phase_start_step = step
        self.last_check = step
        self.success_streak = 0
        self.failure_streak = 0
        self.instruction = self.phase["instruction"]
        self.caption, self.caption_type = None, "none"
        if self.config.caption_mode == "instructor":
            self._caption_future = self._submit(self._caption_executor, "caption", {"images": self.phase_start_images,
                                        "task_description": self.instruction}, step)
        self.events.append({"event": "phase_enter", "step": step, "generation": self.generation,
                            "phase": self.index, "instruction": self.instruction,
                            "caption": self.caption, "caption_type": self.caption_type,
                            "success_criteria": self.phase["success_criteria"], "replans": self.replans})

    def _replan(self, obs, step, feedback):
        self.history.append({"instruction": self.instruction, "outcome": "not_confirmed",
                             "evidence": feedback})
        if self.replans >= self.config.max_replans:
            # Both allowed repair plans have run; another confirmed failure ends the episode.
            self.terminate(step, "stage_replan_budget_exhausted: " + feedback)
            return
        self.replans += 1
        self.total_replans += 1
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
                    # Approved policy: do not use simulator truth to request extra planning.
                    # Future research: diagnose/recover visual-vs-environment disagreement.
                    return self.terminate(step, "monitor_environment_disagreement")
                self.events.append({"event": "budget_reset", "step": step,
                                    "previous_replans": self.replans, "reason": "confirmed_phase_success"})
                self.replans = 0
                self.index += 1
                self._enter(obs, step)
                return True
            feedback = payload
            self._replan(obs, step, feedback)
            return True
        except Exception as exc:
            self._abort(exc, step)
        return self.terminate(step, self.error)

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
            response = future.result()
            verdict = parse_verdict(response)
            if response.get("request_attempts", 1) > 1:
                self.success_streak = self.failure_streak = 0
            # A late response from a previous phase must never alter the new phase.
            if request["phase"] != self.index or request["generation"] != self.generation:
                with self._event_lock:
                    self.events.append({"event": "stale_verify_discarded", "step": step})
                return
            confident = verdict["confidence"] >= self.config.confidence_threshold
            self.success_streak = self.success_streak + 1 if confident and verdict["status"] == "success" else 0
            self.failure_streak = self.failure_streak + 1 if confident and verdict["status"] == "failure" else 0
            self.events.append({"event": "verdict_applied", "step": step, "verdict": verdict,
                                "success_streak": self.success_streak, "failure_streak": self.failure_streak})
            if self.success_streak >= self.config.confirmations:
                self._pending_transition = ("advance", verdict)
            else:
                if self.failure_streak >= self.config.confirmations:
                    self._pending_transition = ("replan", "Failure: Monitor confirmed failure: " + verdict["evidence"])
        except Exception as exc:
            # A failed observation is not evidence of phase success/failure; retry once.
            self.success_streak = self.failure_streak = 0
            self.events.append({"event": "verify_error", "step": step, "error": str(exc)})
            self._abort(exc, step)  # _call already exhausted the one allowed retry.

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
                return self.terminate(step, self.error)
        self._poll_verifier(obs, step)
        if self.stopped:
            return self.terminate(step, self.error) if at_chunk_boundary else False
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
                # Caption is optional; keep executing this phase without stale guidance.
                self.caption, self.caption_type = None, "none"
                self.events.append({"event": "caption_error", "step": step, "error": str(exc)})
        # Retry failed captions once per available chunk boundary, with fresh observations.
        # There is no retry-count limit; phase/version changes invalidate the old request.
        if (at_chunk_boundary and self.config.caption_mode == "instructor"
                and self.caption is None and self._caption_future is None):
            self._caption_future = self._submit(self._caption_executor, "caption", {"images": self._images(obs),
                                        "task_description": self.instruction}, step)
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
            "start_images": self.phase_start_images, "images": self._images(obs),
        }
        self._verify_request = {"phase": self.index, "generation": self.generation, "step": step}
        if self.config.async_verifier:
            self._verify_future = self._submit(self._executor, "verify", payload, step)
        else:
            self._verify_future = Future()
            try:
                self._verify_future.set_result(self._call("verify", payload, step))
            except Exception as exc:
                self._verify_future.set_exception(exc)
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
                "task": self.task, "replans": self.replans, "total_replans": self.total_replans, "events": events,
                "plan_finished": self.finished,
                "inference_seconds": inference_seconds,
                "note": "Visual completion is not benchmark success. Verifier runs asynchronously and transitions apply at chunk boundaries."}
