"""Local generic VLM planner/verifier; run in the existing instructor environment.

Use a general instruction-tuned VLM, NOT the caption-only DeMiAn Instructor
checkpoint. Outputs are validated data, never executable code.
"""
import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from hierarchy_trace import EventLog

import torch
from qwen_instructor_server import CaptionModel, decode_images
from hierarchical_controller import parse_object, parse_plan, parse_verdict


CAPTION_TYPE_DESCRIPTIONS = """The five caption choices use the DeMiAn caption taxonomy:
- current_action: physical-motion caption describing the immediate robot motion.
- environment-simple: simple scene caption describing the relevant objects and environment state.
- agent_arm_state: arm-pose caption describing the robot arm, gripper, and end-effector state.
- reasoning-short: short reasoning caption connecting the current observation, constraint, and next action.
- none: no auxiliary caption is reliable; return an empty caption.
"""


def plan_system():
    return f"""You plan short executable subtasks for a robot controlled by a language-conditioned VLA.
Inspect the labeled left, right, and wrist views independently. Briefly report what
each view shows in view_observations. If the target is visible in ANY view, treat it
as visible; occlusion in another view does not negate positive visual evidence.
Use the complete user task, prior outcomes and
feedback to plan only the REMAINING work. One phase is a skill-sized instruction
(e.g. place an object at a destination), not individual motor commands. Preserve
task constraints and ordering dependencies. Do not invent unavailable objects or
robot capabilities. A missing or empty skills list means NO extra skill-name
constraint, not that the robot has no capabilities. If a non-empty skills list is
supplied, each skill must match it exactly; it describes supported skills, not a
task-specific plan. Temporary lack of visibility
or an inconvenient robot pose does not make the task unplannable: first add a grounded
phase to search, move to a useful viewpoint, approach the workspace, or open an
occluding container as appropriate. Return {{"error": "unplannable", "reason":
"observable reason"}} only when the requested task clearly conflicts with the supplied
skills, available objects, or robot capabilities; missing visual evidence alone is
insufficient.
Output ONLY a JSON object: {{"view_observations": {{"left": "...", "right": "...", "wrist": "..."}},
"phases": [{{"skill": "...", "instruction": "...",
"success_criteria": "observable completed outcome, including release when needed",
"caption_type": "current_action", "caption": "short grounded action/scene guidance"}}]}}.
Return a complete non-empty list of phases when feasible. There is no fixed phase-count cap. Instructions are English and at most 300 characters.
Captions are at most 400 characters; success_criteria at most 600.
{CAPTION_TYPE_DESCRIPTIONS}
Use caption_type none and an empty caption when no reliable extra information is available.
Do not assume future actions already happened. Reasoning should identify a concrete
dependency or constraint; do not write long speculative explanations. Never return Python code."""

VERIFY_SYSTEM = """You judge whether a robot's CURRENT subtask is completed using before/current images.
You are not the planner. A caption is guidance, NOT evidence of success. The first
three images are at phase start; the last three are the current left, right, wrist views.
Inspect all three labeled current views independently. A visible unmet condition in
ANY view rules out success, even if another view looks complete. Evaluate every
requirement in success_criteria using the current observations.
Do not confuse reaching with grasping, proximity with correct placement, or holding
an object over a receptacle with placing and releasing it. Placement requires the
complete object footprint to be visibly supported by the target surface after release;
overlap in one image, edge contact, hovering, or a gripper still holding it is not
enough. If support or release cannot be verified, choose uncertain. A button being approached
does not prove a machine was activated. If the necessary condition is occluded or
not visually observable, choose uncertain. Use failure for visible loss of a required
condition or an invalid plan requiring recovery, not merely unfinished movement.
Output ONLY {"status": "ongoing|success|failure|uncertain", "confidence": 0.0,
"evidence": "brief evidence from each relevant view and any unresolved condition"}.
confidence is a heuristic score, not a calibrated probability. evidence <= 600 characters."""


class InvalidModelOutput(ValueError):
    """The model responded, but its structured output is unusable."""


class HierarchyModel(CaptionModel):
    def __init__(self, model_path, device, max_new_tokens):
        super().__init__(model_path, device, max_new_tokens)

    @torch.no_grad()
    def structured(self, images, payload, operation):
        system = plan_system() if operation == "plan" else VERIFY_SYSTEM
        if operation == "plan" and payload.get("caption_mode") != "planner":
            system += (
                " The controller uses a separate caption source. Set caption_type to none "
                "and caption to an empty string for every phase."
            )
        labels = (["phase start left", "phase start right", "phase start wrist"]
                  if operation == "verify" else []) + ["current left", "current right", "current wrist"]
        content = []
        for label, im in zip(labels, images):
            content.extend([{"type": "text", "text": label}, {"type": "image", "image": im}])
        model_payload = {key: value for key, value in payload.items()
                         if key not in {"request_id", "generation", "session_id", "deadline_unix"}}
        content.append({"type": "text", "text": json.dumps(model_payload, ensure_ascii=False)})
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        generated = self._generate(images, messages)
        try:
            value = parse_object(generated)
            if operation == "plan":
                if value.get("error") == "unplannable":
                    raise ValueError("Planner could not plan reliably: " + str(value.get("reason", "")))
                if payload.get("caption_mode") != "planner" and isinstance(value.get("phases"), list):
                    # Instructor owns captions; Qwen's unused caption fields must not reject a valid plan.
                    value["phases"] = [{**phase, "caption_type": "none", "caption": ""}
                                       if isinstance(phase, dict) else phase for phase in value["phases"]]
                result = {"phases": parse_plan(value, payload.get("skills", []))}
            else:
                result = parse_verdict(value)
            return {**result, "raw_output": generated}
        except ValueError as exc:
            # Preserve invalid output too. Retry belongs to the client: two attempts total.
            raise InvalidModelOutput(json.dumps({"validation_error": str(exc),
                                                 "raw_output": generated})) from exc

    def _generate(self, images, messages):
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        inputs = self.processor(text=[text], images=images, return_tensors="pt", padding=True).to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                     do_sample=False, use_cache=True)
        return self.processor.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:],
                                               skip_special_tokens=True).strip()


def make_handler(model, trace=None):
    # Concurrent HTTP receipt does not mean concurrent GPU generation. One model owner
    # avoids extra VRAM; deadline checks discard requests that expired while queued.
    inference_lock = threading.Lock()
    receipt_lock = threading.Lock()
    latest_generation = {}
    completed = {}  # Last completed request per session/operation, not every monitor frame.
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, value):
            if trace is not None:
                trace.append({"event": "http_response", "status": status, "response": value,
                              "request_id": getattr(self, "request_id", None)})
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                logging.warning("Client disconnected; response retained in service trace")

        def do_GET(self):
            self.respond(200 if self.path == "/healthz" else 404, {"service": "hierarchy"})

        def do_POST(self):
            operation = self.path.strip("/")
            if operation not in {"plan", "verify"}:
                self.respond(404, {"error": "Use /plan or /verify"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8_000_000:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON request object")
                self.request_id = body.get("request_id")
                if trace is not None:
                    trace.append({"event": "http_request", "operation": operation, "payload": body})
                session, generation = body.get("session_id", "legacy"), body.get("generation", 0)
                if not isinstance(session, str) or type(generation) is not int:
                    raise ValueError("Invalid request identity")
                with receipt_lock:
                    latest_generation[session] = max(generation, latest_generation.get(session, generation))
                images = body.pop("images", None)
                start = body.pop("start_images", [])
                if not isinstance(images, list) or len(images) != 3:
                    raise ValueError("Expected three current images")
                if operation == "verify" and (not isinstance(start, list) or len(start) != 3):
                    raise ValueError("Expected three phase-start images")
                decoded = decode_images({"images": (start if operation == "verify" else []) + images})
            except (ValueError, TypeError, OSError) as exc:
                self.respond(400, {"error": str(exc), "error_type": "invalid_request"})
                return
            try:
                received = time.time()
                with inference_lock:
                    started = time.time()
                    with receipt_lock:
                        stale = generation < latest_generation[session]
                    if stale or body.get("deadline_unix", float("inf")) < started:
                        self.respond(408, {"error": "stale_or_expired_before_inference", "request_id": body.get("request_id"),
                                           "queue_seconds": started - received})
                        return
                    cache_key = (session, operation)
                    cached = completed.get(cache_key)
                    if self.request_id and cached and cached[0] == self.request_id:
                        status, result = cached[1], cached[2]
                    else:
                        try:
                            result = model.structured(decoded, body, operation)
                            status = 200
                        except InvalidModelOutput as exc:
                            status, result = 422, {"error": str(exc), "error_type": "invalid_model_output"}
                        except Exception as exc:
                            logging.exception("Hierarchy inference failed")
                            status, result = 500, {"error": str(exc), "error_type": "inference_error"}
                        result["service_timing"] = {"received_unix": received, "started_unix": started,
                                                    "queue_seconds": started - received,
                                                    "inference_seconds": time.time() - started}
                        if self.request_id:
                            completed[cache_key] = (self.request_id, status, result)
                self.respond(status, result)
            except InvalidModelOutput as exc:
                logging.exception("Hierarchy output validation failed")
                self.respond(422, {"error": str(exc), "error_type": "invalid_model_output"})
            except ValueError as exc:
                self.respond(400, {"error": str(exc), "error_type": "invalid_request"})
            except Exception as exc:
                logging.exception("Hierarchy request failed")
                self.respond(500, {"error": str(exc), "error_type": "inference_error"})

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--trace-dir", default="hierarchy-service-" + time.strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    model = HierarchyModel(args.model_path, args.device, args.max_new_tokens)
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(model, EventLog(args.trace_dir, vars(args), retain=False)))
    logging.info("Hierarchy server on %s:%s", args.bind, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
