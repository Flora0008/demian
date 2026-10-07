"""Local generic VLM planner/verifier; run in the existing instructor environment.

Use a general instruction-tuned VLM, NOT the caption-only DeMiAn Instructor
checkpoint. Outputs are validated data, never executable code.
"""
import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from qwen_instructor_server import CaptionModel, decode_images
from hierarchical_controller import MAX_PHASES, parse_object, parse_plan, parse_verdict


CAPTION_TYPE_DESCRIPTIONS = """The five caption choices use the DeMiAn caption taxonomy:
- current_action: physical-motion caption describing the immediate robot motion.
- environment-simple: simple scene caption describing the relevant objects and environment state.
- agent_arm_state: arm-pose caption describing the robot arm, gripper, and end-effector state.
- reasoning-short: short reasoning caption connecting the current observation, constraint, and next action.
- none: no auxiliary caption is reliable; return an empty caption.
"""


def plan_system(max_phases):
    return f"""You plan short executable subtasks for a robot controlled by a language-conditioned VLA.
Observe the current camera images. Use the complete user task, prior outcomes and
feedback to plan only the REMAINING work. One phase is a skill-sized instruction
(e.g. place an object at a destination), not individual motor commands. Preserve
task constraints and ordering dependencies. Do not invent unavailable objects or
robot capabilities. If a skills list is supplied, each skill must match it exactly;
it describes supported skills, not a task-specific plan. If infeasible, return an
empty phases list (the controller will stop with a validation error).
Output ONLY a JSON object: {"phases": [{"skill": "...", "instruction": "...",
"success_criteria": "observable completed outcome, including release when needed",
"caption_type": "current_action", "caption": "short grounded action/scene guidance"}]}.
Return 1 to {max_phases} phases when feasible. Instructions are English and at most 300 characters.
Captions are at most 400 characters; success_criteria at most 600.
{CAPTION_TYPE_DESCRIPTIONS}
Use caption_type none and an empty caption when no reliable extra information is available.
Do not assume future actions already happened. Reasoning should identify a concrete
dependency or constraint; do not write long speculative explanations. Never return Python code."""

VERIFY_SYSTEM = """You judge whether a robot's CURRENT subtask is completed using before/current images.
You are not the planner. A caption is guidance, NOT evidence of success. The first
three images are at phase start; the last three are the current left, right, wrist views.
Evaluate every requirement in success_criteria using the current observations.
Do not confuse reaching with grasping, proximity with correct placement, or holding
an object over a receptacle with placing and releasing it. A button being approached
does not prove a machine was activated. If the necessary condition is occluded or
not visually observable, choose uncertain. Use failure for visible loss of a required
condition or an invalid plan requiring recovery, not merely unfinished movement.
Output ONLY {"status": "ongoing|success|failure|uncertain", "confidence": 0.0,
"evidence": "brief observable evidence and any unresolved condition"}.
confidence is a heuristic score, not a calibrated probability. evidence <= 600 characters."""


class HierarchyModel(CaptionModel):
    def __init__(self, model_path, device, max_new_tokens, max_phases=MAX_PHASES):
        super().__init__(model_path, device, max_new_tokens)
        self.max_phases = max_phases

    @torch.no_grad()
    def structured(self, images, payload, operation):
        requested_max = payload.get("max_phases", self.max_phases)
        if not isinstance(requested_max, int) or not 1 <= requested_max <= self.max_phases:
            raise ValueError(f"max_phases must be an integer in [1, {self.max_phases}]")
        system = plan_system(requested_max) if operation == "plan" else VERIFY_SYSTEM
        if operation == "plan" and payload.get("caption_mode") != "planner":
            system += (
                " The controller uses a separate caption source. Set caption_type to none "
                "and caption to an empty string for every phase."
            )
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": json.dumps(payload, ensure_ascii=False)})
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        inputs = self.processor(text=[text], images=images, return_tensors="pt", padding=True).to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                     do_sample=False, use_cache=True)
        generated = self.processor.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:],
                                                    skip_special_tokens=True).strip()
        value = parse_object(generated)
        if operation == "plan":
            return {"phases": parse_plan(value, payload.get("skills", []), requested_max)}
        return parse_verdict(value)


def make_handler(model):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, value):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

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
                images = body.pop("images", None)
                start = body.pop("start_images", [])
                if not isinstance(images, list) or len(images) != 3:
                    raise ValueError("Expected three current images")
                if operation == "verify" and (not isinstance(start, list) or len(start) != 3):
                    raise ValueError("Expected three phase-start images")
                decoded = decode_images({"images": (start if operation == "verify" else []) + images})
                self.respond(200, model.structured(decoded, body, operation))
            except Exception as exc:
                logging.exception("Hierarchy request failed")
                self.respond(422, {"error": str(exc)})

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-phases", type=int, default=MAX_PHASES,
                        choices=range(1, MAX_PHASES + 1))
    args = parser.parse_args()
    model = HierarchyModel(args.model_path, args.device, args.max_new_tokens, args.max_phases)
    server = HTTPServer((args.bind, args.port), make_handler(model))
    logging.info("Hierarchy server on %s:%s", args.bind, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
