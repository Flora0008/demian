#!/usr/bin/env python3
"""Qwen3.5-2B VLM caption server with image support.

Accepts 3 camera images (base64 JPEG) + task description via HTTP POST,
generates a caption using the fine-tuned VLM with KV caching.

Must be run from a Python environment with latest transformers that
supports the qwen3_5 model type.

Usage:
    python scripts/eval/qwen_instructor_server.py \
        --model-path checkpoints/qwen_instructor/merged \
        --port 8001 --device cuda:1
"""

import argparse
import base64
import io
import json
import logging
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


class CaptionModel:
    SYSTEM_PROMPT = (
        "You are a robot action instructor. Given camera images and a task description "
        "for a robotic manipulation task, generate a concise caption describing the "
        "robot's action and scene. Respond with only the caption text, nothing else."
    )

    def __init__(self, model_path: str, device: str = "cuda:1", max_new_tokens: int = 128):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("Loading model: %s on %s", model_path, device)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device).eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        logger.info("Model loaded successfully.")

    @torch.no_grad()
    def generate(self, images: list[Image.Image], task_description: str) -> str:
        """Generate caption from N camera images + task description.

        N is whatever the client sends — 3 for RoboCasa (agentview L/R + wrist),
        2 for MolmoBot/MolmoSpaces (exo + wrist). The instructor was trained with
        the matching camera count.
        """
        content = [{"type": "image", "image": img} for img in images]
        content.append({
            "type": "text",
            "text": f"Task: {task_description}\nGenerate a concise caption describing the robot's action and scene for this task.",
        })
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt", padding=True).to(self.device)
        output_ids = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False, use_cache=True,
        )
        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


def decode_images(body: dict) -> list[Image.Image] | None:
    """Decode base64-encoded JPEG images from the request body (N >= 1; 3 for
    RoboCasa, 2 for MolmoBot/MolmoSpaces)."""
    images_b64 = body.get("images")
    if not images_b64:
        return None
    images = []
    for b64_str in images_b64:
        img_bytes = base64.b64decode(b64_str)
        images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    return images


def make_handler(caption_model):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            task_desc = body.get("task_description", "")

            t0 = time.monotonic()
            images = decode_images(body)
            if images is None:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "Expected 3 base64 images in images field"}')
                return

            caption = caption_model.generate(images, task_desc)
            dt = time.monotonic() - t0

            logger.info("[gen] %.2fs \"%s\"", dt, caption[:60])
            response = json.dumps({"caption": caption, "generation_time_ms": dt * 1000})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response.encode())

        def do_GET(self):
            if self.path == "/healthz":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    model = CaptionModel(args.model_path, args.device, args.max_new_tokens)

    # Warmup with a dummy image.
    dummy_img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    t0 = time.monotonic()
    text = model.generate([dummy_img, dummy_img, dummy_img], "Close the fridge doors.")
    logger.info("[warmup] %.1fs: %s", time.monotonic() - t0, text[:80])

    server = HTTPServer(("0.0.0.0", args.port), make_handler(model))
    logger.info("Caption server listening on port %d", args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
