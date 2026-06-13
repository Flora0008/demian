#!/usr/bin/env python3
"""Generate GT captions using existing first_frames (no video lookup).
"""
from __future__ import annotations
import argparse, json, logging, re, time
from pathlib import Path

import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


BENCHMARKS = [
    ("pick_std",    "tier3_pick_std_procthor10k_200ep"),
    ("pnp_std",     "tier3_pnp_std_procthor10k_200ep"),
    ("pick_hard",   "tier3_pick_hard_procthor10k_200ep"),
    ("pnp_hard",    "tier3_pnp_hard_procthor10k_200ep"),
    ("pnp_nextto",  "tier3_pnp_nextto_procthor10k_200ep"),
    ("pick_ood",    "tier3_pick_hard_holodeck_200ep"),
    ("pnp_ood",     "tier3_pnp_hard_holodeck_200ep"),
    ("nextto_ood",  "tier3_pnp_nextto_holodeck_200ep"),
    ("pnp_color",   "tier3_pnp_color_objaverse_200ep"),
]

SYSTEM_PROMPT = (
    "You are a robot action instructor. Given camera images and a task description "
    "for a robotic manipulation task, generate a concise caption describing the "
    "robot's action and scene. Respond with only the caption text, nothing else."
)


def parse_pickup_obj(prompt: str) -> str:
    m = re.match(r"pick up (?:the|some|a) (.+?) and place it next to (?:the )?(.+)", prompt, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.match(r"pick up (?:the|some|a) (.+?) and place it (?:in or on|in|on) (?:the )?(.+)", prompt, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.match(r"pick up (?:the|some|a) (.+)", prompt, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return "the object"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instructor-dir", required=True)
    p.add_argument("--benchmarks-dir", default="benchmarks")
    p.add_argument("--caption-key", default="current_action")
    p.add_argument("--output-name", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--rerun", action="store_true")
    args = p.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    log.info(f"Loading instructor: {args.instructor_dir}")
    processor = AutoProcessor.from_pretrained(args.instructor_dir, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        args.instructor_dir, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device).eval()

    benchmarks_dir = Path(args.benchmarks_dir)
    summary = []

    for short, bench in BENCHMARKS:
        bench_dir = benchmarks_dir / bench
        bench_json = bench_dir / "benchmark.json"
        if not bench_json.exists():
            log.warning(f"{short}: missing benchmark.json")
            continue
        eps = json.loads(bench_json.read_text())
        log.info(f"\n=== {short} ({bench}) — {len(eps)} episodes ===")

        out_path = bench_dir / args.output_name
        existing: dict[str, dict[str, str]] = {}
        if out_path.exists() and not args.rerun:
            existing = json.loads(out_path.read_text())
            log.info(f"  loaded {len(existing)} existing captions; will skip those")

        frames_dir = bench_dir / "first_frames"
        wrist_dir = bench_dir / "first_frames_wrist"
        if not frames_dir.exists() or not wrist_dir.exists():
            log.warning(f"  missing first_frames/ or first_frames_wrist/ — skipping benchmark")
            continue

        n_done, n_skip, n_fail = 0, 0, 0
        t0 = time.time()
        for i, ep in enumerate(eps):
            house_idx = ep["house_index"]
            task_desc = ep["language"]["task_description"]
            pickup_obj = parse_pickup_obj(task_desc)
            lookup_key = pickup_obj.lower().strip()

            if (lookup_key in existing and args.caption_key in existing[lookup_key]
                    and not args.rerun):
                n_skip += 1
                continue

            exo_jpg = frames_dir / f"house_{house_idx}.jpg"
            wrist_jpg = wrist_dir / f"house_{house_idx}.jpg"
            if not exo_jpg.exists() or not wrist_jpg.exists():
                log.warning(f"  ep{i} house_{house_idx}: missing frame (exo_exists={exo_jpg.exists()}, wrist_exists={wrist_jpg.exists()})")
                n_fail += 1
                continue

            exo_img = Image.open(exo_jpg).convert("RGB")
            wrist_img = Image.open(wrist_jpg).convert("RGB")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": exo_img},
                    {"type": "image", "image": wrist_img},
                    {"type": "text", "text": (
                        f"Task: {task_desc}\n"
                        "Generate a concise caption describing the robot's action and scene for this task."
                    )},
                ]},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[[exo_img, wrist_img]],
                               return_tensors="pt", padding=True).to(args.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            caption = processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )[0].strip()

            entry = existing.setdefault(lookup_key, {})
            entry[args.caption_key] = caption
            n_done += 1
            if (n_done % 5) == 0:
                out_path.write_text(json.dumps(existing, indent=2))

        out_path.write_text(json.dumps(existing, indent=2))
        elapsed = time.time() - t0
        log.info(f"  saved {len(existing)} → {out_path}  (done={n_done} skip={n_skip} fail={n_fail}, {elapsed:.1f}s)")
        summary.append((short, len(existing), n_done, n_fail))

    log.info("\n=== Summary ===")
    for short, total, new, fail in summary:
        log.info(f"  {short:<14} total={total:>3}  new={new:>3}  fail={fail:>3}")


if __name__ == "__main__":
    main()
