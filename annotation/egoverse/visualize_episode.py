#!/usr/bin/env python3
"""Visualize an EgoVerse episode from local zarr data.

Reads frames, annotations, and (optionally) pose data from a zarr episode
directory. Saves an annotated MP4 video and/or a grid of sampled frames.

Usage:
    python visualize_episode.py --episode 69a6625fb29228ae08b91423
    python visualize_episode.py --episode 69a6625fb29228ae08b91423 --save_grid
    python visualize_episode.py --episode 69a6625fb29228ae08b91423 --fps 15 --output_dir vis_output
    python visualize_episode.py --data_dir /path/to/egoverse_dataset --episode EPISODE_ID
"""

import argparse
import json
import os

import cv2
import numpy as np
import zarr


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_BASE = (
    "/workspace"
    if os.path.isdir("/workspace")
    else "/path/to"
)
DEFAULT_DATA_DIR = os.path.join(_BASE, "data")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_episode(episode_dir: str) -> dict:
    """Load zarr episode metadata, frames, annotations, and poses."""
    store = zarr.open(episode_dir, mode="r")

    with open(os.path.join(episode_dir, "zarr.json")) as f:
        meta = json.load(f)["attributes"]

    # Decode annotations
    raw_annotations = store["annotations"][:]
    annotations = []
    for a in raw_annotations:
        text = a if isinstance(a, str) else a.decode("utf-8")
        annotations.append(json.loads(text))

    # Discover available cameras
    cameras = sorted(k for k in store.keys() if k.startswith("images."))

    return {
        "meta": meta,
        "store": store,
        "annotations": annotations,
        "cameras": cameras,
        "total_frames": meta["total_frames"],
    }


def decode_frame(store, camera: str, idx: int) -> np.ndarray:
    """Decode a single JPEG frame from the zarr store into BGR numpy array."""
    raw = store[camera][idx]
    # Unwrap nested numpy arrays (zarr v3 variable_length_bytes quirk)
    while isinstance(raw, np.ndarray) and raw.dtype == object:
        raw = raw.item()
    jpeg_bytes = bytes(raw)
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Annotation overlay
# ---------------------------------------------------------------------------

def get_annotation_at_frame(annotations: list, frame_idx: int) -> str:
    """Return the annotation text for the given frame index, or empty string."""
    for ann in annotations:
        if ann["start_idx"] <= frame_idx < ann["end_idx"]:
            return ann["text"]
    return ""


def draw_text_overlay(frame: np.ndarray, text: str, position: str = "bottom") -> np.ndarray:
    """Draw text with a background bar on the frame."""
    if not text:
        return frame
    frame = frame.copy()
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, h / 900)
    thickness = max(1, int(h / 500))

    # Wrap text if too long
    max_chars = int(w / (font_scale * 12))
    lines = []
    while text:
        if len(text) <= max_chars:
            lines.append(text)
            break
        split = text[:max_chars].rfind(" ")
        if split == -1:
            split = max_chars
        lines.append(text[:split])
        text = text[split:].lstrip()

    line_height = int(30 * font_scale)
    total_height = line_height * len(lines) + 10

    if position == "bottom":
        y_start = h - total_height - 5
    else:
        y_start = 5

    # Background bar
    cv2.rectangle(frame, (0, y_start), (w, y_start + total_height), (0, 0, 0), -1)

    for i, line in enumerate(lines):
        y = y_start + (i + 1) * line_height
        cv2.putText(frame, line, (8, y), font, font_scale, (255, 255, 255), thickness)

    return frame


def draw_frame_info(frame: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
    """Draw frame counter in the top-right corner."""
    frame = frame.copy()
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, h / 900)
    thickness = max(1, int(h / 500))
    text = f"t={frame_idx}/{total_frames}"
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (w - tw - 16, 4), (w - 4, th + 16), (0, 0, 0), -1)
    cv2.putText(frame, text, (w - tw - 12, th + 10), font, font_scale, (255, 255, 255), thickness)
    return frame


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------

def save_video(episode: dict, output_path: str, camera: str = "images.front_1",
               fps: int = 30, skip: int = 1):
    """Render annotated video for one camera view."""
    store = episode["store"]
    total = episode["total_frames"]
    annotations = episode["annotations"]

    first_frame = decode_frame(store, camera, 0)
    h, w = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for i in range(0, total, skip):
        frame = decode_frame(store, camera, i)
        ann_text = get_annotation_at_frame(annotations, i)
        frame = draw_frame_info(frame, i, total)
        frame = draw_text_overlay(frame, ann_text)
        writer.write(frame)
        if i % 200 == 0:
            print(f"  frame {i}/{total}")

    writer.release()
    print(f"  Saved video: {output_path}")


# ---------------------------------------------------------------------------
# Grid visualization
# ---------------------------------------------------------------------------

def save_grid(episode: dict, output_path: str, camera: str = "images.front_1",
              n_samples: int = 16, cols: int = 4):
    """Save a grid of uniformly sampled frames with annotations."""
    store = episode["store"]
    total = episode["total_frames"]
    annotations = episode["annotations"]

    indices = np.linspace(0, total - 1, n_samples, dtype=int)
    frames = []

    for idx in indices:
        frame = decode_frame(store, camera, idx)
        ann_text = get_annotation_at_frame(annotations, idx)
        frame = draw_frame_info(frame, idx, total)
        frame = draw_text_overlay(frame, ann_text)
        frames.append(frame)

    rows_count = (n_samples + cols - 1) // cols
    h, w = frames[0].shape[:2]

    # Pad to fill grid
    while len(frames) < rows_count * cols:
        frames.append(np.zeros((h, w, 3), dtype=np.uint8))

    rows = []
    for r in range(rows_count):
        row_frames = frames[r * cols : (r + 1) * cols]
        rows.append(np.hstack(row_frames))
    grid = np.vstack(rows)

    cv2.imwrite(output_path, grid)
    print(f"  Saved grid: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize an EgoVerse episode")
    parser.add_argument("--episode", type=str, required=True, help="Episode ID")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Root directory containing episode folders")
    parser.add_argument("--output_dir", type=str, default="vis_output",
                        help="Output directory for generated files")
    parser.add_argument("--camera", type=str, default=None,
                        help="Camera to visualize (default: all available)")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--skip", type=int, default=1, help="Frame skip for video")
    parser.add_argument("--save_grid", action="store_true",
                        help="Save a grid of sampled frames")
    parser.add_argument("--grid_samples", type=int, default=16,
                        help="Number of frames to sample for grid")
    parser.add_argument("--no_video", action="store_true",
                        help="Skip video generation (useful with --save_grid)")
    args = parser.parse_args()

    episode_dir = os.path.join(args.data_dir, args.episode)
    if not os.path.isdir(episode_dir):
        print(f"Episode not found: {episode_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading episode: {args.episode}")
    episode = load_episode(episode_dir)
    meta = episode["meta"]

    print(f"  Task: {meta.get('task_description', 'N/A')}")
    print(f"  Frames: {episode['total_frames']} @ {meta.get('fps', '?')} fps")
    print(f"  Duration: {meta.get('duration', '?')}s")
    print(f"  Cameras: {episode['cameras']}")
    print(f"  Annotations: {len(episode['annotations'])} segments")
    for i, ann in enumerate(episode["annotations"]):
        print(f"    [{i}] {ann['start_idx']}–{ann['end_idx']}: {ann['text']}")

    cameras = [args.camera] if args.camera else episode["cameras"]

    for cam in cameras:
        cam_label = cam.replace("images.", "")
        print(f"\nProcessing camera: {cam}")

        if not args.no_video:
            video_path = os.path.join(args.output_dir, f"{args.episode}_{cam_label}.mp4")
            save_video(episode, video_path, camera=cam, fps=args.fps, skip=args.skip)

        if args.save_grid:
            grid_path = os.path.join(args.output_dir, f"{args.episode}_{cam_label}_grid.jpg")
            save_grid(episode, grid_path, camera=cam, n_samples=args.grid_samples)

    print("\nDone.")


if __name__ == "__main__":
    main()
