#!/usr/bin/env python3
"""Extract the 164 GUI frames and create a numbered preview video.

The sampling method exactly matches generate_masks.py:
np.unique(np.rint(np.linspace(0, total_frames - 1, target_frames))).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract uniformly sampled GUI frames and create a numbered MP4."
    )
    parser.add_argument("--video", required=True, help="Original input video.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for frames, manifest, and numbered video.",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=164,
        help="Number of GUI frames to extract (default: 164).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Playback rate of the numbered preview video (default: 10).",
    )
    parser.add_argument(
        "--output-video",
        default=None,
        help="Optional MP4 path. Default: <output-dir>/<video-name>_164_numbered.mp4",
    )
    parser.add_argument(
        "--gui-only",
        action="store_true",
        help="Write only GUI frame 0-163, without the original source-frame number.",
    )
    return parser.parse_args()


def sample_indices(total: int, target: int) -> np.ndarray:
    if total <= 0:
        raise ValueError("The video contains no readable frames.")
    if target <= 0:
        raise ValueError("--target-frames must be greater than zero.")
    if target >= total:
        return np.arange(total, dtype=int)
    return np.unique(np.rint(np.linspace(0, total - 1, target)).astype(int))


def add_frame_number(
    frame: np.ndarray,
    gui_index: int,
    gui_total: int,
    source_index: int,
    gui_only: bool,
) -> None:
    lines = [f"GUI frame: {gui_index} / {gui_total - 1}"]
    if not gui_only:
        lines.append(f"Original video frame: {source_index}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.85
    thickness = 2
    line_height = 38
    margin = 18
    widths = [cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines]
    box_width = max(widths) + 2 * margin
    box_height = len(lines) * line_height + margin

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_width, 10 + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    for position, line in enumerate(lines):
        y = 10 + margin + 24 + position * line_height
        cv2.putText(
            frame,
            line,
            (10 + margin, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    frames_dir = output_dir / "frames"
    manifest_path = output_dir / "frames_manifest.csv"

    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero.")

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = sample_indices(total, args.target_frames)
    wanted = {int(value): position for position, value in enumerate(indices)}
    selected_total = len(indices)

    output_video = (
        Path(args.output_video).expanduser().resolve()
        if args.output_video
        else output_dir / f"{video_path.stem}_{selected_total}_numbered.mp4"
    )
    output_video.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    rows: list[dict[str, object]] = []
    source_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if source_index not in wanted:
                source_index += 1
                continue

            gui_index = wanted[source_index]
            frame_file = f"frame_{gui_index:05d}.png"
            frame_path = frames_dir / frame_file
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Could not save frame: {frame_path}")

            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(args.fps),
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create video: {output_video}")

            numbered = frame.copy()
            add_frame_number(
                numbered, gui_index, selected_total, source_index, args.gui_only
            )
            writer.write(numbered)
            rows.append(
                {
                    "frame_index": gui_index,
                    "source_frame_index": source_index,
                    "frame_file": frame_file,
                    "source_video": video_path.name,
                }
            )
            source_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if len(rows) != selected_total:
        raise RuntimeError(
            f"Expected {selected_total} sampled frames but read {len(rows)}. "
            "The video metadata may report an incorrect frame count."
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        csv_writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index",
                "source_frame_index",
                "frame_file",
                "source_video",
            ],
        )
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    print(f"[OK] Extracted frames: {len(rows)}")
    print(f"[OK] Frames folder: {frames_dir}")
    print(f"[OK] Frame mapping: {manifest_path}")
    print(f"[OK] Numbered video: {output_video}")


if __name__ == "__main__":
    main()
