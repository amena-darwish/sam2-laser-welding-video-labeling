#!/usr/bin/env python3
"""Create a colored overlay video from a GUI final-dataset export."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


LABEL_COLORS = {
    "weld": (0, 0, 255),       # red in BGR
    "plasma": (0, 255, 255),   # yellow
    "spatter": (0, 255, 0),    # green
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an MP4 overlay from labels_final.csv and final_masks/."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Exported labels_final.csv from the labeling GUI.",
    )
    parser.add_argument(
        "--frames",
        required=True,
        help="Folder containing the exported GUI frames.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output MP4 path. Default: beside labels_final.csv.",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Output playback rate.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask opacity (0 to 1).")
    parser.add_argument(
        "--expected-frames", type=int, default=164, help="Expected GUI frame count."
    )
    return parser.parse_args()


def as_int(value: object, default: int) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def resolve_from_run(run_dir: Path, value: str, fallback_dir: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    direct = run_dir / path
    if direct.exists():
        return direct
    return run_dir / fallback_dir / path.name


def draw_legend(frame: np.ndarray, gui_index: int, total: int) -> None:
    lines = [(f"GUI frame: {gui_index} / {total - 1}", (255, 255, 255))]
    lines.extend((label, color) for label, color in LABEL_COLORS.items())
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (265, 155), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    for index, (text, color) in enumerate(lines):
        y = 38 + index * 32
        cv2.putText(
            frame, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.68, color, 2, cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    csv_path = Path(args.csv).expanduser().resolve()
    frames_dir = Path(args.frames).expanduser().resolve()
    run_dir = csv_path.parent
    if not csv_path.is_file():
        raise FileNotFoundError(f"Final label table not found: {csv_path}")
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    rows_by_frame: dict[str, list[dict[str, str]]] = defaultdict(list)
    frame_indices: dict[str, int] = {}
    kept_mask_paths: set[Path] = set()
    for row in rows:
        frame_file = Path(str(row.get("frame_file", ""))).name
        if frame_file:
            frame_indices[frame_file] = as_int(row.get("frame_index"), len(frame_indices))

        label = str(row.get("label", "")).strip().lower()
        if label not in LABEL_COLORS:
            continue
        if not frame_file:
            continue
        rows_by_frame[frame_file].append(row)
        kept_mask_paths.add(
            resolve_from_run(run_dir, str(row.get("mask_path", "")), "final_masks").resolve()
        )

    # Keep labels_final.csv unchanged. Remove only non-target/stale image masks
    # from final_masks; working masks elsewhere are never touched.
    final_masks_dir = run_dir / "final_masks"
    removed_masks = 0
    if final_masks_dir.is_dir():
        for mask_path in final_masks_dir.iterdir():
            if (
                mask_path.is_file()
                and mask_path.suffix.lower() in IMAGE_SUFFIXES
                and mask_path.resolve() not in kept_mask_paths
            ):
                mask_path.unlink()
                removed_masks += 1
    print(f"[INFO] Removed {removed_masks} non-target/stale masks from: {final_masks_dir}")

    frame_files = sorted(
        (path for path in frames_dir.iterdir()
         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: (frame_indices.get(path.name, 10**9), path.name),
    )
    if not frame_files:
        raise RuntimeError(f"No frame images found in: {frames_dir}")
    if args.expected_frames > 0 and len(frame_files) != args.expected_frames:
        print(f"[WARNING] Expected {args.expected_frames} frames, found {len(frame_files)}.")

    first = cv2.imread(str(frame_files[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Cannot read frame: {frame_files[0]}")
    height, width = first.shape[:2]

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else run_dir / f"{run_dir.name}_final_masks.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output}")

    try:
        for sequence_index, frame_path in enumerate(frame_files):
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Cannot read frame: {frame_path}")
            if frame.shape[:2] != (height, width):
                raise RuntimeError(f"Unexpected frame size: {frame_path}")

            for row in rows_by_frame.get(frame_path.name, []):
                label = str(row["label"]).strip().lower()
                mask_path = resolve_from_run(run_dir, str(row.get("mask_path", "")), "final_masks")
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"[WARNING] Missing mask, skipped: {mask_path}")
                    continue
                if mask.shape != (height, width):
                    print(f"[WARNING] Wrong mask size, skipped: {mask_path}")
                    continue
                selected = mask > 0
                color_layer = np.empty_like(frame)
                color_layer[:] = LABEL_COLORS[label]
                frame[selected] = cv2.addWeighted(
                    frame, 1.0 - args.alpha, color_layer, args.alpha, 0
                )[selected]
                contours, _ = cv2.findContours(
                    selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(frame, contours, -1, LABEL_COLORS[label], 2)

            gui_index = frame_indices.get(frame_path.name, sequence_index)
            draw_legend(frame, gui_index, len(frame_files))
            writer.write(frame)
    finally:
        writer.release()

    print(f"[OK] Created final overlay video: {output}")
    print("[INFO] Included labels: weld, plasma, spatter")
    print(f"[INFO] labels_final.csv was not modified: {csv_path}")


if __name__ == "__main__":
    main()
