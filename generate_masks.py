import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any
import argparse

import cv2
import numpy as np
import pandas as pd
import torch

try:
    import sam2
    from hydra import initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
except ImportError as exc:
    raise ImportError("Install Meta SAM2 and its dependencies before running this script.") from exc


def natural_key(path: Path) -> list[Any]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


@contextmanager
def pushd(path: str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def resolve_repo(repo_dir: str | None) -> str:
    if repo_dir:
        return os.path.abspath(os.path.expanduser(repo_dir))
    env_repo = os.environ.get("SAM2_REPO", "").strip()
    if env_repo:
        return os.path.abspath(os.path.expanduser(env_repo))
    return os.path.abspath(os.path.join(list(sam2.__path__)[0], ".."))


def resolve_checkpoint(repo: str, checkpoint: str) -> str:
    candidate = os.path.abspath(os.path.expanduser(checkpoint))
    if not os.path.isfile(candidate):
        candidate = os.path.join(repo, "checkpoints", checkpoint)
    if not os.path.isfile(candidate):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {candidate}")
    return candidate


def build_generator(args: argparse.Namespace) -> SAM2AutomaticMaskGenerator:
    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable; using CPU.")
        device = "cpu"
    repo = resolve_repo(args.repo_dir)
    checkpoint = resolve_checkpoint(repo, args.checkpoint)
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize_config_module(config_module="sam2", version_base=None)
    with pushd(repo):
        model = build_sam2(
            args.model_cfg,
            checkpoint,
            device=device,
            apply_postprocessing=False,
        )
    print("[SAM2] repository:", repo)
    print("[SAM2] checkpoint:", checkpoint)
    return SAM2AutomaticMaskGenerator(
        model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.generator_pred_iou,
        stability_score_thresh=args.generator_stability,
        min_mask_region_area=args.min_area,
    )


def sample_indices(total: int, target: int) -> np.ndarray:
    if total <= 0:
        raise RuntimeError("Video contains no readable frames.")
    if total <= target:
        return np.arange(total, dtype=int)
    return np.unique(np.rint(np.linspace(0, total - 1, target)).astype(int))


def extract_frames(video: str, frames_dir: Path, target: int) -> pd.DataFrame:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = sample_indices(total, target)
    wanted = set(int(i) for i in indices)
    rows: list[dict[str, Any]] = []
    frames_dir.mkdir(parents=True, exist_ok=True)
    source_video = os.path.basename(video)
    try:
        source_idx = 0
        extracted_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if source_idx in wanted:
                name = f"frame_{extracted_idx:05d}.png"
                path = frames_dir / name
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"Could not write frame: {path}")
                rows.append({
                    "frame_index": extracted_idx,
                    "source_frame_index": source_idx,
                    "frame_file": name,
                    "image_path": str(path.resolve()),
                    "source_video": source_video,
                    "is_review_keyframe": extracted_idx % 10 == 0,
                })
                extracted_idx += 1
            source_idx += 1
    finally:
        cap.release()
    return pd.DataFrame(rows)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union else 0.0


def mask_nms(masks: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    masks = sorted(masks, key=lambda m: int(m["area"]), reverse=True)
    kept: list[dict[str, Any]] = []
    for mask in masks:
        if all(mask_iou(mask["segmentation"], old["segmentation"]) < threshold for old in kept):
            kept.append(mask)
    return kept


def filter_masks(raw: list[dict[str, Any]], shape: tuple[int, int], args: argparse.Namespace):
    max_area = int(args.max_area_fraction * shape[0] * shape[1])
    masks = []
    for ann in raw:
        if "segmentation" not in ann:
            continue
        piou = float(ann.get("predicted_iou", 1.0))
        stability = float(ann.get("stability_score", 1.0))
        seg = np.asarray(ann["segmentation"], dtype=bool)
        area = int(seg.sum())
        if piou < args.min_pred_iou or stability < args.min_stability:
            continue
        if area < args.min_area or area > max_area:
            continue
        masks.append({"segmentation": seg, "area": area,
                      "predicted_iou": piou, "stability_score": stability})
    return mask_nms(masks, args.nms_iou)


def geometry(seg: np.ndarray) -> tuple[int, int, int, int, float, float]:
    x, y, w, h = cv2.boundingRect(seg.astype(np.uint8) * 255)
    ys, xs = np.where(seg)
    return x, y, w, h, float(xs.mean()), float(ys.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract video frames and generate SAM2 candidate masks.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--target-frames", type=int, default=164)
    ap.add_argument("--keyframe-interval", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repo-dir", default=None)
    ap.add_argument("--checkpoint", default="sam2.1_hiera_large.pt")
    ap.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l")
    ap.add_argument("--points-per-side", type=int, default=16)
    ap.add_argument("--generator-pred-iou", type=float, default=0.40)
    ap.add_argument("--generator-stability", type=float, default=0.25)
    ap.add_argument("--min-pred-iou", type=float, default=0.50)
    ap.add_argument("--min-stability", type=float, default=0.50)
    ap.add_argument("--min-area", type=int, default=20)
    ap.add_argument("--max-area-fraction", type=float, default=0.98)
    ap.add_argument("--nms-iou", type=float, default=0.97)
    args = ap.parse_args()

    video = os.path.abspath(os.path.expanduser(args.video))
    output = Path(args.output_dir).expanduser().resolve()
    frames_dir = output / "frames"
    masks_dir = output / "RAW_SAM" / "masks"
    if not os.path.isfile(video):
        raise FileNotFoundError(video)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}. Use a new directory.")
    masks_dir.mkdir(parents=True, exist_ok=True)

    frames = extract_frames(video, frames_dir, args.target_frames)
    frames["is_review_keyframe"] = frames["frame_index"] % args.keyframe_interval == 0
    frames.to_csv(output / "frames_manifest.csv", index=False)
    generator = build_generator(args)

    rows = []
    with torch.inference_mode():
        for _, frame_row in frames.iterrows():
            bgr = cv2.imread(frame_row.image_path, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            masks = filter_masks(generator.generate(rgb), rgb.shape[:2], args)
            base = Path(frame_row.frame_file).stem
            if not masks:
                rows.append({**frame_row.to_dict(), "mask_index": -1, "mask_path": "",
                    "area": 0, "bbox_x": -1, "bbox_y": -1, "bbox_w": -1, "bbox_h": -1,
                    "cx_geo": np.nan, "cy_geo": np.nan, "predicted_iou": np.nan,
                    "stability_score": np.nan, "sam_quality": np.nan, "segment_type": "",
                    "label": "", "label_final": "", "ignore": True,
                    "edited_manually": False, "source": "empty_frame_placeholder"})
            for mask_index, ann in enumerate(masks):
                path = masks_dir / f"{base}_mask_{mask_index:03d}.png"
                cv2.imwrite(str(path), ann["segmentation"].astype(np.uint8) * 255)
                x, y, w, h, cx, cy = geometry(ann["segmentation"])
                rows.append({**frame_row.to_dict(), "mask_index": mask_index,
                    "mask_path": str(path.resolve()), "area": ann["area"],
                    "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
                    "cx_geo": cx, "cy_geo": cy,
                    "predicted_iou": ann["predicted_iou"],
                    "stability_score": ann["stability_score"],
                    "sam_quality": min(ann["predicted_iou"], ann["stability_score"]),
                    "segment_type": "", "label": "", "label_final": "",
                    "ignore": False, "edited_manually": False, "source": "sam2_automatic"})
            print(f"[SAM2] frame {int(frame_row.frame_index)+1}/{len(frames)}: {len(masks)} masks")
    manifest = output / "label_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print("[OK] Frames manifest:", output / "frames_manifest.csv")
    print("[OK] Label manifest:", manifest)


if __name__ == "__main__":
    main()
