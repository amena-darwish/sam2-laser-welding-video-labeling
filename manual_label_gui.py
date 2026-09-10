from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import pandas as pd

APP_CSS = """
#viewer_row {
    gap: 12px !important;
    align-items: stretch !important;
}

#img_col {
    flex: 4 1 0% !important;
    min-width: 320px !important;
}

#table_col {
    flex: 6 1 0% !important;
    min-width: 520px !important;
}

#frame_image {
    min-width: 0 !important;
}

#frame_image img {
    object-fit: contain !important;
    width: 100% !important;
}

#mask_table {
    min-width: 0 !important;
    overflow-x: auto !important;
    max-height: 430px !important;
}

#save_status textarea {
    font-size: 13px !important;
}
"""


# ---------------------------------------------------------------------
# Small path helpers
# ---------------------------------------------------------------------
def _abs_path(base_dir: str, p: Any) -> str:
    if p is None:
        return ""
    s = str(p).strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    s = os.path.expanduser(s)
    if os.path.isabs(s):
        return s
    return os.path.abspath(os.path.join(base_dir, s))


def _mtime(path: str) -> float:
    try:
        return float(os.path.getmtime(path))
    except Exception:
        return 0.0


def sync_label_columns_for_gui(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make the GUI robust to different CSV versions.

    Manual truth should live in label_final. The older/exported column label is
    mirrored from label_final so downstream scripts remain compatible. The
    original segment_type is kept as the model/rule prediction when possible.
    """
    df = df.copy()

    for col in ["label", "label_final", "segment_type"]:
        if col not in df.columns:
            df[col] = pd.NA

    def clean_series(s: pd.Series) -> pd.Series:
        s = s.astype("string").str.strip()
        s = s.mask(s.str.lower().isin(["", "nan", "none", "<na>", "null"]))
        s = s.replace({"plume": "plasma", "plume2": "plasma"})
        return s

    label = clean_series(df["label"])
    label_final = clean_series(df["label_final"])
    segment_type = clean_series(df["segment_type"])

    combined = label_final.combine_first(label).combine_first(segment_type)

    df["label_final"] = combined.fillna("")
    df["label"] = df["label_final"]

    # Keep predicted segment_type, but fill if completely empty.
    df["segment_type"] = segment_type.combine_first(combined).fillna("")

    return df


def get_final_masks_dir(csv_path: str) -> str:
    d = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "final_masks")
    os.makedirs(d, exist_ok=True)
    return d


def get_raw_masks_dir(csv_path: str) -> str:
    d = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "RAW_SAM", "masks")
    os.makedirs(d, exist_ok=True)
    return d


def next_mask_indices_for_frame(df: pd.DataFrame, frame_file: str, n: int = 1) -> List[int]:
    vals = pd.to_numeric(
        df.loc[df["frame_file"] == frame_file, "mask_index"],
        errors="coerce",
    ).dropna()
    start = int(vals.max()) + 1 if len(vals) else 0
    return list(range(start, start + n))


def make_raw_mask_path(raw_masks_dir: str, frame_file: str, mask_index: int) -> str:
    frame_base = os.path.splitext(frame_file)[0]
    return os.path.join(raw_masks_dir, f"{frame_base}_mask_{mask_index:03d}.png")


# ---------------------------------------------------------------------
# Brush / manual helpers
# ---------------------------------------------------------------------
def make_editor_value_for_frame(frame_rgb: np.ndarray) -> np.ndarray:
    return frame_rgb.copy()


def _to_numpy_image(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        arr = np.asarray(x)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return arr


def _extract_layer_mask(layer: Any, shape_hw: Tuple[int, int]) -> np.ndarray:
    H, W = shape_hw
    arr = _to_numpy_image(layer)
    if arr is None:
        return np.zeros((H, W), dtype=bool)

    if arr.ndim == 3 and arr.shape[2] == 4:
        mask = arr[:, :, 3] > 0
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        mask = arr[:, :, :3].max(axis=2) > 0
    else:
        mask = arr > 0

    if mask.shape != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def _editor_to_binary_mask(editor_value: Any, shape_hw: Tuple[int, int]) -> np.ndarray:
    H, W = shape_hw
    out = np.zeros((H, W), dtype=bool)

    if editor_value is None:
        return out

    if isinstance(editor_value, dict):
        layers = editor_value.get("layers", []) or []
        for layer in layers:
            out |= _extract_layer_mask(layer, (H, W))

        if not np.any(out):
            bg = _to_numpy_image(editor_value.get("background"))
            comp = _to_numpy_image(editor_value.get("composite"))
            if bg is not None and comp is not None:
                if bg.shape[:2] != (H, W):
                    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)
                if comp.shape[:2] != (H, W):
                    comp = cv2.resize(comp, (W, H), interpolation=cv2.INTER_LINEAR)

                bg3 = np.stack([bg, bg, bg], axis=2) if bg.ndim == 2 else bg[:, :, :3]
                comp3 = np.stack([comp, comp, comp], axis=2) if comp.ndim == 2 else comp[:, :, :3]

                diff = np.max(np.abs(comp3.astype(np.int16) - bg3.astype(np.int16)), axis=2)
                out |= diff > 0
    else:
        arr = _to_numpy_image(editor_value)
        if arr is not None:
            if arr.ndim == 3 and arr.shape[2] >= 3:
                gray = arr[:, :, :3].max(axis=2)
                out = gray > 0
            else:
                out = arr > 0
            if out.shape != (H, W):
                out = cv2.resize(out.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0

    return out


def clean_manual_mask(seg: np.ndarray, min_area: int = 8) -> np.ndarray:
    seg_u8 = seg.astype(np.uint8) * 255

    k = np.ones((3, 3), np.uint8)
    seg_u8 = cv2.morphologyEx(seg_u8, cv2.MORPH_OPEN, k)
    seg_u8 = cv2.morphologyEx(seg_u8, cv2.MORPH_CLOSE, k)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seg_u8, connectivity=8)
    cleaned = np.zeros_like(seg_u8)
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            cleaned[labels == i] = 255

    return cleaned > 0


# ---------------------------------------------------------------------
# I/O cached by mtime
# ---------------------------------------------------------------------
@lru_cache(maxsize=512)
def _read_rgb_cached(path: str, mtime: float) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_rgb(path: str) -> np.ndarray:
    return _read_rgb_cached(path, _mtime(path))


@lru_cache(maxsize=4096)
def _read_mask_bool_cached(path: str, mtime: float) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise RuntimeError(f"Cannot read mask: {path}")
    return g > 0


def read_mask_bool(path: str) -> np.ndarray:
    return _read_mask_bool_cached(path, _mtime(path))


def write_mask_bool(path: str, seg: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, seg.astype(np.uint8) * 255)
    if not ok:
        raise RuntimeError(f"Could not write mask: {path}")
    try:
        _read_mask_bool_cached.cache_clear()
    except Exception:
        pass


def bbox_from_mask(seg: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(seg)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def centroid(seg: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(seg)
    if len(xs) == 0:
        return (np.nan, np.nan)
    return (float(xs.mean()), float(ys.mean()))


def chebyshev_center(seg: np.ndarray) -> Tuple[float, float, float]:
    """Centre and radius of the largest inscribed circle."""
    mask = np.asarray(seg, dtype=np.uint8)
    if int(mask.sum()) == 0:
        return float("nan"), float("nan"), 0.0
    x, y, w, h = cv2.boundingRect(mask * 255)
    distance = cv2.distanceTransform(mask[y:y + h, x:x + w], cv2.DIST_L2, 5)
    _, radius, _, location = cv2.minMaxLoc(distance)
    return float(x + location[0]), float(y + location[1]), float(radius)


def alpha_blend(rgb: np.ndarray, seg: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    out = rgb.copy()
    m = seg.astype(bool)
    if not np.any(m):
        return out
    c = np.array(color, dtype=np.float32)
    out[m] = ((1 - alpha) * out[m].astype(np.float32) + alpha * c).astype(np.uint8)
    return out


def draw_contour(
    rgb: np.ndarray,
    seg: np.ndarray,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    u8 = seg.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr_color = (int(color[2]), int(color[1]), int(color[0]))
    cv2.drawContours(bgr, cnts, -1, bgr_color, thickness)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_cross(
    rgb: np.ndarray,
    x: int,
    y: int,
    color: Tuple[int, int, int] = (255, 255, 255),
    size: int = 10,
    thickness: int = 2,
) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr_color = (int(color[2]), int(color[1]), int(color[0]))
    cv2.line(bgr, (x - size, y), (x + size, y), bgr_color, thickness)
    cv2.line(bgr, (x, y - size), (x, y + size), bgr_color, thickness)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def polygon_to_mask(shape_hw: Tuple[int, int], points: List[Tuple[int, int]]) -> np.ndarray:
    H, W = shape_hw
    if len(points) < 3:
        return np.zeros((H, W), dtype=bool)

    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask > 0


# ---------------------------------------------------------------------
# Labels / colors
# ---------------------------------------------------------------------
LABELS = ["", "weld", "plasma", "plume", "plume2", "spatter", "static", "dynamic_other"]

LABEL_COLOR = {
    "": (0, 128, 255),
    "weld": (255, 0, 0),
    "plasma": (255, 255, 0),
    "plume": (255, 255, 0),
    "plume2": (255, 255, 0),
    "spatter": (0, 255, 0),
    "static": (140, 140, 140),
    "dynamic_other": (160, 32, 240),
}

UNIQUE_LABELS = {"weld", "plasma", "plume", "plume2"}
UNIQUE_TRACKS = {"weld": 0, "plasma": 1, "plume": 1, "plume2": 2}
DEMOTE_LABEL = "dynamic_other"


# ---------------------------------------------------------------------
# Safety parsers
# ---------------------------------------------------------------------
def norm_label(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null", "<na>"]:
        return ""
    if s == "plume" or s == "plume2":
        return "plasma"
    return s


def origin_tag_for_row(df: pd.DataFrame, rid: int) -> str:
    origin = str(df.at[rid, "review_origin"]) if "review_origin" in df.columns else ""
    source = str(df.at[rid, "source"]) if "source" in df.columns else ""

    if origin == "sam2_video_propagation" or source == "sam2_video_propagation":
        return "PROP"
    if source.startswith("manual"):
        return "MAN"
    if origin == "original_before_propagation" or source in {"raw_sam", "original_pipeline"}:
        return "ORG"
    if source == "empty_frame_placeholder":
        return "EMPTY"
    return ""


def is_propagated_row(df: pd.DataFrame, rid: int) -> bool:
    origin = str(df.at[rid, "review_origin"]) if "review_origin" in df.columns else ""
    source = str(df.at[rid, "source"]) if "source" in df.columns else ""
    return origin == "sam2_video_propagation" or source == "sam2_video_propagation"


def safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return int(v)
    except Exception:
        return default


def safe_float(v: Any, default: float = 0.98) -> float:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default


# ---------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------
def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "label": "",
        "label_final": "",
        "segment_type": "",
        "ignore": False,
        "is_smooth_top10": False,
        "smooth_rank": -1,

        "track_id": -1,
        "old_track_id": -1,
        "static_overlap": 0.0,
        "auto_static": False,
        "track_len": -1,

        "why": "",
        "broken_rule": "",
        "manual_note": "",
        "old_label_final": "",
        "edited_manually": False,

        "cx_geo": np.nan,
        "cy_geo": np.nan,
        "area": -1,
        "bbox_x": -1,
        "bbox_y": -1,
        "bbox_w": -1,
        "bbox_h": -1,

        "source": "raw_sam",
        "review_origin": "",
        "is_manual": False,

        "predicted_iou": np.nan,
        "stability_score": np.nan,
        "sam_quality": np.nan,
        "frame_uncertainty": np.nan,
        "is_review_keyframe": False,
        "needs_review": False,
        "review_reason": "",

        "propagation_seed_frame_index": -1,
        "propagation_seed_frame_file": "",
        "propagation_obj_id": -1,
        "propagation_segment_start_pos": -1,
        "propagation_segment_end_pos": -1,
    }
    for c, d in defaults.items():
        if c not in df.columns:
            df[c] = d

    for c in ["label", "label_final", "segment_type"]:
        df[c] = df[c].apply(norm_label)

    df["ignore"] = df["ignore"].fillna(False).astype(bool)
    df["is_smooth_top10"] = df["is_smooth_top10"].fillna(False).astype(bool)
    df["smooth_rank"] = df["smooth_rank"].fillna(-1).astype(int)

    df["track_id"] = df["track_id"].fillna(-1).astype(int)
    df["old_track_id"] = df["old_track_id"].fillna(-1).astype(int)
    df["static_overlap"] = pd.to_numeric(df["static_overlap"], errors="coerce").fillna(0.0).astype(float)
    df["auto_static"] = df["auto_static"].fillna(False).astype(bool)
    df["track_len"] = df["track_len"].fillna(-1).astype(int)

    df["why"] = df["why"].fillna("").astype(str)
    df["broken_rule"] = df["broken_rule"].fillna("").astype(str)
    df["manual_note"] = df["manual_note"].fillna("").astype(str)
    df["old_label_final"] = df["old_label_final"].fillna("").astype(str)
    df["edited_manually"] = df["edited_manually"].fillna(False).astype(bool)

    for c in ["area", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int)

    df["source"] = df["source"].fillna("raw_sam").astype(str)
    df["review_origin"] = df["review_origin"].fillna("").astype(str)
    df["is_manual"] = df["is_manual"].fillna(False).astype(bool)

    for c in ["predicted_iou", "stability_score", "sam_quality", "frame_uncertainty"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["is_review_keyframe"] = df["is_review_keyframe"].fillna(False).astype(bool)
    df["needs_review"] = df["needs_review"].fillna(False).astype(bool)
    df["review_reason"] = df["review_reason"].fillna("").astype(str)

    for c in [
        "propagation_seed_frame_index",
        "propagation_obj_id",
        "propagation_segment_start_pos",
        "propagation_segment_end_pos",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int)

    df["propagation_seed_frame_file"] = df["propagation_seed_frame_file"].fillna("").astype(str)

    empty = df["label_final"].astype(str).str.strip().isin(["", "nan", "None"])
    df.loc[empty, "label_final"] = df.loc[empty, "segment_type"].apply(norm_label)
    df["label"] = df["label_final"]

    return df


def normalize_paths(df: pd.DataFrame, csv_path: str, frames_dir: Optional[str]) -> Tuple[pd.DataFrame, str]:
    base_dir = os.path.dirname(os.path.abspath(csv_path))
    frames_dir_res = os.path.expanduser(frames_dir) if frames_dir else ""
    frames_dir_res = os.path.abspath(frames_dir_res) if frames_dir_res else ""

    if "mask_path" in df.columns:
        df["mask_path"] = df["mask_path"].apply(lambda p: _abs_path(base_dir, p))
    else:
        df["mask_path"] = ""

    if "image_path" in df.columns:
        df["image_path"] = df["image_path"].apply(lambda p: _abs_path(base_dir, p))
    else:
        df["image_path"] = ""

    if frames_dir_res and "frame_file" in df.columns:
        missing = df["image_path"].astype(str).str.strip().isin(["", "nan", "None"])
        df.loc[missing, "image_path"] = df.loc[missing, "frame_file"].apply(
            lambda f: _abs_path(frames_dir_res, f)
        )

    return df, frames_dir_res


def import_previous_labels(df: pd.DataFrame, pre_csv: Optional[str]) -> pd.DataFrame:
    if not pre_csv:
        return df
    pre_csv = os.path.expanduser(pre_csv)
    if not os.path.exists(pre_csv):
        print(f"[WARN] pre_csv not found: {pre_csv}")
        return df

    prev = pd.read_csv(pre_csv)
    if "frame_file" not in prev.columns or "mask_index" not in prev.columns:
        print("[WARN] pre_csv missing frame_file/mask_index, cannot import labels")
        return df

    src_col = "label_final" if "label_final" in prev.columns else ("segment_type" if "segment_type" in prev.columns else None)
    if src_col is None:
        print("[WARN] pre_csv has no label_final or segment_type")
        return df

    prev = prev[["frame_file", "mask_index", src_col]].copy()
    prev[src_col] = prev[src_col].apply(norm_label)

    key_to_label = {
        (r["frame_file"], int(r["mask_index"])): r[src_col]
        for _, r in prev.iterrows()
        if norm_label(r[src_col]) != ""
    }

    filled = 0
    for i in range(len(df)):
        if norm_label(df.at[i, "label_final"]) != "":
            continue
        k = (df.at[i, "frame_file"], int(df.at[i, "mask_index"]))
        lab = key_to_label.get(k, "")
        if lab:
            df.at[i, "label_final"] = lab
            df.at[i, "label"] = lab
            filled += 1

    if filled:
        print(f"[INFO] Imported {filled} labels from: {pre_csv}")

    return df


def build_frames(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["frame_file", "frame_index", "image_path"]]
        .drop_duplicates()
        .sort_values(["frame_index", "frame_file"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Review queue helpers
# ---------------------------------------------------------------------
def frame_has_label(df: pd.DataFrame, frame_file: str, label: str) -> bool:
    label = norm_label(label)
    if label == "":
        return True

    sub = df[(df["frame_file"] == frame_file) & (~df["ignore"].astype(bool))]
    if sub.empty:
        return False

    labs = sub["label_final"].apply(norm_label)
    segs = sub["segment_type"].apply(norm_label) if "segment_type" in sub.columns else labs

    if label == "plasma":
        return bool(labs.isin(["plasma", "plume", "plume2"]).any() or segs.isin(["plasma", "plume", "plume2"]).any())

    return bool((labs == label).any() or (segs == label).any())


def frame_matches_review_mode(
    df: pd.DataFrame,
    frame_file: str,
    mode: str,
    uncertainty_thr: float = 0.65,
) -> bool:
    mode = str(mode or "All frames")
    sub = df[df["frame_file"] == frame_file]
    if sub.empty:
        return False

    if mode == "All frames":
        return True

    if mode == "Keyframes only":
        return bool(sub["is_review_keyframe"].astype(bool).any())

    if mode == "Needs review":
        return bool(sub["needs_review"].astype(bool).any() or sub["is_review_keyframe"].astype(bool).any())

    if mode == "Uncertain frames":
        vals = pd.to_numeric(sub["frame_uncertainty"], errors="coerce")
        return bool((vals < float(uncertainty_thr)).any())

    if mode == "Spatter frames":
        return frame_has_label(df, frame_file, "spatter")

    return True


def review_frame_positions(
    df: pd.DataFrame,
    frames: pd.DataFrame,
    mode: str,
    uncertainty_thr: float = 0.65,
) -> List[int]:
    out: List[int] = []
    for pos, row in frames.iterrows():
        frame_file = str(row["frame_file"])
        if frame_matches_review_mode(df, frame_file, mode, uncertainty_thr):
            out.append(int(pos))
    return out


def find_next_review_pos(
    df: pd.DataFrame,
    frames: pd.DataFrame,
    current_pos: int,
    mode: str,
    direction: int = 1,
    uncertainty_thr: float = 0.65,
) -> int:
    positions = review_frame_positions(df, frames, mode, uncertainty_thr)
    if not positions:
        return safe_frame_pos(current_pos, len(frames))

    current_pos = safe_frame_pos(current_pos, len(frames))
    direction = 1 if int(direction) >= 0 else -1

    if direction > 0:
        for p in positions:
            if p > current_pos:
                return int(p)
        return int(positions[-1])

    for p in reversed(positions):
        if p < current_pos:
            return int(p)
    return int(positions[0])


# ---------------------------------------------------------------------
# Frame context / rendering
# ---------------------------------------------------------------------
def safe_frame_pos(frame_pos: Any, nframes: int) -> int:
    if nframes <= 0:
        return 0
    if frame_pos is None:
        return 0
    try:
        v = int(frame_pos)
    except Exception:
        return 0
    return int(np.clip(v, 0, nframes - 1))


def load_frame_context(
    df: pd.DataFrame,
    frames: pd.DataFrame,
    frames_dir_fallback: str,
    frame_pos: int,
    only_smooth: bool,
    class_filter: str = "all",
    show_ignored: bool = False,
) -> Dict[str, Any]:
    frame_pos = safe_frame_pos(frame_pos, len(frames))
    frame_file = str(frames.at[frame_pos, "frame_file"])
    frame_index = int(frames.at[frame_pos, "frame_index"])
    img_path = str(frames.at[frame_pos, "image_path"]) if "image_path" in frames.columns else ""

    if not img_path or not os.path.exists(img_path):
        if frames_dir_fallback:
            img_path = os.path.join(frames_dir_fallback, frame_file)

    if not img_path or not os.path.exists(img_path):
        raise RuntimeError(
            f"Cannot locate frame image.\n"
            f"frame_file={frame_file}\n"
            f"image_path={img_path}\n"
            f"frames_dir_fallback={frames_dir_fallback}"
        )

    img = read_rgb(img_path)

    row_ids_all = df.index[df["frame_file"] == frame_file].tolist()

    if show_ignored:
        row_ids_view = list(row_ids_all)
    else:
        row_ids_view = [rid for rid in row_ids_all if not bool(df.at[rid, "ignore"])]

    if only_smooth:
        row_ids_view = [rid for rid in row_ids_view if bool(df.at[rid, "is_smooth_top10"])]

    class_filter = str(class_filter or "all")
    if class_filter != "all":
        filtered = []
        for rid in row_ids_view:
            lab = norm_label(df.at[rid, "label_final"])
            st = norm_label(df.at[rid, "segment_type"]) if "segment_type" in df.columns else ""

            if class_filter == "plasma":
                if lab in ["plasma", "plume", "plume2"] or st in ["plasma", "plume", "plume2"]:
                    filtered.append(rid)
            elif lab == class_filter or st == class_filter:
                filtered.append(rid)
        row_ids_view = filtered

    selected = row_ids_view[0] if row_ids_view else None

    return {
        "frame_pos": frame_pos,
        "frame_index": frame_index,
        "frame_file": frame_file,
        "img_path": img_path,
        "img": img,
        "row_ids_all": row_ids_all,
        "row_ids_view": row_ids_view,
        "selected": selected,
        "checked": set(),
        "hits": [],
        "hit_i": 0,
        "click_mode": "select",
        "seed1": None,
        "seed2": None,
        "poly_points": [],
        "only_smooth": bool(only_smooth),
        "class_filter": class_filter,
        "show_ignored": bool(show_ignored),
        "merge_a": None,
        "merge_b": None,
        "table_row_ids": [],
    }


def build_mask_table(df: pd.DataFrame, ctx: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    selected = ctx.get("selected", None)
    checked = set(ctx.get("checked", set()) or set())

    for rid in ctx["row_ids_view"]:
        lab = norm_label(df.at[rid, "label_final"])
        st = norm_label(df.at[rid, "segment_type"]) if "segment_type" in df.columns else ""
        tid = int(df.at[rid, "track_id"]) if "track_id" in df.columns else -1

        rows.append({
            "row_id": int(rid),
            "checked": int(rid) in checked,
            "selected": rid == selected,
            "mask_index": int(df.at[rid, "mask_index"]),
            "label": lab if lab else "unlabeled",
            "segment_type": st,
            "track_id": tid,
            "origin_tag": origin_tag_for_row(df, rid),
            "review_origin": str(df.at[rid, "review_origin"]) if "review_origin" in df.columns else "",
            "source": str(df.at[rid, "source"]) if "source" in df.columns else "",
            "area": int(df.at[rid, "area"]) if "area" in df.columns else -1,
            "sam_quality": float(df.at[rid, "sam_quality"]) if "sam_quality" in df.columns and pd.notna(df.at[rid, "sam_quality"]) else np.nan,
            "frame_uncertainty": float(df.at[rid, "frame_uncertainty"]) if "frame_uncertainty" in df.columns and pd.notna(df.at[rid, "frame_uncertainty"]) else np.nan,
            "review_reason": str(df.at[rid, "review_reason"]) if "review_reason" in df.columns else "",
            "ignore": bool(df.at[rid, "ignore"]),
            "why": str(df.at[rid, "why"]) if "why" in df.columns else "",
            "broken_rule": str(df.at[rid, "broken_rule"]) if "broken_rule" in df.columns else "",
            "manual_note": str(df.at[rid, "manual_note"]) if "manual_note" in df.columns else "",
            "static_overlap": float(df.at[rid, "static_overlap"]) if "static_overlap" in df.columns else 0.0,
            "smooth": bool(df.at[rid, "is_smooth_top10"]),
            "smooth_rank": int(df.at[rid, "smooth_rank"]),
            "mask_path": str(df.at[rid, "mask_path"]) if "mask_path" in df.columns else "",
        })

    tab = pd.DataFrame(rows)
    if not tab.empty:
        tab = tab.sort_values(
            ["selected", "checked", "origin_tag", "label", "area"],
            ascending=[False, False, True, True, True],
        ).reset_index(drop=True)
        ctx["table_row_ids"] = tab["row_id"].tolist()
    else:
        ctx["table_row_ids"] = []
    return tab


def label_text_for(df: pd.DataFrame, rid: int) -> str:
    lab = norm_label(df.at[rid, "label_final"]) or "unlabeled"
    mi = int(df.at[rid, "mask_index"])
    tid = int(df.at[rid, "track_id"]) if "track_id" in df.columns else -1
    tag = origin_tag_for_row(df, rid)
    tag_txt = f" | {tag}" if tag else ""
    if tid >= 0:
        return f"{lab}{tag_txt} | t{tid} | m{mi}"
    return f"{lab}{tag_txt} | m{mi}"


def render(
    df: pd.DataFrame,
    ctx: Dict[str, Any],
    show_all: bool,
    draw_labels_text: bool,
) -> np.ndarray:
    img = ctx["img"].copy()
    row_ids_view = ctx["row_ids_view"]
    selected = ctx["selected"]
    checked = set(ctx.get("checked", set()) or set())

    draw_ids = row_ids_view if show_all else ([selected] if selected is not None else [])

    for rid in draw_ids:
        if rid is None:
            continue
        try:
            seg = read_mask_bool(str(df.at[rid, "mask_path"]))
        except Exception:
            continue

        lab = norm_label(df.at[rid, "label_final"])
        color = LABEL_COLOR.get(lab, LABEL_COLOR[""])

        if rid == selected:
            alpha = 0.18
        elif rid in checked:
            alpha = 0.30
        else:
            alpha = 0.20

        img = alpha_blend(img, seg, color, alpha)

        # Make propagated masks visually obvious.
        if is_propagated_row(df, rid):
            img = draw_contour(img, seg, color=(0, 255, 255), thickness=3)

    for rid in draw_ids:
        if rid is None or rid == selected or rid not in checked:
            continue
        try:
            seg = read_mask_bool(str(df.at[rid, "mask_path"]))
        except Exception:
            continue
        img = draw_contour(img, seg, color=(0, 255, 255), thickness=2)

    if selected is not None and selected in row_ids_view:
        try:
            seg = read_mask_bool(str(df.at[selected, "mask_path"]))
            sel_lab = norm_label(df.at[selected, "label_final"])
            sel_color = LABEL_COLOR.get(sel_lab, LABEL_COLOR[""])
            img = alpha_blend(img, seg, sel_color, 0.28)
            img = draw_contour(img, seg, color=(255, 255, 255), thickness=5)
            img = draw_contour(img, seg, color=sel_color, thickness=3)

            if is_propagated_row(df, selected):
                img = draw_contour(img, seg, color=(0, 255, 255), thickness=2)

            cx = float(df.at[selected, "cx_geo"]) if "cx_geo" in df.columns else np.nan
            cy = float(df.at[selected, "cy_geo"]) if "cy_geo" in df.columns else np.nan
            if not (np.isfinite(cx) and np.isfinite(cy)):
                cx, cy = centroid(seg)
            if np.isfinite(cx) and np.isfinite(cy):
                img = draw_cross(img, int(round(cx)), int(round(cy)), color=(255, 255, 255), size=12, thickness=2)
        except Exception:
            pass

    if draw_labels_text and len(draw_ids) > 0:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        H, W = bgr.shape[:2]
        for rid in draw_ids:
            if rid is None:
                continue
            try:
                cx = float(df.at[rid, "cx_geo"]) if "cx_geo" in df.columns else np.nan
                cy = float(df.at[rid, "cy_geo"]) if "cy_geo" in df.columns else np.nan
            except Exception:
                cx, cy = np.nan, np.nan

            if not (np.isfinite(cx) and np.isfinite(cy)):
                try:
                    seg = read_mask_bool(str(df.at[rid, "mask_path"]))
                    cx, cy = centroid(seg)
                except Exception:
                    continue
            if not (np.isfinite(cx) and np.isfinite(cy)):
                continue

            x = int(np.clip(int(round(cx)), 0, W - 1))
            y = int(np.clip(int(round(cy)), 0, H - 1))
            txt = label_text_for(df, rid)
            cv2.putText(bgr, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(bgr, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if ctx.get("seed1") is not None:
        x, y = ctx["seed1"]
        cv2.circle(bgr, (x, y), 6, (0, 0, 255), -1)
    if ctx.get("seed2") is not None:
        x, y = ctx["seed2"]
        cv2.circle(bgr, (x, y), 6, (0, 255, 0), -1)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    poly_points = ctx.get("poly_points", []) or []
    if len(poly_points) > 0:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for px, py in poly_points:
            cv2.circle(bgr, (int(px), int(py)), 4, (255, 0, 255), -1)
        if len(poly_points) >= 2:
            pts = np.array(poly_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(bgr, [pts], False, (255, 0, 255), 2)
        if len(poly_points) >= 3:
            x0, y0 = poly_points[0]
            x1, y1 = poly_points[-1]
            cv2.line(bgr, (int(x1), int(y1)), (int(x0), int(y0)), (180, 0, 180), 1)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return img


def find_hits(df: pd.DataFrame, ctx: Dict[str, Any], x: int, y: int) -> List[int]:
    hits = []
    for rid in ctx["row_ids_view"]:
        try:
            seg = read_mask_bool(str(df.at[rid, "mask_path"]))
        except Exception:
            continue
        if 0 <= y < seg.shape[0] and 0 <= x < seg.shape[1] and seg[y, x]:
            hits.append(rid)

    if hits:
        hits = sorted(hits, key=lambda rid: int(df.at[rid, "area"]) if "area" in df.columns else 10**18)
    return hits


# ---------------------------------------------------------------------
# Gradio table utils
# ---------------------------------------------------------------------
def _tab_to_df(tab_value: Any) -> pd.DataFrame:
    if isinstance(tab_value, pd.DataFrame):
        return tab_value
    if isinstance(tab_value, dict) and "data" in tab_value:
        headers = tab_value.get("headers", None)
        data = tab_value.get("data", [])
        return pd.DataFrame(data, columns=headers) if headers else pd.DataFrame(data)
    return pd.DataFrame(tab_value)


# ---------------------------------------------------------------------
# Split via 2 seed points + watershed
# ---------------------------------------------------------------------
def split_mask_watershed_points(
    seg: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    seed_r: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    if seg.sum() == 0:
        return np.zeros_like(seg, bool), np.zeros_like(seg, bool)

    H, W = seg.shape
    (x1, y1), (x2, y2) = p1, p2
    if not (0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H):
        return np.zeros_like(seg, bool), np.zeros_like(seg, bool)
    if not seg[y1, x1] or not seg[y2, x2]:
        return np.zeros_like(seg, bool), np.zeros_like(seg, bool)

    seg_u8 = seg.astype(np.uint8) * 255
    dist = cv2.distanceTransform(seg_u8, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return np.zeros_like(seg, bool), np.zeros_like(seg, bool)

    dist_norm = (dist / (dist.max() + 1e-6) * 255.0).astype(np.uint8)
    inv = 255 - dist_norm
    ws_img = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    markers = np.zeros((H, W), dtype=np.int32)
    cv2.circle(markers, (x1, y1), seed_r, 1, -1)
    cv2.circle(markers, (x2, y2), seed_r, 2, -1)
    markers[~seg] = 0
    cv2.watershed(ws_img, markers)

    m1 = (markers == 1) & seg
    m2 = (markers == 2) & seg

    if m1.sum() < 20 or m2.sum() < 20:
        return np.zeros_like(seg, bool), np.zeros_like(seg, bool)

    return m1, m2


# ---------------------------------------------------------------------
# Propagation helpers
# ---------------------------------------------------------------------
def get_bbox_from_df_or_mask(df: pd.DataFrame, rid: int, seg: Optional[np.ndarray] = None) -> Tuple[int, int, int, int]:
    if all(c in df.columns for c in ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]):
        bx = int(df.at[rid, "bbox_x"])
        by = int(df.at[rid, "bbox_y"])
        bw = int(df.at[rid, "bbox_w"])
        bh = int(df.at[rid, "bbox_h"])
        if bw >= 0 and bh >= 0:
            return (bx, by, bw, bh)
    if seg is None:
        seg = read_mask_bool(str(df.at[rid, "mask_path"]))
    return bbox_from_mask(seg)


def find_best_same_position_match(
    df: pd.DataFrame,
    rid_src: int,
    seg_src: np.ndarray,
    next_frame_index: int,
    iou_thr: float = 0.98,
    require_same_bbox: bool = False,
) -> Optional[int]:
    bbox_src = get_bbox_from_df_or_mask(df, rid_src, seg_src)
    cand_rids = df.index[(df["frame_index"] == next_frame_index) & (~df["ignore"].astype(bool))].tolist()
    if not cand_rids:
        return None

    best_rid = None
    best_iou = -1.0
    for rid in cand_rids:
        try:
            seg_c = read_mask_bool(str(df.at[rid, "mask_path"]))
        except Exception:
            continue
        bbox_c = get_bbox_from_df_or_mask(df, rid, seg_c)
        if require_same_bbox and bbox_c != bbox_src:
            continue
        iou = mask_iou(seg_src, seg_c)
        if iou > best_iou:
            best_iou = iou
            best_rid = rid

    if best_rid is None or best_iou < iou_thr:
        return None
    return int(best_rid)


def propagate_label_forward_same_position(
    df: pd.DataFrame,
    rid_start: int,
    label: str,
    new_track_id: Optional[int] = None,
    iou_thr: float = 0.98,
    require_same_bbox: bool = False,
    only_if_empty: bool = True,
    max_hops: int = 10_000,
) -> int:
    if rid_start is None:
        return 0

    frame_ids = sorted(df["frame_index"].dropna().astype(int).unique().tolist())
    cur_frame = int(df.at[rid_start, "frame_index"])
    if cur_frame not in frame_ids:
        return 0

    try:
        cur_seg = read_mask_bool(str(df.at[rid_start, "mask_path"]))
    except Exception:
        return 0

    start_pos = frame_ids.index(cur_frame)
    cur_rid = int(rid_start)
    added = 0
    hops = 0

    for j in range(start_pos + 1, len(frame_ids)):
        if hops >= max_hops:
            break
        nxt_frame = frame_ids[j]
        nxt_rid = find_best_same_position_match(
            df,
            rid_src=cur_rid,
            seg_src=cur_seg,
            next_frame_index=int(nxt_frame),
            iou_thr=iou_thr,
            require_same_bbox=require_same_bbox,
        )
        if nxt_rid is None:
            break

        existing = norm_label(df.at[nxt_rid, "label_final"])
        if only_if_empty and existing != "":
            break

        df.at[nxt_rid, "label_final"] = label
        df.at[nxt_rid, "label"] = label
        if new_track_id is not None:
            df.at[nxt_rid, "track_id"] = int(new_track_id)
        added += 1

        try:
            cur_seg = read_mask_bool(str(df.at[nxt_rid, "mask_path"]))
        except Exception:
            break
        cur_rid = int(nxt_rid)
        hops += 1

    return added


# ---------------------------------------------------------------------
# Spatter helpers
# ---------------------------------------------------------------------
def next_free_spatter_track_id(df: pd.DataFrame) -> int:
    if "track_id" not in df.columns:
        return 3
    vals = pd.to_numeric(df["track_id"], errors="coerce").fillna(-1).astype(int)
    vals = vals[vals >= 3]
    return int(vals.max()) + 1 if len(vals) else 3


def get_row_center(df: pd.DataFrame, rid: int) -> Tuple[float, float]:
    cx = float(df.at[rid, "cx_geo"]) if "cx_geo" in df.columns else np.nan
    cy = float(df.at[rid, "cy_geo"]) if "cy_geo" in df.columns else np.nan
    if np.isfinite(cx) and np.isfinite(cy):
        return cx, cy
    try:
        seg = read_mask_bool(str(df.at[rid, "mask_path"]))
        return centroid(seg)
    except Exception:
        return (np.nan, np.nan)


def find_matching_spatter_track_id(
    df: pd.DataFrame,
    rid: int,
    max_frame_gap: int = 1,
    max_center_dist: float = 90.0,
    min_iou: float = 0.05,
) -> Optional[int]:
    if rid is None:
        return None

    frame_idx = int(df.at[rid, "frame_index"])
    cx, cy = get_row_center(df, rid)
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return None

    try:
        seg_cur = read_mask_bool(str(df.at[rid, "mask_path"])).astype(bool)
    except Exception:
        seg_cur = None

    candidate_rids = df.index[
        (~df["ignore"].astype(bool))
        & (df.index != rid)
        & (df["label_final"].apply(norm_label) == "spatter")
        & (df["track_id"].fillna(-1).astype(int) >= 3)
        & (df["frame_index"].fillna(-999999).astype(int).sub(frame_idx).abs() <= max_frame_gap)
    ].tolist()

    if not candidate_rids:
        return None

    best_prev = None
    best_prev_score = None
    best_next = None
    best_next_score = None

    for cand_rid in candidate_rids:
        cand_frame = int(df.at[cand_rid, "frame_index"])
        if cand_frame not in (frame_idx - 1, frame_idx + 1):
            continue

        tid = int(df.at[cand_rid, "track_id"])
        c2x, c2y = get_row_center(df, cand_rid)
        if not (np.isfinite(c2x) and np.isfinite(c2y)):
            continue

        dist = float(((cx - c2x) ** 2 + (cy - c2y) ** 2) ** 0.5)
        if dist > max_center_dist:
            continue

        iou = 0.0
        if seg_cur is not None:
            try:
                seg_cand = read_mask_bool(str(df.at[cand_rid, "mask_path"])).astype(bool)
                iou = mask_iou(seg_cur, seg_cand)
            except Exception:
                iou = 0.0

        if not (iou >= min_iou or dist <= 0.5 * max_center_dist):
            continue

        score = dist - 40.0 * iou
        if cand_frame == frame_idx - 1:
            if best_prev is None or score < best_prev_score:
                best_prev = tid
                best_prev_score = score
        elif cand_frame == frame_idx + 1:
            if best_next is None or score < best_next_score:
                best_next = tid
                best_next_score = score

    if best_prev is not None:
        return int(best_prev)
    if best_next is not None:
        return int(best_next)
    return None


def enforce_unique_label_in_frame(df: pd.DataFrame, frame_file: str, keep_rid: int, new_label: str) -> None:
    new_label = norm_label(new_label)
    if new_label not in UNIQUE_LABELS:
        return

    same = df.index[
        (df["frame_file"] == frame_file)
        & (df.index != keep_rid)
        & (~df["ignore"].astype(bool))
        & (df["label_final"].apply(norm_label) == new_label)
    ].tolist()

    for rid in same:
        df.at[rid, "old_label_final"] = norm_label(df.at[rid, "label_final"])
        df.at[rid, "old_track_id"] = int(df.at[rid, "track_id"]) if "track_id" in df.columns else -1
        df.at[rid, "label_final"] = DEMOTE_LABEL
        df.at[rid, "label"] = DEMOTE_LABEL
        df.at[rid, "edited_manually"] = True
        if "track_id" in df.columns:
            df.at[rid, "track_id"] = -1

    if "track_id" in df.columns:
        df.at[keep_rid, "track_id"] = UNIQUE_TRACKS[new_label]




def resolve_propagation_script(explicit_path: Optional[str] = None) -> str:
    """Find the standalone SAM2 keyframe propagation script."""
    candidates: List[str] = []

    if explicit_path:
        candidates.append(os.path.abspath(os.path.expanduser(explicit_path)))

    here = os.path.dirname(os.path.abspath(__file__))
    candidates.extend([
        os.path.join(here, "propagate_keyframes_sam2.py"),
        os.path.join(here, "propagate_keyframes.py"),
        os.path.join(here, "scripts", "propagate_keyframes_sam2.py"),
        os.path.join(here, "scripts", "propagate_keyframes.py"),
    ])

    seen = set()
    for p in candidates:
        p = os.path.abspath(p)
        if p in seen:
            continue
        seen.add(p)
        if os.path.isfile(p):
            return p

    return ""


# ---------------------------------------------------------------------
# App
# ---------------------------------------------------------------------
def app(
    csv_path: str,
    frames_dir: Optional[str],
    pre_csv: Optional[str] = None,
    propagation_script: Optional[str] = None,
    sam2_repo: Optional[str] = None,
    sam2_checkpoint: Optional[str] = None,
    sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l",
    sam2_device: str = "cuda",
    keyframe_interval: int = 10,
):
    csv_path = os.path.abspath(os.path.expanduser(csv_path))
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    initial_csv_path = csv_path
    propagation_script_resolved = resolve_propagation_script(propagation_script)

    raw_masks_dir = get_raw_masks_dir(csv_path)
    df_raw = pd.read_csv(csv_path)
    df_raw = sync_label_columns_for_gui(df_raw)
    df = ensure_cols(df_raw).reset_index(drop=True)
    df, frames_dir_res = normalize_paths(df, csv_path=csv_path, frames_dir=frames_dir)
    df = import_previous_labels(df, pre_csv=pre_csv)
    df.to_csv(csv_path, index=False)

    frames = build_frames(df)
    nframes = len(frames)
    if nframes <= 0:
        raise RuntimeError("No frames in CSV (missing frame_file/frame_index).")

    ctx0 = load_frame_context(
        df,
        frames,
        frames_dir_fallback=frames_dir_res,
        frame_pos=0,
        only_smooth=False,
        class_filter="all",
        show_ignored=False,
    )
    ctx0["nframes"] = nframes

    UI_HEIGHT = 720
    TABLE_HEIGHT = 430

    def save_df():
        df["label"] = df["label_final"].apply(norm_label)
        df.to_csv(csv_path, index=False)

    def refresh(ctx: Dict[str, Any], show_all: bool, draw_labels_text: bool):
        img_out = render(df, ctx, show_all=show_all, draw_labels_text=draw_labels_text)
        tab_out = build_mask_table(df, ctx)
        return img_out, tab_out, ctx

    def refresh_no_table(ctx: Dict[str, Any], show_all: bool, draw_labels_text: bool):
        img_out = render(df, ctx, show_all=show_all, draw_labels_text=draw_labels_text)
        return img_out, ctx

    def pack_full(img_val: np.ndarray, tab_val: pd.DataFrame, ctx_val: Dict[str, Any]):
        brush_mode = ctx_val.get("click_mode", "select") == "brush"
        img_update = gr.update(value=img_val, visible=not brush_mode)
        brush_update = gr.update(value=make_editor_value_for_frame(ctx_val["img"]), visible=brush_mode)
        return img_update, tab_val, ctx_val, brush_update

    def pack_no_table(img_val: np.ndarray, ctx_val: Dict[str, Any]):
        brush_mode = ctx_val.get("click_mode", "select") == "brush"
        img_update = gr.update(value=img_val, visible=not brush_mode)
        brush_update = gr.update(value=make_editor_value_for_frame(ctx_val["img"]), visible=brush_mode)
        return img_update, ctx_val, brush_update

    def reload_current_ctx(old_ctx: Dict[str, Any], click_mode_override: Optional[str] = None):
        ctx2 = load_frame_context(
            df,
            frames,
            frames_dir_fallback=frames_dir_res,
            frame_pos=int(old_ctx["frame_pos"]),
            only_smooth=bool(old_ctx.get("only_smooth", False)),
            class_filter=str(old_ctx.get("class_filter", "all")),
            show_ignored=bool(old_ctx.get("show_ignored", False)),
        )
        ctx2["nframes"] = nframes
        ctx2["click_mode"] = click_mode_override if click_mode_override is not None else old_ctx.get("click_mode", "select")
        ctx2["checked"] = set(old_ctx.get("checked", set()) or set())
        ctx2["merge_a"] = old_ctx.get("merge_a")
        ctx2["merge_b"] = old_ctx.get("merge_b")
        return ctx2

    def assign_track_for_new_row(new_rid: int, lab: str):
        lab = norm_label(lab)
        if lab in UNIQUE_TRACKS:
            df.at[new_rid, "track_id"] = int(UNIQUE_TRACKS[lab])
            enforce_unique_label_in_frame(df, str(df.at[new_rid, "frame_file"]), new_rid, lab)
            return

        if lab == "spatter":
            match_tid = find_matching_spatter_track_id(
                df,
                new_rid,
                max_frame_gap=1,
                max_center_dist=90.0,
                min_iou=0.01,
            )
            if match_tid is None:
                match_tid = next_free_spatter_track_id(df)
            df.at[new_rid, "track_id"] = int(match_tid)

    def build_manual_row(
        frame_file: str,
        frame_index: int,
        img_path: str,
        mask_index: int,
        mask_path: str,
        seg: np.ndarray,
        label: str,
        why: str,
        manual_note: str,
        source: str,
        track_id: int = -1,
    ) -> Dict[str, Any]:
        cx, cy = centroid(seg)
        bx, by, bw, bh = bbox_from_mask(seg)
        return {
            "frame_file": frame_file,
            "frame_index": int(frame_index),
            "image_path": img_path,
            "mask_index": int(mask_index),
            "mask_path": mask_path,
            "area": int(seg.sum()),
            "bbox_x": int(bx),
            "bbox_y": int(by),
            "bbox_w": int(bw),
            "bbox_h": int(bh),
            "cx_geo": float(cx),
            "cy_geo": float(cy),
            "segment_type": "",
            "track_id": int(track_id),
            "static_overlap": 0.0,
            "auto_static": False,
            "track_len": -1,
            "is_smooth_top10": False,
            "smooth_rank": -1,
            "label_final": label,
            "label": label,
            "ignore": False,
            "why": why,
            "broken_rule": "",
            "manual_note": manual_note,
            "old_label_final": "",
            "old_track_id": -1,
            "edited_manually": True,
            "source": source,
            "review_origin": "manual_edit",
            "is_manual": True,
            "predicted_iou": np.nan,
            "stability_score": np.nan,
            "sam_quality": np.nan,
            "frame_uncertainty": np.nan,
            "is_review_keyframe": False,
            "needs_review": False,
            "review_reason": "manual_added",
            "propagation_seed_frame_index": -1,
            "propagation_seed_frame_file": "",
            "propagation_obj_id": -1,
            "propagation_segment_start_pos": -1,
            "propagation_segment_end_pos": -1,
        }

    def _load_ctx_at(
        frame_pos: int,
        only_smooth: bool,
        class_filter: str,
        show_ignored: bool,
        old_ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx2 = load_frame_context(
            df,
            frames,
            frames_dir_fallback=frames_dir_res,
            frame_pos=frame_pos,
            only_smooth=only_smooth,
            class_filter=class_filter,
            show_ignored=show_ignored,
        )
        ctx2["nframes"] = nframes
        if old_ctx is not None:
            ctx2["click_mode"] = old_ctx.get("click_mode", "select")
            ctx2["merge_a"] = old_ctx.get("merge_a")
            ctx2["merge_b"] = old_ctx.get("merge_b")
        return ctx2

    def go_frame(
        frame_pos: int,
        only_smooth: bool,
        class_filter: str,
        show_ignored: bool,
        show_all: bool,
        draw_labels_text: bool,
        ctx: Dict[str, Any],
    ):
        frame_pos = safe_frame_pos(safe_int(frame_pos, 0), nframes)
        ctx2 = _load_ctx_at(
            frame_pos,
            only_smooth,
            class_filter,
            show_ignored,
            ctx,
        )
        return pack_full(*refresh(ctx2, show_all, draw_labels_text))

    def go_relative(
        delta: int,
        only_smooth: bool,
        class_filter: str,
        show_ignored: bool,
        show_all: bool,
        draw_labels_text: bool,
        ctx: Dict[str, Any],
    ):
        new_pos = safe_frame_pos(int(ctx["frame_pos"]) + int(delta), nframes)
        ctx2 = _load_ctx_at(
            new_pos,
            only_smooth,
            class_filter,
            show_ignored,
            ctx,
        )
        img_u, tab_u, ctx_u, brush_u = pack_full(*refresh(ctx2, show_all, draw_labels_text))
        return img_u, tab_u, ctx_u, brush_u, new_pos

    def go_review_relative(
        direction: int,
        review_mode: str,
        uncertainty_thr: float,
        only_smooth: bool,
        class_filter: str,
        show_ignored: bool,
        show_all: bool,
        draw_labels_text: bool,
        ctx: Dict[str, Any],
    ):
        new_pos = find_next_review_pos(
            df=df,
            frames=frames,
            current_pos=int(ctx["frame_pos"]),
            mode=review_mode,
            direction=int(direction),
            uncertainty_thr=float(uncertainty_thr),
        )
        ctx2 = _load_ctx_at(
            new_pos,
            only_smooth,
            class_filter,
            show_ignored,
            ctx,
        )
        img_u, tab_u, ctx_u, brush_u = pack_full(*refresh(ctx2, show_all, draw_labels_text))
        return img_u, tab_u, ctx_u, brush_u, new_pos

    def apply_filters(
        only_smooth: bool,
        class_filter: str,
        show_ignored: bool,
        show_all: bool,
        draw_labels_text: bool,
        ctx: Dict[str, Any],
    ):
        ctx2 = _load_ctx_at(
            int(ctx["frame_pos"]),
            only_smooth,
            class_filter,
            show_ignored,
            ctx,
        )
        ctx2["checked"] = set(ctx.get("checked", set()) or set())
        return pack_full(*refresh(ctx2, show_all, draw_labels_text))

    def save_now(ctx: Dict[str, Any]):
        save_df()
        return f"Saved CSV: {csv_path}"

    def reload_dataset_from_csv(new_csv_path: str) -> None:
        """Switch the running UI to another annotation manifest."""
        nonlocal csv_path, raw_masks_dir, df, frames, nframes

        new_csv_path = os.path.abspath(os.path.expanduser(str(new_csv_path).strip()))
        if not os.path.isfile(new_csv_path):
            raise FileNotFoundError(new_csv_path)

        csv_path = new_csv_path
        raw_masks_dir = get_raw_masks_dir(csv_path)

        df_raw2 = pd.read_csv(csv_path)
        df_raw2 = sync_label_columns_for_gui(df_raw2)
        df2 = ensure_cols(df_raw2).reset_index(drop=True)
        df2, _ = normalize_paths(df2, csv_path=csv_path, frames_dir=frames_dir_res)
        df = df2
        df.to_csv(csv_path, index=False)

        frames = build_frames(df)
        nframes = len(frames)
        if nframes <= 0:
            raise RuntimeError("No frames found in propagated review CSV.")

    def _expected_propagation_segments(interval_value: int) -> int:
        try:
            src_df = pd.read_csv(initial_csv_path)
            if "is_review_keyframe" in src_df.columns:
                flags = src_df["is_review_keyframe"].fillna(False).astype(bool)
                vals = sorted(set(pd.to_numeric(
                    src_df.loc[flags, "frame_index"], errors="coerce"
                ).dropna().astype(int).tolist()))
                if vals:
                    return max(1, len(vals))
        except Exception:
            pass

        step = max(1, safe_int(interval_value, 10))
        return max(1, len(range(0, nframes, step)))

    def run_sam2_keyframe_propagation(
        interval_value: int,
        overwrite_existing: bool,
        prop_script_value: str,
        repo_value: str,
        checkpoint_value: str,
        device_value: str,
        model_cfg_value: str,
    ):
        """
        Save current keyframe edits, launch the standalone propagation script,
        and stream its console output back into the Gradio UI.
        """
        save_df()

        # Propagation should start from the original keyframe-review CSV.
        if os.path.abspath(csv_path) != os.path.abspath(initial_csv_path):
            msg = (
                "The UI is currently showing a propagated review CSV. "
                "Reload the original keyframe CSV before starting a new propagation run."
            )
            yield msg, 0, ""
            return

        script_path = resolve_propagation_script(prop_script_value or propagation_script_resolved)
        if not script_path:
            msg = (
                "Propagation script not found. Set the path to "
                "propagate_keyframes_sam2.py in the SAM2 propagation settings."
            )
            yield msg, 0, ""
            return

        interval_value = max(1, safe_int(interval_value, keyframe_interval))
        run_dir = os.path.dirname(os.path.abspath(initial_csv_path))
        output_dir = os.path.join(run_dir, "sam2_propagated_keyframes")
        combined_csv = os.path.join(
            output_dir,
            "labels_manifest_combined_original_plus_propagated.csv",
        )

        if os.path.isdir(output_dir) and not overwrite_existing:
            msg = (
                f"Propagation output already exists:\n{output_dir}\n\n"
                "Enable 'Overwrite existing propagation output' and run again "
                "if you want to replace it."
            )
            yield msg, 0, combined_csv if os.path.isfile(combined_csv) else ""
            return

        frames_arg = frames_dir_res
        if not frames_arg:
            # Fall back to the directory containing the first frame.
            try:
                frames_arg = os.path.dirname(str(frames.iloc[0]["image_path"]))
            except Exception:
                frames_arg = ""

        if not frames_arg or not os.path.isdir(frames_arg):
            yield f"Frames directory not found: {frames_arg}", 0, ""
            return

        cmd = [
            sys.executable,
            "-u",
            script_path,
            "--csv", initial_csv_path,
            "--frames", frames_arg,
            "--output-dir", output_dir,
            "--keyframe-interval", str(interval_value),
            "--manual-only",
            "--device", str(device_value or sam2_device),
            "--model-cfg", str(model_cfg_value or sam2_model_cfg),
        ]

        repo_value = str(repo_value or sam2_repo or "").strip()
        checkpoint_value = str(checkpoint_value or sam2_checkpoint or "").strip()

        if repo_value:
            cmd.extend(["--repo-dir", repo_value])
        if checkpoint_value:
            cmd.extend(["--checkpoint-path", checkpoint_value])
        if overwrite_existing:
            cmd.append("--overwrite")

        total_segments = _expected_propagation_segments(interval_value)
        segment_count = 0
        log_lines: List[str] = []

        pretty_cmd = subprocess.list2cmdline(cmd)
        log_lines.append("Saving current manual labels...")
        log_lines.append("Starting SAM2 keyframe propagation...")
        log_lines.append(pretty_cmd)
        yield "\n".join(log_lines), 1, ""

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
            )
        except Exception as e:
            yield f"Could not start propagation:\n{e}", 0, ""
            return

        assert proc.stdout is not None

        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            if not line:
                continue

            log_lines.append(line)
            # Keep the visible log responsive instead of growing forever.
            if len(log_lines) > 120:
                log_lines = log_lines[-120:]

            if line.startswith("[SEGMENT]"):
                segment_count += 1

            # Reserve the last few percent for CSV/report generation.
            progress_value = min(
                95,
                max(2, int(round(95.0 * segment_count / max(1, total_segments)))),
            )

            yield "\n".join(log_lines), progress_value, ""

        return_code = proc.wait()

        if return_code != 0:
            log_lines.append(f"\n[FAILED] Propagation exited with code {return_code}.")
            yield "\n".join(log_lines[-120:]), min(95, max(1, segment_count)), ""
            return

        if not os.path.isfile(combined_csv):
            log_lines.append(
                "\n[FAILED] Propagation finished, but the combined review CSV was not found:\n"
                + combined_csv
            )
            yield "\n".join(log_lines[-120:]), 100, ""
            return

        log_lines.append("\n[OK] SAM2 propagation completed.")
        log_lines.append("[OK] Click 'Open propagated review' to inspect the propagated frames.")
        yield "\n".join(log_lines[-120:]), 100, combined_csv

    def open_propagated_review(
        combined_csv_value: str,
        only_smooth: bool,
        class_filter: str,
        show_ignored: bool,
        show_all: bool,
        draw_labels_text: bool,
        uncertainty_value: float,
        ctx: Dict[str, Any],
    ):
        combined_csv_value = str(combined_csv_value or "").strip()
        if not combined_csv_value or not os.path.isfile(combined_csv_value):
            img_u, tab_u, ctx_u, brush_u = pack_full(*refresh(ctx, show_all, draw_labels_text))
            return (
                img_u,
                tab_u,
                ctx_u,
                brush_u,
                gr.update(value=int(ctx.get("frame_pos", 0))),
                gr.update(value="Needs review"),
                "Combined propagated review CSV not found. Run propagation first.",
            )

        reload_dataset_from_csv(combined_csv_value)

        start_pos = 0
        review_positions = review_frame_positions(
            df,
            frames,
            "Needs review",
            float(uncertainty_value),
        )
        if review_positions:
            start_pos = int(review_positions[0])

        ctx2 = _load_ctx_at(
            start_pos,
            only_smooth,
            class_filter,
            show_ignored,
            ctx,
        )
        ctx2["click_mode"] = "select"
        ctx2["checked"] = set()

        img_u, tab_u, ctx_u, brush_u = pack_full(
            *refresh(ctx2, show_all, draw_labels_text)
        )

        return (
            img_u,
            tab_u,
            ctx_u,
            brush_u,
            gr.update(
                value=start_pos,
                label=f"Frame index (0 to {max(0, nframes - 1)})",
            ),
            gr.update(value="Needs review"),
            f"Loaded propagated review CSV: {csv_path}",
        )

    def _fix_click_xy(x: int, y: int, img: np.ndarray) -> Tuple[int, int]:
        H, W = img.shape[:2]
        if 0 <= x < W and 0 <= y < H:
            return x, y
        if 0 <= y < W and 0 <= x < H:
            return y, x
        return int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))

    def on_click(show_all: bool, draw_labels_text: bool, ctx: Dict[str, Any], evt: gr.SelectData):
        if evt is None or evt.index is None:
            return pack_full(*refresh(ctx, show_all, draw_labels_text))

        x, y = int(evt.index[0]), int(evt.index[1])
        x, y = _fix_click_xy(x, y, ctx["img"])
        mode = ctx.get("click_mode", "select")

        if mode == "seed1":
            ctx["seed1"] = (x, y)
            return pack_full(*refresh(ctx, show_all, draw_labels_text))

        if mode == "seed2":
            ctx["seed2"] = (x, y)
            return pack_full(*refresh(ctx, show_all, draw_labels_text))

        if mode == "polygon":
            pts = list(ctx.get("poly_points", []) or [])
            pts.append((x, y))
            ctx["poly_points"] = pts
            return pack_full(*refresh(ctx, show_all, draw_labels_text))

        hits = find_hits(df, ctx, x, y)
        ctx["hits"] = hits
        ctx["hit_i"] = 0
        if hits:
            ctx["selected"] = hits[0]
        return pack_full(*refresh(ctx, show_all, draw_labels_text))

    def cycle_hit(delta: int, show_all: bool, draw_labels_text: bool, ctx: Dict[str, Any]):
        hits = ctx.get("hits", [])
        if hits:
            ctx["hit_i"] = (ctx.get("hit_i", 0) + delta) % len(hits)
            ctx["selected"] = hits[ctx["hit_i"]]
        return pack_full(*refresh(ctx, show_all, draw_labels_text))

    def set_click_mode(mode: str, show_all: bool, draw_labels_text: bool, ctx: Dict[str, Any]):
        ctx["click_mode"] = mode
        if mode != "polygon":
            ctx["poly_points"] = []
        return pack_full(*refresh(ctx, show_all, draw_labels_text))

    def clear_seeds(show_all: bool, draw_labels_text: bool, ctx: Dict[str, Any]):
        ctx["seed1"] = None
        ctx["seed2"] = None
        return pack_full(*refresh(ctx, show_all, draw_labels_text))

    def clear_checked(show_all: bool, draw_labels_text: bool, ctx: Dict[str, Any]):
        ctx["checked"] = set()
        return pack_full(*refresh(ctx, show_all, draw_labels_text))

    def undo_polygon_point(show_all: bool, draw_labels_text: bool, ctx: Dict[str, Any]):
        pts = list(ctx.get("poly_points", []) or [])
        if pts:
            pts.pop()
        ctx["poly_points"] = pts
        return pack_full(*refresh(ctx, show_all, draw_labels_text))

    def clear_polygon_points(show_all: bool, draw_labels_text: bool, ctx: Dict[str, Any]):
        ctx["poly_points"] = []
        return pack_full(*refresh(ctx, show_all, draw_labels_text))

    def add_polygon_mask(
        poly_label: str,
        manual_note: str,
        show_all: bool,
        draw_labels_text: bool,
        ctx: Dict[str, Any],
    ):
        pts = list(ctx.get("poly_points", []) or [])
        if len(pts) < 3:
            return pack_full(*refresh(ctx, show_all, draw_labels_text))

        lab = norm_label(poly_label) or "spatter"
        frame_file = str(ctx["frame_file"])
        frame_index = int(ctx["frame_index"])
        img_path = str(ctx.get("img_path", ""))
        H, W = ctx["img"].shape[:2]

        seg = polygon_to_mask((H, W), pts)
        seg = clean_manual_mask(seg, min_area=8)
        if int(seg.sum()) <= 0:
            return pack_full(*refresh(ctx, show_all, draw_labels_text))

        new_idx = next_mask_indices_for_frame(df, frame_file, n=1)[0]
        new_path = make_raw_mask_path(raw_masks_dir, frame_file, new_idx)
        write_mask_bool(new_path, seg)

        note = str(manual_note).strip() if str(manual_note).strip() else "Added manually with polygon"
        new_row = build_manual_row(
            frame_file=frame_file,
            frame_index=frame_index,
            img_path=img_path,
            mask_index=new_idx,
            mask_path=new_path,
            seg=seg,
            label=lab,
            why="manual_polygon_add",
            manual_note=note,
            source="manual_polygon",
        )
        df.loc[len(df)] = new_row
        new_rid = int(df.index[-1])
        assign_track_for_new_row(new_rid, lab)
        save_df()

        ctx2 = reload_current_ctx(ctx, click_mode_override="select")
        ctx2["selected"] = new_rid
        return pack_full(*refresh(ctx2, show_all, draw_labels_text))

    def clear_brush(ctx: Dict[str, Any]):
        return gr.update(value=make_editor_value_for_frame(ctx["img"]), visible=True)

    def add_brush_mask(
        brush_value: Any,
        brush_label: str,
        manual_note: str,
        show_all: bool,
        draw_labels_text: bool,
        ctx: Dict[str, Any],
    ):
        lab = norm_label(brush_label) or "spatter"
        frame_file = str(ctx["frame_file"])
        frame_index = int(ctx["frame_index"])
        img_path = str(ctx.get("img_path", ""))
        H, W = ctx["img"].shape[:2]

        seg_all = _editor_to_binary_mask(brush_value, (H, W))
        seg_all = clean_manual_mask(seg_all, min_area=8)

        if int(seg_all.sum()) <= 0:
            img_out, tab_out, ctx_out = refresh(ctx, show_all, draw_labels_text)
            img_upd, tab_out, ctx_out, brush_upd = pack_full(img_out, tab_out, ctx_out)
            if ctx_out.get("click_mode") == "brush":
                brush_upd = gr.update(value=make_editor_value_for_frame(ctx_out["img"]), visible=True)
            return img_upd, tab_out, ctx_out, brush_upd

        # Split one brush drawing into separate connected masks
        seg_u8 = seg_all.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seg_u8, connectivity=8)

        components = []
        for comp_id in range(1, num_labels):
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area < 8:
                continue
            comp_mask = labels == comp_id
            components.append(comp_mask)

        if len(components) == 0:
            return pack_full(*refresh(ctx, show_all, draw_labels_text))

        new_indices = next_mask_indices_for_frame(df, frame_file, n=len(components))
        new_rids = []

        note = str(manual_note).strip() if str(manual_note).strip() else "Added manually with brush"

        for seg, new_idx in zip(components, new_indices):
            new_path = make_raw_mask_path(raw_masks_dir, frame_file, new_idx)
            write_mask_bool(new_path, seg)

            new_row = build_manual_row(
                frame_file=frame_file,
                frame_index=frame_index,
                img_path=img_path,
                mask_index=new_idx,
                mask_path=new_path,
                seg=seg,
                label=lab,
                why="manual_brush_add_separate_component",
                manual_note=note,
                source="manual_brush",
            )

            df.loc[len(df)] = new_row
            new_rid = int(df.index[-1])
            assign_track_for_new_row(new_rid, lab)
            new_rids.append(new_rid)

        save_df()

        ctx2 = reload_current_ctx(
            ctx,
            click_mode_override="select",
        )
        ctx2["selected"] = new_rids[-1] if new_rids else None

        return pack_full(*refresh(ctx2, show_all, draw_labels_text))
    def exit_brush_mode(show_all: bool, draw_labels_text: bool, ctx: Dict[str, Any]):
        ctx["click_mode"] = "select"
        return pack_full(*refresh(ctx, show_all, draw_labels_text))

    def on_table_change(tab_value: Any, show_all: bool, draw_labels_text: bool, ctx: Dict[str, Any]):
        tab = _tab_to_df(tab_value)
        checked_ids: List[int] = []
        if not tab.empty and "checked" in tab.columns and "row_id" in tab.columns:
            try:
                v = tab["checked"].astype(str).str.lower().isin(["true", "1", "yes"])
                checked_ids = tab.loc[v, "row_id"].astype(int).tolist()
            except Exception:
                checked_ids = []
        ctx["checked"] = set(checked_ids)
        return pack_no_table(*refresh_no_table(ctx, show_all, draw_labels_text))

    def set_label(
        label: str,
        auto_prop: bool,
        prop_iou: float,
        prop_same_bbox: bool,
        apply_to_checked: bool,
        broken_rule: str,
        manual_note: str,
        show_all: bool,
        draw_labels_text: bool,
        ctx: Dict[str, Any],
    ):
        prop_iou = safe_float(prop_iou, 0.98)
        lab = norm_label(label)

        checked = ctx.get("checked", set()) or set()
        if apply_to_checked and checked:
            targets = sorted([int(x) for x in checked])
        else:
            rid = ctx.get("selected")
            targets = [int(rid)] if rid is not None else []

        if not targets:
            return pack_full(*refresh(ctx, show_all, draw_labels_text))

        def get_assigned_tid_for_row(row_id: int) -> Optional[int]:
            if lab in UNIQUE_TRACKS:
                return int(UNIQUE_TRACKS[lab])
            if lab == "spatter":
                match_tid = find_matching_spatter_track_id(
                    df,
                    row_id,
                    max_frame_gap=1,
                    max_center_dist=90.0,
                    min_iou=0.05,
                )
                if match_tid is not None:
                    return int(match_tid)
                return int(next_free_spatter_track_id(df))
            return None

        if lab in UNIQUE_LABELS:
            by_frame: Dict[str, List[int]] = {}
            for rid in targets:
                if bool(df.at[rid, "ignore"]):
                    continue
                ff = str(df.at[rid, "frame_file"])
                by_frame.setdefault(ff, []).append(int(rid))

            for frame_file, group in by_frame.items():
                sel = ctx.get("selected")
                keep_rid = int(sel) if sel in group else int(group[0])
                old_lab = norm_label(df.at[keep_rid, "label_final"])
                old_tid = int(df.at[keep_rid, "track_id"]) if "track_id" in df.columns else -1
                new_tid = get_assigned_tid_for_row(keep_rid)

                df.at[keep_rid, "old_label_final"] = old_lab
                df.at[keep_rid, "old_track_id"] = old_tid
                df.at[keep_rid, "edited_manually"] = True
                df.at[keep_rid, "broken_rule"] = str(broken_rule).strip()
                df.at[keep_rid, "manual_note"] = str(manual_note).strip()
                df.at[keep_rid, "label_final"] = lab
                df.at[keep_rid, "label"] = lab
                if new_tid is not None:
                    df.at[keep_rid, "track_id"] = int(new_tid)

                enforce_unique_label_in_frame(df, frame_file, keep_rid, lab)

            if auto_prop:
                print(f"[INFO] Auto-propagation skipped for unique label '{lab}' to avoid duplicates.")
            save_df()
            return pack_full(*refresh(ctx, show_all, draw_labels_text))

        for rid in targets:
            old_lab = norm_label(df.at[rid, "label_final"])
            old_tid = int(df.at[rid, "track_id"]) if "track_id" in df.columns else -1
            new_tid = get_assigned_tid_for_row(rid)

            df.at[rid, "old_label_final"] = old_lab
            df.at[rid, "old_track_id"] = old_tid
            df.at[rid, "edited_manually"] = True
            df.at[rid, "broken_rule"] = str(broken_rule).strip()
            df.at[rid, "manual_note"] = str(manual_note).strip()
            df.at[rid, "label_final"] = lab
            df.at[rid, "label"] = lab
            if new_tid is not None:
                df.at[rid, "track_id"] = int(new_tid)

            if auto_prop and lab != "":
                added = propagate_label_forward_same_position(
                    df,
                    rid_start=int(rid),
                    label=lab,
                    new_track_id=new_tid,
                    iou_thr=float(prop_iou),
                    require_same_bbox=bool(prop_same_bbox),
                    only_if_empty=True,
                )
                if added:
                    print(f"[INFO] Propagated label '{lab}' from row {rid} to {added} masks forward.")

        save_df()
        return pack_full(*refresh(ctx, show_all, draw_labels_text))

    def quick_label(
        lab: str,
        auto_forward: bool,
        review_mode: str,
        uncertainty_thr: float,
        auto_prop: bool,
        prop_iou: float,
        prop_same_bbox: bool,
        apply_to_checked: bool,
        broken_rule: str,
        manual_note: str,
        only_smooth: bool,
        class_filter: str,
        show_ignored: bool,
        show_all: bool,
        draw_labels_text: bool,
        ctx: Dict[str, Any],
    ):
        img_upd, tab_out, ctx_out, brush_upd = set_label(
            lab,
            auto_prop,
            prop_iou,
            prop_same_bbox,
            apply_to_checked,
            broken_rule,
            manual_note,
            show_all,
            draw_labels_text,
            ctx,
        )
        new_pos = int(ctx_out.get("frame_pos", ctx.get("frame_pos", 0)))

        if auto_forward:
            candidate_pos = find_next_review_pos(
                df=df,
                frames=frames,
                current_pos=int(ctx_out["frame_pos"]),
                mode=review_mode,
                direction=1,
                uncertainty_thr=float(uncertainty_thr),
            )
            if candidate_pos != int(ctx_out["frame_pos"]):
                new_pos = int(candidate_pos)
                ctx2 = _load_ctx_at(
                    new_pos,
                    only_smooth,
                    class_filter,
                    show_ignored,
                    ctx_out,
                )
                ctx2["click_mode"] = "select"
                ctx2["checked"] = set()
                img_upd, tab_out, ctx_out, brush_upd = pack_full(*refresh(ctx2, show_all, draw_labels_text))

        return img_upd, tab_out, ctx_out, brush_upd, new_pos

    def set_track_id(
        new_track_id: int,
        apply_to_checked: bool,
        show_all: bool,
        draw_labels_text: bool,
        ctx: Dict[str, Any],
    ):
        tid = safe_int(new_track_id, -1)
        checked = ctx.get("checked", set()) or set()
        if apply_to_checked and checked:
            targets = sorted([int(x) for x in checked])
        else:
            rid = ctx.get("selected")
            targets = [int(rid)] if rid is not None else []

        if not targets:
            return pack_full(*refresh(ctx, show_all, draw_labels_text))

        for rid in targets:
            df.at[rid, "track_id"] = tid
            df.at[rid, "edited_manually"] = True
        save_df()
        return pack_full(*refresh(ctx, show_all, draw_labels_text))

    def toggle_ignore(show_all: bool, draw_labels_text: bool, ctx: Dict[str, Any]):
        rid = ctx.get("selected")
        if rid is not None:
            df.at[rid, "ignore"] = not bool(df.at[rid, "ignore"])
            df.at[rid, "edited_manually"] = True
            save_df()
            ctx2 = reload_current_ctx(ctx)
            return pack_full(*refresh(ctx2, show_all, draw_labels_text))
        return pack_full(*refresh(ctx, show_all, draw_labels_text))

    def soft_delete_selected(show_all: bool, draw_labels_text: bool, ctx: Dict[str, Any]):
        rid = ctx.get("selected")
        if rid is None:
            return pack_full(*refresh(ctx, show_all, draw_labels_text))

        df.at[rid, "ignore"] = True
        df.at[rid, "why"] = str(df.at[rid, "why"]) + " | manually_deleted"
        df.at[rid, "edited_manually"] = True
        save_df()

        checked = set(ctx.get("checked", set()) or set())
        checked.discard(int(rid))
        ctx2 = reload_current_ctx(ctx)
        ctx2["checked"] = checked
        return pack_full(*refresh(ctx2, show_all, draw_labels_text))

    def soft_delete_unlabeled_dynamic_in_frame(show_all: bool, draw_labels_text: bool, ctx: Dict[str, Any]):
        frame_file = str(ctx["frame_file"])
        target_rids = df.index[(df["frame_file"] == frame_file) & (~df["ignore"].astype(bool))].tolist()
        deleted = []
        for rid in target_rids:
            lab = norm_label(df.at[rid, "label_final"])
            if lab in ("", "dynamic_other"):
                df.at[rid, "ignore"] = True
                df.at[rid, "why"] = str(df.at[rid, "why"]) + " | bulk_deleted_unlabeled_dynamic"
                df.at[rid, "edited_manually"] = True
                deleted.append(int(rid))
        save_df()

        ctx2 = reload_current_ctx(ctx)
        return pack_full(*refresh(ctx2, show_all, draw_labels_text))

    def on_table_select(show_all: bool, draw_labels_text: bool, ctx: Dict[str, Any], evt: gr.SelectData):
        if evt is None or evt.index is None:
            return pack_no_table(*refresh_no_table(ctx, show_all, draw_labels_text))

        row = int(evt.index[0]) if isinstance(evt.index, (list, tuple)) else int(evt.index)
        ids = ctx.get("table_row_ids", [])
        if 0 <= row < len(ids):
            rid = int(ids[row])
            ctx["selected"] = rid
            ctx["hits"] = [rid]
            ctx["hit_i"] = 0
        return pack_no_table(*refresh_no_table(ctx, show_all, draw_labels_text))

    def do_split(show_all: bool, draw_labels_text: bool, ctx: Dict[str, Any]):
        rid = ctx.get("selected")
        if rid is None:
            img_out, tab_out, ctx_ = refresh(ctx, show_all, draw_labels_text)
            img_upd, tab_out, ctx_, brush_upd = pack_full(img_out, tab_out, ctx_)
            return img_upd, "No mask selected.", tab_out, ctx_, brush_upd

        if ctx.get("seed1") is None or ctx.get("seed2") is None:
            img_out, tab_out, ctx_ = refresh(ctx, show_all, draw_labels_text)
            img_upd, tab_out, ctx_, brush_upd = pack_full(img_out, tab_out, ctx_)
            return img_upd, "Need seed1 and seed2 (set click mode seed1/seed2 and click).", tab_out, ctx_, brush_upd

        seg = read_mask_bool(str(df.at[rid, "mask_path"]))
        m1, m2 = split_mask_watershed_points(seg, ctx["seed1"], ctx["seed2"])
        if m1.sum() == 0 or m2.sum() == 0:
            img_out, tab_out, ctx_ = refresh(ctx, show_all, draw_labels_text)
            img_upd, tab_out, ctx_, brush_upd = pack_full(img_out, tab_out, ctx_)
            return img_upd, "Split failed (make sure both seeds are inside the mask and far enough).", tab_out, ctx_, brush_upd

        frame_file = str(df.at[rid, "frame_file"])
        idx1, idx2 = next_mask_indices_for_frame(df, frame_file, n=2)
        p1 = make_raw_mask_path(raw_masks_dir, frame_file, idx1)
        p2 = make_raw_mask_path(raw_masks_dir, frame_file, idx2)
        write_mask_bool(p1, m1)
        write_mask_bool(p2, m2)

        df.at[rid, "ignore"] = True
        df.at[rid, "edited_manually"] = True

        img_path = str(df.at[rid, "image_path"]) if "image_path" in df.columns else ctx.get("img_path", "")
        parent_label = norm_label(df.at[rid, "label_final"])
        parent_tid = int(df.at[rid, "track_id"]) if "track_id" in df.columns else -1

        row1 = build_manual_row(
            frame_file=frame_file,
            frame_index=int(df.at[rid, "frame_index"]),
            img_path=img_path,
            mask_index=idx1,
            mask_path=p1,
            seg=m1,
            label=parent_label,
            why="manual_split",
            manual_note="Created by split",
            source="manual_split",
            track_id=parent_tid,
        )
        row2 = build_manual_row(
            frame_file=frame_file,
            frame_index=int(df.at[rid, "frame_index"]),
            img_path=img_path,
            mask_index=idx2,
            mask_path=p2,
            seg=m2,
            label=parent_label,
            why="manual_split",
            manual_note="Created by split",
            source="manual_split",
            track_id=parent_tid,
        )
        df.loc[len(df)] = row1
        df.loc[len(df)] = row2
        save_df()

        ctx2 = reload_current_ctx(ctx, click_mode_override="select")
        ctx2["seed1"] = None
        ctx2["seed2"] = None
        img_out, tab_out, ctx_ = refresh(ctx2, show_all, draw_labels_text)
        img_upd, tab_out, ctx_, brush_upd = pack_full(img_out, tab_out, ctx_)
        return img_upd, f"Split OK: created {os.path.basename(p1)} and {os.path.basename(p2)} (parent ignored).", tab_out, ctx_, brush_upd

    def set_merge_a(show_all: bool, draw_labels_text: bool, ctx: Dict[str, Any]):
        ctx["merge_a"] = ctx.get("selected")
        return pack_full(*refresh(ctx, show_all, draw_labels_text))

    def set_merge_b(show_all: bool, draw_labels_text: bool, ctx: Dict[str, Any]):
        ctx["merge_b"] = ctx.get("selected")
        return pack_full(*refresh(ctx, show_all, draw_labels_text))

    def do_merge(show_all: bool, draw_labels_text: bool, ctx: Dict[str, Any]):
        a = ctx.get("merge_a")
        b = ctx.get("merge_b")
        if a is None or b is None or a == b:
            img_out, tab_out, ctx_ = refresh(ctx, show_all, draw_labels_text)
            img_upd, tab_out, ctx_, brush_upd = pack_full(img_out, tab_out, ctx_)
            return img_upd, "Merge needs two different masks: set Merge A and Merge B.", tab_out, ctx_, brush_upd

        seg_a = read_mask_bool(str(df.at[a, "mask_path"]))
        seg_b = read_mask_bool(str(df.at[b, "mask_path"]))
        merged = clean_manual_mask(seg_a | seg_b, min_area=8)
        frame_file = str(df.at[a, "frame_file"])
        new_idx = next_mask_indices_for_frame(df, frame_file, n=1)[0]
        new_path = make_raw_mask_path(raw_masks_dir, frame_file, new_idx)
        write_mask_bool(new_path, merged)

        df.at[a, "ignore"] = True
        df.at[b, "ignore"] = True
        df.at[a, "edited_manually"] = True
        df.at[b, "edited_manually"] = True

        lab_a = norm_label(df.at[a, "label_final"])
        lab_b = norm_label(df.at[b, "label_final"])
        new_lab = lab_a if lab_a else lab_b
        tid_a = int(df.at[a, "track_id"]) if "track_id" in df.columns else -1
        tid_b = int(df.at[b, "track_id"]) if "track_id" in df.columns else -1
        new_tid = tid_a if tid_a >= 0 else tid_b
        img_path = str(df.at[a, "image_path"]) if "image_path" in df.columns else ctx.get("img_path", "")

        new_row = build_manual_row(
            frame_file=frame_file,
            frame_index=int(df.at[a, "frame_index"]),
            img_path=img_path,
            mask_index=new_idx,
            mask_path=new_path,
            seg=merged,
            label=new_lab,
            why="manual_merge",
            manual_note="Created by merge",
            source="manual_merge",
            track_id=new_tid,
        )
        df.loc[len(df)] = new_row
        new_rid = int(df.index[-1])
        if new_lab in UNIQUE_LABELS:
            enforce_unique_label_in_frame(df, frame_file, new_rid, new_lab)
        save_df()

        ctx2 = reload_current_ctx(ctx)
        ctx2["merge_a"] = None
        ctx2["merge_b"] = None
        img_out, tab_out, ctx_ = refresh(ctx2, show_all, draw_labels_text)
        img_upd, tab_out, ctx_, brush_upd = pack_full(img_out, tab_out, ctx_)
        return img_upd, f"Merged OK: created {os.path.basename(new_path)} (parents ignored).", tab_out, ctx_, brush_upd

    def export_final_dataset():
        save_df()
        run_dir = os.path.dirname(os.path.abspath(csv_path))
        out_csv = os.path.join(run_dir, "labels_final.csv")
        final_masks_dir = get_final_masks_dir(csv_path)
        os.makedirs(final_masks_dir, exist_ok=True)
        frames_out_dir = os.path.join(run_dir, "frames")
        os.makedirs(frames_out_dir, exist_ok=True)

        out_df = df.copy()
        out_df["label_final"] = out_df["label_final"].apply(norm_label)
        out_df = out_df[~out_df["ignore"].astype(bool)].copy()
        out_df = out_df[out_df["label_final"] != ""].copy()
        if len(out_df) == 0:
            return "No rows to export."

        exported_rows = []
        missing_masks = 0
        missing_frames = 0
        for rid, row in out_df.iterrows():
            src_mask = str(row.get("mask_path", "")).strip()
            if not src_mask or not os.path.exists(src_mask):
                missing_masks += 1
                continue

            frame_file = str(row["frame_file"])
            frame_base = os.path.splitext(frame_file)[0]
            mask_index = int(row["mask_index"])
            dst_mask = os.path.join(final_masks_dir, f"{frame_base}_rid_{int(rid):06d}_mask_{mask_index:03d}.png")
            if os.path.abspath(src_mask) != os.path.abspath(dst_mask):
                shutil.copy2(src_mask, dst_mask)

            src_img = str(row.get("image_path", "")).strip()
            if not src_img or not os.path.isabs(src_img) or not os.path.exists(src_img):
                if frames_dir_res:
                    src_img = os.path.join(frames_dir_res, frame_file)
            dst_img = os.path.join(frames_out_dir, frame_file)
            if os.path.exists(src_img):
                if os.path.abspath(src_img) != os.path.abspath(dst_img):
                    shutil.copy2(src_img, dst_img)
            else:
                missing_frames += 1

            rr = row.copy()
            rr["image_path"] = os.path.join("frames", frame_file).replace("\\", "/")
            rr["mask_path"] = os.path.join("final_masks", os.path.basename(dst_mask)).replace("\\", "/")
            lab = norm_label(rr.get("label_final", ""))
            rr["label_final"] = "plasma" if lab in ("plume", "plume2") else lab

            try:
                seg_final = read_mask_bool(dst_mask)
                cx_core, cy_core, core_r = chebyshev_center(seg_final)
                rr["cx_core"] = float(cx_core)
                rr["cy_core"] = float(cy_core)
                rr["core_r"] = float(core_r)
            except Exception:
                rr["cx_core"] = float(rr.get("cx_geo", np.nan))
                rr["cy_core"] = float(rr.get("cy_geo", np.nan))
                rr["core_r"] = float(rr.get("core_r", 0.0) or 0.0)
            exported_rows.append(rr)

        if len(exported_rows) == 0:
            return f"No rows exported. Missing masks={missing_masks}"

        export_df = pd.DataFrame(exported_rows)
        keep_cols = [
            "frame_file", "frame_index", "image_path",
            "source_video", "source_frame_index",
            "mask_index", "mask_path", "label_final",
            "track_id", "ignore", "area",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "cx_geo", "cy_geo", "cx_core", "cy_core", "core_r",
        ]
        keep_cols = [c for c in keep_cols if c in export_df.columns]
        export_df = export_df[keep_cols].rename(columns={"label_final": "label"})
        export_df = export_df.sort_values(["frame_index", "mask_index"]).reset_index(drop=True)
        export_df.to_csv(out_csv, index=False)

        return (
            f"Saved {len(export_df)} rows to {out_csv}. "
            f"Final masks folder: {final_masks_dir}. "
            f"Missing masks skipped: {missing_masks}. "
            f"Missing frames (could not copy): {missing_frames}"
        )

    def make_mask_table():
        base_kwargs = dict(
            headers=[
                "row_id", "checked", "selected", "mask_index",
                "label", "segment_type", "track_id",
                "origin_tag", "review_origin", "source",
                "area", "sam_quality", "frame_uncertainty", "review_reason",
                "ignore", "why", "broken_rule", "manual_note",
                "static_overlap", "smooth", "smooth_rank", "mask_path",
            ],
            datatype=[
                "number", "bool", "bool", "number",
                "str", "str", "number",
                "str", "str", "str",
                "number", "number", "number", "str",
                "bool", "str", "str", "str",
                "number", "bool", "number", "str",
            ],
            label="Masks in this frame (PROP rows have cyan outline; click row to select; use checked for batch label)",
            interactive=True,
            elem_id="mask_table",
        )
        candidates = [
            dict(
                wrap=True,
                max_height=TABLE_HEIGHT,
                row_count=16,
                column_widths=[
                    70, 80, 80, 90,
                    130, 150, 90,
                    85, 230, 190,
                    90, 110, 140, 260,
                    80, 340, 220, 220,
                    120, 80, 90, 360,
                ],
            ),
            dict(wrap=True, row_count=16),
            dict(wrap=True),
            dict(),
        ]
        for extra in candidates:
            try:
                return gr.Dataframe(**base_kwargs, **extra)
            except TypeError:
                continue
        return gr.Dataframe(**base_kwargs)

    def make_image():
        base_kwargs = dict(
            type="numpy",
            label="Frame + overlay (cyan outline = SAM2 video propagated mask)",
            interactive=True,
            elem_id="frame_image",
            visible=True,
        )
        for extra in [dict(height=UI_HEIGHT), dict()]:
            try:
                return gr.Image(**base_kwargs, **extra)
            except TypeError:
                continue
        return gr.Image(**base_kwargs)

    def make_brush_editor(initial_value: Any):
        base_kwargs = dict(
            value=initial_value,
            label="Brush on current frame",
            interactive=True,
            visible=False,
            elem_id="brush_editor",
        )
        candidates = [
            dict(type="numpy", image_mode="RGBA", sources=(), transforms=(), height=UI_HEIGHT),
            dict(type="numpy", image_mode="RGBA", height=UI_HEIGHT),
            dict(type="numpy", height=UI_HEIGHT),
            dict(height=UI_HEIGHT),
            dict(),
        ]
        for extra in candidates:
            try:
                return gr.ImageEditor(**base_kwargs, **extra)
            except Exception:
                continue
        raise RuntimeError("Could not create gr.ImageEditor. Your Gradio version may not support it as expected.")

    with gr.Blocks(fill_width=True) as demo:
        gr.Markdown(
            "## Manual Label Tool — review dashboard + final truth dataset\n"
            "**ORG** = original pipeline mask, **PROP** = SAM2 video propagated mask, **MAN** = manual mask. "
            "PROP masks are drawn with a cyan outline."
        )
        state = gr.State(ctx0)

        with gr.Row():
            prev_frame_btn = gr.Button("← Prev frame")
            next_frame_btn = gr.Button("Next frame →")
            prev_review_btn = gr.Button("← Prev review")
            next_review_btn = gr.Button("Next review →")
            save_btn = gr.Button("Save now")

        with gr.Row():
            frame_slider = gr.Number(
                value=0,
                precision=0,
                label=f"Frame index (0 to {max(0, nframes - 1)})",
            )
            review_mode = gr.Dropdown(
                ["All frames", "Keyframes only", "Needs review", "Uncertain frames", "Spatter frames"],
                value="Needs review",
                label="Review queue",
            )
            uncertainty_thr = gr.Number(value=0.65, precision=3, label="Uncertainty threshold")
            auto_forward_review = gr.Checkbox(False, label="Auto-forward to next review after label")

        with gr.Row():
            only_smooth = gr.Checkbox(False, label="Show only smooth-top10")
            class_filter = gr.Dropdown(
                ["all", "weld", "plasma", "spatter", "static", "dynamic_other"],
                value="all",
                label="Class filter",
            )
            show_ignored = gr.Checkbox(False, label="Show ignored masks")
            show_all = gr.Checkbox(True, label="Draw visible masks")
            draw_labels_text = gr.Checkbox(True, label="Draw labels text")

        with gr.Row(equal_height=True, elem_id="viewer_row"):
            with gr.Column(scale=4, min_width=320, elem_id="img_col"):
                img = make_image()
                brush_editor = make_brush_editor(make_editor_value_for_frame(ctx0["img"]))

            with gr.Column(scale=6, min_width=520, elem_id="table_col"):
                mask_table = make_mask_table()


        with gr.Row():
            quick_weld_btn = gr.Button("Set → weld")
            quick_plasma_btn = gr.Button("Set → plasma")
            quick_spatter_btn = gr.Button("Set → spatter")
            quick_static_btn = gr.Button("Set → static")
            quick_dynamic_btn = gr.Button("Set → dynamic_other")
            ignore_btn = gr.Button("Toggle ignore")
            soft_delete_btn = gr.Button("Soft delete selected")

        with gr.Row():
            prev_hit = gr.Button("Overlap prev (cycle)")
            next_hit = gr.Button("Overlap next (cycle)")
            soft_delete_unlabeled_btn = gr.Button("Delete unlabeled dynamic in frame")
            clear_checked_btn = gr.Button("Clear checked")
            
        with gr.Row():
            label_dd = gr.Dropdown(LABELS, value="", label="Set label / polygon label / brush label")
            apply_to_checked = gr.Checkbox(True, label="Apply to CHECKED (if any), else selected")
            auto_propagate = gr.Checkbox(False, label="Auto-propagate label forward")
            prop_iou = gr.Number(value=0.98, precision=3, label="Propagate IoU threshold")
            prop_same_bbox = gr.Checkbox(False, label="Propagation requires exact same bbox")
            set_label_btn = gr.Button("Apply label")

        with gr.Row():
            track_id_box = gr.Number(value=-1, precision=0, label="Set track_id")
            set_track_btn = gr.Button("Apply track_id")
            save_status = gr.Textbox(label="Save status", lines=1, elem_id="save_status")

        with gr.Row():
            broken_rule_box = gr.Textbox(label="Broken rule", lines=2, scale=1)
            manual_note_box = gr.Textbox(label="Manual note", lines=2, scale=1)

        with gr.Row():
            click_mode = gr.Radio(["select", "seed1", "seed2", "polygon", "brush"], value="select", label="Click mode")
            clear_seed_btn = gr.Button("Clear seeds")
            split_btn = gr.Button("Split selected mask (needs seed1+seed2)")
        split_msg = gr.Textbox(label="Split/Merge messages", lines=2)

        with gr.Row():
            poly_undo_btn = gr.Button("Undo polygon point")
            poly_clear_btn = gr.Button("Clear polygon")
            poly_save_btn = gr.Button("Save polygon as new RAW mask")

        with gr.Row():
            brush_clear_btn = gr.Button("Clear brush")
            brush_save_btn = gr.Button("Save brush as new RAW mask")
            brush_cancel_btn = gr.Button("Exit brush mode")

        with gr.Row():
            merge_a_btn = gr.Button("Set Merge A = selected")
            merge_b_btn = gr.Button("Set Merge B = selected")
            merge_btn = gr.Button("Merge A + B (union)")

        gr.Markdown(
            "### SAM2 keyframe propagation\n"
            "After reviewing the keyframes, save the labels and run SAM2 propagation "
            "to fill the frames between the reviewed keyframes."
        )

        with gr.Row():
            propagation_interval = gr.Number(
                value=int(keyframe_interval),
                precision=0,
                label="Keyframe interval",
            )
            propagation_overwrite = gr.Checkbox(
                False,
                label="Overwrite existing propagation output",
            )
            run_propagation_btn = gr.Button(
                "Save + Run SAM2 keyframe propagation",
                variant="primary",
            )

        with gr.Row():
            propagation_script_box = gr.Textbox(
                value=propagation_script_resolved,
                label="Propagation script",
            )
            propagation_repo_box = gr.Textbox(
                value=str(sam2_repo or os.environ.get("SAM2_REPO", "")),
                label="SAM2 repo (optional)",
            )
            propagation_checkpoint_box = gr.Textbox(
                value=str(sam2_checkpoint or ""),
                label="SAM2 checkpoint (optional)",
            )

        with gr.Row():
            propagation_device_box = gr.Dropdown(
                ["cuda", "cpu"],
                value=str(sam2_device or "cuda"),
                label="Device",
            )
            propagation_model_cfg_box = gr.Textbox(
                value=str(sam2_model_cfg),
                label="SAM2 model config",
            )
            propagation_progress = gr.Slider(
                minimum=0,
                maximum=100,
                value=0,
                step=1,
                interactive=False,
                label="Propagation progress (%)",
            )

        propagation_log = gr.Textbox(
            label="SAM2 propagation log",
            lines=10,
            max_lines=14,
        )
        propagation_combined_csv = gr.Textbox(
            label="Combined propagated review CSV",
            interactive=False,
        )
        open_propagated_btn = gr.Button("Open propagated review")

        with gr.Row():
            export_btn = gr.Button("Export final dataset")
        export_msg = gr.Textbox(label="Export message", lines=2)

        img0 = render(df, ctx0, show_all=True, draw_labels_text=True)
        tab0 = build_mask_table(df, ctx0)
        img.value = img0
        mask_table.value = tab0

        # Navigation and filters
        frame_slider.change(
            go_frame,
            inputs=[frame_slider, only_smooth, class_filter, show_ignored, show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor],
        )
        only_smooth.change(
            apply_filters,
            inputs=[only_smooth, class_filter, show_ignored, show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor],
        )
        class_filter.change(
            apply_filters,
            inputs=[only_smooth, class_filter, show_ignored, show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor],
        )
        show_ignored.change(
            apply_filters,
            inputs=[only_smooth, class_filter, show_ignored, show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor],
        )
        show_all.change(
            lambda sa, dl, ctx: pack_full(*refresh(ctx, sa, dl)),
            inputs=[show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor],
        )
        draw_labels_text.change(
            lambda sa, dl, ctx: pack_full(*refresh(ctx, sa, dl)),
            inputs=[show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor],
        )

        prev_frame_btn.click(
            lambda os_, cf, si, sa, dl, ctx: go_relative(-1, os_, cf, si, sa, dl, ctx),
            inputs=[only_smooth, class_filter, show_ignored, show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor, frame_slider],
        )
        next_frame_btn.click(
            lambda os_, cf, si, sa, dl, ctx: go_relative(+1, os_, cf, si, sa, dl, ctx),
            inputs=[only_smooth, class_filter, show_ignored, show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor, frame_slider],
        )
        prev_review_btn.click(
            lambda rm, ut, os_, cf, si, sa, dl, ctx: go_review_relative(-1, rm, ut, os_, cf, si, sa, dl, ctx),
            inputs=[review_mode, uncertainty_thr, only_smooth, class_filter, show_ignored, show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor, frame_slider],
        )
        next_review_btn.click(
            lambda rm, ut, os_, cf, si, sa, dl, ctx: go_review_relative(+1, rm, ut, os_, cf, si, sa, dl, ctx),
            inputs=[review_mode, uncertainty_thr, only_smooth, class_filter, show_ignored, show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor, frame_slider],
        )
        save_btn.click(save_now, inputs=[state], outputs=[save_status])

        # Image/table interaction
        img.select(
            on_click,
            inputs=[show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor],
        )
        mask_table.select(
            on_table_select,
            inputs=[show_all, draw_labels_text, state],
            outputs=[img, state, brush_editor],
        )
        mask_table.change(
            on_table_change,
            inputs=[mask_table, show_all, draw_labels_text, state],
            outputs=[img, state, brush_editor],
        )

        prev_hit.click(
            lambda sa, dl, ctx: cycle_hit(-1, sa, dl, ctx),
            inputs=[show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor],
        )
        next_hit.click(
            lambda sa, dl, ctx: cycle_hit(+1, sa, dl, ctx),
            inputs=[show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor],
        )

        click_mode.change(
            lambda mode, sa, dl, ctx: set_click_mode(mode, sa, dl, ctx),
            inputs=[click_mode, show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor],
        )

        # Quick labels + auto-forward
        def _quick(label_value: str):
            return lambda afr, rm, ut, ap, pi, psb, atc, br, mn, os_, cf, si, sa, dl, ctx: quick_label(
                label_value,
                afr,
                rm,
                ut,
                ap,
                pi,
                psb,
                atc,
                br,
                mn,
                os_,
                cf,
                si,
                sa,
                dl,
                ctx,
            )

        quick_inputs = [
            auto_forward_review,
            review_mode,
            uncertainty_thr,
            auto_propagate,
            prop_iou,
            prop_same_bbox,
            apply_to_checked,
            broken_rule_box,
            manual_note_box,
            only_smooth,
            class_filter,
            show_ignored,
            show_all,
            draw_labels_text,
            state,
        ]
        quick_outputs = [img, mask_table, state, brush_editor, frame_slider]

        quick_weld_btn.click(_quick("weld"), inputs=quick_inputs, outputs=quick_outputs)
        quick_plasma_btn.click(_quick("plasma"), inputs=quick_inputs, outputs=quick_outputs)
        quick_spatter_btn.click(_quick("spatter"), inputs=quick_inputs, outputs=quick_outputs)
        quick_static_btn.click(_quick("static"), inputs=quick_inputs, outputs=quick_outputs)
        quick_dynamic_btn.click(_quick("dynamic_other"), inputs=quick_inputs, outputs=quick_outputs)

        set_label_btn.click(
            lambda lab, afr, rm, ut, ap, pi, psb, atc, br, mn, os_, cf, si, sa, dl, ctx: quick_label(
                lab,
                afr,
                rm,
                ut,
                ap,
                pi,
                psb,
                atc,
                br,
                mn,
                os_,
                cf,
                si,
                sa,
                dl,
                ctx,
            ),
            inputs=[
                label_dd,
                auto_forward_review,
                review_mode,
                uncertainty_thr,
                auto_propagate,
                prop_iou,
                prop_same_bbox,
                apply_to_checked,
                broken_rule_box,
                manual_note_box,
                only_smooth,
                class_filter,
                show_ignored,
                show_all,
                draw_labels_text,
                state,
            ],
            outputs=quick_outputs,
        )

        ignore_btn.click(
            lambda sa, dl, ctx: toggle_ignore(sa, dl, ctx),
            inputs=[show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor],
        )
        soft_delete_btn.click(
            lambda sa, dl, ctx: soft_delete_selected(sa, dl, ctx),
            inputs=[show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor],
        )
        soft_delete_unlabeled_btn.click(
            lambda sa, dl, ctx: soft_delete_unlabeled_dynamic_in_frame(sa, dl, ctx),
            inputs=[show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor],
        )
        clear_checked_btn.click(
            lambda sa, dl, ctx: clear_checked(sa, dl, ctx),
            inputs=[show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor],
        )

        set_track_btn.click(
            lambda tid, atc, sa, dl, ctx: set_track_id(tid, atc, sa, dl, ctx),
            inputs=[track_id_box, apply_to_checked, show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor],
        )

        clear_seed_btn.click(
            lambda sa, dl, ctx: clear_seeds(sa, dl, ctx),
            inputs=[show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor],
        )
        split_btn.click(
            lambda sa, dl, ctx: do_split(sa, dl, ctx),
            inputs=[show_all, draw_labels_text, state],
            outputs=[img, split_msg, mask_table, state, brush_editor],
        )

        poly_undo_btn.click(
            lambda sa, dl, ctx: undo_polygon_point(sa, dl, ctx),
            inputs=[show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor],
        )
        poly_clear_btn.click(
            lambda sa, dl, ctx: clear_polygon_points(sa, dl, ctx),
            inputs=[show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor],
        )
        poly_save_btn.click(
            lambda lab, mn, sa, dl, ctx: add_polygon_mask(lab, mn, sa, dl, ctx),
            inputs=[label_dd, manual_note_box, show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor],
        )

        brush_clear_btn.click(clear_brush, inputs=[state], outputs=[brush_editor])
        brush_save_btn.click(
            lambda bv, lab, mn, sa, dl, ctx: add_brush_mask(bv, lab, mn, sa, dl, ctx),
            inputs=[brush_editor, label_dd, manual_note_box, show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor],
        )
        brush_cancel_btn.click(
            lambda sa, dl, ctx: exit_brush_mode(sa, dl, ctx),
            inputs=[show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor],
        )

        merge_a_btn.click(
            lambda sa, dl, ctx: set_merge_a(sa, dl, ctx),
            inputs=[show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor],
        )
        merge_b_btn.click(
            lambda sa, dl, ctx: set_merge_b(sa, dl, ctx),
            inputs=[show_all, draw_labels_text, state],
            outputs=[img, mask_table, state, brush_editor],
        )
        merge_btn.click(
            lambda sa, dl, ctx: do_merge(sa, dl, ctx),
            inputs=[show_all, draw_labels_text, state],
            outputs=[img, split_msg, mask_table, state, brush_editor],
        )

        run_propagation_btn.click(
            run_sam2_keyframe_propagation,
            inputs=[
                propagation_interval,
                propagation_overwrite,
                propagation_script_box,
                propagation_repo_box,
                propagation_checkpoint_box,
                propagation_device_box,
                propagation_model_cfg_box,
            ],
            outputs=[
                propagation_log,
                propagation_progress,
                propagation_combined_csv,
            ],
        )

        open_propagated_btn.click(
            open_propagated_review,
            inputs=[
                propagation_combined_csv,
                only_smooth,
                class_filter,
                show_ignored,
                show_all,
                draw_labels_text,
                uncertainty_thr,
                state,
            ],
            outputs=[
                img,
                mask_table,
                state,
                brush_editor,
                frame_slider,
                review_mode,
                save_status,
            ],
        )

        export_btn.click(export_final_dataset, inputs=[], outputs=[export_msg])

    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="label_manifest.csv / labels manifest")
    ap.add_argument("--frames", default=None, help="frames directory (optional if CSV has image_path)")
    ap.add_argument("--pre_csv", default=None, help="Optional older CSV to import labels from")
    ap.add_argument(
        "--propagation-script",
        default=None,
        help="Path to propagate_keyframes_sam2.py (auto-detected when omitted)",
    )
    ap.add_argument("--sam2-repo", default=None, help="Optional SAM2 repository path")
    ap.add_argument("--sam2-checkpoint", default=None, help="Optional SAM2 checkpoint path")
    ap.add_argument(
        "--sam2-model-cfg",
        default="configs/sam2.1/sam2.1_hiera_l",
        help="SAM2 model config",
    )
    ap.add_argument("--sam2-device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--keyframe-interval", type=int, default=10)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    demo = app(
        args.csv,
        args.frames,
        pre_csv=args.pre_csv,
        propagation_script=args.propagation_script,
        sam2_repo=args.sam2_repo,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_model_cfg=args.sam2_model_cfg,
        sam2_device=args.sam2_device,
        keyframe_interval=args.keyframe_interval,
    )
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=False,
        css=APP_CSS,
    )
