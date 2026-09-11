# scripts/propagate_keyframes_sam2.py
import argparse
import json
import os
import shutil
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

try:
    import sam2  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Could not import sam2. Run this inside the same environment where your SAM2 pipeline works."
    ) from e

from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2_video_predictor


LABEL_ALIASES = {
    "plume": "plasma",
    "plume2": "plasma",
}

DEFAULT_LABELS = ["weld", "plasma", "spatter"]


@contextmanager
def pushd(path: str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def norm_label(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""
    return LABEL_ALIASES.get(s, s)


def safe_int(x: Any, default: int = -1) -> int:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return int(x)
    except Exception:
        return default


def resolve_repo_dir(repo_dir: Optional[str]) -> str:
    if repo_dir:
        return os.path.abspath(os.path.expanduser(repo_dir))

    env_repo = os.environ.get("SAM2_REPO", "").strip()
    if env_repo:
        return os.path.abspath(os.path.expanduser(env_repo))

    sam2_pkg_dir = list(sam2.__path__)[0]
    return os.path.abspath(os.path.join(sam2_pkg_dir, ".."))


def resolve_checkpoint_path(repo_dir: str, checkpoint_name: str, checkpoint_path: Optional[str]) -> str:
    if checkpoint_path:
        ckpt = os.path.abspath(os.path.expanduser(checkpoint_path))
    elif os.path.isfile(os.path.expanduser(checkpoint_name)):
        ckpt = os.path.abspath(os.path.expanduser(checkpoint_name))
    else:
        ckpt = os.path.join(repo_dir, "checkpoints", checkpoint_name)

    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            "SAM2 checkpoint not found.\n"
            f"  tried: {ckpt}\n"
            f"  repo_dir: {repo_dir}\n"
            "Fix by passing --checkpoint-path or setting SAM2_REPO."
        )
    return ckpt



def combine_original_and_propagated_csv(
    original_csv: str,
    propagated_csv: str,
    output_csv: str,
) -> str:
    """
    Create a clean combined review CSV:

        original masks
        +
        real SAM2 propagated masks

    Empty placeholder rows are removed from the combined file.
    They are only useful in the propagated-only CSV so the GUI can show
    frames where no propagated mask exists.
    """
    original_csv = os.path.abspath(os.path.expanduser(original_csv))
    propagated_csv = os.path.abspath(os.path.expanduser(propagated_csv))
    output_csv = os.path.abspath(os.path.expanduser(output_csv))

    orig = pd.read_csv(original_csv).copy()
    prop = pd.read_csv(propagated_csv).copy()

    # Remove empty placeholder rows from propagated CSV before combining.
    if "seg_key" in prop.columns:
        prop = prop[
            prop["seg_key"].fillna("").astype(str) != "empty_frame_placeholder"
        ].copy()

    if "source" in prop.columns:
        prop = prop[
            prop["source"].fillna("").astype(str) != "empty_frame_placeholder"
        ].copy()

    if "area" in prop.columns:
        prop["area"] = pd.to_numeric(prop["area"], errors="coerce").fillna(0)
        prop = prop[prop["area"] > 0].copy()

    # Mark where each row came from.
    orig["review_origin"] = "original_before_propagation"
    prop["review_origin"] = "sam2_video_propagation"

    if "source" not in orig.columns:
        orig["source"] = "original_pipeline"
    else:
        orig["source"] = orig["source"].fillna("").astype(str)
        orig.loc[orig["source"].str.strip() == "", "source"] = "original_pipeline"

    if "source" not in prop.columns:
        prop["source"] = "sam2_video_propagation"
    else:
        prop["source"] = prop["source"].fillna("").astype(str)
        prop.loc[prop["source"].str.strip() == "", "source"] = "sam2_video_propagation"

    # Match columns.
    all_cols = sorted(set(orig.columns) | set(prop.columns))

    for col in all_cols:
        if col not in orig.columns:
            orig[col] = np.nan
        if col not in prop.columns:
            prop[col] = np.nan

    orig = orig[all_cols]
    prop = prop[all_cols]

    # Normalize frame/mask indices.
    orig["frame_index"] = pd.to_numeric(
        orig["frame_index"], errors="coerce"
    ).fillna(-1).astype(int)

    prop["frame_index"] = pd.to_numeric(
        prop["frame_index"], errors="coerce"
    ).fillna(-1).astype(int)

    orig["mask_index"] = pd.to_numeric(
        orig["mask_index"], errors="coerce"
    ).fillna(-1).astype(int)

    prop["mask_index"] = pd.to_numeric(
        prop["mask_index"], errors="coerce"
    ).fillna(-1).astype(int)

    # Avoid mask_index collisions:
    # original keeps its mask_index,
    # propagated starts after max original mask_index per frame.
    max_orig_by_frame = orig.groupby("frame_index")["mask_index"].max().to_dict()

    new_prop_mask_indices: list[int] = []
    counter_by_frame: Dict[int, int] = {}

    for _, row in prop.iterrows():
        frame_index = int(row["frame_index"])
        old_midx = int(row.get("mask_index", -1))

        if old_midx < 0:
            new_prop_mask_indices.append(-1)
            continue

        start = int(max_orig_by_frame.get(frame_index, -1)) + 1
        count = int(counter_by_frame.get(frame_index, 0))

        new_prop_mask_indices.append(start + count)
        counter_by_frame[frame_index] = count + 1

    prop["mask_index"] = new_prop_mask_indices

    combined = pd.concat([orig, prop], ignore_index=True)

    def origin_order(x: object) -> int:
        s = str(x)
        if s == "original_before_propagation":
            return 0
        if s == "sam2_video_propagation":
            return 1
        return 2

    combined["_origin_order"] = combined["review_origin"].apply(origin_order)

    combined = (
        combined
        .sort_values(["frame_index", "_origin_order", "mask_index"])
        .drop(columns=["_origin_order"])
        .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    combined.to_csv(output_csv, index=False)

    print("[OK] Combined original + propagated CSV:", output_csv)
    print("[INFO] Original rows:", len(orig))
    print("[INFO] Real propagated rows added:", len(prop))
    print("[INFO] Combined rows:", len(combined))
    print("[INFO] Combined unique frames:", combined["frame_index"].nunique())

    if len(prop):
        print(
            "[INFO] Propagated frame range:",
            int(prop["frame_index"].min()),
            "to",
            int(prop["frame_index"].max()),
        )

    return output_csv

def build_video_predictor(
    device: str,
    repo_dir: Optional[str],
    checkpoint_name: str,
    checkpoint_path: Optional[str],
    model_cfg: str,
    vos_optimized: bool = False,
):
    if "cuda" in str(device).lower() and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    sam2_repo = resolve_repo_dir(repo_dir)
    ckpt = resolve_checkpoint_path(sam2_repo, checkpoint_name, checkpoint_path)

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    initialize_config_module(config_module="sam2", version_base=None)

    with pushd(sam2_repo):
        try:
            predictor = build_sam2_video_predictor(
                model_cfg,
                ckpt,
                device=device,
                vos_optimized=bool(vos_optimized),
            )
        except TypeError:
            predictor = build_sam2_video_predictor(
                model_cfg,
                ckpt,
                device=device,
            )

    print("[SAM2] repo:", sam2_repo)
    print("[SAM2] checkpoint:", ckpt)
    print("[SAM2] model_cfg:", model_cfg)
    return predictor, sam2_repo, ckpt


def read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read frame image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_mask_bool(path: str) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise RuntimeError(f"Could not read mask: {path}")
    return g > 0


def write_mask(path: str, mask: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, (mask.astype(np.uint8) * 255))
    if not ok:
        raise RuntimeError(f"Could not write mask: {path}")


def bbox_from_mask(seg: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(seg.astype(bool))
    if xs.size == 0:
        return 0, 0, 0, 0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def centroid(seg: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(seg.astype(bool))
    if xs.size == 0:
        return float("nan"), float("nan")
    return float(xs.mean()), float(ys.mean())


def clean_mask(seg: np.ndarray, min_area: int = 5) -> np.ndarray:
    seg = seg.astype(bool)
    if seg.sum() <= 0:
        return seg

    u8 = seg.astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    out = np.zeros_like(seg, dtype=bool)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            out[labels == i] = True
    return out


def prepare_video_frames_jpg(
    frames_df: pd.DataFrame,
    out_dir: str,
    jpg_quality: int = 95,
) -> Dict[int, int]:
    """
    Create a JPEG frame folder for SAM2 video predictor.

    Returns:
        original frame_index -> sam2 video position
    """
    os.makedirs(out_dir, exist_ok=True)

    frame_index_to_pos: Dict[int, int] = {}

    for pos, row in frames_df.reset_index(drop=True).iterrows():
        frame_index = int(row["frame_index"])
        src = str(row["image_path"])
        rgb = read_rgb(src)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        dst = os.path.join(out_dir, f"{int(pos):06d}.jpg")
        cv2.imwrite(dst, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
        frame_index_to_pos[frame_index] = int(pos)

    return frame_index_to_pos


def normalize_paths(df: pd.DataFrame, csv_path: str, frames_dir: Optional[str]) -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.abspath(csv_path))
    frames_dir_abs = os.path.abspath(os.path.expanduser(frames_dir)) if frames_dir else ""

    def abs_path(p: Any) -> str:
        s = str(p or "").strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return ""
        s = os.path.expanduser(s)
        if os.path.isabs(s):
            return s
        return os.path.abspath(os.path.join(base_dir, s))

    if "mask_path" not in df.columns:
        raise RuntimeError("CSV must contain mask_path")
    if "frame_file" not in df.columns:
        raise RuntimeError("CSV must contain frame_file")
    if "frame_index" not in df.columns:
        raise RuntimeError("CSV must contain frame_index")

    if "image_path" not in df.columns:
        df["image_path"] = ""

    df["mask_path"] = df["mask_path"].apply(abs_path)
    df["image_path"] = df["image_path"].apply(abs_path)

    missing = df["image_path"].astype(str).str.strip().isin(["", "nan", "None"])
    if frames_dir_abs:
        df.loc[missing, "image_path"] = df.loc[missing, "frame_file"].apply(
            lambda f: os.path.abspath(os.path.join(frames_dir_abs, str(f)))
        )

    return df


def build_frames(df: pd.DataFrame) -> pd.DataFrame:
    frame_columns = ["frame_index", "frame_file", "image_path"]
    for optional in ("source_video", "source_frame_index"):
        if optional in df.columns:
            frame_columns.append(optional)
    frames = (
        df[frame_columns]
        .drop_duplicates()
        .sort_values(["frame_index", "frame_file"])
        .reset_index(drop=True)
    )
    if frames.empty:
        raise RuntimeError("No frames found in CSV.")
    return frames


def keyframe_positions_from_csv(
    df: pd.DataFrame,
    frames_df: pd.DataFrame,
    interval: int,
    use_review_keyframes: bool,
    exact_n_keyframes: int = 0,
) -> List[int]:
    frame_indices = frames_df["frame_index"].astype(int).tolist()
    frame_index_to_pos = {int(fi): i for i, fi in enumerate(frame_indices)}

    selected_indices: List[int] = []

    if use_review_keyframes and "is_review_keyframe" in df.columns:
        sub = df[df["is_review_keyframe"].fillna(False).astype(bool)]
        selected_indices = sorted(set(sub["frame_index"].astype(int).tolist()))

    if not selected_indices:
        if exact_n_keyframes and exact_n_keyframes > 1:
            positions = np.linspace(0, len(frames_df) - 1, int(exact_n_keyframes))
            return sorted(set(int(round(p)) for p in positions))

        step = max(1, int(interval))
        return list(range(0, len(frames_df), step))

    out = []
    for fi in selected_indices:
        if int(fi) in frame_index_to_pos:
            out.append(frame_index_to_pos[int(fi)])

    return sorted(set(out))


def merge_label_masks_for_keyframe(
    df: pd.DataFrame,
    frames_df: pd.DataFrame,
    frame_pos: int,
    labels: List[str],
    min_area: int,
    manual_only: bool,
    max_spatter_per_keyframe: int,
) -> List[Dict[str, Any]]:
    """
    Build seed objects for one keyframe.

    weld/plasma are merged per label.
    spatter remains one object per row.
    """
    frame_index = int(frames_df.at[frame_pos, "frame_index"])
    frame_file = str(frames_df.at[frame_pos, "frame_file"])

    sub = df[
        (df["frame_index"].astype(int) == frame_index)
        & (~df.get("ignore", False).fillna(False).astype(bool))
    ].copy()

    if manual_only and "edited_manually" in sub.columns:
        sub = sub[sub["edited_manually"].fillna(False).astype(bool)].copy()

    sub["label_norm"] = sub["label_final"].apply(norm_label)

    seed_objects: List[Dict[str, Any]] = []

    # Unique/large objects: one mask per label
    for label in ["weld", "plasma"]:
        if label not in labels:
            continue

        rows = sub[sub["label_norm"] == label]
        if rows.empty:
            continue

        union_mask: Optional[np.ndarray] = None
        source_rows: List[int] = []
        track_ids: List[int] = []

        for rid, row in rows.iterrows():
            try:
                m = read_mask_bool(str(row["mask_path"]))
            except Exception as e:
                print(f"[WARN] Cannot read seed mask row={rid}: {e}")
                continue

            if union_mask is None:
                union_mask = np.zeros_like(m, dtype=bool)

            if m.shape != union_mask.shape:
                print(f"[WARN] Skipping seed row={rid} due to shape mismatch.")
                continue

            union_mask |= m.astype(bool)
            source_rows.append(int(rid))
            track_ids.append(safe_int(row.get("track_id", -1), -1))

        if union_mask is not None:
            union_mask = clean_mask(union_mask, min_area=min_area)
            if int(union_mask.sum()) >= min_area:
                default_tid = 0 if label == "weld" else 1
                tid = next((t for t in track_ids if t >= 0), default_tid)
                seed_objects.append(
                    {
                        "label": label,
                        "mask": union_mask,
                        "source_rows": source_rows,
                        "source_track_id": int(tid),
                        "source_frame_index": frame_index,
                        "source_frame_file": frame_file,
                    }
                )

    # Spatter: one object per row
    if "spatter" in labels:
        sp_rows = sub[sub["label_norm"] == "spatter"].copy()
        if not sp_rows.empty:
            sp_rows["area_num"] = pd.to_numeric(sp_rows.get("area", 0), errors="coerce").fillna(0)
            sp_rows = sp_rows.sort_values("area_num", ascending=False)

        count = 0
        for rid, row in sp_rows.iterrows():
            if count >= int(max_spatter_per_keyframe):
                break

            try:
                m = read_mask_bool(str(row["mask_path"]))
            except Exception as e:
                print(f"[WARN] Cannot read spatter seed mask row={rid}: {e}")
                continue

            m = clean_mask(m, min_area=min_area)
            if int(m.sum()) < min_area:
                continue

            seed_objects.append(
                {
                    "label": "spatter",
                    "mask": m,
                    "source_rows": [int(rid)],
                    "source_track_id": safe_int(row.get("track_id", -1), -1),
                    "source_frame_index": frame_index,
                    "source_frame_file": frame_file,
                }
            )
            count += 1

    return seed_objects


def mask_logits_to_bool(mask_logits: Any, min_area: int) -> np.ndarray:
    if hasattr(mask_logits, "detach"):
        arr = mask_logits.detach().cpu().numpy()
    else:
        arr = np.asarray(mask_logits)

    arr = np.squeeze(arr)
    mask = arr > 0.0
    return clean_mask(mask, min_area=min_area)


def make_manifest_row(
    frame_file: str,
    frame_index: int,
    image_path: str,
    mask_index: int,
    mask_path: str,
    label: str,
    track_id: int,
    mask: np.ndarray,
    source_keyframe: int,
    source_frame_file: str,
    obj_id: int,
    segment_start: int,
    segment_end: int,
    source_video: str = "",
    source_frame_index: int = -1,
) -> Dict[str, Any]:
    bx, by, bw, bh = bbox_from_mask(mask)
    cx, cy = centroid(mask)
    area = int(mask.sum())

    return {
        "frame_file": frame_file,
        "frame_index": int(frame_index),
        "source_video": str(source_video),
        "source_frame_index": int(source_frame_index),
        "image_path": os.path.abspath(image_path),
        "mask_index": int(mask_index),
        "mask_path": os.path.abspath(mask_path),

        "seg_key": f"sam2prop_{label}_obj{obj_id}",
        "segment_type": label,
        "label": label,
        "label_final": label,
        "ignore": False,

        "track_id": int(track_id),
        "why": "sam2_video_propagated_from_corrected_keyframe",

        "area": int(area),
        "bbox_x": int(bx),
        "bbox_y": int(by),
        "bbox_w": int(bw),
        "bbox_h": int(bh),
        "cx_geo": float(cx),
        "cy_geo": float(cy),
        "cx_core": float(cx),
        "cy_core": float(cy),
        "core_r": float("nan"),

        "source": "sam2_video_propagation",
        "is_manual": False,
        "edited_manually": False,

        "propagation_seed_frame_index": int(source_keyframe),
        "propagation_seed_frame_file": str(source_frame_file),
        "propagation_obj_id": int(obj_id),
        "propagation_segment_start_pos": int(segment_start),
        "propagation_segment_end_pos": int(segment_end),

        "is_review_keyframe": False,
        "needs_review": True,
        "review_reason": "review_after_sam2_video_propagation",
    }


def overlay_preview(
    frame_rgb: np.ndarray,
    masks: Iterable[Tuple[str, np.ndarray]],
) -> np.ndarray:
    color_map = {
        "weld": (255, 0, 0),
        "plasma": (255, 255, 0),
        "spatter": (0, 255, 0),
    }
    out = frame_rgb.copy()

    for label, mask in masks:
        color = np.array(color_map.get(label, (0, 128, 255)), dtype=np.float32)
        m = mask.astype(bool)
        out[m] = (0.70 * out[m].astype(np.float32) + 0.30 * color).astype(np.uint8)

        cnts, _ = cv2.findContours((m.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.drawContours(bgr, cnts, -1, (255, 255, 255), 2)
        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return out


def propagate_segment(
    predictor: Any,
    video_state: Any,
    frames_df: pd.DataFrame,
    video_pos_to_frame_index: Dict[int, int],
    output_masks_dir: str,
    seed_pos: int,
    end_pos: int,
    seed_objects: List[Dict[str, Any]],
    min_area: int,
    save_overlays: bool,
    overlay_dir: str,
) -> List[Dict[str, Any]]:
    """
    Reset SAM2 state, add keyframe masks as prompts, and propagate seed_pos -> end_pos.
    """
    if not seed_objects:
        return []

    predictor.reset_state(video_state)

    obj_id_to_seed: Dict[int, Dict[str, Any]] = {}

    for obj_id, seed in enumerate(seed_objects, start=1):
        obj_id_to_seed[obj_id] = seed
        predictor.add_new_mask(
            inference_state=video_state,
            frame_idx=int(seed_pos),
            obj_id=int(obj_id),
            mask=seed["mask"].astype(bool),
        )

    max_frames = int(end_pos - seed_pos + 1)

    rows: List[Dict[str, Any]] = []
    next_mask_index_by_frame: Dict[int, int] = {}

    frame_masks_for_overlay: Dict[int, List[Tuple[str, np.ndarray]]] = {}

    for out_frame_pos, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        video_state,
        start_frame_idx=int(seed_pos),
        max_frame_num_to_track=max_frames,
        reverse=False,
    ):
        out_frame_pos = int(out_frame_pos)

        if out_frame_pos < seed_pos or out_frame_pos > end_pos:
            continue

        if out_frame_pos not in video_pos_to_frame_index:
            continue

        frame_index = int(video_pos_to_frame_index[out_frame_pos])
        frame_row = frames_df[frames_df["frame_index"].astype(int) == frame_index].iloc[0]
        frame_file = str(frame_row["frame_file"])
        image_path = str(frame_row["image_path"])

        if frame_index not in next_mask_index_by_frame:
            next_mask_index_by_frame[frame_index] = 0

        if hasattr(out_obj_ids, "detach"):
            obj_ids_list = [int(x) for x in out_obj_ids.detach().cpu().numpy().tolist()]
        else:
            obj_ids_list = [int(x) for x in list(out_obj_ids)]

        for local_i, obj_id in enumerate(obj_ids_list):
            seed = obj_id_to_seed.get(int(obj_id))
            if seed is None:
                continue

            mask = mask_logits_to_bool(out_mask_logits[local_i], min_area=min_area)
            if int(mask.sum()) < int(min_area):
                continue

            label = str(seed["label"])
            track_id = int(seed.get("source_track_id", -1))

            # keep deterministic IDs for unique labels
            if label == "weld":
                track_id = 0
            elif label == "plasma":
                track_id = 1
            elif label == "spatter" and track_id < 0:
                track_id = 10000 + int(seed_pos) * 1000 + int(obj_id)

            mask_index = next_mask_index_by_frame[frame_index]
            next_mask_index_by_frame[frame_index] += 1

            base = os.path.splitext(frame_file)[0]
            mask_name = f"{base}_sam2prop_{label}_obj{int(obj_id):03d}_mask_{mask_index:03d}.png"
            mask_path = os.path.join(output_masks_dir, mask_name)
            write_mask(mask_path, mask)

            rows.append(
                make_manifest_row(
                    frame_file=frame_file,
                    frame_index=frame_index,
                    image_path=image_path,
                    mask_index=mask_index,
                    mask_path=mask_path,
                    label=label,
                    track_id=track_id,
                    mask=mask,
                    source_keyframe=int(seed["source_frame_index"]),
                    source_frame_file=str(seed["source_frame_file"]),
                    obj_id=int(obj_id),
                    segment_start=int(seed_pos),
                    segment_end=int(end_pos),
                    source_video=str(frame_row.get("source_video", "")),
                    source_frame_index=safe_int(frame_row.get("source_frame_index", -1), -1),
                )
            )

            if save_overlays:
                frame_masks_for_overlay.setdefault(frame_index, []).append((label, mask))

    if save_overlays:
        os.makedirs(overlay_dir, exist_ok=True)
        for frame_index, masks in frame_masks_for_overlay.items():
            frame_row = frames_df[frames_df["frame_index"].astype(int) == int(frame_index)].iloc[0]
            rgb = read_rgb(str(frame_row["image_path"]))
            ov = overlay_preview(rgb, masks)
            out_path = os.path.join(overlay_dir, f"{os.path.splitext(str(frame_row['frame_file']))[0]}_sam2prop_overlay.png")
            cv2.imwrite(out_path, cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Propagate manually corrected keyframes using SAM2 video predictor."
    )

    ap.add_argument("--csv", required=True, help="Corrected keyframe CSV, e.g. labels_manifest_tracked.csv")
    ap.add_argument("--frames", required=True, help="Frames folder used by the GUI")
    ap.add_argument("--output-dir", default=None, help="Output folder. Default: <run_dir>/sam2_propagated_keyframes")

    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS, help="Labels to propagate")
    ap.add_argument("--keyframe-interval", type=int, default=10, help="Use every Nth frame if is_review_keyframe is unavailable")
    ap.add_argument("--use-review-keyframes", action="store_true", default=True, help="Use is_review_keyframe column when available")
    ap.add_argument("--no-use-review-keyframes", action="store_false", dest="use_review_keyframes")
    ap.add_argument("--exact-n-keyframes", type=int, default=0, help="Optional: use exactly N evenly spaced keyframes instead of interval")

    ap.add_argument("--manual-only", action="store_true", help="Use only rows with edited_manually=True as seed masks")
    ap.add_argument("--min-area", type=int, default=10)
    ap.add_argument("--max-spatter-per-keyframe", type=int, default=50)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repo-dir", default=None)
    ap.add_argument("--checkpoint-name", default="sam2.1_hiera_large.pt")
    ap.add_argument("--checkpoint-path", default=None)
    ap.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l")
    ap.add_argument("--vos-optimized", action="store_true", help="Use vos_optimized=True if your SAM2 install supports it")

    ap.add_argument("--jpg-quality", type=int, default=95)
    ap.add_argument("--save-overlays", action="store_true", help="Save propagated overlay previews")
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    csv_path = os.path.abspath(os.path.expanduser(args.csv))
    frames_dir = os.path.abspath(os.path.expanduser(args.frames))

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)
    if not os.path.isdir(frames_dir):
        raise NotADirectoryError(frames_dir)

    run_dir = os.path.dirname(csv_path)
    output_dir = (
        os.path.abspath(os.path.expanduser(args.output_dir))
        if args.output_dir
        else os.path.join(run_dir, "sam2_propagated_keyframes")
    )

    if os.path.exists(output_dir) and args.overwrite:
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    masks_dir = os.path.join(output_dir, "typed_masks")
    overlay_dir = os.path.join(output_dir, "overlays")
    video_frames_dir = os.path.join(output_dir, "_sam2_video_frames_jpg")
    os.makedirs(masks_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = normalize_paths(df, csv_path=csv_path, frames_dir=frames_dir)

    if "ignore" not in df.columns:
        df["ignore"] = False
    if "label_final" not in df.columns:
        if "label" in df.columns:
            df["label_final"] = df["label"]
        elif "segment_type" in df.columns:
            df["label_final"] = df["segment_type"]
        else:
            raise RuntimeError("CSV needs label_final, label, or segment_type.")

    df["label_final"] = df["label_final"].apply(norm_label)

    frames_df = build_frames(df)

    missing_images = [p for p in frames_df["image_path"].astype(str).tolist() if not os.path.exists(p)]
    if missing_images:
        raise RuntimeError(
            "Some frame images are missing. First missing image:\n"
            f"{missing_images[0]}"
        )

    labels = [norm_label(x) for x in args.labels]
    labels = [x for x in labels if x]
    if not labels:
        raise RuntimeError("No labels selected.")

    key_positions = keyframe_positions_from_csv(
        df=df,
        frames_df=frames_df,
        interval=int(args.keyframe_interval),
        use_review_keyframes=bool(args.use_review_keyframes),
        exact_n_keyframes=int(args.exact_n_keyframes),
    )

    if 0 not in key_positions:
        key_positions = [0] + key_positions

    key_positions = sorted(set([p for p in key_positions if 0 <= p < len(frames_df)]))

    print("[INFO] Frames:", len(frames_df))
    print("[INFO] Keyframe positions:", key_positions)
    print("[INFO] Labels:", labels)
    print("[INFO] Output:", output_dir)

    frame_index_to_video_pos = prepare_video_frames_jpg(
        frames_df=frames_df,
        out_dir=video_frames_dir,
        jpg_quality=int(args.jpg_quality),
    )
    video_pos_to_frame_index = {v: k for k, v in frame_index_to_video_pos.items()}

    predictor, sam2_repo, ckpt = build_video_predictor(
        device=args.device,
        repo_dir=args.repo_dir,
        checkpoint_name=args.checkpoint_name,
        checkpoint_path=args.checkpoint_path,
        model_cfg=args.model_cfg,
        vos_optimized=bool(args.vos_optimized),
    )

    report: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []

    manifest_checkpoint = os.path.join(output_dir, "labels_manifest_propagated_checkpoint.csv")

    with torch.inference_mode():
        if "cuda" in str(args.device).lower() and torch.cuda.is_available():
            autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            autocast_ctx = torch.autocast("cpu", dtype=torch.bfloat16, enabled=False)

        with autocast_ctx:
            video_state = predictor.init_state(video_path=video_frames_dir)

            for i, seed_pos in enumerate(key_positions):
                next_seed = key_positions[i + 1] if i + 1 < len(key_positions) else len(frames_df)
                end_pos = int(next_seed - 1)

                if end_pos < seed_pos:
                    continue

                seed_objects = merge_label_masks_for_keyframe(
                    df=df,
                    frames_df=frames_df,
                    frame_pos=int(seed_pos),
                    labels=labels,
                    min_area=int(args.min_area),
                    manual_only=bool(args.manual_only),
                    max_spatter_per_keyframe=int(args.max_spatter_per_keyframe),
                )

                seed_frame_index = int(frames_df.at[int(seed_pos), "frame_index"])
                seed_frame_file = str(frames_df.at[int(seed_pos), "frame_file"])

                print(
                    f"\n[SEGMENT] seed_pos={seed_pos}, end_pos={end_pos}, "
                    f"seed_frame={seed_frame_index}, objects={len(seed_objects)}"
                )

                if not seed_objects:
                    report.append(
                        {
                            "seed_pos": int(seed_pos),
                            "end_pos": int(end_pos),
                            "seed_frame_index": int(seed_frame_index),
                            "seed_frame_file": seed_frame_file,
                            "n_seed_objects": 0,
                            "n_rows_written": 0,
                            "status": "skipped_no_seed_objects",
                        }
                    )
                    continue

                seg_rows = propagate_segment(
                    predictor=predictor,
                    video_state=video_state,
                    frames_df=frames_df,
                    video_pos_to_frame_index=video_pos_to_frame_index,
                    output_masks_dir=masks_dir,
                    seed_pos=int(seed_pos),
                    end_pos=int(end_pos),
                    seed_objects=seed_objects,
                    min_area=int(args.min_area),
                    save_overlays=bool(args.save_overlays),
                    overlay_dir=overlay_dir,
                )

                all_rows.extend(seg_rows)

                report.append(
                    {
                        "seed_pos": int(seed_pos),
                        "end_pos": int(end_pos),
                        "seed_frame_index": int(seed_frame_index),
                        "seed_frame_file": seed_frame_file,
                        "n_seed_objects": int(len(seed_objects)),
                        "n_rows_written": int(len(seg_rows)),
                        "status": "ok",
                    }
                )

                pd.DataFrame(all_rows).to_csv(manifest_checkpoint, index=False)
                pd.DataFrame(report).to_csv(os.path.join(output_dir, "propagation_report_checkpoint.csv"), index=False)
                print("[checkpoint] rows:", len(all_rows))

    out_csv = os.path.join(output_dir, "labels_manifest_propagated.csv")
    out_report = os.path.join(output_dir, "propagation_report.csv")
    combined_csv = os.path.join(
        output_dir,
        "labels_manifest_combined_original_plus_propagated.csv",
    )

    out_df = pd.DataFrame(all_rows)

    # ---------------------------------------------------------
    # IMPORTANT:
    # Keep ALL original frames in the propagated-only CSV.
    # SAM2 sometimes produces no mask for a frame. Without a row,
    # the GUI cannot show that frame because it builds its frame list
    # from the CSV. We add one ignored placeholder row for each missing
    # frame so the propagated-only CSV still shows all 164 frames.
    #
    # The combined CSV removes these placeholder rows because the original
    # CSV already contains all frames.
    # ---------------------------------------------------------
    if out_df.empty:
        out_df = pd.DataFrame(columns=[
            "frame_file", "frame_index", "image_path",
            "mask_index", "mask_path",
            "seg_key", "segment_type", "label", "label_final",
            "ignore", "track_id", "why",
            "area", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "cx_geo", "cy_geo", "cx_core", "cy_core", "core_r",
            "source", "is_manual", "edited_manually",
            "propagation_seed_frame_index", "propagation_seed_frame_file",
            "propagation_obj_id", "propagation_segment_start_pos",
            "propagation_segment_end_pos",
            "is_review_keyframe", "needs_review", "review_reason",
        ])

    present_frames = set()
    if "frame_index" in out_df.columns and not out_df.empty:
        present_frames = set(out_df["frame_index"].astype(int).unique())

    placeholder_rows: List[Dict[str, Any]] = []

    for _, fr in frames_df.iterrows():
        frame_index = int(fr["frame_index"])
        frame_file = str(fr["frame_file"])
        image_path = str(fr["image_path"])

        if frame_index in present_frames:
            continue

        placeholder_rows.append({
            "frame_file": frame_file,
            "frame_index": frame_index,
            "image_path": os.path.abspath(image_path),
            "mask_index": -1,
            "mask_path": "",

            "seg_key": "empty_frame_placeholder",
            "segment_type": "",
            "label": "",
            "label_final": "",
            "ignore": True,

            "track_id": -1,
            "why": "empty_frame_placeholder_added_during_propagation",

            "area": 0,
            "bbox_x": -1,
            "bbox_y": -1,
            "bbox_w": -1,
            "bbox_h": -1,
            "cx_geo": np.nan,
            "cy_geo": np.nan,
            "cx_core": np.nan,
            "cy_core": np.nan,
            "core_r": np.nan,

            "source": "empty_frame_placeholder",
            "is_manual": False,
            "edited_manually": False,

            "propagation_seed_frame_index": -1,
            "propagation_seed_frame_file": "",
            "propagation_obj_id": -1,
            "propagation_segment_start_pos": -1,
            "propagation_segment_end_pos": -1,

            "is_review_keyframe": bool(frame_index % max(1, int(args.keyframe_interval)) == 0),
            "needs_review": True,
            "review_reason": "empty_frame_needs_manual_review",
        })

    if placeholder_rows:
        print(f"[INFO] Adding {len(placeholder_rows)} empty placeholder frame rows for GUI.")
        out_df = pd.concat([out_df, pd.DataFrame(placeholder_rows)], ignore_index=True)

    out_df = out_df.sort_values(["frame_index", "mask_index"]).reset_index(drop=True)
    out_df.to_csv(out_csv, index=False)

    print("[INFO] Propagated-only CSV unique frames:", out_df["frame_index"].nunique())

    combine_original_and_propagated_csv(
        original_csv=csv_path,
        propagated_csv=out_csv,
        output_csv=combined_csv,
    )

    report_df = pd.DataFrame(report)
    report_df.to_csv(out_report, index=False)

    meta = {
        "input_csv": csv_path,
        "frames_dir": frames_dir,
        "output_dir": output_dir,
        "labels": labels,
        "key_positions": key_positions,
        "sam2_repo": sam2_repo,
        "checkpoint": ckpt,
        "model_cfg": args.model_cfg,
        "manual_only": bool(args.manual_only),
        "min_area": int(args.min_area),
        "max_spatter_per_keyframe": int(args.max_spatter_per_keyframe),
        "propagated_only_csv": out_csv,
        "combined_csv": combined_csv,
        "report_csv": out_report,
    }
    with open(os.path.join(output_dir, "propagation_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n[OK] Saved propagated-only CSV:", out_csv)
    print("[OK] Saved combined original + propagated CSV:", combined_csv)
    print("[OK] Saved report:", out_report)
    print("[OK] Masks:", masks_dir)

    print("\nOpen the COMBINED result with GUI:")
    print(
        f"python scripts/manual_label_web_2.py "
        f"--csv '{combined_csv}' "
        f"--frames '{frames_dir}' "
        f"--port 7860"
    )


if __name__ == "__main__":
    main()
