# SAM2 Laser-Welding Video Labeling

This repository provides a standalone workflow for creating and reviewing segmentation labels in laser-welding videos with Meta SAM2. It:

- extracts frames uniformly across an entire video;
- preserves every frame's exact index in the source video;
- generates automatic SAM2 candidate masks;
- supports manual `weld`, `plasma`, and `spatter` labeling;
- lets users review keyframes, such as every tenth extracted frame;
- propagates corrected masks with the SAM2 video predictor; and
- supports final review and dataset export.

The defaults are 164 extracted frames and a keyframe interval of 10. Both are configurable.

## Workflow

1. Generate frames and candidate masks with `generate_masks.py`.
2. Open `manual_label_gui.py` and review **Keyframes only**.
3. Propagate the corrected masks with `propagate_keyframes.py`.
4. Open the propagated manifest for final review and export.

## Included files

| File | Purpose |
|---|---|
| `generate_masks.py` | Extract frames, preserve source indices, and generate SAM2 candidates |
| `manual_label_gui.py` | Manual labeling and review interface |
| `propagate_keyframes.py` | Propagate corrected masks between keyframes |
| `requirements.txt` | UI and image-processing dependencies |

## Requirements

- Python 3.10 or 3.11 recommended
- Git
- PyTorch and TorchVision suitable for the computer
- Meta SAM2 and a compatible checkpoint
- NVIDIA GPU recommended; CPU works but Hiera Large can be very slow

All paths are supplied by the user. The code does not depend on a particular username, drive, Linux/WSL folder, or video filename.

## 1. Download this repository

```bash
git clone <THIS_REPOSITORY_URL>
cd standalone_sam2_labeling
```

Alternatively, download and extract the repository ZIP from GitHub.

## 2. Create an environment

The following Conda commands work on Windows, Linux, and macOS:

```bash
conda create -n sam2_labeling python=3.11 -y
conda activate sam2_labeling
python -m pip install --upgrade pip
```

A normal Python virtual environment can also be used.

## 3. Install PyTorch

PyTorch is intentionally excluded from `requirements.txt` because the correct build depends on the operating system and hardware. Get the current installation command from:

https://pytorch.org/get-started/locally/

NVIDIA GPU example (the versions offered by PyTorch may change):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

CPU-only example:

```bash
pip install torch torchvision
```

The CUDA version displayed by `nvidia-smi` is the maximum supported by the driver. It does not need to exactly match the CUDA runtime bundled with PyTorch. Use a build offered by the official PyTorch installer.

Verify the installation:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

GPU users should see `CUDA available: True`.

## 4. Install Meta SAM2

Choose any convenient directory, then run:

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ..
```

If Git is unavailable, download SAM2 from https://github.com/facebookresearch/sam2, extract it, open a terminal inside it, and run `pip install -e .`.

Native Windows may report that an optional CUDA extension could not be compiled. Follow the current SAM2 documentation if installation fails. WSL/Linux may be easier for some configurations, but this labeling package does not require Linux-specific paths.

## 5. Download a SAM2 checkpoint

Follow the official instructions at https://github.com/facebookresearch/sam2#download-checkpoints.

The package defaults to:

```text
sam2.1_hiera_large.pt
```

Place it at:

```text
<SAM2_REPO>/checkpoints/sam2.1_hiera_large.pt
```

The checkpoint is intentionally excluded from Git. It can be stored elsewhere if its full path is passed to the scripts.

## 6. Install the application dependencies

Return to this repository and run:

```bash
pip install -r requirements.txt
```

## 7. Generate frames and masks

Use a new output directory for every video. It must be empty or not yet exist.

### Windows Command Prompt

```cmd
python generate_masks.py ^
  --video "C:\path\to\video.avi" ^
  --output-dir "C:\path\to\output\video_name" ^
  --repo-dir "C:\path\to\sam2" ^
  --checkpoint "C:\path\to\sam2\checkpoints\sam2.1_hiera_large.pt" ^
  --target-frames 164 ^
  --keyframe-interval 10 ^
  --device cuda
```

### Linux, WSL, or macOS

```bash
python generate_masks.py \
  --video "/path/to/video.avi" \
  --output-dir "/path/to/output/video_name" \
  --repo-dir "/path/to/sam2" \
  --checkpoint "/path/to/sam2/checkpoints/sam2.1_hiera_large.pt" \
  --target-frames 164 \
  --keyframe-interval 10 \
  --device cuda
```

Use `--device cpu` when CUDA is unavailable.

`--repo-dir` can be omitted if `SAM2_REPO` is set:

```cmd
set SAM2_REPO=C:\path\to\sam2
```

```bash
export SAM2_REPO="/path/to/sam2"
```

Expected output:

```text
output/video_name/
|-- frames/
|-- RAW_SAM/
|   `-- masks/
|-- frames_manifest.csv
`-- label_manifest.csv
```

## 8. Label the keyframes

### Windows Command Prompt

```cmd
python manual_label_gui.py ^
  --csv "C:\path\to\output\video_name\label_manifest.csv" ^
  --frames "C:\path\to\output\video_name\frames" ^
  --host 127.0.0.1 ^
  --port 7860
```

### Linux, WSL, or macOS

```bash
python manual_label_gui.py \
  --csv "/path/to/output/video_name/label_manifest.csv" \
  --frames "/path/to/output/video_name/frames" \
  --host 127.0.0.1 \
  --port 7860
```

Open http://127.0.0.1:7860, then:

1. Select **Keyframes only**.
2. Correct or assign masks as `weld`, `plasma`, or `spatter`.
3. Add missing masks with the polygon or brush tools.
4. Click **Save now**.
5. Stop the server with `Ctrl+C`.

With 164 extracted frames and interval 10, approximately 17 frames are reviewed manually.

## 9. Propagate corrected masks

### Windows Command Prompt

```cmd
python propagate_keyframes.py ^
  --csv "C:\path\to\output\video_name\label_manifest.csv" ^
  --frames "C:\path\to\output\video_name\frames" ^
  --repo-dir "C:\path\to\sam2" ^
  --checkpoint-path "C:\path\to\sam2\checkpoints\sam2.1_hiera_large.pt" ^
  --keyframe-interval 10 ^
  --manual-only ^
  --device cuda
```

### Linux, WSL, or macOS

```bash
python propagate_keyframes.py \
  --csv "/path/to/output/video_name/label_manifest.csv" \
  --frames "/path/to/output/video_name/frames" \
  --repo-dir "/path/to/sam2" \
  --checkpoint-path "/path/to/sam2/checkpoints/sam2.1_hiera_large.pt" \
  --keyframe-interval 10 \
  --manual-only \
  --device cuda
```

`--manual-only` prevents unreviewed automatic masks from becoming propagation seeds. The default output is:

```text
output/video_name/sam2_propagated_keyframes/
```

Use `--overwrite` only when intentionally replacing an existing propagation run.

## 10. Final review and export

### Windows Command Prompt

```cmd
python manual_label_gui.py ^
  --csv "C:\path\to\output\video_name\sam2_propagated_keyframes\labels_manifest_combined_original_plus_propagated.csv" ^
  --frames "C:\path\to\output\video_name\frames" ^
  --host 127.0.0.1 ^
  --port 7860
```

### Linux, WSL, or macOS

```bash
python manual_label_gui.py \
  --csv "/path/to/output/video_name/sam2_propagated_keyframes/labels_manifest_combined_original_plus_propagated.csv" \
  --frames "/path/to/output/video_name/frames" \
  --host 127.0.0.1 \
  --port 7860
```

Review the propagated masks, correct errors, and click **Export final dataset**. The export includes `labels_final.csv`, final masks, and copies of the referenced frames.

## Preserved frame indices

| Field | Meaning |
|---|---|
| `frame_index` | Position in the extracted sequence |
| `source_frame_index` | Exact frame index in the original video |
| `source_video` | Original video filename |

These fields remain available through generation, propagation, review, and final export. Do not treat the extracted filename alone as the original frame index.

## Default settings

| Parameter | Default |
|---|---:|
| Model | SAM2.1 Hiera Large |
| Points per side | 16 |
| Generator predicted-IoU threshold | 0.40 |
| Generator stability threshold | 0.25 |
| Retained predicted-IoU threshold | 0.50 |
| Retained stability threshold | 0.50 |
| Minimum mask area | 20 px |
| Maximum frame-area fraction | 0.98 |
| Duplicate-mask NMS IoU | 0.97 |
| Extracted frames | 164 |
| Keyframe interval | 10 |

To see all configurable options:

```bash
python generate_masks.py --help
python manual_label_gui.py --help
python propagate_keyframes.py --help
```

## Do not commit to GitHub

- SAM2 checkpoints
- input videos
- extracted frames and generated masks
- labeling output directories
- machine-specific absolute paths
- Conda environments or Python caches

The included `.gitignore` excludes common generated files and model weights.

## Troubleshooting

- **`CUDA available: False`**: check `nvidia-smi`, then install PyTorch using its official selector. Do not select a wheel only by matching the CUDA number printed by `nvidia-smi`.
- **`Could not import sam2`**: activate the correct environment, enter the SAM2 repository, and rerun `pip install -e .`.
- **Checkpoint not found**: verify `--checkpoint` or `--checkpoint-path` and confirm that the download completed.
- **Output directory is not empty**: choose a new directory to protect existing labels.
- **Port 7860 is busy**: use `--port 7861` and open http://127.0.0.1:7861.
- **GPU out of memory**: close GPU applications, reduce `--points-per-side`, or use a smaller SAM2 model with its matching checkpoint and configuration.
- **CPU is very slow**: this is expected with Hiera Large; use a supported GPU or a smaller matching model.
- **Video cannot be read**: confirm the path and permissions. If OpenCV lacks the codec, convert the video to a common AVI or MP4 codec.

When reporting an issue, include the operating system, Python and PyTorch versions, CUDA availability, GPU name, exact command, and full terminal error.
