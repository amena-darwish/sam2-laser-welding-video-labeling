# Standalone SAM2 Labeling for Laser-Welding Videos

This repository provides a standalone workflow for creating and reviewing segmentation labels in laser-welding videos with Meta SAM2. It:

- extracts frames uniformly across an entire video;
- preserves every frame's exact index in the source video;
- generates automatic SAM2 candidate masks;
- supports manual `weld`, `plasma`, and `spatter` labeling;
- lets users review keyframes, such as every tenth extracted frame;
- propagates corrected masks with the SAM2 video predictor; and
- supports final review and dataset export.

The defaults are 164 extracted frames and a keyframe interval of 10. Both are configurable.

**GitHub repository:** [amena-darwish/sam2-laser-welding-video-labeling](https://github.com/amena-darwish/sam2-laser-welding-video-labeling)

## Dataset

The associated laser-welding video annotation dataset is distributed separately from the source code.

**Dataset:** [Add the published dataset link here](DATASET_URL_HERE)

After publishing the dataset, replace `DATASET_URL_HERE` with its DOI or repository URL (for example, Zenodo, Figshare, Mendeley Data, or an institutional repository).

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
| `requirements-tested.txt` | Exact non-PyTorch versions from the verified Windows environment |

## Requirements

- Python 3.10 or 3.11 recommended
- Git
- PyTorch and TorchVision suitable for the computer
- Meta SAM2 and a compatible checkpoint
- NVIDIA GPU recommended; CPU works but Hiera Large can be very slow

All paths are supplied by the user. The code does not depend on a particular username, drive, Linux/WSL folder, or video filename.

## 1. Download this repository

```bash
git clone https://github.com/amena-darwish/sam2-laser-welding-video-labeling.git
cd sam2-laser-welding-video-labeling
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
pip uninstall -y hf-gradio
pip install -r requirements.txt
```

`requirements.txt` pins the UI stack to the versions used by the original
working labeling interface. `hf-gradio` is not used by this project and should
not be installed because it can interfere with Gradio's API generation.

To reproduce the exact verified application/web stack instead, use:

```bash
pip install -r requirements-tested.txt
```

PyTorch and SAM2 are intentionally excluded from both files. PyTorch must match
the target CPU/GPU platform, and SAM2 is installed from Meta's repository.

### Verified environment

The SAM2 workflow was tested on an NVIDIA RTX A6000. The original working UI
was verified with the following interface stack:

| Package | Tested version |
|---|---:|
| Python | 3.11 environment |
| NumPy | 2.2.6 |
| pandas | 2.2.3 |
| OpenCV Python | 5.0.0.93 |
| Gradio | 6.5.1 |
| Gradio Client | 2.0.3 |
| FastAPI | 0.129.0 |
| Starlette | 0.52.1 |
| Pydantic | 2.12.5 |
| hf-gradio | Not installed |
| Hydra Core | 1.3.6 |
| OmegaConf | 2.3.1 |
| PyTorch | 2.11.0+cu128 |
| TorchVision | 0.26.0+cu128 |
| SAM2 | 1.0 |

The tested PyTorch versions are evidence of one successful CUDA setup, not a
universal requirement. New users should select PyTorch from the official
installer for their own hardware and operating system.

## 7. Generate frames and masks

Use a new output directory for every video. It must be empty or not yet exist.

### Windows Command Prompt

```cmd
python generate_masks.py --video "C:\path\to\video.avi" --output-dir "C:\path\to\output\video_name" --repo-dir "C:\path\to\sam2" --checkpoint "C:\path\to\sam2\checkpoints\sam2.1_hiera_large.pt" --target-frames 164 --keyframe-interval 10 --device cuda
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
python manual_label_gui.py --csv "C:\path\to\output\video_name\label_manifest.csv" --frames "C:\path\to\output\video_name\frames" --host 127.0.0.1 --port 7860
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
2. Select masks by clicking the image or using the mask table.
3. Correct or assign masks as `weld`, `plasma`, or `spatter`.
4. Add missing masks with the polygon or brush tools.
5. Use split or merge when a candidate mask has the wrong shape.
6. Click **Save now** regularly.
7. Stop the server with `Ctrl+C` after finishing the keyframes.

### Manual labeling controls

- **Previous/next frame** navigates through all extracted frames.
- **Previous/next review** follows the selected review queue.
- **Keyframes only** limits review navigation to the configured keyframes.
- **Set → weld/plasma/spatter** assigns the selected or checked masks.
- **Polygon** creates a new mask from clicked boundary points.
- **Brush** creates one or more masks from painted connected regions.
- **Split** separates a selected mask using two seed points.
- **Merge** combines two selected masks.
- **Save now** writes changes to the active manifest CSV.
- **Export final dataset** writes the reviewed CSV, masks, and frame copies.

The interface intentionally does not display the former large diagnostic
**Info** panel. Essential operation feedback remains in the compact save,
split/merge, and export message fields.

With 164 extracted frames and interval 10, approximately 17 frames are reviewed manually.

### Keyframe-labeling example

![Manual labeling steps 1–4](./images/manual_labeling_steps_1_4.png)

## 9. Propagate corrected masks

### Windows Command Prompt

```cmd
python propagate_keyframes.py --csv "C:\path\to\output\video_name\label_manifest.csv" --frames "C:\path\to\output\video_name\frames" --repo-dir "C:\path\to\sam2" --checkpoint-path "C:\path\to\sam2\checkpoints\sam2.1_hiera_large.pt" --keyframe-interval 10 --manual-only --device cuda
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

### Propagation and final-review example

![SAM2 propagation steps 5–8](./images/sam2_propagation_steps_5_8.png)

> **Important:** A reviewed keyframe may contain both its manually corrected mask and a propagated mask. Inspect these keyframes and delete or ignore the duplicate mask before final export.

## 10. Final review and export

### Windows Command Prompt

```cmd
python manual_label_gui.py --csv "C:\path\to\output\video_name\sam2_propagated_keyframes\labels_manifest_combined_original_plus_propagated.csv" --frames "C:\path\to\output\video_name\frames" --host 127.0.0.1 --port 7860
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

Before exporting, confirm that image selection, table selection, class buttons,
brush saving, polygon saving, frame navigation, and **Save now** work in the
installed Gradio environment.

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
- **Gradio API/schema errors mentioning `hf_gradio`, `DataFrame`, `TemplateResponse`, or `localhost is not accessible`**: run `pip uninstall -y gradio gradio-client hf-gradio`, then reinstall the verified stack with `pip install --upgrade --force-reinstall -r requirements.txt`.
- **GPU out of memory**: close GPU applications, reduce `--points-per-side`, or use a smaller SAM2 model with its matching checkpoint and configuration.
- **CPU is very slow**: this is expected with Hiera Large; use a supported GPU or a smaller matching model.
- **Video cannot be read**: confirm the path and permissions. If OpenCV lacks the codec, convert the video to a common AVI or MP4 codec.

When reporting an issue, include the operating system, Python and PyTorch versions, CUDA availability, GPU name, exact command, and full terminal error.
