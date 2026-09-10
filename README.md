# Standalone SAM2 Labeling for Laser Welding Videos

This repository provides a standalone human-in-the-loop workflow for creating reviewed segmentation annotations from laser welding videos using SAM2 and a custom manual-labeling interface.

The final annotation classes are:

- `weld`
- `plasma`
- `spatter`

SAM2 supports automatic mask generation and propagation, while the final reference annotations are manually reviewed and corrected before export.

---

## 1. First-time installation

The steps below only need to be done once on a new computer.

### 1.1 Install Anaconda or Miniconda

Install Anaconda or Miniconda if Conda is not already available.

After installation, open **Anaconda Prompt** or **Command Prompt** with Conda available.

### 1.2 Create the environment

```cmd
conda create -n sam2_labeling python=3.10 -y
conda activate sam2_labeling
```

Upgrade `pip`:

```cmd
python -m pip install --upgrade pip
```

### 1.3 Install PyTorch

Install PyTorch first:

```cmd
pip install torch torchvision
```

Check that PyTorch is installed and that the NVIDIA GPU is available:

```cmd
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

For SAM2 propagation, an NVIDIA GPU is strongly recommended.

If `CUDA available` is `False` on a computer with an NVIDIA GPU, install the PyTorch build recommended for the computer from:

https://pytorch.org/get-started/locally/

### 1.4 Install Gradio and the other Python packages

```cmd
pip install gradio opencv-python pandas numpy hydra-core iopath tqdm
```

Check Gradio:

```cmd
python -c "import gradio as gr; print('Gradio:', gr.__version__)"
```

### 1.5 Install SAM2

Clone the official SAM2 repository:

```cmd
git clone https://github.com/facebookresearch/sam2.git
```

Move into the SAM2 repository:

```cmd
cd sam2
```

Install SAM2 in the current Conda environment:

```cmd
pip install -e .
```

### 1.6 Download the SAM2 checkpoint

Download the SAM2.1 Hiera Large checkpoint:

```text
sam2.1_hiera_large.pt
```

Place it inside:

```text
sam2/
└── checkpoints/
    └── sam2.1_hiera_large.pt
```

The checkpoint can be obtained from the official SAM2 repository:

https://github.com/facebookresearch/sam2

### 1.7 Check the installation

Run:

```cmd
python -c "import torch, gradio, cv2, pandas, numpy, sam2; print('Installation OK'); print('CUDA available:', torch.cuda.is_available())"
```

If this prints:

```text
Installation OK
```

the main dependencies are available.

---

## 2. Repository files

Keep the standalone labeling scripts together in one project folder, for example:

```text
standalone_sam2_labeling/
├── generate_masks.py
├── manual_label_gui.py
├── propagate_keyframes_sam2.py
└── ...
```

The SAM2 repository can be stored separately, for example:

```text
C:\path\to\sam2\
```

The UI uses `propagate_keyframes_sam2.py` for the integrated SAM2 keyframe-propagation step, so keep the propagation script together with the standalone labeling code.

---

## 3. Workflow

```text
Original welding video
        ↓
Extract frames + generate initial SAM2 masks
        ↓
Open the manual-labeling UI
        ↓
Review and label keyframes
(for example every 10 frames)
        ↓
Save + Run SAM2 keyframe propagation
        ↓
Open propagated review
        ↓
Inspect and correct propagated masks
        ↓
Remove duplicate PROP masks on keyframes
        ↓
Export final dataset
        ↓
frames/
final_masks/
labels_final.csv
```

---

## 4. Activate the environment

Each time the labeling workflow is used, activate the environment:

```cmd
conda activate sam2_labeling
```

Move to the standalone repository:

```cmd
cd /d Path\standalone_sam2_labeling
```

Replace the example paths below with the paths used on your computer.

---

## 5. Extract frames and generate initial SAM2 masks

Run `generate_masks.py` on the original welding video.

Example:

```cmd
python generate_masks.py ^
  --video "D:\path\to\video.avi" ^
  --output-dir "D:\path\to\output\video_name" ^
  --target-frames 164 ^
  --keyframe-interval 10 ^
  --sam2-repo "C:\path\to\sam2"
```

If the available arguments differ, check them with:

```cmd
python generate_masks.py -h
```

Typical output:

```text
video_name/
├── frames/
├── RAW_SAM/
├── frames_manifest.csv
└── label_manifest.csv
```

---

## 6. Open the manual-labeling UI

Start the interface with the generated `label_manifest.csv`:

```cmd
python manual_label_gui.py ^
  --csv "D:\path\to\output\video_name\label_manifest.csv" ^
  --frames "D:\path\to\output\video_name\frames" ^
  --host 127.0.0.1 ^
  --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

---

## 7. Label the keyframes

Set:

```text
Review queue → Keyframes only
```

With a keyframe interval of 10, review frames such as:

```text
0, 10, 20, 30, ..., 150, 160
```

Review and correct the selected keyframes only. The frames between the keyframes will be filled later by SAM2 propagation.

### Keyframe labeling example

![Manual labeling steps 1–4](Images/manual_labeling_steps_1_4.png)

### Steps 1–4

1. **Choose the frame.**  
   Use the frame index or review navigation. For example, review every 10 frames.

2. **Select the mask.**  
   Click once on the mask in the image, or select its row in the table.

3. **Assign the class.**  
   For example, if the selected mask is the weld region, click **Set → weld**.

4. **Save your work.**  
   Click **Save now** regularly.

Repeat the same procedure for all selected keyframes.

The three final classes are:

- `weld`
- `plasma`
- `spatter`

The interface may also contain working categories such as `static` and `dynamic_other`. These are review categories and are not final training classes.

---

## 8. Useful manual-editing tools

The interface supports:

- selecting masks directly from the image;
- selecting masks from the table;
- assigning `weld`, `plasma`, or `spatter`;
- ignoring or soft-deleting incorrect masks;
- polygon drawing;
- brush editing;
- splitting one mask using two seed points;
- merging two masks;
- assigning track IDs.

Use these tools when an automatically generated mask is missing, incorrect, split, or merged with another object.

---

## 9. Important: UI label-forward is not SAM2 propagation

The option:

```text
Auto-propagate label forward
```

is different from SAM2 keyframe propagation. It only carries a class label forward to sufficiently similar candidate masks using an IoU rule.

For the keyframe workflow, use the dedicated **SAM2 keyframe propagation** section.

---

## 10. Run SAM2 keyframe propagation inside the UI

After reviewing the keyframes, use the propagation section in the interface.

![SAM2 propagation steps 5–8](Images/sam2_propagation_steps_5_8.png)

### Steps 5–8

5. **Run SAM2 propagation.**  
   Click **Save + Run SAM2 keyframe propagation**.

6. **Check the progress.**  
   Wait until the propagation progress reaches `100%`.

7. **Open propagated review.**  
   Click **Open propagated review**. Inspect and correct the propagated masks in the frames between the manually reviewed keyframes. Also check the keyframes themselves for duplicate masks.

8. **Export the final dataset.**  
   Only after the propagated masks have been reviewed and corrected, click **Export final dataset**.

You do **not** need to close the UI before running propagation.

---

## 11. Important: check duplicate masks on keyframes

After SAM2 propagation, a reviewed keyframe can contain two masks for the same object:

- `ORG` or `MAN` — the original or manually reviewed mask
- `PROP` — the SAM2 propagated mask

This can happen because the reviewed keyframe is also used as a propagation seed.

For the same physical object, keep only one final mask.

Normally:

1. keep the manually reviewed `ORG` or `MAN` mask if it is correct;
2. select the duplicated `PROP` mask;
3. use **Soft delete selected** or **Toggle ignore**;
4. confirm that only one mask remains for that object.

Do not keep both masks because this would create a duplicate annotation for the same object.

For `spatter`, several masks may be correct in one frame when they represent different physical spatter objects. Remove only masks that represent the same object twice.

---

## 12. What propagation creates

The propagation step creates a folder such as:

```text
video_name/
└── sam2_propagated_keyframes/
```

Typical intermediate files include:

```text
labels_manifest_propagated.csv
labels_manifest_combined_original_plus_propagated.csv
propagation_report.csv
propagation_meta.json
typed_masks/
```

The combined CSV is loaded by **Open propagated review**.

---

## 13. Final review

During the final review:

1. inspect the propagated masks;
2. remove duplicate masks on the keyframes;
3. correct wrong boundaries;
4. add missing regions;
5. remove unwanted masks;
6. correct wrong class labels;
7. confirm that the final objects use only `weld`, `plasma`, and `spatter`;
8. save regularly.

When the review is complete, click **Export final dataset**.

---

## 14. Final annotation dataset

The clean annotation folder used for model development should contain:

```text
video_name/
├── frames/
├── final_masks/
└── labels_final.csv
```

If a frame-mapping file is available, it can also be kept to link the extracted annotation frames to their original positions in the complete video:

```text
video_name/
├── frames/
├── final_masks/
├── labels_final.csv
└── frame_mapping.csv
```

### Final files

- `frames/` — image frames used for annotation and model input.
- `final_masks/` — final reviewed segmentation masks.
- `labels_final.csv` — final labels linking each object to its frame and mask.
- `frame_mapping.csv` — optional mapping to original video-frame indices.

---

## 15. Files not required in the clean public annotation package

After the final export has been checked, intermediate files can be excluded from the clean release copy.

Examples:

```text
RAW_SAM/
typed_masks/
_sam2_video_frames_jpg/
labels_manifest*.csv
labels_final_before_*.csv
propagation_report*.csv
propagation_meta.json
temporary overlay images
checkpoint files
*.Zone.Identifier
```

Keep the original working annotation folder as a backup until the clean release copy has been verified.

---

## 16. Verify the final dataset

Before publishing or training from the cleaned annotations, verify that:

1. every `image_path` in `labels_final.csv` points to an existing file in `frames/`;
2. every `mask_path` points to an existing file in `final_masks/`;
3. the final labels contain only `weld`, `plasma`, and `spatter`;
4. duplicate masks from the keyframe propagation have been removed;
5. ignored or deleted rows are not included in the final export;
6. the paths remain valid if the dataset folder is moved.

Preferred path format:

```text
frames/frame_00010.png
final_masks/frame_00010_obj_001.png
```

Avoid machine-specific absolute paths in the released CSV.

---

## 17. Recommended public annotation structure

```text
annotations/
├── model_development/
│   ├── DoE3_9/
│   ├── DoE3_16/
│   ├── DoE3_22/
│   ├── DoE3_24/
│   ├── DoE3_25/
│   └── DoE3_26/
│
└── independent_evaluation/
    ├── DoE3_19/
    └── DoE3_23/
```

---

## 18. Annotation principle

SAM2 assists mask generation and temporal propagation. Automatically generated masks are not treated as ground truth by themselves.

The released reference annotations are the masks retained after manual inspection, correction, final review, duplicate removal, and export.
