# Create a video from the exported GUI masks

First, finish reviewing the labels in the GUI and click **Export final
dataset**. The export must contain:

- `labels_final.csv`
- `frames/`
- `final_masks/`

Run the following command from the repository folder.

## Windows Command Prompt

```cmd
python create_final_mask_video.py --csv "C:\path\to\video_name\labels_final.csv" --frames "C:\path\to\video_name\frames"
```

## Linux, WSL, or macOS

```bash
python create_final_mask_video.py \
  --csv "/path/to/video_name/labels_final.csv" \
  --frames "/path/to/video_name/frames"
```

The script finds `final_masks/` automatically beside `labels_final.csv`.
Before creating the video, it deletes mask images from `final_masks/` that are
not used by the three final classes:

- `weld`
- `plasma`
- `spatter`

The CSV file is not changed. Working masks in `RAW_SAM/` and propagation
folders are not changed.

The video uses the original frame resolution and shows:

- weld in red;
- plasma in yellow; and
- spatter in green.

The default output is saved beside the CSV as:

```text
video_name_final_masks.mp4
```

Optional settings:

```cmd
python create_final_mask_video.py --csv "C:\path\to\labels_final.csv" --frames "C:\path\to\frames" --fps 10 --alpha 0.45 --output "C:\path\to\final_video.mp4"
```
