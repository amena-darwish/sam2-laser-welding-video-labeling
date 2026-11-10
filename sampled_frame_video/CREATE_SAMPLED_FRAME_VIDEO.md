
# Create the 164 GUI frames from an original video

This script uses the same uniform frame-sampling method as `generate_masks.py`.
It produces:

- `frames/` containing the 164 clean PNG frames used by the GUI;
- `frames_manifest.csv` linking GUI frames to original video frames; and
- an MP4 preview with the GUI frame number written on each frame.

## Windows Command Prompt

```cmd
python create_sampled_frame_video.py --video "D:\path\to\original_video.avi" --output-dir "D:\path\to\output\video_name"
```

## Linux, WSL, or macOS

```bash
python create_sampled_frame_video.py \
  --video "/path/to/original_video.avi" \
  --output-dir "/path/to/output/video_name"
```

By default, the video shows both:

- the GUI frame number (`0` to `163`); and
- the corresponding frame number in the original video.

To show only the GUI number:

```cmd
python create_sampled_frame_video.py --video "D:\path\to\original_video.avi" --output-dir "D:\path\to\output\video_name" --gui-only
```

The default preview speed is 10 fps. It can be changed with `--fps`.
The original image resolution is preserved.
