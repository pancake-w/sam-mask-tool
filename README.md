# SAM Mask Tool

Small utility for extracting robot-arm masks from calibration images with
Ultralytics SAM. It supports two workflows:

- `click`: interactively add foreground/background points in an OpenCV window.
- `prompt`: detect an object with YOLOWorld text prompts, then segment it with SAM.

This repository is intended to stay lightweight enough to publish publicly and
to use as a Git submodule from another project.

## Repository Layout

```text
.
├── sam_mask_tool.py          # CLI entrypoint
├── examples/images/          # Small input images for quick checks
├── requirements.txt          # Python runtime dependencies
└── outputs/                  # Generated masks/cutouts, ignored by Git
```

Model checkpoints such as `sam2_s.pt` and `yolov8s-world.pt` are intentionally
ignored and should not be committed to the public repository. Put them in the
repository root or pass explicit paths with `--sam-model` and `--world-model`.

## Install

Use Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Linux, `click` mode needs a desktop session with a working GUI display.
If OpenCV cannot open a window, use `prompt` mode with `--no-preview`.

## Usage

Interactive point-based mask extraction:

```bash
python sam_mask_tool.py \
  --mode click \
  --image examples/images/thirdview_calibrate_2.png \
  --sam-model sam2_s.pt
```

Prompt-based mask extraction without opening a preview window:

```bash
python sam_mask_tool.py \
  --mode prompt \
  --image examples/images/thirdview_calibrate_2.png \
  --prompt "robot arm" \
  --sam-model sam2_s.pt \
  --world-model yolov8s-world.pt \
  --no-preview
```

Generated files are written to `outputs/` by default:

- `<image>_mask.png`: binary mask.
- `<image>_cutout.png`: BGRA cutout with the mask in the alpha channel.

## Using As A Submodule

From the parent `mask` repository:

```bash
git submodule add <repo-url> third_party/sam-mask-tool
git submodule update --init --recursive
pip install -r third_party/sam-mask-tool/requirements.txt
```

Then call the script from the parent project:

```bash
python third_party/sam-mask-tool/sam_mask_tool.py \
  --mode prompt \
  --image path/to/input.png \
  --sam-model path/to/sam2_s.pt \
  --world-model path/to/yolov8s-world.pt \
  --output-dir path/to/output \
  --no-preview
```

If the submodule path is added to `PYTHONPATH`, avoid treating this repository
as the external `ultralytics` Python package. The actual package still comes
from PyPI through `requirements.txt`.

## Public Release Notes

- Keep model weights, run outputs, caches, and local editor state out of Git.
- Add a license before accepting external use or contributions.
- Do not commit private calibration images if future examples contain sensitive
  lab or customer data.
