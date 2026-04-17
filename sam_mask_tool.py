from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

if "QT_QPA_FONTDIR" not in os.environ:
    default_qt_font_dir = Path("/usr/share/fonts/truetype/dejavu")
    if default_qt_font_dir.exists():
        os.environ["QT_QPA_FONTDIR"] = str(default_qt_font_dir)

import cv2
import numpy as np


WINDOW_NAME = "SAM Mask Tool"
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = REPO_ROOT / "examples" / "images" / "thirdview_calibrate_2.png"
DEFAULT_SAM_MODEL = REPO_ROOT / "sam2_s.pt"
DEFAULT_WORLD_MODEL = REPO_ROOT / "yolov8s-world.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"


@dataclass
class ClickState:
    image_bgr: np.ndarray
    predictor: object
    mask_index: int
    output_dir: Path
    image_path: Path
    points: List[List[int]] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    last_mask: Optional[np.ndarray] = None
    current_label: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use SAM to extract a robot arm mask by clicking or by text prompt."
    )
    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE),
        help="Input image path.",
    )
    parser.add_argument(
        "--sam-model",
        default=str(DEFAULT_SAM_MODEL),
        help="SAM checkpoint path.",
    )
    parser.add_argument(
        "--mode",
        choices=("click", "prompt"),
        default="click",
        help="click: interactive point prompts. prompt: text prompt via YOLOWorld + SAM.",
    )
    parser.add_argument(
        "--prompt",
        default="robot arm",
        help="Text prompt used in prompt mode.",
    )
    parser.add_argument(
        "--world-model",
        default=str(DEFAULT_WORLD_MODEL),
        help="YOLOWorld checkpoint used in prompt mode.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Confidence threshold for prompt mode detection.",
    )
    parser.add_argument(
        "--mask-index",
        type=int,
        default=0,
        help="Which SAM candidate mask to keep if multiple masks are returned.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save mask and cutout PNGs.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Do not open an OpenCV preview window in prompt mode.",
    )
    return parser.parse_args()


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def ensure_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def can_use_gui() -> bool:
    display = os.environ.get("DISPLAY")
    if display:
        try:
            result = subprocess.run(
                ["xdpyinfo", "-display", display],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode == 0:
                return True
        except FileNotFoundError:
            return False

    return bool(os.environ.get("WAYLAND_DISPLAY"))


def opencv_window_flags() -> int:
    return cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_GUI_NORMAL", 0)


def load_sam_predictor(sam_model: str | Path) -> object:
    try:
        from ultralytics import SAM
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed. Please run `pip install ultralytics` first."
        ) from exc

    sam_model = str(expand_path(sam_model))
    sam = SAM(sam_model)
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        imgsz=1024,
        model=sam_model,
        save=False,
        verbose=False,
    )
    predictor_cls = sam.task_map["segment"]["predictor"]
    predictor = predictor_cls(overrides=overrides)
    predictor.setup_model(model=sam.model, verbose=False)
    predictor.args.save = False
    predictor.args.verbose = False
    return predictor


def extract_mask(result: object, mask_index: int) -> np.ndarray:
    if result.masks is None or result.masks.data is None:
        raise RuntimeError("SAM did not return any mask.")

    mask_data = result.masks.data
    if hasattr(mask_data, "detach"):
        mask_data = mask_data.detach().cpu().numpy()
    else:
        mask_data = np.asarray(mask_data)

    if mask_data.ndim == 2:
        mask = mask_data
    elif mask_data.ndim == 3:
        index = max(0, min(mask_index, mask_data.shape[0] - 1))
        mask = mask_data[index]
    else:
        raise RuntimeError(f"Unexpected mask shape: {mask_data.shape}")

    return (mask > 0).astype(np.uint8)


def save_outputs(
    mask: np.ndarray,
    image_bgr: np.ndarray,
    image_path: Path,
    output_dir: Path,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / f"{image_path.stem}_mask.png"
    cutout_path = output_dir / f"{image_path.stem}_cutout.png"

    mask_u8 = (mask * 255).astype(np.uint8)
    cv2.imwrite(str(mask_path), mask_u8)

    cutout = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    cutout[:, :, 3] = mask_u8
    cv2.imwrite(str(cutout_path), cutout)

    return mask_path, cutout_path


def prompt_candidates(prompt: str) -> List[str]:
    prompt = prompt.strip()
    variants = [prompt]
    alias_map = {
        "gripper": [
            "end effector",
            "robotic gripper",
            "robot gripper",
            "robotic arm",
            "robot arm",
        ],
        "robot arm": [
            "robotic arm",
            "industrial robot arm",
            "manipulator",
            "end effector",
        ],
        "robotic arm": ["robot arm", "industrial robot arm", "manipulator", "end effector"],
        "end effector": ["gripper", "robotic gripper", "robot arm"],
    }
    prompt_lower = prompt.lower()
    variants.extend(alias_map.get(prompt_lower, []))

    # Support partial prompts like "end" -> "end effector".
    known_terms = list(alias_map.keys()) + [
        item for values in alias_map.values() for item in values
    ]
    for term in known_terms:
        term_lower = term.lower()
        if prompt_lower and (prompt_lower in term_lower or term_lower in prompt_lower):
            variants.append(term)
            variants.extend(alias_map.get(term_lower, []))

    deduped: List[str] = []
    seen = set()
    for item in variants:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def detect_best_prompt_box(
    detector: object,
    image_path: Path,
    prompt: str,
    conf: float,
) -> Tuple[str, np.ndarray, float]:
    def detect_candidate(candidate: str) -> Optional[Tuple[np.ndarray, float]]:
        detector.set_classes([candidate])
        det_results = detector.predict(source=str(image_path), conf=conf, verbose=False)
        if not det_results or det_results[0].boxes is None or len(det_results[0].boxes) == 0:
            return None

        boxes = det_results[0].boxes.xyxy
        scores = det_results[0].boxes.conf
        if hasattr(boxes, "detach"):
            boxes = boxes.detach().cpu().numpy()
        else:
            boxes = np.asarray(boxes)
        if hasattr(scores, "detach"):
            scores = scores.detach().cpu().numpy()
        else:
            scores = np.asarray(scores)

        idx = int(np.argmax(scores))
        return np.asarray(boxes[idx]), float(scores[idx])

    exact = detect_candidate(prompt)
    if exact is not None:
        box, score = exact
        return prompt, box, score

    best_prompt = ""
    best_box = None
    best_score = -1.0
    for candidate in prompt_candidates(prompt)[1:]:
        detected = detect_candidate(candidate)
        if detected is None:
            continue
        box, score = detected
        if score > best_score:
            best_prompt = candidate
            best_box = box
            best_score = score

    if best_box is None:
        raise RuntimeError(
            f"YOLOWorld could not find `{prompt}` in {image_path}. "
            f"Tried prompts: {', '.join(prompt_candidates(prompt))}. "
            "Try click mode or lower --conf."
        )

    return best_prompt, np.asarray(best_box), best_score


def build_overlay(
    image_bgr: np.ndarray,
    mask: Optional[np.ndarray],
    points: Sequence[Sequence[int]],
    labels: Sequence[int],
    current_label: int,
) -> np.ndarray:
    canvas = image_bgr.copy()

    if mask is not None:
        colored = np.zeros_like(canvas)
        colored[:, :, 1] = 255
        mask_bool = mask.astype(bool)
        canvas[mask_bool] = cv2.addWeighted(canvas, 0.35, colored, 0.65, 0)[mask_bool]

    for (x, y), label in zip(points, labels):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(canvas, (x, y), 6, color, -1)
        cv2.circle(canvas, (x, y), 10, (255, 255, 255), 1)

    hints = [
        f"Current click mode: {'foreground' if current_label == 1 else 'background'}",
        "Left click: add point   F: foreground   B: background",
        "U: undo   R: reset   S: save   Q/ESC: quit",
    ]
    for row, text in enumerate(hints):
        cv2.putText(
            canvas,
            text,
            (12, 28 + row * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


def rerun_with_clicks(state: ClickState) -> None:
    if not state.points:
        state.last_mask = None
        return

    results = state.predictor(
        state.image_bgr,
        points=[state.points],
        labels=[state.labels],
    )
    state.last_mask = extract_mask(results[0], state.mask_index)
    mask_path, cutout_path = save_outputs(
        state.last_mask,
        state.image_bgr,
        state.image_path,
        state.output_dir,
    )
    print(f"Saved mask to: {mask_path}")
    print(f"Saved cutout to: {cutout_path}")


def on_mouse(event: int, x: int, y: int, _flags: int, state: ClickState) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        state.points.append([x, y])
        state.labels.append(state.current_label)
        rerun_with_clicks(state)


def run_click_mode(args: argparse.Namespace) -> None:
    if not can_use_gui():
        raise RuntimeError(
            "Click mode requires a GUI display, but no DISPLAY/WAYLAND_DISPLAY is available. "
            "Use prompt mode, or run in a desktop session."
        )

    image_path = expand_path(args.image)
    image_bgr = ensure_image(image_path)
    predictor = load_sam_predictor(args.sam_model)
    predictor.set_image(image_bgr)

    state = ClickState(
        image_bgr=image_bgr,
        predictor=predictor,
        mask_index=args.mask_index,
        output_dir=expand_path(args.output_dir),
        image_path=image_path,
    )

    cv2.namedWindow(WINDOW_NAME, opencv_window_flags())
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, state)
    print("Controls:")
    print("  Left click -> add point")
    print("  F -> foreground mode")
    print("  B -> background mode")
    print("  U -> undo last point")
    print("  R -> reset all points")
    print("  S -> save current mask")
    print("  Q / ESC -> quit")

    while True:
        canvas = build_overlay(
            state.image_bgr,
            state.last_mask,
            state.points,
            state.labels,
            state.current_label,
        )
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q")):
            break
        if key == ord("f"):
            state.current_label = 1
        if key == ord("b"):
            state.current_label = 0
        if key == ord("r"):
            state.points.clear()
            state.labels.clear()
            state.last_mask = None
        if key == ord("u") and state.points:
            state.points.pop()
            state.labels.pop()
            rerun_with_clicks(state)
        if key == ord("s"):
            if state.last_mask is None:
                print("No mask yet. Click on the image first.")
            else:
                mask_path, cutout_path = save_outputs(
                    state.last_mask,
                    state.image_bgr,
                    state.image_path,
                    state.output_dir,
                )
                print(f"Saved mask to: {mask_path}")
                print(f"Saved cutout to: {cutout_path}")

    cv2.destroyAllWindows()


def run_prompt_mode(args: argparse.Namespace) -> None:
    image_path = expand_path(args.image)
    image_bgr = ensure_image(image_path)
    predictor = load_sam_predictor(args.sam_model)
    predictor.set_image(image_bgr)

    try:
        from ultralytics import YOLOWorld
    except ImportError as exc:
        raise ImportError(
            "Prompt mode requires ultralytics with YOLOWorld support. "
            "Install it with `pip install ultralytics`."
        ) from exc

    detector = YOLOWorld(str(expand_path(args.world_model)))
    matched_prompt, best_box, best_score = detect_best_prompt_box(
        detector,
        image_path,
        args.prompt,
        args.conf,
    )
    best_box = best_box.tolist()
    sam_results = predictor(image_bgr, bboxes=best_box)
    mask = extract_mask(sam_results[0], args.mask_index)
    mask_path, cutout_path = save_outputs(
        mask,
        image_bgr,
        image_path,
        expand_path(args.output_dir),
    )

    preview = build_overlay(image_bgr, mask, [], [], 1)
    x1, y1, x2, y2 = map(int, best_box)
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(
        preview,
        f"Prompt: {matched_prompt} ({best_score:.3f})",
        (12, preview.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if matched_prompt.lower() != args.prompt.lower():
        print(
            f"Prompt `{args.prompt}` had no good box. "
            f"Fell back to `{matched_prompt}` (score={best_score:.3f})."
        )
    print(f"Saved mask to: {mask_path}")
    print(f"Saved cutout to: {cutout_path}")
    if args.no_preview or not can_use_gui():
        print("Preview skipped because GUI display is unavailable or --no-preview was set.")
        return

    cv2.namedWindow(WINDOW_NAME, opencv_window_flags())
    cv2.imshow(WINDOW_NAME, preview)
    print("Press any key in the preview window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    if args.mode == "click":
        run_click_mode(args)
    else:
        run_prompt_mode(args)


if __name__ == "__main__":
    main()
