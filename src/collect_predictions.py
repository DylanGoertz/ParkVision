"""Collect detector predictions at the evaluation sample times.

This script runs the same detection pipeline as main.py (score, temporal
smoothing, and hysteresis) over every frame of the video, without opening a
display window. At each sample time it:

  * records the detector's prediction for every space in
    results/evaluation_results.csv (actual_status is left blank so it can be
    filled in by visual inspection),
  * saves an unannotated close-up of the parking row with space numbers to
    output/labeling/ for manual labeling,
  * saves the annotated detector frame to output/annotated/.

It also saves the beginning, middle, and end screenshots to
results/screenshots/.

Run from the main ParkVision folder:
    python src/collect_predictions.py
"""

import csv
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from detector import classify_spaces, draw_results, load_parking_spaces


CONFIG_FILE = "config.json"
SAMPLE_TIMES_SECONDS = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]

# These must match the values in main.py.
SMOOTHING_FACTOR = 0.20
HYSTERESIS_MARGIN = 0.02

RESULTS_CSV = Path("results/evaluation_results.csv")
SCREENSHOT_DIR = Path("results/screenshots")
LABELING_DIR = Path("output/labeling")
ANNOTATED_DIR = Path("output/annotated")

CSV_COLUMNS = [
    "time_seconds",
    "frame_number",
    "space_id",
    "actual_status",
    "predicted_status",
    "correct",
]


def status_text(occupied: bool) -> str:
    return "Occupied" if occupied else "Available"


def annotate(frame: np.ndarray, results: List, frame_number: int,
             total_frames: int) -> np.ndarray:
    """Draw the detector overlay exactly as main.py does."""
    annotated = draw_results(frame, results)
    cv2.putText(annotated, f"Frame: {frame_number}/{total_frames}",
                (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


def save_labeling_image(frame: np.ndarray, spaces: List, path: Path) -> None:
    """Save a close-up of the parking spaces without predictions.

    Spaces are outlined in yellow (not red/green) so the person labeling
    the frame judges each space by eye, not by the detector's answer.
    """
    image = frame.copy()

    for space_id, polygon in enumerate(spaces, start=1):
        points = np.array(polygon, dtype=np.int32)
        cv2.polylines(image, [points], True, (0, 255, 255), 1, cv2.LINE_AA)
        x = int(points[:, 0].mean()) - 10
        y = int(points[:, 1].min()) - 12
        cv2.putText(image, str(space_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, str(space_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2, cv2.LINE_AA)

    all_points = np.array([p for polygon in spaces for p in polygon])
    height, width = image.shape[:2]
    x0 = max(int(all_points[:, 0].min()) - 80, 0)
    x1 = min(int(all_points[:, 0].max()) + 80, width)
    y0 = max(int(all_points[:, 1].min()) - 120, 0)
    y1 = min(int(all_points[:, 1].max()) + 80, height)

    crop = image[y0:y1, x0:x1]
    crop = cv2.resize(crop, None, fx=1.5, fy=1.5,
                      interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(path), crop)


def main() -> None:
    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = json.load(file)

    threshold = float(config.get("occupied_threshold", 0.14))
    spaces = load_parking_spaces(config["spaces_file"])

    capture = cv2.VideoCapture(config["input_video"])
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {config['input_video']}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # frame_number matches the "Frame: N" counter main.py draws on screen
    # (the first frame of the video is frame 1).
    sample_frames: Dict[int, int] = {}
    for seconds in SAMPLE_TIMES_SECONDS:
        frame_number = int(round(seconds * fps))
        if total_frames and frame_number > total_frames:
            print(f"Warning: {seconds}s (frame {frame_number}) is past the "
                  f"end of the video ({total_frames} frames). Skipping it.")
            continue
        sample_frames[frame_number] = seconds

    # Beginning = first sample time, middle = halfway through the video.
    # The end screenshot is taken from the last frame that is actually read.
    screenshot_frames: Dict[int, str] = {}
    if sample_frames:
        screenshot_frames[min(sample_frames)] = "beginning"
    if total_frames > 0:
        screenshot_frames[total_frames // 2] = "middle"

    for folder in (RESULTS_CSV.parent, SCREENSHOT_DIR, LABELING_DIR,
                   ANNOTATED_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    print(f"Video: {config['input_video']} ({fps:.2f} fps, "
          f"{total_frames} frames)")
    print(f"Spaces: {len(spaces)} | Threshold: {threshold}")
    print("Processing every frame (this keeps smoothing identical to "
          "main.py)...")

    smoothed_scores: Dict[int, float] = {}
    space_states: Dict[int, bool] = {}
    rows = []
    frame_number = 0
    last_frame = None
    last_results = None

    while True:
        success, frame = capture.read()
        if not success:
            break
        frame_number += 1

        results = classify_spaces(frame, spaces, threshold)

        # Same smoothing and hysteresis logic as main.py.
        for result in results:
            space_id = int(result["id"])
            current_score = float(result["score"])

            if space_id not in smoothed_scores:
                smoothed_scores[space_id] = current_score
            else:
                smoothed_scores[space_id] = (
                    SMOOTHING_FACTOR * current_score
                    + (1.0 - SMOOTHING_FACTOR) * smoothed_scores[space_id]
                )

            smoothed_score = smoothed_scores[space_id]
            result["score"] = smoothed_score

            previous_state = space_states.get(
                space_id, smoothed_score >= threshold)

            if previous_state:
                occupied = smoothed_score >= threshold - HYSTERESIS_MARGIN
            else:
                occupied = smoothed_score >= threshold

            space_states[space_id] = occupied
            result["occupied"] = occupied

        last_frame, last_results = frame, results

        needs_annotation = (frame_number in sample_frames
                            or frame_number in screenshot_frames)
        if not needs_annotation:
            continue

        annotated = annotate(frame, results, frame_number, total_frames)

        if frame_number in sample_frames:
            seconds = sample_frames[frame_number]
            for result in results:
                rows.append({
                    "time_seconds": seconds,
                    "frame_number": frame_number,
                    "space_id": result["id"],
                    "actual_status": "",
                    "predicted_status": status_text(result["occupied"]),
                    "correct": "",
                })
            name = f"t{seconds:02d}s_frame{frame_number}.png"
            save_labeling_image(frame, spaces, LABELING_DIR / name)
            cv2.imwrite(str(ANNOTATED_DIR / name), annotated)
            print(f"  Sampled {seconds}s (frame {frame_number})")

        if frame_number in screenshot_frames:
            name = screenshot_frames[frame_number]
            cv2.imwrite(str(SCREENSHOT_DIR / f"{name}.png"), annotated)

    capture.release()

    if last_frame is not None:
        cv2.imwrite(str(SCREENSHOT_DIR / "end.png"),
                    annotate(last_frame, last_results, frame_number,
                             total_frames))

    if RESULTS_CSV.exists() and any(
            row.get("actual_status") for row in csv.DictReader(
                RESULTS_CSV.open(encoding="utf-8"))):
        backup = RESULTS_CSV.with_suffix(".backup.csv")
        RESULTS_CSV.replace(backup)
        print(f"Existing labeled CSV moved to {backup} so it is not lost.")

    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Wrote {len(rows)} rows to {RESULTS_CSV}")
    print(f"Labeling images: {LABELING_DIR}/")
    print(f"Screenshots: {SCREENSHOT_DIR}/")
    print("Next: open each labeling image, fill in actual_status, then run "
          "python src/evaluate.py")


if __name__ == "__main__":
    main()
