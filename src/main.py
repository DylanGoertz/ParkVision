"""Main video-processing program for ParkVision."""

import json
import time
from pathlib import Path
from typing import Any, Dict

import cv2

from detector import (
    classify_spaces,
    draw_results,
    load_parking_spaces,
)


CONFIG_FILE = "config.json"


def load_config(config_file: str) -> Dict[str, Any]:
    """Load project settings from a JSON configuration file."""
    path = Path(config_file)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file was not found: {config_file}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def run_parkvision() -> None:
    """Process a parking-lot video and display occupancy results."""
    config = load_config(CONFIG_FILE)

    input_video = config["input_video"]
    spaces_file = config["spaces_file"]
    output_video = config["output_video"]
    threshold = float(config.get("occupied_threshold", 0.14))
    display_video = bool(config.get("display_video", True))

    spaces = load_parking_spaces(spaces_file)

    print(f"Loaded {len(spaces)} parking spaces.")
    print(f"Occupancy threshold: {threshold}")
    print(f"Opening video: {input_video}")

    capture = cv2.VideoCapture(input_video)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("The video has invalid dimensions.")

    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    codec = cv2.VideoWriter_fourcc(*"MJPG")

    writer = cv2.VideoWriter(
        str(output_path),
        codec,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not create output video: {output_video}"
        )

    # Store previous measurements and classifications.
    smoothed_scores: Dict[int, float] = {}
    space_states: Dict[int, bool] = {}

    # Temporal smoothing reduces rapid score changes.
    smoothing_factor = 0.20

    # Hysteresis prevents scores near the threshold from flickering.
    hysteresis_margin = 0.02

    frame_number = 0
    start_time = time.time()
    paused = False
    processed_frame = None

    print("Processing video...")
    print("Press Q in the video window to stop.")
    print("Press the space bar to pause or resume.")

    while True:
        if not paused:
            success, frame = capture.read()

            if not success:
                break

            frame_number += 1

            results = classify_spaces(
                frame,
                spaces,
                threshold
            )

            for result in results:
                space_id = int(result["id"])
                current_score = float(result["score"])

                if space_id not in smoothed_scores:
                    smoothed_scores[space_id] = current_score
                else:
                    previous_score = smoothed_scores[space_id]

                    smoothed_scores[space_id] = (
                        smoothing_factor * current_score
                        + (1.0 - smoothing_factor) * previous_score
                    )

                smoothed_score = smoothed_scores[space_id]
                result["score"] = smoothed_score

                previous_state = space_states.get(
                    space_id,
                    smoothed_score >= threshold
                )

                if previous_state:
                    # Once occupied, the score must fall clearly below
                    # the threshold before becoming available.
                    occupied = smoothed_score >= (
                        threshold - hysteresis_margin
                    )
                else:
                    # An available space becomes occupied when its
                    # score reaches the normal threshold.
                    occupied = smoothed_score >= threshold

                space_states[space_id] = occupied
                result["occupied"] = occupied

            processed_frame = draw_results(frame, results)

            progress_text = (
                f"Frame: {frame_number}/{total_frames}"
            )

            cv2.putText(
                processed_frame,
                progress_text,
                (25, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            writer.write(processed_frame)

        if display_video and processed_frame is not None:
            preview_width = 1152
            preview_height = int(
                height * (preview_width / width)
            )

            preview = cv2.resize(
                processed_frame,
                (preview_width, preview_height)
            )

            cv2.imshow(
                "ParkVision Occupancy Detection",
                preview
            )

            key = cv2.waitKey(
                1 if not paused else 30
            ) & 0xFF

            if key == ord("q"):
                print("Processing stopped by user.")
                break

            if key == 32:
                paused = not paused

        elif not display_video:
            paused = False

    elapsed_time = time.time() - start_time

    capture.release()
    writer.release()
    cv2.destroyAllWindows()

    print()
    print("ParkVision processing complete.")
    print(f"Frames processed: {frame_number}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Output saved to: {output_video}")


if __name__ == "__main__":
    run_parkvision()