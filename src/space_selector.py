"""Interactive parking-space polygon selector for ParkVision."""

import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.json"

WINDOW_NAME = "ParkVision Space Selector"
REFERENCE_TIME_SECONDS = 5.0

spaces: list[dict[str, Any]] = []
current_points: list[list[int]] = []
reference_frame: Optional[np.ndarray] = None
spaces_path: Path


def load_config() -> dict[str, Any]:
    """Load the ParkVision configuration file."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_reference_frame(video_path: Path) -> np.ndarray:
    """Load a frame from the configured reference video."""
    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    video.set(cv2.CAP_PROP_POS_MSEC, REFERENCE_TIME_SECONDS * 1000)
    success, frame = video.read()
    video.release()

    if not success or frame is None:
        raise RuntimeError("Could not read a reference frame from the video.")

    return frame


def load_existing_spaces(file_path: Path) -> list[dict[str, Any]]:
    """Load previously selected parking spaces if the file exists."""
    if not file_path.exists():
        return []

    with file_path.open("r", encoding="utf-8") as file:
        loaded_spaces = json.load(file)

    print(f"Loaded {len(loaded_spaces)} existing parking spaces.")
    return loaded_spaces


def save_spaces(file_path: Path) -> None:
    """Save parking-space polygons to a JSON file."""
    if current_points:
        print("Finish or undo the incomplete polygon before saving.")
        return

    if not spaces:
        print("No parking spaces have been selected.")
        return

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(spaces, file, indent=2)

    print(f"Saved {len(spaces)} parking spaces to {file_path}")


def renumber_spaces() -> None:
    """Assign sequential IDs after removing a parking space."""
    for index, space in enumerate(spaces, start=1):
        space["id"] = index


def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
    """Add polygon points when the user left-clicks the image."""
    del flags, param

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    current_points.append([x, y])
    print(f"Selected point: ({x}, {y})")

    if len(current_points) == 4:
        space = {
            "id": len(spaces) + 1,
            "points": current_points.copy(),
        }
        spaces.append(space)
        current_points.clear()
        print(f"Completed parking space {space['id']}")


def draw_interface(frame: np.ndarray) -> np.ndarray:
    """Draw existing polygons, current points, and instructions."""
    display = frame.copy()

    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (display.shape[1], 75), (0, 0, 0), -1)
    display = cv2.addWeighted(overlay, 0.65, display, 0.35, 0)

    instructions = (
        "Click 4 corners | S: Save | U: Undo | R: Reset | Q: Quit"
    )
    cv2.putText(
        display,
        instructions,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    status = (
        f"Completed spaces: {len(spaces)} | "
        f"Current polygon points: {len(current_points)}/4"
    )
    cv2.putText(
        display,
        status,
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for space in spaces:
        polygon = np.array(space["points"], dtype=np.int32)

        cv2.polylines(
            display,
            [polygon],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

        center = polygon.mean(axis=0).astype(int)

        cv2.putText(
            display,
            str(space["id"]),
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for point in current_points:
        cv2.circle(display, tuple(point), 5, (0, 255, 255), -1)

    if len(current_points) > 1:
        partial_polygon = np.array(current_points, dtype=np.int32)
        cv2.polylines(
            display,
            [partial_polygon],
            isClosed=False,
            color=(0, 255, 255),
            thickness=2,
        )

    return display


def undo_last_action() -> None:
    """Undo the most recently selected point or completed polygon."""
    if current_points:
        removed_point = current_points.pop()
        print(f"Removed point: {removed_point}")
    elif spaces:
        removed_space = spaces.pop()
        renumber_spaces()
        print(f"Removed parking space {removed_space['id']}")
    else:
        print("Nothing to undo.")


def run_selector() -> None:
    """Run the interactive parking-space selection window."""
    global spaces, reference_frame, spaces_path

    config = load_config()

    video_path = PROJECT_ROOT / config["input_video"]
    spaces_path = PROJECT_ROOT / config["spaces_file"]

    reference_frame = load_reference_frame(video_path)
    spaces = load_existing_spaces(spaces_path)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("\nParkVision parking-space selector")
    print("Left click: Select one polygon corner")
    print("S: Save polygons")
    print("U: Undo last point or polygon")
    print("R: Reset all polygons")
    print("Q: Quit\n")

    while True:
        display = draw_interface(reference_frame)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(20) & 0xFF

        if key == ord("s"):
            save_spaces(spaces_path)

        elif key == ord("u"):
            undo_last_action()

        elif key == ord("r"):
            spaces.clear()
            current_points.clear()
            print("All parking spaces were cleared.")

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_selector()