"""Parking-space occupancy detection for ParkVision."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]
Polygon = List[Point]


def load_parking_spaces(spaces_file: str) -> List[Polygon]:
    """Load parking-space polygons from a JSON file."""
    path = Path(spaces_file)

    if not path.exists():
        raise FileNotFoundError(
            f"Parking-space file was not found: {spaces_file}"
        )

    with path.open("r", encoding="utf-8") as file:
        data: Any = json.load(file)

    # Support several possible JSON formats.
    if isinstance(data, dict):
        raw_spaces = data.get("spaces", data.get("polygons", []))
    else:
        raw_spaces = data

    spaces: List[Polygon] = []

    for item in raw_spaces:
        if isinstance(item, dict):
            raw_points = item.get("points", item.get("polygon", []))
        else:
            raw_points = item

        polygon = [
            (int(point[0]), int(point[1]))
            for point in raw_points
        ]

        if len(polygon) >= 3:
            spaces.append(polygon)

    if not spaces:
        raise ValueError(
            f"No valid parking spaces were found in {spaces_file}"
        )

    return spaces


def calculate_occupancy_score(
    frame: np.ndarray,
    polygon: Polygon
) -> float:
    """
    Calculate an occupancy score for one parking space.

    Cars generally create more edges and brightness variation than empty
    pavement. The score combines those two measurements.
    """
    polygon_array = np.array(polygon, dtype=np.int32)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_array], 255)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)
    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

    region_area = cv2.countNonZero(mask)

    if region_area == 0:
        return 0.0

    edge_pixels = cv2.countNonZero(masked_edges)
    edge_ratio = edge_pixels / region_area

    _, standard_deviation = cv2.meanStdDev(
        blurred,
        mask=mask
    )

    brightness_variation = float(standard_deviation[0][0]) / 255.0

    score = edge_ratio + (0.75 * brightness_variation)

    return min(score, 1.0)


def classify_spaces(
    frame: np.ndarray,
    spaces: List[Polygon],
    threshold: float
) -> List[Dict[str, Any]]:
    """Classify every selected space as occupied or available."""
    results: List[Dict[str, Any]] = []

    for index, polygon in enumerate(spaces, start=1):
        score = calculate_occupancy_score(frame, polygon)
        occupied = score >= threshold

        results.append(
            {
                "id": index,
                "polygon": polygon,
                "score": score,
                "occupied": occupied,
            }
        )

    return results


def draw_results(
    frame: np.ndarray,
    results: List[Dict[str, Any]]
) -> np.ndarray:
    """Draw occupancy results and totals onto a video frame."""
    output = frame.copy()
    occupied_count = 0

    for result in results:
        polygon = np.array(result["polygon"], dtype=np.int32)
        occupied = bool(result["occupied"])

        if occupied:
            color = (0, 0, 255)
            status = "Occupied"
            occupied_count += 1
        else:
            color = (0, 255, 0)
            status = "Available"

        cv2.polylines(
            output,
            [polygon],
            isClosed=True,
            color=color,
            thickness=3
        )

        center = np.mean(polygon, axis=0).astype(int)
        label = (
            f'{result["id"]}: {status} '
            f'({result["score"]:.2f})'
        )

        cv2.putText(
            output,
            label,
            (center[0] - 45, center[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA
        )

    total_spaces = len(results)
    available_count = total_spaces - occupied_count

    summary = (
        f"Available: {available_count} | "
        f"Occupied: {occupied_count} | "
        f"Total: {total_spaces}"
    )

    cv2.rectangle(output, (15, 15), (570, 60), (0, 0, 0), -1)

    cv2.putText(
        output,
        summary,
        (25, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return output