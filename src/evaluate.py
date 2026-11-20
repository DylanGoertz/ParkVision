"""Evaluate ParkVision predictions against manually verified labels."""

import csv
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # Save images without opening a window.
import matplotlib.pyplot as plt  # noqa: E402


RESULTS_CSV = Path("results/evaluation_results.csv")
RESULTS_DIR = Path("results")
VALID_STATUSES = {"Occupied", "Available"}
CSV_COLUMNS = [
    "time_seconds",
    "frame_number",
    "space_id",
    "actual_status",
    "predicted_status",
    "correct",
]

# Chart styling.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"


def safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or 0.0 when the denominator is 0."""
    return numerator / denominator if denominator else 0.0


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        sys.exit(f"Could not find {path}. Run src/collect_predictions.py "
                 "first or create the CSV by hand.")

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = [c for c in CSV_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"{path} is missing columns: {', '.join(missing)}")
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
            if any((value or "").strip() for value in row.values())
        ]

    if not rows:
        sys.exit(f"{path} has no data rows.")

    return rows


def validate_and_fill(rows: List[Dict[str, str]]) -> bool:
    """Check every row. Fill blank 'correct' values. Return True if filled."""
    problems = []
    filled = False

    for line, row in enumerate(rows, start=2):  # Line 1 is the header.
        actual = row["actual_status"]
        predicted = row["predicted_status"]

        for column, value in (("actual_status", actual),
                              ("predicted_status", predicted)):
            if value not in VALID_STATUSES:
                problems.append(
                    f"line {line}: {column} is '{value}' "
                    "(must be Occupied or Available)")

        if actual not in VALID_STATUSES or predicted not in VALID_STATUSES:
            continue

        expected = "1" if actual == predicted else "0"

        if row["correct"] == "":
            row["correct"] = expected
            filled = True
        elif row["correct"] != expected:
            problems.append(
                f"line {line}: correct is {row['correct']} but actual="
                f"{actual} and predicted={predicted} (should be {expected})")

    if problems:
        print("Please fix these rows in the CSV and run again:")
        for problem in problems:
            print("  " + problem)
        sys.exit(1)

    return filled


def write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in CSV_COLUMNS})


def calculate_metrics(rows: List[Dict[str, str]]) -> Dict[str, float]:
    tp = tn = fp = fn = 0

    for row in rows:
        actual_occupied = row["actual_status"] == "Occupied"
        predicted_occupied = row["predicted_status"] == "Occupied"

        if actual_occupied and predicted_occupied:
            tp += 1
        elif not actual_occupied and not predicted_occupied:
            tn += 1
        elif not actual_occupied and predicted_occupied:
            fp += 1
        else:
            fn += 1

    total = len(rows)
    correct = tp + tn
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    return {
        "total": total,
        "correct": correct,
        "accuracy": safe_divide(correct, total),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(2 * precision * recall, precision + recall),
    }


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.title.set_color(TEXT_PRIMARY)


def plot_confusion_matrix(metrics: Dict[str, float], path: Path) -> None:
    # Rows = actual, columns = predicted. Occupied first (positive class).
    cells = [
        [("TP", metrics["tp"], "Occupied, predicted Occupied"),
         ("FN", metrics["fn"], "Occupied, predicted Available")],
        [("FP", metrics["fp"], "Available, predicted Occupied"),
         ("TN", metrics["tn"], "Available, predicted Available")],
    ]
    values = [[cell[1] for cell in row] for row in cells]
    largest = max(max(row) for row in values) or 1

    fig, ax = plt.subplots(figsize=(6.4, 5.6), facecolor=SURFACE)
    ax.imshow(values, cmap="Blues", vmin=0, vmax=largest * 1.15)

    for r, row in enumerate(cells):
        for c, (name, count, meaning) in enumerate(row):
            dark_cell = count > largest * 0.55
            color = "white" if dark_cell else TEXT_PRIMARY
            ax.text(c, r - 0.12, f"{name}: {count}", ha="center",
                    va="center", fontsize=18, fontweight="bold", color=color)
            ax.text(c, r + 0.18, meaning, ha="center", va="center",
                    fontsize=9, color=color)

    ax.set_xticks([0, 1], labels=["Occupied", "Available"])
    ax.set_yticks([0, 1], labels=["Occupied", "Available"])
    ax.set_xlabel("Predicted status", fontsize=11)
    ax.set_ylabel("Actual status", fontsize=11)
    ax.set_title(f"ParkVision Confusion Matrix "
                 f"({metrics['total']} classifications)",
                 fontsize=13, pad=12)
    style_axes(ax)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([0.5], minor=True)
    ax.set_yticks([0.5], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="minor", length=0)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_metrics_summary(metrics: Dict[str, float], path: Path) -> None:
    names = ["Accuracy", "Precision", "Recall", "F1 score"]
    values = [metrics["accuracy"], metrics["precision"],
              metrics["recall"], metrics["f1"]]

    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=SURFACE)
    bars = ax.bar(names, values, color=BLUE, width=0.55, zorder=3)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02,
                f"{value:.1%}", ha="center", va="bottom", fontsize=11,
                color=TEXT_PRIMARY)

    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0],
                  labels=["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Score")
    ax.set_title("ParkVision Performance Metrics (Occupied = positive class)",
                 fontsize=13, pad=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_occupancy_over_time(rows: List[Dict[str, str]], path: Path) -> None:
    actual_counts: "OrderedDict[float, int]" = OrderedDict()
    predicted_counts: "OrderedDict[float, int]" = OrderedDict()
    space_ids = set()

    for row in sorted(rows, key=lambda r: float(r["time_seconds"])):
        time = float(row["time_seconds"])
        space_ids.add(row["space_id"])
        actual_counts.setdefault(time, 0)
        predicted_counts.setdefault(time, 0)
        actual_counts[time] += row["actual_status"] == "Occupied"
        predicted_counts[time] += row["predicted_status"] == "Occupied"

    times = list(actual_counts)

    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    ax.plot(times, list(actual_counts.values()), color=BLUE, linewidth=2,
            marker="o", markersize=8, label="Actual occupied (manual)",
            zorder=3)
    ax.plot(times, list(predicted_counts.values()), color=ORANGE,
            linewidth=2, linestyle="--", marker="s", markersize=8,
            label="Predicted occupied (ParkVision)", zorder=4)

    ax.set_xticks(times, labels=[f"{t:g}" for t in times])
    ax.set_ylim(-0.3, len(space_ids) + 0.5)
    ax.set_yticks(range(0, len(space_ids) + 1))
    ax.set_xlabel("Video time (seconds)")
    ax.set_ylabel("Occupied spaces")
    ax.set_title(f"Occupied Spaces Over Time: Actual vs Predicted "
                 f"({len(space_ids)} spaces)", fontsize=13, pad=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.legend(frameon=False, labelcolor=TEXT_SECONDARY, loc="lower right")
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def print_report(metrics: Dict[str, float], rows: List[Dict[str, str]]) -> None:
    frames = len({row["time_seconds"] for row in rows})
    spaces = len({row["space_id"] for row in rows})

    print()
    print("ParkVision Evaluation")
    print("=" * 40)
    print(f"Sampled frames:            {frames}")
    print(f"Spaces per frame:          {spaces}")
    print(f"Total classifications:     {metrics['total']}")
    print(f"Correct classifications:   {metrics['correct']}")
    print(f"Accuracy:                  {metrics['accuracy']:.2%}")
    print()
    print(f"True positives  (TP):      {metrics['tp']}")
    print(f"True negatives  (TN):      {metrics['tn']}")
    print(f"False positives (FP):      {metrics['fp']}"
          "   (available, predicted occupied)")
    print(f"False negatives (FN):      {metrics['fn']}"
          "   (occupied, predicted available)")
    print()
    print(f"Precision:                 {metrics['precision']:.2%}")
    print(f"Recall:                    {metrics['recall']:.2%}")
    print(f"F1 score:                  {metrics['f1']:.2%}")

    errors = [row for row in rows if row["correct"] == "0"]
    if errors:
        print()
        print("Misclassified rows (use these for 'Observed Errors'):")
        for row in errors:
            print(f"  {row['time_seconds']}s, space {row['space_id']}: "
                  f"actual {row['actual_status']}, "
                  f"predicted {row['predicted_status']}")


def main() -> None:
    rows = load_rows(RESULTS_CSV)

    if validate_and_fill(rows):
        write_rows(RESULTS_CSV, rows)
        print(f"Filled in the 'correct' column in {RESULTS_CSV}")

    metrics = calculate_metrics(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(metrics, RESULTS_DIR / "confusion_matrix.png")
    plot_metrics_summary(metrics, RESULTS_DIR / "metrics_summary.png")
    plot_occupancy_over_time(rows, RESULTS_DIR / "occupancy_over_time.png")

    print_report(metrics, rows)
    print()
    print("Saved charts:")
    for name in ("confusion_matrix.png", "metrics_summary.png",
                 "occupancy_over_time.png"):
        print(f"  {RESULTS_DIR / name}")


if __name__ == "__main__":
    main()
