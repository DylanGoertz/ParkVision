# ParkVision: Parking-Space Occupancy Detection

ParkVision uses Python, OpenCV, and NumPy to estimate which marked spaces in a fixed-camera parking lot are available. It draws a green outline around an available space and a red outline around an occupied one, and displays both counts. It monitors the selected spaces, not every space visible in the lot.

## Setup

From the repository root on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the reported metrics (no video required)

Run `python src/evaluate.py`. This compares the stored predictions with the manual labels in [the evaluation CSV](results/evaluation_results.csv), prints the confusion counts and metrics, and regenerates the three charts in `results/` using Matplotlib. It does not run the detector or verify the labels against the footage.

## Run the detector on video

1. Place the original clip at `data/primary_parking.mp4` ([input details](data/README.md)). The included `spaces.json` already marks eight spaces for this view.
2. From the repository root, run `python src/main.py`. Press **Space** to pause or resume the preview and **Q** to stop early. For a run without a preview, set `display_video` to `false` in `config.json`.
3. Open `output/parkvision_output.avi` to see the annotated video. Let the program reach the end of the clip for a complete output.

For a different camera view, point `input_video` in `config.json` to the new clip and set `spaces_file` to a new filename so you keep the original polygons. Run `python src/space_selector.py`, click the four corners of each space in order around its boundary, then press **S** to save and **Q** to quit. Mark visible empty **and** occupied spaces.

## How detection and evaluation work

For each polygon, the detector combines edge density with brightness variation. It smooths the score across frames and uses separate cutoffs to switch to occupied (at least 0.14) or back to available (below 0.12). This reduces flicker when a score is close to the threshold.

To inspect the original sampling process, run `python src/collect_predictions.py` with the original clip. It processes frames in sequence, saves numbered crops without predictions in `output/labeling/`, annotated samples in `output/annotated/`, and refreshes the three screenshots in `results/screenshots/`. **It leaves an already-labeled CSV unchanged.** For a new evaluation, work in a separate copy without an existing results CSV, run the collector, fill in `actual_status` from the numbered crops, then run `python src/evaluate.py`.

## Contributions

- **Dylan Goertz:** parking-space selection, occupancy scoring and stabilization, and annotated video output.
- **Shusan:** prediction sampling, manual labeling, evaluation metrics, charts, screenshots, and evaluation write-up.

## Evaluation Results

ParkVision was evaluated on **8** unobstructed parking spaces across **10** sampled frames (6 to 60 seconds of `primary_parking.mp4`), for **80** manually verified classifications. Occupied is treated as the positive class.

| Metric | Value |
|---|---|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1 score | 100% |

- [Confusion matrix](results/confusion_matrix.png)
- [Metrics summary](results/metrics_summary.png)
- [Occupied spaces over time](results/occupancy_over_time.png)

Full details are in [results/EVALUATION.md](results/EVALUATION.md).

**Limitations:** no car arrived in or left the 8 evaluated spaces during the sampled period, so these results do not measure how the detector handles occupancy changes. The detector uses edge density and brightness variation with a fixed threshold, so it is sensitive to shadows, lighting changes, dark vehicles, occlusion, and camera angle, and it depends on manually drawn space boundaries. Results come from a single video and may not generalize to other parking lots or conditions.
