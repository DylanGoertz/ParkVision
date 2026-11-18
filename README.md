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
