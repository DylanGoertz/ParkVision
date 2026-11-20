# Input video

The detector expects the original, unannotated fixed-camera clip at `data/primary_parking.mp4`. The clip is not included in Git; request it from the project team to reproduce the detector run. Use the original 1920 × 1080 framing because the coordinates in `spaces.json` are measured in pixels. The annotated AVI is an output, not a substitute for the input.

If you only want to reproduce the metrics and charts, no video is needed: from the repository root, run `python src/evaluate.py` after installing the [project dependencies](../requirements.txt).
