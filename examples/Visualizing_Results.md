---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.0
  kernelspec:
    display_name: Python [conda env:.conda-raains]
    language: python
    name: conda-env-.conda-raains-py
---

# Visualizing Results

```python
from detection_utils.demo.pytorch import ShapeDetectionModel
from detection_utils.demo.plot import draw_detections, plot_img
from pathlib import Path
import matplotlib as plt

%matplotlib notebook


```

```python
loaded = ShapeDetectionModel.load_from_checkpoint(
    "./lightning_logs/version_28/checkpoints/epoch=187.ckpt", data_experiment_path="./data"
)

loaded.setup("fit")
```

```python
boxes, labels, scores = zip(*loaded.get_detections(loaded.val_images[:], nms_threshold=0.1))
```

```python
img_id = -10

fig, ax = plot_img(loaded.val_images[img_id])
draw_detections(ax, boxes=boxes[img_id], labels=labels[img_id])
```