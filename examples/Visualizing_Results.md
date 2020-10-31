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
from detection_utils.demo.boxes import compute_batch_stats

from pathlib import Path
import matplotlib as plt

%matplotlib notebook


```

```python
loaded = ShapeDetectionModel(data_experiment_path="./data")
loaded.setup("fit")
```

```python
loaded = ShapeDetectionModel.load_from_checkpoint(
    "./lightning_logs/version_33/checkpoints/last.ckpt", data_experiment_path="./data"
)

loaded.setup("fit")
```

```python

start = 0
stop = 1
imgs = loaded.val_images[start:stop]

class_predictions, regression_predictions = loaded(imgs)
confusion_matrix, precision, recall = compute_batch_stats(
    class_predictions=class_predictions,
    regression_predictions=regression_predictions,
    boxes=loaded.val_boxes[start:stop],
    labels=loaded.val_labels[start:stop],
    feature_map_width=imgs.shape[2] // 16,  # backbone downsamples by factor 16
    nms_iou_threshold=0.1)
```

```python
precision.mean(), recall.mean()
```

```python
boxes, labels, scores = zip(*loaded.get_detections(loaded.val_images[:10], nms_threshold=0.1))
```

```python
confusion_matrix
```

```python
img_id = 0

fig, ax = plot_img(loaded.val_images[img_id])
draw_detections(ax, boxes=boxes[img_id], labels=labels[img_id])
```

```python
img_id = 0

fig, ax = plot_img(loaded.val_images[img_id])
draw_detections(ax, boxes=boxes[img_id], labels=labels[img_id])
```
