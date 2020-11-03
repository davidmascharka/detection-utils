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

# Visualizing Example Detections

```python
from pathlib import Path
import matplotlib as plt
import pickle

from detection_utils.demo.plot import draw_detections, plot_img

%matplotlib inline
from ipywidgets import interact, widgets
from IPython.display import display
import matplotlib.pyplot as plt


with open(r"./lightning_logs/version_49/val_img_1_info.pkl", "rb") as f:
    boxes_labels_scores = pickle.load(f)
    
from detection_utils.demo.pytorch import ShapeDetectionModel
loaded = ShapeDetectionModel(data_experiment_path="./data")
loaded.setup("fit")

img = loaded.val_images[0].numpy()
img = img.transpose((1, 2, 0))
```

```python
def f(epoch: int):
    ax.cla()
    ax.imshow(img)
    b, l, s = boxes_labels_scores[epoch - 1]
    draw_detections(ax, boxes=b, labels=l, label_fontsize=18)
    fig.canvas.draw()
    display(fig)

fig, ax = plot_img(img, figsize=(8, 8))
```

```python
interact(f, epoch=widgets.IntSlider(min=1, max=len(boxes_labels_scores), step=1, value=0));
```
