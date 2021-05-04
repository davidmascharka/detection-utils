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

# Visualizing Confusion Matrix

```python
from pathlib import Path
import matplotlib as plt
import pickle

from detection_utils.demo.plot import plot_confusion_matrix
%matplotlib inline

from ipywidgets import interact, widgets
from IPython.display import display
import matplotlib.pyplot as plt
import warnings


with open(r"./lightning_logs/version_53/confusion_matrices.pkl", "rb") as f:
    conf_matrices = pickle.load(f)
```

```python
kwargs = dict(figsize=(8, 8), font_size=15)


def f(epoch):
    ax.cla()
    plot_confusion_matrix(conf_matrices[epoch - 1], ax=ax, **kwargs)
    fig.canvas.draw()
    display(fig)


fig, ax = plot_confusion_matrix(conf_matrices[0], **kwargs)
```

```python
interact(f, epoch=widgets.IntSlider(min=1, max=len(conf_matrices), step=1, value=0));
```
