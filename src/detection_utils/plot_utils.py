# utilities for visualizing toy problem

from typing import Dict, Optional, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as tr
from matplotlib.patches import Rectangle

__all__ = ["plot_img", "draw_detections"]


def asarray(x: Union[np.ndarray, tr.Tensor]) -> np.ndarray:
    if isinstance(x, tr.Tensor):
        return x.detach().cpu().numpy()
    else:
        return np.asarray(x)


label_lookup: Dict[int, str] = {1: "rectangle", 2: "triangle", 3: "circle"}


def plot_img(
    img: Union[np.ndarray, tr.Tensor], ax: Optional[plt.Axes] = None, **plt_kwargs
) -> Tuple[Optional[plt.Figure], plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(**plt_kwargs)
    else:
        fig = None

    img = asarray(img)
    assert img.ndim == 3

    if img.shape[0] == 3:
        img = img.transpose((1, 2, 0))

    ax.imshow(img)

    return fig, ax


def draw_detections(
    ax, *, boxes: np.ndarray, labels: Optional[np.ndarray] = None, color="r", lw=2
):

    assert boxes.ndim == 2 and boxes.shape[1] == 4

    if labels is None:
        labels = [None] * len(boxes)

    assert len(boxes) == len(labels)

    for class_pred, box_pred in zip(labels, boxes):
        if class_pred is not None and class_pred == 0:
            continue

        x1, y1, x2, y2 = box_pred
        ax.add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, color=color, fill=None, lw=lw)
        )
        if class_pred is not None:
            label = label_lookup[int(class_pred)]
            ax.annotate(label, (x1, y1), color="r", fontsize=24)
