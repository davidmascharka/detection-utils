# utilities for visualizing toy problem

from typing import Dict, Optional, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as tr
from matplotlib.patches import Rectangle

__all__ = ["plot_img", "draw_detections", "plot_confusion_matrix"]


def asarray(x: Union[np.ndarray, tr.Tensor]) -> np.ndarray:
    if isinstance(x, tr.Tensor):
        return x.detach().cpu().numpy()
    else:
        return np.asarray(x)


label_lookup: Dict[int, str] = {
    0: "background",
    1: "rectangle",
    2: "triangle",
    3: "circle",
}


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
    else:
        # filter non-background for efficient slider
        not_background = labels.squeeze() != 0
        boxes = boxes[not_background]
        labels = labels[not_background]

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


def plot_confusion_matrix(
    matrix: np.ndarray, font_size: Optional[int] = None, ax=None, **plt_kwargs
) -> Tuple[Optional[plt.Figure], plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(**plt_kwargs)
    else:
        fig = None

    ax.set_title("confusion matrix")
    labels = [label_lookup[i] for i in sorted(label_lookup)]

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.xaxis.tick_top()

    if font_size:
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(font_size)

    im = ax.imshow(matrix, vmin=0.0, vmax=1.0)
    if fig is not None:
        fig.colorbar(im)
    return fig, ax
