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
    ax: plt.Axes,
    *,
    boxes: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_associations: Dict[int, str] = dict(label_lookup),
    box_color: str = "r",
    font_color: Optional[str] = None,
    box_line_width: int = 2,
    label_fontsize: int = 24
):
    """
    Draws detection boxes/labels on an existing image.

    Parameters
    ----------
    ax : Axes
        The image axis object on which the detections will be drawn.

    boxes : ndarray, shape-(N, 4)
        The detection boxes, each box is formatted as (xlo, ylo, xhi, yhi)
        in pixel space.

    labels : Optional[ndarray], shape-(N,)
        The integer classification label associated with each box.

    label_associations : Dict[int, str]
        int -> label

    box_color : str, optional (default=red)

    font_color : Optional[str]
        If not specified, matches ``box_color``

    box_line_width : int, optional (default=2)

    label_fontsize : int, optional (default=24)
    """
    assert boxes.ndim == 2 and boxes.shape[1] == 4

    if labels is None:
        labels = [None] * len(boxes)
    else:
        # filter non-background for efficient slider
        not_background = labels.squeeze() != 0
        boxes = boxes[not_background]
        labels = labels[not_background]

    assert len(boxes) == len(labels)

    if font_color is None:
        font_color = box_color

    for class_pred, box_pred in zip(labels, boxes):
        if class_pred is not None and class_pred == 0:
            continue

        x1, y1, x2, y2 = box_pred
        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                color=box_color,
                fill=None,
                lw=box_line_width,
            )
        )
        if class_pred is not None:
            label = label_associations[int(class_pred)]
            ax.annotate(label, (x1, y1), color=font_color, fontsize=label_fontsize)


def plot_confusion_matrix(
    matrix: np.ndarray,
    ax: Optional[plt.Axes] = None,
    font_size: Optional[int] = None,
    include_colorbar: bool = True,
    label_associations: Dict[int, str] = dict(label_lookup),
    **plt_kwargs
) -> Tuple[Optional[plt.Figure], plt.Axes]:
    """
    Plots a confusion matrix.

    Parameters
    ----------
    matrix : ndarray, shape-(N_class, N_class)
        The confusion matrix

    ax : Optional[Axes]
        If specified, the axis object on which the confusion matrix
        will be drawn. Otherwise a new figure/axes pair will be created

    font_size : Optional[int]

    include_colorbar: bool, optional (default=True)

    label_associations : Dict[int, str]
        int -> label

    plt_kwargs
        Keyword arguments passed to ``plt.subplots(...)``; used only
        if ``ax`` is not specified.

    Returns
    -------
    Tuple[Optional[plt.Figure], plt.Axes]
        The figure and axis object associated with the plot. Figure is
        ``None`` if the user supplied ``ax``.
    """
    if ax is None:
        fig, ax = plt.subplots(**plt_kwargs)
    else:
        fig = None

    ax.set_title("confusion matrix")
    labels = [label_associations[i] for i in sorted(label_associations)]

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
    if fig is not None and include_colorbar:
        fig.colorbar(im)
    return fig, ax
