from typing import Optional
import numpy as np
import torch as tr

from ..boxes import non_max_suppression

DEFAULT_BOX_STEP = 16
DEFAULT_BOX_SIZE = 32


def make_anchor_boxes(
    *,
    image_height: int,
    image_width: int,
    box_size=DEFAULT_BOX_SIZE,
    box_stride=DEFAULT_BOX_STEP
) -> np.ndarray:
    """

    Parameters
    ----------
    image_height : int
    image_width : int
    box_size : int
    box_stride : int

    Returns
    -------
    anchor_boxes : numpy.ndarray, shape-(K, 4)
        K strided anchor boxes, specified in row-major order. Each anchore
        box is specified as (x-low, y-low, x-high, y-high)
    """
    anchor_boxes = []
    for y in range(0, image_height, box_stride):
        for x in range(0, image_width, box_stride):
            anchor_boxes.append(
                np.array([-box_size // 2, -box_size // 2, box_size // 2, box_size // 2])
                + np.array([x, y, x, y])
            )
    return np.vstack(anchor_boxes)


def compute_detections(
    classifications: tr.Tensor,
    regressions: tr.Tensor,
    feature_map_width: int,
    anchor_box_step: int = DEFAULT_BOX_STEP,
    anchor_box_size: int = DEFAULT_BOX_SIZE,
    score_threshold: Optional[float] = None,
    nms_threshold: float = 0.3,
):
    """Compute a set of boxes, class predictions, and foreground scores from
    detection model outputs.

    Weak and redundant detections are filtered via score thresholding and NMS,
    respectively.

    Parameters
    ----------
    classifications : torch.Tensor, shape=(N, R*C, # classes)
        A set of class predictions at each spatial location.

    regressions : torch.Tensor, shape=(N, R*C, 4)
        A set of predicted box offsets, in (x, y, w, h) at each spatial location.

    feature_map_width : int
        The number of pixels in the feature map, along the x direction.

    anchor_box_step : int
        The number of pixels (in image space) between each anchor box.

    anchor_box_size : int
        The side length of the anchor box.

    score_threshold: Optional[float]
        If specified, detections with foreground scores below this
        threshold are ignored

    nms_threshold: float, optional (default=0.3)
        The IoU threshold to use for NMS, above which one of two box will be suppressed.

    Returns
    -------
    Tuple[numpy.ndarray shape=(R*C, 4), numpy.ndarray shape=(R*C, 1), numpy.ndarray shape=(R*C,)]
        The (boxes, class predictions, foreground scores) at each spatial location.
    """
    box_predictions = np.empty((len(regressions), 4), dtype=np.float32)
    scores = tr.softmax(classifications, dim=-1).detach().cpu().numpy()
    scores = 1 - scores[:, 0]  # foreground score

    class_predictions = (
        classifications.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    )
    regressions = regressions.detach().cpu().numpy()

    y, x = np.divmod(
        np.arange(len(classifications)), feature_map_width, dtype=np.float32
    )

    # transform (R*C, 4) to (4, R*C) for assignment
    (x_reg, y_reg, w_reg, h_reg,) = regressions.T
    x = anchor_box_step * x + anchor_box_size * x_reg
    y = anchor_box_step * y + anchor_box_size * y_reg

    half_w = np.clip(np.exp(w_reg), 0, 10 ** 6) * anchor_box_size / 2
    half_h = np.clip(np.exp(h_reg), 0, 10 ** 6) * anchor_box_size / 2

    box_predictions[:, 0] = x - half_w  # x1
    box_predictions[:, 1] = y - half_h  # y1
    box_predictions[:, 2] = x + half_w  # x2
    box_predictions[:, 3] = y + half_h  # y2

    if score_threshold is not None:
        keep = scores > score_threshold
        box_predictions = box_predictions[keep]
        class_predictions = class_predictions[keep]
        scores = scores[keep]

    keep_idxs = non_max_suppression(box_predictions, scores, threshold=nms_threshold)
    box_predictions = box_predictions[keep_idxs]
    class_predictions = class_predictions[keep_idxs]

    return box_predictions, class_predictions, scores
