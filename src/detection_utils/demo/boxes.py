from typing import Optional, Tuple

import numpy as np
import torch as tr

from ..boxes import non_max_suppression, DEFAULT_NMS_THRESHOLD
from ..metrics import (
    confusion_matrix,
    precision_and_recall,
)

DEFAULT_BOX_STEP = 16
DEFAULT_BOX_SIZE = 32
DEFAULT_BOX_OFFSET = 6


def make_anchor_boxes(
    *,
    image_height: int,
    image_width: int,
    box_size=DEFAULT_BOX_SIZE,
    box_stride=DEFAULT_BOX_STEP,
    box_offset=DEFAULT_BOX_OFFSET,
) -> np.ndarray:
    """

    Parameters
    ----------
    image_height : int
    image_width : int
    box_size : int, optional
    box_stride : int, optional
    box_offset : float, optional
        Offsets all anchor boxes by a constant offset in order
        to center boxes on receptive field.

    Returns
    -------
    anchor_boxes : numpy.ndarray, shape-(K, 4)
        K strided anchor boxes, specified in row-major order. Each anchor
        box is specified as (x-low, y-low, x-high, y-high)
    """
    anchor_boxes = []
    for y in range(0, image_height, box_stride):
        for x in range(0, image_width, box_stride):
            anchor_boxes.append(
                np.array([-box_size // 2, -box_size // 2, box_size // 2, box_size // 2])
                + np.array([x, y, x, y])
            )
    return np.vstack(anchor_boxes) + box_offset


def compute_detections(
    classifications: tr.Tensor,
    regressions: tr.Tensor,
    feature_map_width: int,
    anchor_box_step: int = DEFAULT_BOX_STEP,
    anchor_box_size: int = DEFAULT_BOX_SIZE,
    anchor_box_offset: int = DEFAULT_BOX_OFFSET,
    score_threshold: Optional[float] = None,
    nms_threshold: float = DEFAULT_NMS_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a set of boxes, class predictions, and foreground scores from
    detection model outputs.

    Weak and redundant detections are filtered via score thresholding and NMS,
    respectively.

    Parameters
    ----------
    classifications : torch.Tensor, shape=(R*C, # classes)
        The class predictions at each spatial location.

    regressions : torch.Tensor, shape=(R*C, 4)
        The predicted box offsets, in (x, y, w, h) at each spatial location.

    feature_map_width : int
        The number of pixels in the feature map, along the x direction.

    anchor_box_step : int, optional
        The number of pixels (in image space) between each anchor box.

    anchor_box_size : int, optional
        The side length of the anchor box.

    anchor_box_offset : float, optional
        Offsets all anchor boxes by a constant offset in order
        to center boxes on receptive field.

    score_threshold: Optional[float]
        If specified, detections with foreground scores below this
        threshold are ignored

    nms_threshold: float, optional
        The IoU threshold to use for NMS, above which one of two box will be suppressed.

    Returns
    -------
    Tuple[numpy.ndarray shape=(N_det, 4), numpy.ndarray shape=(N_det, 1), numpy.ndarray shape=(N_det,)]
        The (boxes, class predictions, foreground scores) for each detection above the specified threshold(s)

    Notes
    -----
    Boxes are reported as (xlo, ylo, xhi, yhi).
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
    x = anchor_box_step * x + anchor_box_offset + anchor_box_size * x_reg
    y = anchor_box_step * y + anchor_box_offset + anchor_box_size * y_reg

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
    scores = scores[keep_idxs]

    return box_predictions, class_predictions, scores


def compute_batch_stats(
    class_predictions: tr.Tensor,
    regression_predictions: tr.Tensor,
    boxes: np.ndarray,
    labels: np.ndarray,
    feature_map_width: int,
    anchor_box_step: int = DEFAULT_BOX_STEP,
    anchor_box_size: int = DEFAULT_BOX_SIZE,
    anchor_box_offset: int = DEFAULT_BOX_OFFSET,
    score_threshold: Optional[float] = None,
    nms_iou_threshold: float = DEFAULT_NMS_THRESHOLD,
) -> Tuple[tr.Tensor, tr.Tensor, tr.Tensor]:
    """Compute the batch statistics (AP and AR) given a batch of predictions and truth.

    Parameters
    ----------
    class_predictions : Tensor, shape=(N, K, C)
        The predicted class scores of each of N images at each of K anchor boxes.

    regression_predictions : Tensor, shape=(N, K, 4)
        The predicted regression values of each of N images at each of K anchor boxes.

    boxes : numpy.ndarray, shape=(N,)
        The truth boxes for each image. Note that each of the N elements is of
        shape (W_i, 4), where W_i is the number of objects in image i.

    labels : numpy.ndarray, shape=(N,)
        The truth labels for each image. Note that each of the N elements is of
        shape (W_i,), where  W_i is the number of objects in image i.

    feature_map_width : int
        The width of the feature map.

    anchor_box_step : int, optional
        The stride across the image at which anchor boxes are placed.

    anchor_box_size : int, optional
        The side length of each anchor box.

    anchor_box_offset : float, optional
        Offsets all anchor boxes by a constant offset in order
        to center boxes on receptive field.

    score_threshold: Optional[float]
        If specified, detections with foreground scores below this
        threshold are ignored

    nms_iou_threshold: float, optional
        The IoU threshold to use for NMS, above which one of two box will be suppressed.


    Returns
    -------
    Tuple[tr.Tensor, tr.Tensor, tr.Tensor]
        - confusion matrix accumulated over images, shape-(N_class, N_class)
        - precision for each image, shape-(N,)
        - recall for each image, shape-(N,)
    """
    precisions, recalls = [], []
    accumed_confusion = np.zeros((4, 4), dtype=np.int32)

    for i in range(len(class_predictions)):
        truth_detections = np.hstack((boxes[i], labels[i][:, None]))

        box_preds, class_preds, scores = compute_detections(
            class_predictions[i],
            regression_predictions[i],
            feature_map_width,
            anchor_box_step=anchor_box_step,
            anchor_box_size=anchor_box_size,
            anchor_box_offset=anchor_box_offset,
            score_threshold=score_threshold,
            nms_threshold=nms_iou_threshold,
        )

        detections = np.hstack((box_preds, class_preds))
        conf = confusion_matrix(
            prediction_detections=detections, truth_detections=truth_detections
        )
        stats = precision_and_recall(conf)
        precisions.append(stats.precision)
        recalls.append(stats.recall)
        accumed_confusion += conf

    return (
        tr.tensor(accumed_confusion),
        tr.tensor(precisions, dtype=tr.float32),
        tr.tensor(recalls, dtype=tr.float32),
    )
