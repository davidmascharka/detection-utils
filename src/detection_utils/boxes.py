# DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.
#
# This material is based upon work supported by the Assistant Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8721-05-C-0002 and/or FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the author(s) and
# do not necessarily reflect the views of the Assistant Secretary of Defense for Research and
# Engineering.
#
# © 2019 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
# 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
# defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than
# as specifically authorized by the U.S. Government may violate any copyrights that exist in this
# work.

from typing import Tuple

import numba
import numpy as np
from numpy import ndarray


@numba.njit
def box_overlaps(predicted: ndarray, truth: ndarray, eps: float = 1e-12) -> ndarray:
    """ Return the overlap between two lists of boxes.

    Calculates the intersection over union between a list of predicted boxes and a list of ground-truth boxes.

    Parameters
    ----------
    boxes : numpy.ndarray, shape=(N, 4)
        The predicted boxes, in xyxy format.

    truth : numpy.ndarray, shape=(K, 4)
        The ground-truth boxes, in xyxy format.

    eps : Real, optional (default=1e-12)
        The epsilon value to apply to the intersection over union computation for stability.

    Returns
    -------
    numpy.ndarray, shape=(N, K)
        The overlap between the predicted and ground-truth boxes

    Notes
    -----
    The format referred to, xyxy format, indicates (left, top, right, bottom) in pixel space.

    Examples
    --------
    >>> from detection_utils.boxes import box_overlaps
    >>> import numpy as np
    >>> predicted_boxes = np.array([[0, 0, 10, 10], # left, top, right, bottom (xyxy) format
    ...                             [3, 3,  7,  7]])
    >>> true_boxes = np.array([[2, 3, 6, 7]])
    >>> box_overlaps(predicted_boxes, true_boxes)
    array([[0.16], [0.6])
    """
    N = predicted.shape[0]
    K = truth.shape[0]
    ious = np.zeros((N, K), dtype=np.float32)

    for k in range(K):
        truth_area = (truth[k, 2] - truth[k, 0]) * (truth[k, 3] - truth[k, 1])
        for n in range(N):
            width_overlap = min(predicted[n, 2], truth[k, 2]) - max(predicted[n, 0], truth[k, 0])
            if width_overlap > 0:
                height_overlap = min(predicted[n, 3], truth[k, 3]) - max(predicted[n, 1], truth[k, 1])
                if height_overlap > 0:
                    overlap_area = width_overlap * height_overlap
                    box_area = (predicted[n, 2] - predicted[n, 0]) * (predicted[n, 3] - predicted[n, 1])
                    union = box_area + truth_area - overlap_area
                    ious[n, k] = overlap_area / (union + eps)
    return ious


def generate_targets(
        anchor_boxes: ndarray,
        truth_boxes: ndarray,
        labels: ndarray,
        pos_thresh: float = 0.3,
        neg_thresh: float = 0.2,
        eps: float = 1e-12,
) -> Tuple[ndarray, ndarray]:
    """ Generate classification and regression targets from ground-truth boxes.

    Each regression target is matched to its highest-overlapping ground-truth box. Those targets with less than a
    `pos_thresh` IoU are marked as background. Targets with `neg_thresh` <= IoU < `pos_thresh` are flagged as
    ignore boxes. Boxes are regressed based on their centers and widths/heights.

    Parameters
    ----------
    anchor_boxes : numpy.ndarray, shape=(N, 4)
        Anchor boxes in xyxy format.

    truth_boxes : numpy.ndarray, shape=(K, 4)
        Ground-truth boxes in xyxy format.

    labels : numpy.ndarray, shape=(K,)
        The labels associated with each ground-truth box.

    pos_thresh : Real
        The minimum overlap threshold between a truth and anchor box for that truth box to be 'responsible' for
        detecting the anchor.

    neg_thresh : Real
        The maximum overlap threshold between a truth and anchor box for that anchor box to be called a negative.
        Those anchor boxes with overlap greater than this but less than `pos_thresh` will be marked as ignored.

    eps : Real, optional (default=1e-12)
        The epsilon to use for numerical stability.

    Returns
    -------
    Tuple[numpy.ndarray shape=(N,), numpy.ndarray shape=(N, 4)]
        The classification and bounding box regression targets for each anchor box. Regressions are of format
        (x-center, y-center, width, height). Classification targets of 0 indicate background, while targets of -1
        indicate that this prediction should be ignored as a difficult case.

    Examples
    --------
    >>> from detection_utils.boxes import generate_targets
    >>> import numpy as np
    >>> anchors = np.array([[-0.5,   -0.5,   0.5,   0.5],
    ...                     [ 0.0,   -0.5,   1.0,   1.5],
    ...                     [ 0.5,    0.0,   1.5,   1.0]])
    >>> targets = np.array([[0, 0, 1, 1]])
    >>> labels = np.array([1])
    >>> generate_targets(anchors, targets, labels)
    (array([0, 1]),
     array([[ 5.000000e-01,  5.000000e-01, -1.110223e-16, -1.110223e-16],
        [ 0.000000e+00,  0.000000e+00, -1.110223e-16, -6.931472e-01],
        [-5.000000e-01,  0.000000e+00, -1.110223e-16, -1.110223e-16]]))
    """
    if truth_boxes.size == 0:
        targets_reg = np.zeros_like(anchor_boxes, dtype=np.float32)
        targets_cls = np.zeros(anchor_boxes.shape[0], dtype=np.int64)
        return targets_cls, targets_reg

    ious = box_overlaps(anchor_boxes, truth_boxes)  # NxK
    max_ious = ious.max(axis=1)                     # N IoUs
    max_idxs = ious.argmax(axis=1)                  # N indices

    target_boxes = truth_boxes[max_idxs]

    target_centers = (target_boxes[:, :2] + target_boxes[:, 2:]) / 2
    anchor_centers = (anchor_boxes[:, :2] + anchor_boxes[:, 2:]) / 2
    target_wh = target_boxes[:, 2:] - target_boxes[:, :2]
    anchor_wh = anchor_boxes[:, 2:] - anchor_boxes[:, :2]

    xy = (target_centers - anchor_centers) / anchor_wh
    wh = np.log(target_wh / (anchor_wh + eps) + eps)

    targets_reg = np.hstack([xy, wh])
    targets_cls = labels[max_idxs]
    targets_cls[max_ious < pos_thresh] = -1
    targets_cls[max_ious < neg_thresh] = 0

    targets_cls = targets_cls.reshape(-1).astype(np.int32)
    targets_reg = targets_reg.reshape(-1, 4).astype(np.float32)

    return targets_cls, targets_reg


def non_max_suppression(
        boxes: ndarray,
        scores: ndarray,
        threshold: float = 0.7,
        clip_value: float = 1e6,
        eps: float = 1e-12,
) -> ndarray:
    """ Return the indices of non-suppressed detections after applying non-maximum suppression with the given threshold.

    Parameters
    ----------
    boxes : np.ndarray[Real], shape=(N, 4)
        The detection boxes to which to apply NMS, in (left, top, right, bottom) format.

    scores : np.ndarray[Real], shape=(N,)
        The detection score for each box.

    threshold : float ∈ [0, 1], optional (default=0.7)
        The IoU threshold to use for NMS, above which one of two box will be suppressed.

    clip_value : Real, optional (default=1e6)
        The maximum width or height overlap, for numerical stability.

    eps : Real, optional (default=1e-12)
        The epsilon value to use in IoU calculation, for numerical stability.

    Returns
    -------
    np.ndarray[int], shape=(k,)
        The (sorted) subset of detections to keep, where k is the number of non-suppressed inputs and k <= N.

    Examples
    --------
    >>> from detection_utils.boxes import non_max_suppression
    >>> import numpy as np
    >>> boxes = np.array([[  0,   0,   1,   1],
    ...                   [0.5, 0.5, 0.9, 0.9]])
    >>> scores = np.array([0, 1])
    >>> non_max_suppression(boxes, scores)
    array([0, 1])

    # our default threshold is 0.7 and our IoU between these is 0.16; let's try a lower threshold
    >>> non_max_suppression(boxes, scores, threshold=0.15)
    array([1])
    """
    x1s, y1s, x2s, y2s = boxes.T

    areas = np.clip(x2s - x1s, 0, clip_value) * np.clip(y2s - y1s, 0, clip_value)
    order = scores.argsort()[::-1]  # highest to lowest score

    keep = []  # which detections are we going to keep?
    while order.size > 0:
        i = order[0]
        keep.append(i)
        all_others = order[1:]  # everything except the current box

        width_overlaps = np.maximum(0, np.minimum(x2s[i], x2s[all_others]) - np.maximum(x1s[i], x1s[all_others]))
        width_overlaps = np.clip(width_overlaps, 0, clip_value)

        height_overlaps = np.maximum(0, np.minimum(y2s[i], y2s[all_others]) - np.maximum(y1s[i], y1s[all_others]))
        height_overlaps = np.clip(height_overlaps, 0, clip_value)

        intersections = width_overlaps * height_overlaps
        ious = intersections / (areas[i] + areas[all_others] - intersections + eps)

        # +1 to counteract the offset all_others = order[1:]
        order = order[np.where(ious <= threshold)[0] + 1]

    return np.array(sorted(keep), dtype=np.int32)


def xywh_to_xyxy(boxes: ndarray) -> ndarray:
    """ Convert boxes from xywh to xyxy.

    Parameters
    ----------
    boxes : numpy.ndarray, shape=(N, 4)
        Boxes, in xywh format.

    Returns
    -------
    numpy.ndarray, shape=(N, 4)
        Boxes in xyxy format

    Examples
    --------
    >>> from detection_utils.boxes import xywh_to_xyxy
    >>> import numpy as np
    >>> boxes = np.array([[0, 0, 2, 3],  # left, top, width, height
    ...                   [5, 6, 7, 8]])
    >>> xywh_to_xyxy(boxes)
    array([[0, 0,  2, 3],
           [5, 6, 12, 14]])
    """
    temp = np.empty_like(boxes)
    if temp.size > 0:
        temp[:, :2] = boxes[:, :2]
        temp[:, 2:] = boxes[:, :2] + boxes[:, 2:]
    return temp


def xyxy_to_xywh(boxes: ndarray) -> ndarray:
    """ Convert boxes from xyxy to xywh.

    Parameters
    ----------
    boxes : numpy.ndarray, shape=(N, 4)
        Boxes, in xyxy format.

    Returns
    -------
    numpy.ndarray, shape=(N, 4)
        Boxes in xywh format

    Examples
    --------
    >>> from detection_utils.boxes import xyxy_to_xywh
    >>> import numpy as np
    >>> boxes = np.array([[0, 0,  2,  3],  # left, top, right, bottom
    ...                   [5, 6, 12, 14]])
    >>> xyxy_to_xywh(boxes)
    array([[0, 0,  2, 3],
           [5, 6, 7, 8]])
    """
    temp = np.empty_like(boxes)
    if temp.size > 0:
        temp[:, :2] = boxes[:, :2]
        temp[:, 2:] = boxes[:, 2:] - boxes[:, :2]
    return temp
