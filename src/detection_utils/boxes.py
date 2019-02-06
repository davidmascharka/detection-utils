import numba
import numpy as np


@numba.njit
def box_overlaps(predicted, truth, eps=1e-12):
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


def compute_precision(prediction_detections, truth_detections, threshold=0.5):
    """ Compute the average precision of predictions given targets.

    Precision is defined as the number of true positive predictions divided by the number of total positive predictions.

    Parameters
    ----------
    prediction_detections : numpy.ndarray, shape=(N, 5)
        The predicted objects, in (left, top, right, bottom, class) format.

    truth_detections : numpy.ndarray, shape=(K, 5)
        The ground-truth objects in (left, top, right, bottom, class) format.

    threshold : Real, optional (default=0.5)
        The IoU threshold at which to compute precision.

    Returns
    -------
    float
        The average precision (AP) for the given detections and truth.

    Notes
    -----
    This function operates such that when there are zero predictions, precision is 1.
    """
    # we can short-circuit with a couple special cases to improve our efficiency
    predictions = prediction_detections[:, -1]
    truths = truth_detections[:, -1]

    if predictions.sum() == 0:
        return 1  # (0 TP) / (0 TP + 0 FP) is counted as 100% correct

    if truths.sum() == 0:
        return 0  # we've already handled the case where we found 0/0 relevant objects above

    ious = box_overlaps(prediction_detections[:, :4], truth_detections[:, :4])
    max_ious = ious.max(axis=1)
    max_idxs = ious.argmax(axis=1)

    target_labels = truths[max_idxs]
    true_positives = (predictions == target_labels)[np.logical_and(max_ious >= threshold, target_labels > 0)].sum()
    predicted_positives = (predictions > 0).sum()

    return true_positives / predicted_positives


def compute_recall(prediction_detections, truth_detections, threshold=0.5):
    """ Compute the average recall of predictions given targets.

    Recall is defined as the number true positive predictions divided by the number of ground-truth targets.

    Parameters
    ----------
    prediction_detections : numpy.ndarray, shape=(N, 5)
        The predicted objects, in (left, top, right, bottom, class) format.

    truth_detections : numpy.ndarray, shape=(K, 5)
        The ground-truth objects in (left, top, right, bottom, class) format.

    threshold : Real, optional (default=0.5)
        The IoU threshold at which to compute recall.

    Returns
    -------
    float
        The average recall (AR) for the given detections and truth.

    Notes
    -----
    This function operates such that when there are zero targets, recall is 1 regardless of predictions.
    """
    predictions = prediction_detections[:, -1]
    truths = truth_detections[:, -1]

    if truths.sum() == 0:
        return 1  # if there are no targets, then by definition we've found all the targets

    if predictions.sum() == 0:
        return 0  # if there are targets and we predict there are none, we can short circuit

    ious = box_overlaps(prediction_detections[:, :4], truth_detections[:, :4])
    max_ious = ious.max(axis=1)
    max_idxs = ious.argmax(axis=1)

    target_labels = truths[max_idxs]
    true_positives = (predictions == target_labels)[np.logical_and(max_ious >= threshold, target_labels > 0)].sum()
    false_negatives = (predictions != target_labels)[max_ious >= threshold].sum()
    false_negatives += (ious.max(axis=0) < threshold).sum()

    return true_positives / (true_positives + false_negatives)


def generate_targets(anchor_boxes, truth_boxes, labels, pos_thresh=0.3, neg_thresh=0.2, eps=1e-12):
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


def non_max_suppression(detections, threshold=0.7, clip_value=1e6, eps=1e-12):
    """ Apply non-maximum suppression to the detections provided with a given threshold.

    Parameters
    ----------
    detections : np.ndarray[Real], shape=(N, 5)
        The detection boxes to which to apply NMS, in (left, top, right, bottom, score) format.

    threshold : float ∈ [0, 1], optional (default=0.7)
        The IoU threshold to use for NMS, above which one of two box will be suppressed.

    clip_value : Real, optional (default=1e6)
        The maximum width or height overlap, for numerical stability.

    eps : Real, optional (default=1e-12)
        The epsilon value to use in IoU calculation, for numerical stability.

    Returns
    -------
    numpy.ndarray[int], shape=(k,)
        The indices of `detections` to keep, where k is the number of non-suppressed inputs and k <= N.
    """
    x1s = detections[:, 0]
    y1s = detections[:, 1]
    x2s = detections[:, 2]
    y2s = detections[:, 3]

    areas = np.clip(x2s - x1s, 0, 10**6) * np.clip(y2s - y1s, 0, 10**6)
    order = detections[:, 4].argsort()[::-1]  # highest to lowest score

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
        ious = intersections / (areas[i] + areas[all_others] - intersections + 1e-12)

        # +1 to counteract the offset all_others = order[1:]
        order = order[np.where(ious <= threshold)[0] + 1]

    return np.array(keep)


def xywh_to_xyxy(boxes):
    """ Convert boxes from xywh to xyxy.

    Parameters
    ----------
    boxes : numpy.ndarray, shape=(N, 4)
        Boxes, in xywh format.

    Returns
    -------
    numpy.ndarray, shape=(N, 4)
        Boxes in xyxy format
    """
    temp = np.empty_like(boxes)
    if temp.size > 0:
        temp[:, :2] = boxes[:, :2]
        temp[:, 2:] = boxes[:, :2] + boxes[:, 2:]
    return temp


def xyxy_to_xywh(boxes):
    """ Convert boxes from xyxy to xywh.
    Parameters
    ----------
    boxes : numpy.ndarray, shape=(N, 4)
        Boxes, in xyxy format.
    Returns
    -------
    numpy.ndarray, shape=(N, 4)
        Boxes in xywh format
    """
    temp = np.empty_like(boxes)
    if temp.size > 0:
        temp[:, :2] = boxes[:, :2]
        temp[:, 2:] = boxes[:, 2:] - boxes[:, :2]
    return temp
