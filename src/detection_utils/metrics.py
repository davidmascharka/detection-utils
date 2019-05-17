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

import numpy as np
from numpy import ndarray

from .boxes import box_overlaps


def compute_precision(
        prediction_detections: ndarray,
        truth_detections: ndarray,
        threshold: float = 0.5,
) -> float:
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

    Examples
    --------
    >>> from detection_utils.metrics import compute_precision
    >>> import numpy as np
    >>> predictions = np.array([[0, 0, 10, 10, 1], [3, 3, 7, 7, 1]])  # left, top, right, bottom, class prediction
    >>> actual = np.array([[2, 3, 6, 7, 1]])
    >>> compute_precision(predictions, actual)
    0.5

    # Our IoUs are 0.16, 0.6
    >>> compute_precision(predictions, actual, threshold=0.15)
    1.0

    >>> compute_precision(predictions, actual, threshold=0.75)
    0.0
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


def compute_recall(
        prediction_detections: ndarray,
        truth_detections: ndarray,
        threshold: float = 0.5,
) -> float:
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

    Examples
    --------
    >>> from detection_utils.metrics import compute_recall
    >>> import numpy as np
    >>> predictions = np.array([[0, 0, 10, 10, 1], [3, 3, 7, 7, 1]])  # left, top, right, bottom, class prediction
    >>> actual = np.array([[2, 3, 6, 7, 1]])
    >>> compute_recall(predictions, actual)
    1.0

    # Our highest IoU is 0.6 so let's set our threshold above that
    >>> compute_recall(predictions, actual, threshold=0.75)
    0.0
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
