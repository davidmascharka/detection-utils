# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or
# recommendations expressed in this material are those of the author(s) and do not necessarily
# reflect the views of the Under Secretary of Defense for Research and Engineering.
#
# © 2019 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
# 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
# defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other
# than as specifically authorized by the U.S. Government may violate any copyrights that exist in
# this work.

import numpy as np
from numpy import ndarray

from .boxes import box_overlaps
from detection_utils.boxes import DEFAULT_POS_THRESHOLD

from typing import NamedTuple


def confusion_matrix(
    prediction_detections: ndarray,
    truth_detections: ndarray,
    threshold: float = DEFAULT_POS_THRESHOLD,
    num_foreground_classes: int = 3,
) -> np.ndarray:
    """ Compute confusion matrix to evaluate the accuracy of a classification.

    By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in group :math:`i` and
    predicted to be in group :math:`j`.

    Thus in binary classification, the count of true negatives is
    :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
    :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.

    Parameters
    ----------
    prediction_detections : numpy.ndarray, shape=(N, 5)
        The predicted objects, in (left, top, right, bottom, class) format.

    truth_detections : numpy.ndarray, shape=(K, 5)
        The ground-truth objects in (left, top, right, bottom, class) format.

    threshold : Real, optional (default=0.5)
        The IoU threshold above which a predicted box will be associated
        with an overlapping truth box.

    num_foreground_classes: int, optional (default=3)
        The number of foreground class in the problem

    Returns
    -------
    conf_matrix : numpy.ndarray, shape-(N_class, N_class)
        Confusion matrix whose i-th row and j-th column entry
        indicates the number of samples with true label being i-th
        class and predicted label being j-th class.

    Notes
    -----
    The class IDs must be consecutive integers, starting with 0, which
    must be associated with the background
    """

    predictions = prediction_detections[:, -1]  # shape-(N,) labels
    truths = truth_detections[:, -1]  # shape-(N,) labels

    ious = box_overlaps(prediction_detections[:, :4], truth_detections[:, :4])

    # shape-(N,)
    max_ious = ious.max(axis=1)
    # index of highest-overlap truth box associated with each prediction
    max_idxs = ious.argmax(axis=1)  # shape-(N,)

    target_labels = truths[max_idxs]  # shape-(N,)

    # boxes with insufficient overlap are associated with background
    target_labels[max_ious < threshold] = 0

    conf_mat = np.zeros(
        (num_foreground_classes + 1, num_foreground_classes + 1), dtype=np.int32
    )
    np.add.at(conf_mat, (target_labels, predictions), 1)
    return conf_mat


class DetectionStats(NamedTuple):
    precision: float
    recall: float


def compute_precision_and_recall(conf_matrix: np.ndarray) -> DetectionStats:
    # TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1:].sum()
    FN = conf_matrix[1:, 0].sum()
    TP = conf_matrix[1:, 1:].sum()
    return DetectionStats(precision=(TP / (TP + FP)), recall=(TP / (TP + FN)))


def compute_precision(
    prediction_detections: ndarray, truth_detections: ndarray, threshold: float = 0.5,
) -> float:
    """ Compute the average precision of predictions given targets.

    Precision is defined as the number of true positive predictions divided by the number
    of total positive predictions.

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
    >>> predictions = np.array([[0, 0, 10, 10, 1], [3, 3, 7, 7, 1]])  # left, top, right, bottom, class
    >>> actual = np.array([[2, 3, 6, 7, 1]])
    >>> compute_precision(predictions, actual)
    0.5

    Our IoUs are 0.16, 0.6
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
    true_positive_idxs = np.logical_and(max_ious >= threshold, target_labels > 0)
    num_true_positives = (predictions == target_labels)[true_positive_idxs].sum()
    num_predicted_positives = (predictions > 0).sum()

    return num_true_positives / num_predicted_positives


def compute_recall(
    prediction_detections: ndarray, truth_detections: ndarray, threshold: float = 0.5,
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
    >>> predictions = np.array([[0, 0, 10, 10, 1], [3, 3, 7, 7, 1]])  # left, top, right, bottom, class
    >>> actual = np.array([[2, 3, 6, 7, 1]])
    >>> compute_recall(predictions, actual)
    1.0

    Our highest IoU is 0.6 so let's set our threshold above that
    >>> compute_recall(predictions, actual, threshold=0.75)
    0.0
    """
    predictions = prediction_detections[:, -1]
    truths = truth_detections[:, -1]

    if truths.sum() == 0:
        return (
            1  # if there are no targets, then by definition we've found all the targets
        )

    if predictions.sum() == 0:
        return 0  # if there are targets and we predict there are none, we can short circuit

    ious = box_overlaps(prediction_detections[:, :4], truth_detections[:, :4])
    max_ious = ious.max(axis=1)
    max_idxs = ious.argmax(axis=1)

    target_labels = truths[max_idxs]
    true_positive_idxs = np.logical_and(max_ious >= threshold, target_labels > 0)
    num_true_positives = (predictions == target_labels)[true_positive_idxs].sum()
    num_false_negatives = (predictions != target_labels)[max_ious >= threshold].sum()
    num_false_negatives += (ious.max(axis=0) < threshold).sum()

    return num_true_positives / (num_true_positives + num_false_negatives)
