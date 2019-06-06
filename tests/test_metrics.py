import pytest

import numpy as np
from numpy.testing import assert_allclose

from detection_utils.metrics import compute_precision, compute_recall


class Test_precision_recall:
    """ Ensure that precision and recall calculations are correct. """

    detections_a = np.array([[0, 0, 1, 1, 1]])
    detections_b = np.array([[0, 0, 1, 1, 2]])  # identical box, but different class from a
    detections_c = np.array([[1, 1, 2, 2, 1]])  # no overlap with a/b, different class from b
    empty_detect = np.empty((0, 5))

    @pytest.mark.parametrize(
        ("prediction_detections", "truth_detections", "desired_precision", "description"),
        [(detections_a, detections_a, 1.0, "identical boxes"),
         (detections_a, detections_b, 0.0, "identical box with different class"),
         (detections_a, detections_c, 0.0, "non-overlapping boxes"),
         (detections_b, detections_c, 0.0, "non-overlapping boxes with different classes"),
         (detections_a, empty_detect, 0.0, "empty truth and non-empty predictions"),
         (empty_detect, detections_a, 1.0, "empty predictions and non-empty truth"),
         (empty_detect, empty_detect, 1.0, "empty predictions and empty truth")])
    def test_known_precision(self, prediction_detections, truth_detections, desired_precision, description):
        """ Ensure that compute_precision works with known box/label pairs. """
        msg = f'compute_recall failed to report precision of {desired_precision} for {description}'
        assert_allclose(actual=compute_precision(prediction_detections, truth_detections),
                        desired=desired_precision, err_msg=msg)

    @pytest.mark.parametrize(
        ("prediction_detections", "truth_detections", "desired_recall", "description"),
        [(detections_a, detections_a, 1.0, "identical boxes"),
         (detections_a, detections_b, 0.0, "identical box with different class"),
         (detections_a, detections_c, 0.0, "non-overlapping boxes"),
         (detections_b, detections_c, 0.0, "non-overlapping boxes with different classes"),
         (detections_a, empty_detect, 1.0, "empty truth and non-empty predictions"),
         (empty_detect, detections_a, 0.0, "empty predictions and non-empty truth"),
         (empty_detect, empty_detect, 1.0, "empty predictions and empty truth")])
    def test_known_recall(self, prediction_detections, truth_detections, desired_recall, description):
        """ Ensure that compute_recall works with known box/label pairs. """
        msg = f'compute_recall failed to report recall of {desired_recall} for {description}'
        assert_allclose(actual=compute_recall(prediction_detections, truth_detections),
                        desired=desired_recall, err_msg=msg)
