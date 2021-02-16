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

from typing import List

import pytest

import numpy as np
from numpy import ndarray
from numpy.testing import assert_allclose, assert_array_equal

import hypothesis.strategies as st
from hypothesis import given, settings
import hypothesis.extra.numpy as hnp

from detection_utils.boxes import (
    box_overlaps,
    xywh_to_xyxy,
    xyxy_to_xywh,
    generate_targets,
)
from detection_utils.boxes import non_max_suppression


class Test_box_transforms:
    """ Ensure that the basic box manipulation routines work correctly. """

    def test_xyxy_to_xywh_static(self):
        """ Ensure that transforming xyxy to xywh format works with known values. """
        xyxy_box = np.array([[-0.5, -0.5, 0.5, 0.5]])
        xywh_box = np.array([[-0.5, -0.5, 1.0, 1.0]])
        msg = "xyxy_to_xywh failed to produce a known-correct output"
        assert_allclose(actual=xyxy_to_xywh(xyxy_box), desired=xywh_box, err_msg=msg)

    def test_xywh_to_xyxy_static(self):
        """ Ensure that transforming xywh to xyxy format works with known values. """
        xyxy_box = np.array([[-0.5, -0.5, 0.5, 0.5]])
        xywh_box = np.array([[-0.5, -0.5, 1.0, 1.0]])
        msg = "xywh_to_xyxy failed to produce a known-correct output"
        assert_allclose(actual=xywh_to_xyxy(xywh_box), desired=xyxy_box, err_msg=msg)

    @given(
        rand_xyxy_boxes=hnp.arrays(
            dtype=float,
            shape=st.tuples(st.integers(0, 20), st.just(4)),
            elements=st.floats(-100, 100),
        )
    )
    def test_xywh_to_xyxy(self, rand_xyxy_boxes: ndarray):
        """ Ensure that xywh_to_xyxy and xyxy_to_xywh are inverses (we can round-trip). """
        rand_xyxy_boxes[2:] = np.abs(rand_xyxy_boxes[2:])  # ensure h/w are positive

        msg = "xyxy_to_xywh failed to invert xywh_to_xyxy"
        assert_allclose(
            actual=xyxy_to_xywh(xywh_to_xyxy(rand_xyxy_boxes)),
            desired=rand_xyxy_boxes,
            atol=1e-5,
            rtol=1e-5,
            err_msg=msg,
        )


class Test_box_overlaps:
    """ Ensure that the box_overlaps function correctly computes IoU. """

    a = np.array([[-100, -100, -50, -50]])  # xyxy box
    b = np.array([[0, 0, 50, 50]])  # xyxy box
    A = b
    B = np.vstack((a[0], b[0], b[0] / 2))

    @pytest.mark.parametrize(
        ("predicted", "truth", "overlap"),
        [
            (a, b, np.array([[0.00]])),  # no-overlap
            (b, a, np.array([[0.00]])),  # no-overlap
            (a, a, np.array([[1.00]])),  # exact-overlap
            (b, b, np.array([[1.00]])),  # exact-overlap
            (b, b / 2, np.array([[0.25]])),  # quarter-overlap
            (b, b * 2, np.array([[0.25]])),  # quarter-overlap
            (b / 2, b, np.array([[0.25]])),  # quarter-overlap
            (b * 2, b, np.array([[0.25]])),  # quarter-overlap
            (A, B, np.array([[0, 1, 0.25]])),  # mixed-overlap
            (B, A, np.array([[0, 1, 0.25]]).T),  # mixed-overlap
        ],
    )
    def test_known_overlaps(self, predicted, truth, overlap):
        """ Ensures that correctness for hand-crafted overlapping boxes. """
        assert_allclose(actual=box_overlaps(predicted, truth), desired=overlap)

    @settings(deadline=500)
    @given(
        boxes=hnp.arrays(dtype=float, shape=st.tuples(st.integers(0, 100), st.just(4))),
        truth=hnp.arrays(dtype=float, shape=st.tuples(st.integers(0, 100), st.just(4))),
    )
    def test_shapes(self, boxes: ndarray, truth: ndarray):
        """ Ensures that the shape returned by box_overlaps is correct, even in edge cases with no boxes. """
        N = boxes.shape[0]
        K = truth.shape[0]
        msg = "box_overlaps did not produce an empty array of the correct shape"
        assert box_overlaps(boxes, truth).shape == (N, K), msg


class Test_generate_targets:
    """ Ensure that the generate_targets function produces the correct target values. """

    @given(
        boxes=hnp.arrays(
            dtype=float,
            shape=st.tuples(st.integers(0, 3), st.just(4)),
            elements=st.floats(1, 100),
            unique=True,
        ),
        truth=hnp.arrays(
            dtype=float,
            shape=st.tuples(st.integers(0, 3), st.just(4)),
            elements=st.floats(1, 100),
            unique=True,
        ),
        data=st.data(),
    )
    def test_shapes(self, boxes: ndarray, truth: ndarray, data: st.SearchStrategy):
        """ Ensure the shape returned by generate_targets is correct, even in edge cases producing empty arrays. """
        boxes = boxes.cumsum(axis=1)  # to ensure we don't hit 0-width or -height boxes
        truth = truth.cumsum(axis=1)  # to ensure we don't hit 0-width or -height boxes
        N = boxes.shape[0]
        K = truth.shape[0]
        labels = data.draw(hnp.arrays(dtype=int, shape=(K,)))
        cls, reg = generate_targets(boxes, truth, labels, 0.5, 0.4)

        msg = "generate_targets failed to produce classification targets of the correct shape"
        assert cls.shape == (N,), msg

        msg = "generate_targets failed to produce regression targets of the correct shape"
        assert reg.shape == (N, 4), msg

    @given(x=hnp.arrays(dtype=float, shape=(5, 4), elements=st.floats(1, 100)))
    def test_identical_proposed_and_truth(self, x: ndarray):
        """ Ensure that generate_targets produces regression targets that are zero for identical proposal and truth. """
        x = x.cumsum(axis=1)  # ensure (l, t, r , b)
        labels = np.array([0] * 5)
        _, reg = generate_targets(x, x, labels, 0.5, 0.4)
        msg = "generate_targets failed to produce the expected output when the proposed boxes are identical to ground truth"
        assert_allclose(
            actual=reg, desired=np.zeros_like(x), atol=1e-5, rtol=1e-5, err_msg=msg
        )

    @given(shuffle_inds=st.permutations(np.arange(3)))
    def test_known_regression_values(self, shuffle_inds: List[int]):
        """ Ensure that generate_targets works for known values. Ensure that datum ordering does not matter. """
        prop = np.array(
            [
                [-0.5, -0.5, 0.5, 0.5],     # neither axis matches truth
                [0, -0.5, np.exp(1), 0.5],  # x matches truth
                [-0.5, 0, 0.5, np.exp(1)],  # y matches truth
            ]
        )
        truth = np.array([[0, 0, np.exp(1), np.exp(1)]])
        labels = np.array([1.0])

        out_reg = np.array(
            [
                [np.exp(1) / 2, np.exp(1) / 2, 1, 1],
                [0, np.exp(1) / 2, 0, 1],
                [np.exp(1) / 2, 0, 1, 0],
            ]
        )

        _, reg = generate_targets(prop[shuffle_inds], truth, labels, 0.5, 0.4)
        msg = "generate_targets failed to produce known-correct regression values"
        assert_allclose(
            actual=reg, desired=out_reg[shuffle_inds], atol=1e-5, rtol=1e-5, err_msg=msg
        )

    @given(
        label0=st.integers(1, 10),
        label1=st.integers(1, 10),
        shuffle_inds=st.permutations(np.arange(4)),
    )
    def test_label_invariance(self, label0: int, label1: int, shuffle_inds: List[int]):
        """ Ensure that datum ordering doesn't matter for generate_targets. """
        # xyxy format
        prop = np.array(
            [
                [-0.5, -0.5, 0.5, 0.5],  # iou = 1 (truth 0) should be marked poitiive
                [0.0, -0.5, 0.49, 0.5],  # iou = 0.5  (truth 0) should be marked ignore
                [0.0, -0.5, 0.39, 0.5],  # iou = 0.39  (truth 0) should be marked negative
                [10.0, 10.0, 11, 11],
            ]
        )  # iou = 1 (truth 1) should be marked positive

        # xyxy format
        truth = np.array([
            [-0.5, -0.5, 0.5, 0.5],
            [10.0, 10.0, 11, 11]
        ])

        labels = np.array([label0, label1])

        out_labels = np.array(
            [label0, -1, 0, label1]
        )  # truth 0 / ignore / background / truth 1 from above

        labels, reg = generate_targets(prop[shuffle_inds], truth, labels, 0.5, 0.4)
        msg = "generate_targets is not invariant to datum-ordering"
        assert_allclose(actual=labels, desired=out_labels[shuffle_inds], err_msg=msg)


class Test_non_max_suppression:
    """ Ensure that non-maximum suppression (NMS) correctly suppresses expected values. """

    @given(
        boxes=hnp.arrays(
            dtype=float,
            shape=st.tuples(st.integers(0, 100), st.just(5)),
            elements=st.floats(1e-05, 100),
            unique=True,
        ),
        data=st.data(),
    )
    def test_shapes(self, boxes: ndarray, data: st.SearchStrategy):
        """ Ensure that non_max_suppression produces the correct shape output, even for empty inputs. """
        scores = boxes[:, 4]
        boxes = boxes[:, :4].cumsum(axis=1)  # ensures no 0-width or -height boxes

        N = scores.shape[0]
        nms = non_max_suppression(boxes, scores)

        assert (
            nms.shape[0] <= N
        )  # we're suppressing, so we can never end up with more things than we started with
        assert nms.ndim == 1

    def test_empty(self):
        """ Ensure that non_max_suppression works correctly with zero detections. """
        x = np.empty((0, 4))
        scores = np.empty((0,))
        nms = non_max_suppression(x, scores)
        msg = "non_max_suppression failed to produce the expected output for zero detections"
        assert nms.shape == (0,), msg

    @given(
        x=hnp.arrays(dtype=float, shape=(1, 4), elements=st.floats(1e-05, 100)),
        score=st.floats(0, 1),
        rep=st.integers(2, 100),
    )
    def test_identical(self, x, score, rep):
        """ Ensure that non_max_suppression works correctly for identical boxes and that ordering doesn't matter. """
        x = x.cumsum(axis=1)
        x = x.repeat(rep).reshape(x.shape[1], rep).T
        score = np.array([score] * rep)
        idx = np.random.randint(len(x))
        score[idx] = 1000

        nms = non_max_suppression(x, score)
        msg = "non_max_suppression failed to produce the expected output when all detections are identical"
        assert_array_equal(nms, np.array([idx]), msg)

        nms = non_max_suppression(x, score, threshold=1)
        msg = "non_max_suppression failed to produce the expected output for identical detections with threshold 1"
        assert_array_equal(nms, np.array(range(len(x))), msg)

    @given(
        x=hnp.arrays(dtype=float, shape=(1, 4), elements=st.floats(1e-05, 100)),
        score=st.floats(0, 1),
    )
    def test_single_detections(self, x: ndarray, score):
        """ Ensure that a single detection is not suppressed. """
        nms = non_max_suppression(x, np.array([score]))
        msg = "non_max_suppression failed to produce the expected output for a single detection"
        assert_array_equal(nms, np.array([0]), msg)

    @pytest.mark.parametrize(
        ("threshold", "desired_nms"),
        [(0.5, np.array([0, 1])),
         (0.25, np.array([0, 1])),
         (0.15, np.array([1]))],
    )
    def test_known_results(self, threshold, desired_nms):
        """ Ensures that non_max_suppression works correctly for known values. """
        boxes = np.array([
            [0, 0, 1, 1],
            [0.5, 0.5, 0.9, 0.9]
        ])
        scores = np.array([0, 1])

        actual_nms = non_max_suppression(boxes, scores, threshold=threshold)
        assert_array_equal(actual_nms, desired_nms)
