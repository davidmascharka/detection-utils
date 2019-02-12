from typing import List

import numpy as np
from numpy import ndarray
from numpy.testing import assert_allclose, assert_array_equal

import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp

from detection_utils.boxes import box_overlaps, xywh_to_xyxy, xyxy_to_xywh, generate_targets
from detection_utils.boxes import compute_precision, compute_recall, non_max_suppression


class Test_box_transforms:
    """ Ensure that the basic box manipulation routines work as intended. """

    def test_xyxy_to_xywh_static(self):
        """ Ensure that transforming xyxy to xywh format works with known values. """
        xyxy_box = np.array([[-.5, -.5, .5, .5]])
        xywh_box = np.array([[-.5, -.5, 1., 1.]])
        msg = 'xyxy_to_xywh failed to produce a known-correct output'
        assert_allclose(actual=xyxy_to_xywh(xyxy_box), desired=xywh_box, err_msg=msg)

    def test_xywh_to_xyxy_static(self):
        """ Ensure that transforming xywh to xyxy format works with known values. """
        xyxy_box = np.array([[-.5, -.5, .5, .5]])
        xywh_box = np.array([[-.5, -.5, 1., 1.]])
        msg = 'xywh_to_xyxy failed to produce a known-correct output'
        assert_allclose(actual=xywh_to_xyxy(xywh_box), desired=xyxy_box, err_msg=msg)

    @given(rand_xyxy_boxes=hnp.arrays(dtype=float,
                                      shape=st.tuples(st.integers(0, 20), st.just(4)),
                                      elements=st.floats(-100, 100)))
    def test_xywh_to_xyxy(self, rand_xyxy_boxes: ndarray):
        """ Ensure that xywh_to_xyxy and xyxy_to_xywh are inverses. """
        rand_xyxy_boxes[2:] = np.abs(rand_xyxy_boxes[2:])  # ensure h/w are positive

        msg = 'xyxy_to_xywh failed to invert xywh_to_xyxy'
        assert_allclose(actual=xyxy_to_xywh(xywh_to_xyxy(rand_xyxy_boxes)), desired=rand_xyxy_boxes,
                        atol=1e-5, rtol=1e-5, err_msg=msg)


class Test_compute_precision:
    """ Ensure that the compute_precision function works as intended. """

    def test_known_precision(self):
        """ Ensure that compute_precision works with known box/label pairs. """
        a = np.array([[0, 0, 1, 1, 1]])
        msg = 'compute_precision failed to report precision of 1 for identical boxes'
        assert_allclose(actual=compute_precision(a, a), desired=1, err_msg=msg)

        b = np.array([[0, 0, 1, 1, 2]])
        msg = 'compute_precision failed to report precision of 0 for identical box with different class'
        assert_allclose(actual=compute_precision(a, b), desired=0, err_msg=msg)

        b = np.array([[1, 1, 2, 2, 1]])
        msg = 'compute_precision failed to report precision of 0 for non-overlapping boxes'
        assert_allclose(actual=compute_precision(a, b), desired=0, err_msg=msg)

        b = np.empty((0, 5))
        msg = 'compute_precision failed to report precision of 0 for empty truth and non-empty predictions'
        assert_allclose(actual=compute_precision(a, b), desired=0, err_msg=msg)

        msg = 'compute_precision failed to repoort precision of 1 for empty predictions and non-empty truth'
        assert_allclose(actual=compute_precision(b, a), desired=1, err_msg=msg)

        msg = 'compute_precision failed to report precision of 1 for empty predictions and empty truth'
        assert_allclose(actual=compute_precision(b, b), desired=1, err_msg=msg)


class Test_compute_recall:
    """ Ensure that the compute_recall function works as intended. """

    def test_known_recall(self):
        """ Ensure that compute_recall works with known box/label pairs. """
        a = np.array([[0, 0, 1, 1, 1]])
        msg = 'compute_recall failed to report recall of 1 for identical boxes'
        assert_allclose(actual=compute_recall(a, a), desired=1, err_msg=msg)

        b = np.array([[0, 0, 1, 1, 2]])
        msg = 'compute_recall failed to report recall of 0 for identical box with different class'
        assert_allclose(actual=compute_recall(a, b), desired=0, err_msg=msg)

        b = np.array([[1, 1, 2, 2, 1]])
        msg = 'compute_recall failed to report recall of 0 for non-overlapping boxes'
        assert_allclose(actual=compute_recall(a, b), desired=0, err_msg=msg)

        b = np.empty((0, 5))
        msg = 'compute_recall failed to report recall of 1 for empty truth and non-empty predictions'
        assert_allclose(actual=compute_recall(a, b), desired=1, err_msg=msg)

        msg = 'compute_recall failed to repoort recall of 0 for empty predictions and non-empty truth'
        assert_allclose(actual=compute_recall(b, a), desired=0, err_msg=msg)

        msg = 'compute_recall failed to report recall of 1 for empty predictions and empty truth'
        assert_allclose(actual=compute_recall(b, b), desired=1, err_msg=msg)


class Test_box_overlaps:
    """ Ensure that the box_overlaps function works as intended. """

    def test_known_overlaps(self):
        """ Ensures that box_overlaps works for known box overlaps. """
        a = np.array([[-100, -100, -50, -50]])  # xyxy box
        b = np.array([[0, 0, 50, 50]])          # xyxy box

        # no overlap
        msg = 'box_overlaps produced the wrong output for a no-overlap case'
        assert_allclose(actual=box_overlaps(a, b), desired=np.array([[0]]), err_msg=msg)
        assert_allclose(actual=box_overlaps(b, a), desired=np.array([[0]]), err_msg=msg)

        # exact overlap
        msg = 'box_overlaps failed to produce the expected output for an exact-overlap case'
        assert_allclose(actual=box_overlaps(a, a), desired=np.array([[1]]), err_msg=msg)
        assert_allclose(actual=box_overlaps(b, b), desired=np.array([[1]]), err_msg=msg)

        # quarter overlap
        msg = 'box_overlaps failed to produce the expected output for a quarter-overlap case'
        assert_allclose(actual=box_overlaps(b, b / 2), desired=np.array([[0.25]]), err_msg=msg)
        assert_allclose(actual=box_overlaps(b, b * 2), desired=np.array([[0.25]]), err_msg=msg)
        assert_allclose(actual=box_overlaps(b / 2, b), desired=np.array([[0.25]]), err_msg=msg)
        assert_allclose(actual=box_overlaps(b * 2, b), desired=np.array([[0.25]]), err_msg=msg)

        # mixed overlap
        A = b
        B = np.vstack((a[0], b[0], b[0] / 2))
        msg = 'box_overlaps failed to produce the expected output for a known-overlap case'
        assert_allclose(actual=box_overlaps(A, B), desired=np.array([[0, 1, 0.25]]), err_msg=msg)
        assert_allclose(actual=box_overlaps(B, A), desired=np.array([[0, 1, 0.25]]).T, err_msg=msg)

    @given(boxes=hnp.arrays(dtype=float, shape=st.tuples(st.integers(0, 100), st.just(4))),
           truth=hnp.arrays(dtype=float, shape=st.tuples(st.integers(0, 100), st.just(4))))
    def test_shapes(self, boxes: ndarray, truth: ndarray):
        """ Ensures that the shape returned by box_overlaps is correct, even in edge cases with no boxes. """
        N = boxes.shape[0]
        K = truth.shape[0]
        msg = 'box_overlaps did not produce an empty array of the correct shape'
        assert box_overlaps(boxes, truth).shape == (N, K), msg


class Test_generate_targets:
    """ Ensure that the generate_targets function works as intended. """

    @given(boxes=hnp.arrays(dtype=float, shape=st.tuples(st.integers(0, 3), st.just(4)),
                            elements=st.floats(1, 100), unique=True),
           truth=hnp.arrays(dtype=float, shape=st.tuples(st.integers(0, 3), st.just(4)),
                            elements=st.floats(1, 100), unique=True),
           data=st.data())
    def test_shapes(self, boxes: ndarray, truth: ndarray, data: st.SearchStrategy):
        """ Ensure the shape returned by generate_targets is correct, even in edge cases producing empty arrays. """
        boxes = boxes.cumsum(axis=1)
        truth = truth.cumsum(axis=1)
        N = boxes.shape[0]
        K = truth.shape[0]
        labels = data.draw(hnp.arrays(dtype=int, shape=(K,)))
        cls, reg = generate_targets(boxes, truth, labels, 0.5, 0.4)
        assert cls.shape == (N,), 'generate_targets failed to produce empty an array of the correct shape'
        assert reg.shape == (N, 4), 'generate_targets failed to produce empty an array of the correct shape'

    @given(x=hnp.arrays(dtype=float, shape=(5, 4), elements=st.floats(1, 100)))
    def test_identical_proposed_and_truth(self, x: ndarray):
        # identical proposed and ground truth boxes
        x = x.cumsum(axis=1)  # ensure (l, t, r , b)
        labels = np.array([0] * 5)
        _, reg = generate_targets(x, x, labels, 0.5, 0.4)
        msg = 'generate_targets failed to produce the expected output when the proposed boxes are identical to ground truth'
        assert_allclose(actual=reg, desired=np.zeros_like(x), atol=1e-5, rtol=1e-5, err_msg=msg)

    @given(shuffle_inds=st.permutations(np.arange(3)))
    def test_known_regression_values(self, shuffle_inds: List[int]):
        # check explicit mathematical form of regression
        # ensure that datum-ordering doesn't matter
        prop = np.array([[-0.5, -0.5, 0.5, 0.5],      # neither axis matches truth
                         [0, -0.5, np.exp(1), 0.5],   # x matches truth
                         [-0.5, 0, 0.5, np.exp(1)]])  # y matches truth
        truth = np.array([[0, 0, np.exp(1), np.exp(1)]])
        labels = np.array([1.0])

        out_reg = np.array([[np.exp(1) / 2, np.exp(1) / 2, 1, 1],
                            [0, np.exp(1) / 2, 0, 1],
                            [np.exp(1) / 2, 0, 1, 0]])

        _, reg = generate_targets(prop[shuffle_inds], truth, labels, 0.5, 0.4)
        msg = 'generate_targets failed to produce known-correct regression values'
        assert_allclose(actual=reg, desired=out_reg[shuffle_inds], atol=1e-5, rtol=1e-5, err_msg=msg)

    @given(label0=st.integers(1, 10), label1=st.integers(1, 10),
           shuffle_inds=st.permutations(np.arange(4)))
    def test_label_invariance(self, label0: int, label1: int, shuffle_inds: List[int]):
        ''' check explicit mathematical form of regression
            ensure that datum-ordering doesn't matter '''
        # xyxy format
        prop = np.array([[-.5, -.5, .5, .5],    # iou = 1 (truth 0)
                         [0., -.5, 0.49, 0.5],  # iou = 0.5  (truth 0)
                         [0., -.5, 0.39, 0.5],  # iou = 0.39  (truth 0)
                         [10., 10., 11, 11]])   # iou = 1 (truth 1)

        # xyxy format
        truth = np.array([[-.5, -.5, .5, .5],
                          [10., 10., 11, 11]])

        labels = np.array([label0, label1])

        out_labels = np.array([label0, -1, 0, label1])

        labels, reg = generate_targets(prop[shuffle_inds], truth, labels, 0.5, 0.4)
        msg = 'generate_targets is not invariant to datum-ordering'
        assert_allclose(actual=labels, desired=out_labels[shuffle_inds], err_msg=msg)


class Test_non_max_suppression:
    @given(boxes=hnp.arrays(dtype=float, shape=st.tuples(st.integers(0, 100), st.just(5)),
                            elements=st.floats(1e-05, 100), unique=True),
           data=st.data())
    def test_shapes(self, boxes: ndarray, data: st.SearchStrategy):
        scores = boxes[:, 4]
        boxes = boxes[:, :4].cumsum(axis=1)

        N = scores.shape[0]
        nms = non_max_suppression(boxes, scores)

        assert nms.shape[0] <= N

    def test_empty(self):
        x = np.empty((0, 4))
        scores = np.empty((0,))
        nms = non_max_suppression(x, scores)
        msg = 'non_max_suppression failed to produce the expected output for zero detections'
        assert nms.shape == (0,), msg

    @given(x=hnp.arrays(dtype=float, shape=(1, 4), elements=st.floats(1e-05, 100)),
           score=st.floats(0, 1),
           rep=st.integers(2, 100))
    def test_identical(self, x, score, rep):
        # identical detections
        x = x.cumsum(axis=1)  # ensure (l, t, r, b)
        x = x.repeat(rep).reshape(x.shape[1], rep).T
        score = np.array([score] * rep)
        idx = np.random.randint(len(x))
        score[idx] = 1000

        nms = non_max_suppression(x, score)
        msg = 'non_max_suppression failed to produce the expected output when all detections are identical'
        assert_array_equal(nms, np.array([idx]), msg)

        nms = non_max_suppression(x, score, threshold=1)
        msg = 'non_max_suppression failed to produce the expected output with threshold 1'
        assert_array_equal(nms, np.array(range(len(x))), msg)

    @given(x=hnp.arrays(dtype=float, shape=(1, 4), elements=st.floats(1e-05, 100)),
           score=st.floats(0, 1))
    def test_single_detections(self, x: ndarray, score):
        nms = non_max_suppression(x, np.array([score]))
        msg = 'non_max_suppression failed to produce the expected output for a single detection'
        assert_array_equal(nms, np.array([0]), msg)

    def test_known_results(self):
        x = np.array([[0, 0, 1, 1],
                      [0.5, 0.5, 0.9, 0.9]])
        scores = np.array([0, 1])
        nms = non_max_suppression(x, scores, threshold=0.5)
        msg = 'non_max_suppression failed to produce the expected output with threshold 0.5'
        assert_array_equal(nms, np.array([0, 1]), msg)

        nms = non_max_suppression(x, scores, threshold=0.25)
        msg = 'non_max_suppression failed to produce the expected output with threshold 0.25'
        assert_array_equal(nms, np.array([0, 1]), msg)

        nms = non_max_suppression(x, scores, threshold=0.15)
        msg = 'non_max_suppression failed to produce the expected output with threshold 0.15'
        assert_array_equal(nms, np.array([1]), msg)
