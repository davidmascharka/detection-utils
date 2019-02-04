import pytest

import numpy as np
from numpy import ndarray
from numpy.testing import assert_allclose

import hypothesis.strategies as st
from hypothesis import given, settings
import hypothesis.extra.numpy as hnp

try:
    import torch
    from torch import tensor
    import torch.nn.functional as F

    from detection_utils.pytorch import softmax_focal_loss
    has_torch = True
except ImportError:
    has_torch = False


@pytest.mark.skipif(not has_torch, reason='PyTorch not available. Skipping test_focal_loss...')
class Test_focal_loss:
    @settings(derandomize=True, database=None)
    @given(inputs=hnp.arrays(dtype=float, shape=(3, 5), elements=st.floats(0.01, 10)),
           targets=hnp.arrays(dtype=int, shape=(3,), elements=st.integers(0, 2)))
    def test_default_args(self, inputs: ndarray, targets: ndarray):
        """ Ensures default arguments have not changed """
        inputs = tensor(inputs)
        targets = tensor(targets)
        assert_allclose(desired=softmax_focal_loss(inputs, targets, alpha=1.0, gamma=0.0),
                        actual=softmax_focal_loss(inputs, targets),
                        err_msg="`softmax_focal_loss default args changed")

    @given(inputs=hnp.arrays(dtype=float, shape=hnp.array_shapes(min_dims=2, max_dims=2),
                             elements=st.floats(-1e3, 1e3)),
           alpha=st.floats(-10, 10),
           dtype=st.sampled_from((torch.float32, torch.float64)),
           data=st.data())
    def test_matches_crossentropy(self, inputs: ndarray, alpha: float,
                                  dtype: torch.dtype, data: st.SearchStrategy):
        """ ensure that focal loss w/ gamma=0 matches softmax cross-entropy (scaled by alpha)"""
        targets = data.draw(hnp.arrays(dtype=int,
                                       shape=(inputs.shape[0],),
                                       elements=st.integers(0, inputs.shape[1] - 1)),
                            label="targets")

        inputs = tensor(inputs, dtype=dtype)
        targets = tensor(targets)
        assert_allclose(desired=alpha * F.cross_entropy(inputs, targets),
                        actual=softmax_focal_loss(inputs, targets, alpha=alpha, gamma=0.),
                        atol=1e-6, rtol=1e-6, err_msg="Focal loss with gamma=0 fails to match cross-entropy loss.")

    @given(inputs=hnp.arrays(dtype=float, shape=hnp.array_shapes(min_dims=2, max_dims=2, min_side=2),
                             elements=st.floats(0, 1e1)),
           alpha=st.floats(-10, 10),
           gamma=st.floats(0, 10),
           dtype=st.sampled_from((torch.float32, torch.float64)),
           data=st.data())
    def test_matches_simple_implementation(self, inputs: ndarray, alpha: float, gamma: float, dtype: torch.dtype, data: st.SearchStrategy):
        """ Ensures that our numerically-stable focal loss matches a naive-implementation on
            a domain where numerical stability is not critical."""
        targets = data.draw(hnp.arrays(dtype=int,
                                       shape=(inputs.shape[0],),
                                       elements=st.integers(0, inputs.shape[1] - 1)),
                            label="targets")

        inputs1 = tensor(inputs, dtype=dtype, requires_grad=True)
        inputs2 = tensor(inputs, dtype=dtype, requires_grad=True)
        targets = tensor(targets, dtype=torch.int64)

        # numerically-stable focal loss
        loss = softmax_focal_loss(inputs1, targets, alpha=alpha, gamma=gamma)
        loss.backward()

        # naive focal loss
        input = F.softmax(inputs2, dim=1)
        pc = input[(range(len(targets)), targets)]
        naive_loss = (-alpha * (1 - pc)**gamma * torch.log(pc)).mean()
        naive_loss.backward()

        assert_allclose(actual=loss.detach().numpy(),
                        desired=naive_loss.detach().numpy(),
                        atol=1e-5, rtol=1e-5,
                        err_msg="focal loss does not match naive implementation on "
                                "numerically-stable domain")
        assert_allclose(actual=inputs1.grad.numpy(),
                        desired=inputs2.grad.numpy(),
                        atol=1e-5, rtol=1e-5,
                        err_msg="focal loss gradient does not match that of naive loss on "
                                "numerically-stable domain")

    @given(pc=st.floats(1e-5, 1 - 1e-5), alpha=st.floats(-10, 10), gamma=st.floats(0, 10))
    def test_matches_binary_classification(self, pc: float, alpha: float, gamma: float):
        """ Ensures that our focal loss matches the explicit binary-classification
        formulation of focal loss included in paper"""
        loss = -alpha * (1 - pc)**gamma * np.log(pc)
        inputs = tensor([[np.log(pc), np.log(1 - pc)]])
        targets = tensor([0])
        assert_allclose(desired=loss,
                        actual=softmax_focal_loss(inputs, targets, alpha=alpha, gamma=gamma),
                        atol=1e-5, rtol=1e-5,
                        err_msg="focal loss does not reduce to binary-classification form")

    @given(inputs=hnp.arrays(dtype=float, shape=hnp.array_shapes(min_dims=2, max_dims=2),
                             elements=st.floats(-1e6, 1e6)),
           alpha=st.floats(-100, 100),
           gamma=st.floats(0, 20),
           dtype=st.sampled_from((torch.float32, torch.float64)),
           data=st.data())
    def test_nan_in_grad(self, inputs: ndarray, alpha: float, gamma: float, dtype: torch.dtype,
                         data: st.SearchStrategy):
        """ Ensures, across a wide range of inputs, that the focal loss gradient is not nan"""
        targets = data.draw(hnp.arrays(dtype=int,
                                       shape=(inputs.shape[0],),
                                       elements=st.integers(0, inputs.shape[1] - 1)),
                            label="targets")

        inputs = tensor(inputs, dtype=dtype, requires_grad=True)
        targets = tensor(targets)
        loss = softmax_focal_loss(inputs, targets, alpha=alpha, gamma=gamma)

        loss.backward()
        assert not np.any(np.isnan(inputs.grad.numpy())), "focal loss gradient is nan"
