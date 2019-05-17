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

import torch
from torch import Tensor
import torch.nn.functional as F


def softmax_focal_loss(
        inputs: Tensor,
        targets: Tensor,
        alpha: float = 1,
        gamma: float = 0,
        reduction: str = 'mean',
) -> Tensor:
    """ Returns the focal loss as described in https://arxiv.org/abs/1708.02002 which is given by -ɑ(1-p)ˠlog(p).

    Parameters
    ----------
    input : torch.Tensor, shape=(N, C)
        The C class scores for each of the N pieces of data.

    target : torch.Tensor, shape=(N,)
        The correct class indices, in [0, C), for each datum.

    alpha : Real, optional (default=1)
        The ɑ weighting factor used in the loss formulation.

    gamma : Real, optional (default=0)
        The Ɣ focusing parameter.

    reduction : str ∈ {'mean', 'sum', 'none'}, optional (default='mean')
        How to reduce the loss to a scalar, or 'none' to return the per-item loss.

    Returns
    -------
    torch.Tensor, shape=() if reduction is 'none' otherwise shape=(N,)
        The mean focal loss.

    Notes
    -----
    When ɑ=1 and Ɣ=0, this is equivalent to softmax cross-entropy.
    """
    if reduction not in {'mean', 'sum', 'none'}:
        raise ValueError('Valid reduction strategies are "mean," "sum," and "none"')

    inputs = F.log_softmax(inputs, dim=1)
    logpc = inputs[(range(len(targets)), targets)]
    one_m_pc = (-1 * torch.expm1(logpc)).clamp(min=1e-14, max=1.0)
    loss = -alpha * one_m_pc**gamma * logpc
    return loss if reduction == 'none' else loss.mean() if reduction == 'mean' else loss.sum()
