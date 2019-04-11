import torch
import torch.nn.functional as F


def softmax_focal_loss(input, target, alpha=1, gamma=0, reduction='mean'):
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

    input = F.log_softmax(input, dim=1)
    logpc = input[(range(len(target)), target)]
    one_m_pc = (-1 * torch.expm1(logpc)).clamp(min=1e-14, max=1.0)
    loss = -alpha * one_m_pc**gamma * logpc
    return loss if reduction == 'none' else loss.mean() if reduction == 'mean' else loss.sum()
