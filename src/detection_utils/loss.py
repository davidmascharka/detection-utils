try:
    import torch
    import torch.nn.functional as F
except ImportError:
    raise ImportError('You must have PyTorch installed to utilize functions in detection_utils.pytorch')


def softmax_focal_loss(input, target, alpha=1, gamma=0):
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

    Returns
    -------
    torch.Tensor
        The mean focal loss.

    Notes
    -----
    When ɑ=1 and Ɣ=0, this is equivalent to softmax cross-entropy.
    """
    input = F.log_softmax(input, dim=1)
    logpc = input[(range(len(target)), target)]
    one_m_pc = (-1 * torch.expm1(logpc)).clamp(min=1e-14, max=1.0)
    loss = -alpha * one_m_pc**gamma * logpc
    return loss.mean()
