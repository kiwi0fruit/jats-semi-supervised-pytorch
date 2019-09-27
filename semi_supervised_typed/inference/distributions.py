import math
import torch as tr
from torch import Tensor

δ = 1e-8


def log_standard_gaussian(x: Tensor) -> Tensor:
    """
    Evaluates the log pdf of a standard normal distribution at x.

    :param x: point to evaluate
    :return: log N(x|0,I)
    """
    return (-x**2 / 2 - 0.5 * math.log(2 * math.pi)).view(x.size(0), -1).sum(dim=1)  # was .sum(dim=-1)


def log_gaussian(x: Tensor, μ: Tensor, log_σ: Tensor) -> Tensor:
    """
    Returns the log pdf of a normal distribution parametrised
    by μ and log_σ evaluated at x.

    :param x: point to evaluate
    :param μ: mean of distribution
    :param log_σ: log σ of distribution
    :return: log N(x|µ,σ)
    """
    log_pdf = (log_σ * 2 + (x - μ)**2 / tr.exp(log_σ * 2) + math.log(2 * math.pi)) * (-0.5)
    return log_pdf.view(log_pdf.size(0), -1).sum(dim=1)  # was .sum(dim=-1)


def log_standard_categorical(p: Tensor) -> Tensor:
    """
    Calculates the cross entropy between a (one-hot) categorical vector
    and a standard (uniform) categorical distribution.

    :param p: one-hot categorical distribution
    :return: H(p, u)
    """
    # Uniform prior over y
    prior = tr.softmax(tr.ones_like(p), dim=1).to(dtype=p.dtype, device=p.device)
    cross_entropy = -tr.sum(p * tr.log(prior + δ), dim=1)
    return cross_entropy
