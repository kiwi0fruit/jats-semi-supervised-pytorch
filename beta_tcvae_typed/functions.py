from numbers import Number
import torch as tr
from torch import Tensor
from kiwi_bugfix_typechecker import math


def logsumexp(value: Tensor, dim: int=None, keepdim: bool=False) -> Tensor:
    """
    Code was taken from https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py

    Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim=True).log()
    """
    if dim is not None:
        m, _ = tr.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + tr.log(tr.sum(tr.exp(value0), dim=dim, keepdim=keepdim))

    m = tr.max(value)
    sum_exp = tr.sum(tr.exp(value - m))
    if isinstance(sum_exp, Number):
        return m + math.log(sum_exp)
    return m + tr.log(sum_exp)


def log_importance_weight_matrix(batch_size: int, dataset_size: int) -> Tensor:
    """
    Code was taken from https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py
    """
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = tr.empty(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M+1] = 1 / N
    W.view(-1)[1::M+1] = strat_weight
    W[M-1, 0] = strat_weight
    return W.log()
