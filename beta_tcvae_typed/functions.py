import torch as tr
from torch import Tensor


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
