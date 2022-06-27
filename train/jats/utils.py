# import torch as tr
from torch import Tensor


def probs_temper(probs: Tensor) -> Tensor:
    """ (EP/-IR, IJ/+IR, IP/-ER, EJ/+ER) """
    probs_ = probs.clone()
    masks = [probs == i for i in range(16)]
    probs_[masks[1 - 1] | masks[5 - 1] | masks[9 - 1] | masks[13 - 1]] = 0
    probs_[masks[2 - 1] | masks[6 - 1] | masks[10 - 1] | masks[14 - 1]] = 1
    probs_[masks[3 - 1] | masks[7 - 1] | masks[11 - 1] | masks[15 - 1]] = 2
    probs_[masks[4 - 1] | masks[8 - 1] | masks[12 - 1] | masks[16 - 1]] = 3
    return probs_


def probs_quadraclub(probs: Tensor) -> Tensor:
    """ (NTL, SFL, STC, NFC, SFC, NTC, NFL, STL) """
    probs_ = probs.clone()
    masks = [probs == i for i in range(16)]
    probs_[masks[1 - 1] | masks[2 - 1]] = 0
    probs_[masks[3 - 1] | masks[4 - 1]] = 1
    probs_[masks[5 - 1] | masks[6 - 1]] = 2
    probs_[masks[7 - 1] | masks[8 - 1]] = 3
    probs_[masks[9 - 1] | masks[10 - 1]] = 4
    probs_[masks[11 - 1] | masks[12 - 1]] = 5
    probs_[masks[13 - 1] | masks[14 - 1]] = 6
    probs_[masks[15 - 1] | masks[16 - 1]] = 7
    return probs_


def expand_quadraclub(probs: Tensor) -> Tensor:
    """ (NTL, SFL, STC, NFC, SFC, NTC, NFL, STL) """
    n, m = probs.shape
    return probs.view(n, m, 1).expand(n, m, 2).reshape(n, m * 2)


def expand_temper(probs: Tensor) -> Tensor:
    """ (EP/-IR, IJ/+IR, IP/-ER, EJ/+ER) """
    n, m = probs.shape
    return probs.view(n, 1, m).expand(n, 4, m).reshape(n, m * 4)


def expand_temper_to_stat_dyn(probs: Tensor) -> Tensor:
    """ (EP/-IR, IJ/+IR, IP/-ER, EJ/+ER) """
    n, m = probs.shape[0], probs.shape[1] // 2
    probs_ = probs.view(n, m, 2).sum(dim=-1).view(n, m, 1).expand(n, m, 2).reshape(n, m * 2)
    n, m = probs_.shape
    return probs_.view(n, 1, m).expand(n, 4, m).reshape(n, m * 4)
