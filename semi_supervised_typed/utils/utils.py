from typing import Callable
import torch as tr
from torch import Tensor

δ = 1e-8


def enumerate_discrete(x: Tensor, y_dim: int) -> Tensor:
    """
    Generates a `torch.Tensor` of size batch_size x n_labels of
    the given label.

    Example: generate_label(2, 1, 3) #=> torch.Tensor([[0, 1, 0],
                                                       [0, 1, 0]])
    :param x: tensor with batch size to mimic
    :param y_dim: number of total labels
    :return variable
    """
    def batch(batch_size_: int, label: int) -> Tensor:
        labels = (tr.ones(batch_size_, 1) * label).to(dtype=tr.long)
        y = tr.zeros((batch_size_, y_dim))
        y.scatter_(1, labels, 1)
        return y.to(dtype=tr.long)

    batch_size = x.size(0)
    generated = tr.cat([batch(batch_size, i) for i in range(y_dim)])
    return generated.to(device=x.device, dtype=x.dtype)


def encode(label: int, k: int) -> Tensor:
    y = tr.zeros(k)
    if label < k:
        y[label] = 1
    return y


def onehot(k: int) -> Callable[[int], Tensor]:
    """
    Converts a number to its one-hot or 1-of-k representation
    vector.
    :param k: (int) length of vector
    :return: onehot function
    """
    return lambda label: encode(label, k)
