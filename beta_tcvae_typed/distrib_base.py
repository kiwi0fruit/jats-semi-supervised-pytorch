# noinspection PyPep8Naming
from typing import Union, Tuple, Optional as Opt, List
from abc import abstractmethod
from torch import Tensor, Size
import torch as tr
from normalizing_flows_typed import Flow, SequentialFlow
from .distrib_base_type import DistribBase

Flows = Union[Flow, SequentialFlow]
SizeType = Union[Size, List[int], Tuple[int, ...]]


def kernel(x: Tensor, y: Tensor) -> Tensor:
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) * (1. / dim)
    return tr.exp(-kernel_input)  # (x_size, y_size)


def mmd(x: Tensor, y: Tensor) -> Tensor:
    return kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean()


class Distrib(DistribBase):
    inv_pz_flow: Opt[Flows]
    min_mmd_batch_size: int = 128

    def __init__(self):
        super(Distrib, self).__init__()
        self.inv_pz_flow = None

    @property
    @abstractmethod
    def nparams(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_reparameterizable(self) -> bool:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def sample(params: Tuple[Tensor, ...]) -> Tensor:
        """ Samples x of the same dimension as the size of the first parameter.
            (Assumes that all parameters have the same size). """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def rsample(params: Tuple[Tensor, ...]) -> Tensor:
        """ Samples x of the same dimension as the size of the first parameter.
            (Assumes that all parameters have the same size). """
        raise NotImplementedError

    @abstractmethod
    def log_p(self, x: Tensor, p_params: Tuple[Tensor, ...]=None) -> Tensor:
        """ returns unreduced log probability without flow density.
            If p_params==None then distribution used DOES NOT depend on class instance parameters. """
        raise NotImplementedError

    def log_prob(self, x: Tensor, p_params: Tuple[Tensor, ...]=None) -> Tensor:
        """ returns unreduced log probability without flow density.
            If p_params==None then distribution used DOES depend on class instance parameters. """
        p_params_ = p_params if (p_params is not None) else self.get_params(x.size())
        return self.log_p(x, p_params_)

    def forward_(self, x: Tensor, p_params: Tuple[Tensor, ...]=None) -> Tensor:
        """ returns unreduced log probability without flow density. """
        return self.log_prob(x, p_params)

    @staticmethod
    @abstractmethod
    def kl(q_params: Tuple[Tensor, ...], p_params: Tuple[Tensor, ...]=None) -> Tensor:
        """
        Computes KL(q||p). Default p is the standard distribution family
        (DOES NOT use parameters of the class instance).
        """
        raise NotImplementedError

    @abstractmethod
    def get_params(self, size: Size) -> Tuple[Tensor, ...]:
        raise NotImplementedError

    def kld(self, q_params: Tuple[Tensor, ...], p_params: Tuple[Tensor, ...]=None) -> Tensor:
        """
        Computes KL(q||p). Uses parameters of the class instance as p_params for defaults.
        """
        p_params_ = p_params if (p_params is not None) else self.get_params(q_params[0].size())
        return self.kl(q_params, p_params_)

    def log_prob_reduced(self, x: Tensor, p_params: Tuple[Tensor, ...]=None) -> Tensor:
        """ returns reduced log probability with flow density added. """

        batch_size = x.shape[0]
        if self.inv_pz_flow is None:
            return self.log_prob(x=x, p_params=p_params).view(batch_size, -1).sum(dim=1)

        # sladetj := sum log abs det jacobian
        # log_pz0 == log_pzk - sladetj_inv
        zk = x
        z0, sladetj_inv = self.inv_pz_flow.__call__(zk)
        sladetj_inv = sladetj_inv.view(batch_size, -1).sum(dim=1)
        log_pz0 = self.log_prob(x=z0, p_params=p_params).view(batch_size, -1).sum(dim=1)

        log_pzk = log_pz0 + sladetj_inv
        return log_pzk

    def set_inv_pz_flow(self, flow: Flows=None):
        self.inv_pz_flow = flow

    @abstractmethod
    def nll(self, params: Tuple[Tensor, ...], sample_params: Tuple[Tensor, ...]=None) -> Tensor:
        raise NotImplementedError

    def mmd(self, z: Tensor) -> Tensor:
        batch_size, z_dim = z.shape
        batch_size = max(batch_size, self.min_mmd_batch_size)
        return mmd(self.rsample(self.get_params(tr.Size((batch_size, z_dim)))), z)


def test_rsample(x: Tensor) -> None:
    import numpy as np
    # noinspection PyPep8Naming
    from numpy import ndarray as Array
    import matplotlibhelper as mh
    mh.ready()
    import seaborn as sns
    import matplotlib.pyplot as plt

    x = x.detach()
    print(x.mean(0), x.std(0))

    samples: Array = x.numpy()
    fig = plt.figure(figsize=mh.figsize(w=6))
    sns.distplot(samples[:, 0], bins=30)
    Y, X = np.histogram(samples[:, 0], bins=15)
    X = np.array([(X[i] + X[i - 1]) / 2 for i in range(1, len(X))])
    Y = Y / np.sum(Y)
    plt.plot(X, Y, marker='o', linestyle='')
    fig.tight_layout()

    print(samples.shape)
    mh.img()
