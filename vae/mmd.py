from typing import Dict, Callable
import copy
import torch as tr
from torch import Tensor

from .mmd_types import ModuleZToX


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


class MMDLoss(ModuleZToX):
    dtype: tr.dtype
    batch_size: int
    z_dists: Dict[int, str]

    def __init__(self, batch_size: int, z_dists: Dict[int, str]=None):
        """ Values of ``z_dists`` should be names of this class methods:
            'humps2', 'camel2',
            'standard_normal' (default for not set dims). """
        super(MMDLoss, self).__init__()
        self.batch_size = batch_size
        self.z_dists = z_dists if (z_dists is not None) else dict()

    def set_z_dists(self, z_dists: Dict[int, str]=None):
        self.z_dists = z_dists if (z_dists is not None) else dict()

    @staticmethod
    def _bimodal_normal(batch_size: int, z_dim: int, μ: float, σ: float, dtype: tr.dtype) -> Tensor:
        # sample -μ and μ:
        μ_ = tr.randint(0, 2, (batch_size, z_dim), dtype=dtype) * (2 * μ) - μ  # like range(0, 2)
        return σ * tr.randn(batch_size, z_dim) + μ_

    @staticmethod
    def humps2(batch_size: int, z_dim: int, dtype: tr.dtype) -> Tensor:
        """ μ ≃ ±1.4423, σ ≃ 0.5192, std ≃ 1.5329, 99.73% ≃ 3 """
        return MMDLoss._bimodal_normal(batch_size=batch_size, z_dim=z_dim,
                                       μ=1.4423073041362258, σ=0.519230898621258, dtype=dtype)

    @staticmethod
    def camel2(batch_size: int, z_dim: int, dtype: tr.dtype) -> Tensor:
        """ μ ≃ ±1.0714, σ ≃ 0.6429, std ≃ 1.2495, 99.73% ≃ 3 """
        return MMDLoss._bimodal_normal(batch_size=batch_size, z_dim=z_dim,
                                       μ=1.07142821441647, σ=0.6428572618611768, dtype=dtype)

    @staticmethod
    def standard_normal(batch_size: int, z_dim: int, dtype: tr.dtype) -> Tensor:
        return tr.randn(batch_size, z_dim, dtype=dtype)

    def mixed_sampler(self, z_dim: int, dtype: tr.dtype) -> Tensor:
        ret = tr.empty(self.batch_size, z_dim)
        dists = copy.deepcopy(self.z_dists)
        for i in range(z_dim):
            dists.setdefault(i, 'standard_normal')
        for attr in set(attr_ for ax, attr_ in dists.items()):
            sampler: Callable[[int, int, tr.dtype], Tensor] = getattr(self, attr)
            axes = [ax for ax, attr_ in dists.items() if attr_ == attr]
            samples = sampler(self.batch_size, len(axes), dtype)
            for i, ax in enumerate(axes):
                ret[:, ax] = samples[:, i]
        return ret

    @staticmethod
    def transition_dists_dict(z_dists: Dict[int, str]) -> Dict[int, str]:
        return {k: v.replace('camel2', 'standard_normal').replace('humps2', 'camel2')
                for k, v in z_dists.items()}

    def forward_(self, z: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        """ Returns scalar MMD loss. """
        z_dim = z.size(1)
        randn_samples = (self.mixed_sampler(z_dim=z_dim, dtype=z.dtype) if self.z_dists else
                         self.standard_normal(batch_size=self.batch_size, z_dim=z_dim, dtype=z.dtype))
        randn_samples = randn_samples.to(dtype=z.dtype, device=z.device)
        return mmd(randn_samples, z)


def test(samples_: Tensor) -> None:
    import numpy as np
    # noinspection PyPep8Naming
    from numpy import ndarray as Array
    import matplotlibhelper as mh
    mh.ready()
    import seaborn as sns
    import matplotlib.pyplot as plt
    from vae.utils import density_extremum

    print(samples_.mean(0), samples_.std(0))

    samples: Array = samples_.numpy()
    fig = plt.figure(figsize=mh.figsize(w=6))
    sns.distplot(samples[:, 0], bins=30)
    Y, X = np.histogram(samples[:, 0], bins=15)
    X = np.array([(X[i] + X[i - 1]) / 2 for i in range(1, len(X))])
    Y = Y / np.sum(Y)
    plt.plot(X, Y, marker='o', linestyle='')
    fig.tight_layout()

    dens_extr = density_extremum(samples)
    print(samples.shape, dens_extr.shape, dens_extr)
    mh.img(plt)


# bimodal = MMDLoss(1000, z_dists={0: 'humps2'})
# test(bimodal.standard_normal(1000, 3))
# test(bimodal.mixed_sampler(2, dtype=tr.float))
# test(bimodal.camel2(100000, 2))
# x = bimodal.mixed_sampler(100000, 6, {2: 'camel2', 4: 'camel2', 0: 'humps2', 5: 'humps2'})
# test(x[:, [3]])
