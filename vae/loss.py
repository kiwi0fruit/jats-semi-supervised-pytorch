from typing import Tuple, Union, Sequence, Optional as Opt
from abc import abstractmethod
import math
import torch as tr
from torch import Tensor
from kiwi_bugfix_typechecker import test_assert
from beta_tcvae_typed import Normal
from .loss_types import BaseBaseWeightedLoss, BaseTrimLoss

test_assert()


class BaseWeightedLoss(BaseBaseWeightedLoss):
    x_params_multiplier: int = 1
    features_size: Opt[Tuple[int, ...]] = None
    classes: Union[None, int, Sequence[float]] = None
    reshape_size: Opt[Tuple[int, ...]] = None

    # noinspection PyMissingConstructor
    def __init__(self, classes: Union[int, Sequence[float]]=None, features_size: Tuple[int, ...]=None):
        """
        See the second parent class documentation.
        Abstract class for losses for x from [0, 1] line segment.
        Intended to be the first parent class and __init__ should call the second class constructor.
        The second class should be a child of the torch.nn.Module (and inherit __call__ method).

        :param classes: Number of classes / buckets or buckets values.
            If L == classes or L == len(classes) then should be L > 1.
            Reshape to (batch_size, L, features_size_0, ...) if (classes is not None)
            else (batch_size, features_size_0, ...).
        :param features_size: tuple of ints used for reshaping output. For example (100, -1) or (100, 4).
            Then it would be (batch_size, L, 100, -1) if (classes is not None) else (batch_size, 100, -1).
            If None then no reshaping except for classes:
            (batch_size, L, -1) if (classes is not None) else (batch_size, -1) aka no reshape
        """
        if classes is None:
            pass
        elif isinstance(classes, int):
            if classes < 2:
                raise ValueError(f'Bad classes value: {classes} < 2')
            self.x_params_multiplier = classes
        else:
            if len(classes) < 2:
                raise ValueError(f'Bad classes value: {classes}; len(classes) < 2')
            self.x_params_multiplier = len(classes)

        self.features_size = features_size
        self.classes = classes

        if features_size is None:
            self.reshape_size = None if (self.x_params_multiplier == 1) else (self.x_params_multiplier, -1)
        else:
            self.reshape_size = features_size if (self.x_params_multiplier == 1) else (
                    (self.x_params_multiplier,) + features_size)

    def reshape(self, x: Tensor) -> Tensor:
        if self.reshape_size is None:
            return x
        return x.view(x.size(0), *self.reshape_size)

    @abstractmethod
    def forward_(self, x_params: Tensor, target: Tensor, weight: Tensor=None) -> Tensor:
        """
        "..." in common case would be nothing.

        :param x_params: flat Tensor of size (batch_n, x_params_multiplier * features_size_0 * ...)
            that is the output of the nn.Linear. It would be reshaped to
            (batch_size, x_params_multiplier, features_size_0, ...) if (x_params_multiplier > 1) else
            (batch_size, features_size_0, ...)
        :param target: of the size (batch_size, features_size_0, ...)
        :param weight: of the size (batch_size, features_size_0, ...)
        :return: Tensor of the size (batch_size,)
        """
        raise NotImplementedError

    @abstractmethod
    def x_prob_params(self, x_params: Tensor) -> Tensor:
        """
        :param x_params: flat Tensor of size (batch_size, x_params_multiplier * features_size_0 * ...)
            that is the output of the nn.Linear
        :return: distribution parameters of size (batch_size, x_params_multiplier, features_size_0, ...)
            if (x_params_multiplier > 1) else (batch_size, features_size_0, ...)
        """
        raise NotImplementedError

    @abstractmethod
    def x_recon(self, x_params: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param x_params: flat Tensor of size (batch_size, x_params_multiplier * features_size_0 * ...)
            that is the output of the nn.Linear
        :return: tuple (x_rec_continuous, x_rec_sample). Both are of size (batch_size, features_size_0, ...)
        """
        raise NotImplementedError


class TrimLoss(BaseTrimLoss):
    def __init__(self, μ_scl: float=0, σ_scl: float=0, max_abs_μ: float=3, inv_min_σ: float=5 * 6,
                 basis_scl: float=0, anti_basis: Tuple[int, ...]=(), inv_max_σ: float=1,
                 μ_norm_scl: float=0, μ_norm_std: float=1):
        """
        Presumably works really badly with normalizing flows for q(z|x).
        Should work OK with normalizing flows for p(z).

        ``ReLU_trim.sum(1).mean()``

        μ and σ trim of the q_dist via special ReLU Trim Penalty (ReLU_trim).

        ``ReLU_trim(μ,log_σ; μ_scl,σ_scl,μ_thr,σ_thr) =``

        ``μ_scl * (ReLU(μ - max_abs_μ) + ReLU(-μ - max_abs_μ))``

        ``+ σ_scl * ReLU(-log_σ + log(1/inv_min_σ))``

        ReLU_trim loss affects only deterministic encoder behaviour (q_dist)
        hence ELBO would stay ELBO.
        """
        super(TrimLoss, self).__init__()

        self.μ_scl = μ_scl
        self.use_μ = self.μ_scl > 0
        assert max_abs_μ > 0
        self.max_abs_μ = max_abs_μ

        self.σ_scl = σ_scl
        self.use_σ = self.σ_scl > 0
        self.min_log_σ = math.log(inv_min_σ**-1)
        self.max_log_σ = math.log(inv_max_σ**-1)

        self.basis_scl = basis_scl
        self.anti_basis = anti_basis

        self.μ_norm_scl = μ_norm_scl
        self.μ_norm_std = μ_norm_std
        self.use_μ_norm = self.μ_norm_scl > 0

    def set_anti_basis(self, anti_basis: Tuple[int, ...]):
        self.anti_basis = anti_basis

    def set_μ_scl(self, μ_scl: float) -> None:
        self.μ_scl = μ_scl
        self.use_μ = self.μ_scl > 0

    def forward_(self, μ: Tensor, log_σ: Tensor) -> Union[Tensor, int]:
        max_μ = self.max_abs_μ
        ret: Union[Tensor, int]
        if self.use_μ and self.use_σ:
            ret = (
                    (tr.relu(μ - max_μ) + tr.relu(-μ - max_μ)) * self.μ_scl
                    + (tr.relu(-log_σ + self.min_log_σ) + tr.relu(log_σ - self.max_log_σ)) * self.σ_scl
            ).view(μ.size(0), -1).sum(dim=1).mean()
        elif self.use_σ:
            ret = (tr.relu(-log_σ + self.min_log_σ) + tr.relu(log_σ - self.max_log_σ)
                   ).view(μ.size(0), -1).sum(dim=1).mean() * self.σ_scl
        elif self.use_μ:
            ret = (tr.relu(μ - max_μ) + tr.relu(-μ - max_μ)).view(μ.size(0), -1).sum(dim=1).mean() * self.μ_scl
        else:
            ret = 0

        if self.anti_basis:
            kld_mat = Normal.kl((μ, log_σ))
            ret = kld_mat[:, self.anti_basis].view(μ.size(0), -1).sum(dim=1).mean() * self.basis_scl + ret

        if self.use_μ_norm:
            ret = (μ.std() - self.μ_norm_std)**2 * self.μ_norm_scl + ret
        return ret
