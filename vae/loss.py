from typing import Tuple, Union, Sequence, Optional as Opt, List
from abc import abstractmethod
from torch import Tensor
from beta_tcvae_typed import Normal
from .loss_types import BaseBaseWeightedLoss, BaseTrimLoss


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


def kld_basis_strip(μ: Tensor, log_σ: Tensor, strip_scl: float=0,
                    basis: List[int]=None, kld: Tensor=None) -> Union[int, Tensor]:
    """
    kld_strip = 0 if strip_scl == 0 or not basis

    if μ size is (batch_size, feature_size_0, ...) then returns Tensor of the size (batch_size, ...).
    "..." in common case would be nothing.
    """
    if (strip_scl == 0) or not basis:
        return 0

    kld_mat = Normal.kl((μ, log_σ)) if (kld is None) else kld
    drop = [i for i in range(μ.size(1)) if i not in basis]
    return kld_mat[:, drop].view(μ.size(0), -1).sum(dim=1) * strip_scl


class TrimLoss(BaseTrimLoss):
    def __init__(self, strip_scl: float=0, basis: List[int]=None):
        """
        Presumably works really badly with normalizing flows for q(z|x).
        Should work OK with normalizing flows for p(z).

        ``KLD_strip.sum(1).mean()``

        KLD_strip is a basis strip via chery-picking particular latent dims and
        setting high enough KLD term for them (that is calculated via closed
        form assuming standard normal priors for the dimensions):

        ``KLD_strip(μ,log_σ; strip_scl,basis) = strip_scl * KLD(μ,log_σ)[:, not_basis]``

        ``KLD(μ,log_σ) = (log_σ * 2 + 1 - μ**2 - exp(log_σ * 2)) * (-0.5)``

        KLD_strip loss affects only deterministic encoder behaviour (q_dist)
        hence presumably ELBO would stay ELBO.
        """
        super(TrimLoss, self).__init__()
        self.strip_scl = strip_scl
        self.basis = basis

    def set_strip_scl(self, strip_scl: float) -> None:
        self.strip_scl = strip_scl

    def set_basis(self, basis: List[int]=None):
        self.basis = basis

    def forward_(self, μ: Tensor, log_σ: Tensor, kld: Tensor=None) -> Union[Tensor, int]:
        basis_strip = kld_basis_strip(μ=μ, log_σ=log_σ, strip_scl=self.strip_scl, basis=self.basis, kld=kld)
        if isinstance(basis_strip, int):
            return basis_strip
        return basis_strip.mean()
