from typing import Tuple, List
import torch as tr
from torch import Tensor, nn
from beta_tcvae_typed import Normal, Laplace, Distrib

from .stochastic_types import BaseGaussianMerge, ModuleXToXTupleYi, XTupleYi

δ = 1e-8


# noinspection PyAbstractClass
class BaseSample(ModuleXToXTupleYi):  # pylint: disable=abstract-method
    # noinspection PyUnusedLocal
    def __init__(self, in_features: int, out_features: int):  # pylint: disable=unused-argument
        """
        Base stochastic layer that defines interfaces.
        """
        super(BaseSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features


class GaussianSample(BaseSample):
    def __init__(self, in_features: int, out_features: int):
        """
        Layer that represents a sample from a
        Gaussian distribution.

        Base stochastic layer that uses the
        reparametrization trick [Kingma 2013]
        to draw a sample from a distribution
        parametrised by μ and log_σ.
        """
        super(GaussianSample, self).__init__(in_features, out_features)
        self.μ = nn.Linear(in_features, out_features)
        self.log_σ = nn.Linear(in_features, out_features)

    def μ_log_σ(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        μ = self.μ.__call__(x)
        log_σ = self.log_σ.__call__(x)
        return μ, log_σ

    def forward_(self, x: Tensor) -> XTupleYi:
        μ, log_σ = self.μ_log_σ(x)
        return self.reparametrize(μ, log_σ=log_σ), (μ, log_σ)

    @staticmethod
    def reparametrize(μ: Tensor, log_σ: Tensor) -> Tensor:
        ε = tr.randn(μ.size(), dtype=μ.dtype, device=μ.device)
        z = tr.exp(log_σ) * ε + μ
        return z


class GaussianMerge(BaseGaussianMerge):
    def __init__(self, in_features: int, out_features: int):
        """
        Precision weighted merging of two Gaussian
        distributions.
        Merges information from z into the given
        mean and log variance and produces
        a sample from this new distribution.
        """
        super(GaussianMerge, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.μ = nn.Linear(in_features, out_features)
        self.log_σ = nn.Linear(in_features, out_features)

    def forward_(self, z: Tensor, μ0: Tensor, log_σ0: Tensor) -> XTupleYi:
        # Calculate precision of each distribution
        # (inverse variance)
        μ1 = self.μ(z)
        log_σ1 = self.log_σ.__call__(z)
        precision1, precision2 = (tr.exp(log_σ0 * 2)**-1, tr.exp(log_σ1 * 2)**-1)

        # Merge distributions into a single new distribution
        μ = ((μ0 * precision1) + (μ1 * precision2)) / (precision1 + precision2)

        var = (precision1 + precision2)**-1
        log_σ = tr.log(var + δ) / 2

        return GaussianSample.reparametrize(μ, log_σ), (μ, log_σ)


class GumbelSoftmax(BaseSample):
    def __init__(self, in_features: int, out_features: int, n_distributions: int, τ: float=1.0):
        """
        Layer that represents a sample from a categorical
        distribution. Enables sampling and stochastic
        backpropagation using the Gumbel-Softmax trick.
        """
        super(GumbelSoftmax, self).__init__(in_features=in_features, out_features=out_features)
        self.n_distributions = n_distributions
        self.logits = nn.Linear(in_features, n_distributions * out_features)
        self.τ = τ

    def set_τ(self, τ: float=1.0):
        self.τ = τ

    def forward_(self, x: Tensor) -> XTupleYi:
        logits = self.logits.__call__(x).view(-1, self.n_distributions)

        # variational distribution over categories
        probs = tr.softmax(logits, dim=-1)  # q_y
        sample = self.reparametrize(probs, self.τ).view(-1, self.n_distributions, self.out_features).mean(dim=1)
        return sample, (probs,)

    @staticmethod
    def reparametrize(probs: Tensor, τ: float=1.0) -> Tensor:
        ε = tr.rand(probs.size(), dtype=probs.dtype, device=probs.device)
        # Gumbel distributed noise
        gumbel = -tr.log(-tr.log(ε + δ) + δ)
        # Softmax as a continuous approximation of argmax
        y = tr.softmax((tr.log(probs + δ) + gumbel) / τ, dim=1)
        return y


class Sample(BaseSample):
    params: List[nn.Linear]
    dist: Distrib

    def __init__(self, in_features: int, out_features: int, dist: Distrib=Normal()):
        super(Sample, self).__init__(in_features, out_features)
        params: List[nn.Linear] = [nn.Linear(in_features, out_features) for _ in range(dist.nparams)]
        # noinspection PyTypeChecker
        self.params = nn.ModuleList(params)  # type: ignore
        self.dist = dist

    def forward_(self, x: Tensor) -> XTupleYi:
        params = tuple(param.__call__(x) for param in self.params)
        z = self.dist.rsample(params=params)
        return z, params


class NormalSample(Sample):
    def __init__(self, in_features: int, out_features: int):
        super(NormalSample, self).__init__(in_features=in_features, out_features=out_features, dist=Normal())


class LaplaceSample(Sample):
    def __init__(self, in_features: int, out_features: int):
        super(LaplaceSample, self).__init__(in_features=in_features, out_features=out_features, dist=Laplace())
