# noinspection PyPep8Naming
from typing import Tuple, Union, List
import math
import numpy as np
import torch as tr
from torch.nn import Parameter
from torch import Tensor, Size
from kiwi_bugfix_typechecker import test_assert
from .distrib_base import Distrib

δ = 1e-8
SizeType = Union[Size, List[int], Tuple[int, ...]]
test_assert()


class Normal(Distrib):
    normalization: Tensor
    μ: Tensor
    log_σ: Tensor

    def __init__(self, μ: float=0, σ: float=1):
        """
        RSamples from a Normal distribution using the reparameterization trick.

        ``params`` for methods should contain ``μ`` and ``log_σ``
        """
        super(Normal, self).__init__()
        self.register_buffer('normalization', tr.tensor([np.log(2 * np.pi)]))
        self.register_buffer('μ', tr.tensor([float(μ)]))
        self.register_buffer('log_σ', tr.tensor([math.log(σ)]))

    @staticmethod
    def rsample(params: Tuple[Tensor, ...]) -> Tensor:
        """ params should contain ``μ`` and ``log_σ`` """
        μ, log_σ = params
        ε = tr.randn(μ.size()).to(dtype=μ.dtype, device=μ.device)
        return tr.exp(log_σ) * ε + μ

    @staticmethod
    def sample(params: Tuple[Tensor, ...]) -> Tensor:
        """ params should contain ``μ`` and ``log_σ`` """
        with tr.no_grad():
            μ, log_σ = params
            ret = tr.normal(μ, log_σ.exp())
        return ret

    def log_p(self, x: Tensor, p_params: Tuple[Tensor, ...]=None) -> Tensor:
        """ Computes log p(x). Default p_params is standard normal distribution. """
        if p_params is None:
            return (x**2 + self.normalization) * (-0.5)
        μ, log_σ = p_params
        return (log_σ * 2 + (x - μ)**2 / tr.exp(log_σ * 2) + self.normalization) * (-0.5)

    def get_params(self, size: Size) -> Tuple[Tensor, ...]:
        μ = self.μ.expand(size)
        log_σ = self.log_σ.expand(size)
        return μ, log_σ

    def nll(self, params: Tuple[Tensor, ...], sample_params: Tuple[Tensor, ...]=None) -> Tensor:
        """
        Analytically computes
            E_N(mu_2,sigma_2^2) [ - log N(mu_1, sigma_1^2) ]
        If mu_2, and sigma_2^2 are not provided, defaults to entropy.

        Params and sample_params should be of the same size.
        """
        μ, log_σ = params
        if sample_params is not None:
            sample_μ, sample_log_σ = sample_params
        else:
            sample_μ, sample_log_σ = params

        c = self.normalization
        nll_ = ((log_σ * 2).exp() * (sample_μ - μ)**2
                + tr.exp(sample_log_σ * 2 - log_σ * 2)
                + 2 * log_σ + c) * 0.5
        return nll_

    @staticmethod
    def kl(q_params: Tuple[Tensor, ...], p_params: Tuple[Tensor, ...]=None) -> Tensor:
        """
        Computes KL(q||p). Default p is the standard Normal distribution
        (μ = log_σ = 0). DOES NOT use parameters of the class instance.
        """
        q_μ, q_log_σ = q_params
        if p_params is None:
            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(σ^2) - μ^2 - σ^2)
            return (q_log_σ * 2 + 1 - q_μ**2 - tr.exp(q_log_σ * 2)) * (-0.5)

        # https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html
        # @register_kl(Normal, Normal)
        # def _kl_normal_normal(p, q):
        # mind that here is KL(q||p)
        p_μ, p_log_σ = p_params
        log_var_ratio = (p_log_σ - q_log_σ) * 2
        return (log_var_ratio.exp() + (q_μ - p_μ)**2 / tr.exp(p_log_σ * 2) - 1 - log_var_ratio) * 0.5

    @property
    def nparams(self):
        return 2

    @property
    def is_reparameterizable(self):
        return True

    def __repr__(self):
        return self.__class__.__name__ + ' (μ={:.3f}, σ={:.3f})'.format(self.μ[0].item(), self.log_σ.exp()[0].item())


class ZeroSymmetricBimodalNormal(Normal):
    mode: int
    subdims_from_beginning: bool

    def __init__(self,
                 z_dim: int=None,
                 μ: Union[float, Tuple[float, ...]]=None,
                 σ: Union[float, Tuple[float, ...]]=None,
                 learnable_μ: bool=False,
                 learnable_σ: bool=False,
                 mode: int=2,
                 subdims_from_beginning: bool=False):
        """
        RSamples from a zero symmetric bimodal normal distribution
        using the reparameterization trick.

        Assumes linear z_dim!

        Defaults ``μ = 0`` and ``σ = 1`` would lead to unimodal normal distribution.

        For all ``μ``: ``μ >= 0`` (for each dimension both modes have same ``σ`` std and ``±μ`` mean).

        If z_dim is None then scalar parameters are assumed instead of tuples.

        if ``mode`` is 1 then all distributions are assumed to be shifted unimodal.

        When params size (batch_size, z_dim_1) provided is less than distrib. size (batch_size, z_dim_0)
        then only subdim of the z_dim_0 is used. ``subdims_from_beginning`` governs should subdim be placed
        at the beginning or the end of the z_dim_0.
        """
        super(ZeroSymmetricBimodalNormal, self).__init__()

        if z_dim is None:
            assert ((μ is None) or isinstance(μ, (int, float))) and ((σ is None) or isinstance(σ, (int, float)))
            μ_0 = float(μ) if isinstance(μ, (int, float)) else 0.
            log_σ_0 = math.log(σ) if isinstance(σ, (int, float)) else 0.
            _μ, _log_σ = tr.tensor([μ_0]), tr.tensor([log_σ_0])

        else:
            assert ((μ is None) or isinstance(μ, tuple)) and ((σ is None) or isinstance(σ, tuple))
            μ_ = [float(μi) for μi in μ] if isinstance(μ, tuple) else tuple(0. for _ in range(z_dim))
            log_σ_ = [math.log(σi) for σi in σ] if isinstance(σ, tuple) else tuple(0. for _ in range(z_dim))
            assert μ_ and log_σ_ and (len(μ_) == len(log_σ_) == z_dim)
            _μ, _log_σ = tr.tensor(μ_), tr.tensor(log_σ_)

        del self.μ
        del self.log_σ
        assert mode in (1, 2)
        self.mode = mode
        self.subdims_from_beginning = subdims_from_beginning

        if learnable_μ:
            self.μ = Parameter(_μ, requires_grad=True)
        else:
            self.register_buffer('μ', _μ)
        if learnable_σ:
            self.log_σ = Parameter(_log_σ, requires_grad=True)
        else:
            self.register_buffer('log_σ', _log_σ)

    @staticmethod
    def rsample(params: Tuple[Tensor, ...]) -> Tensor:
        """ params should contain ``μ`` and ``log_σ`` """
        μ, log_σ = params
        # sample 0 and 1:
        bit = tr.randint(0, 2, μ.size(), dtype=μ.dtype, device=μ.device)  # like range(0, 2)
        # sample from standard normal dist:
        ε = tr.randn(μ.size(), dtype=μ.dtype, device=μ.device)
        return tr.exp(log_σ) * ε + μ * 2 * bit - μ

    @staticmethod
    def sample(params: Tuple[Tensor, ...]) -> Tensor:
        """ params should contain ``μ`` and ``log_σ`` """
        with tr.no_grad():
            ret = ZeroSymmetricBimodalNormal.rsample(params)
        return ret

    def log_p1(self, x: Tensor, μ: Tensor, log_σ: Tensor) -> Tensor:
        return (log_σ * 2 + (x - μ)**2 / tr.exp(log_σ * 2) + self.normalization) * (-0.5)

    def log_p2(self, x: Tensor, μ: Tensor, log_σ: Tensor) -> Tensor:
        a = (log_σ * 2 + self.normalization) * (-0.5)
        b = tr.exp(log_σ * 2) * (-2)
        log_p_pos_μ = a + (x - μ)**2 / b
        log_p_neg_μ = a + (x + μ)**2 / b
        return tr.logsumexp(tr.stack((log_p_pos_μ, log_p_neg_μ)), dim=0) - math.log(2)

    def log_p(self, x: Tensor, p_params: Tuple[Tensor, ...]=None) -> Tensor:
        if p_params is None:
            raise NotImplementedError
        μ, log_σ = p_params
        if self.mode == 2:
            return self.log_p2(x, μ, log_σ)
        if self.mode == 1:
            return self.log_p1(x, μ, log_σ)
        raise RuntimeError('Unknown bug.')

    def μ__log_σ(self) -> Tuple[Tensor, Tensor]:
        return self.μ, self.log_σ

    def get_params(self, size: Size) -> Tuple[Tensor, ...]:
        μ, log_σ = self.μ__log_σ()
        z_dim = size[-1]
        if (μ.shape[0] > 1) and (μ.shape[0] > z_dim):
            if self.subdims_from_beginning:
                μ, log_σ = μ[:z_dim], log_σ[:z_dim]
            else:
                μ, log_σ = μ[-z_dim:], log_σ[-z_dim:]
        μ = tr.zeros(*size, dtype=μ.dtype, device=μ.device) + μ
        log_σ = tr.zeros(*size, dtype=log_σ.dtype, device=log_σ.device) + log_σ
        return μ, log_σ

    def nll(self, params: Tuple[Tensor, ...], sample_params: Tuple[Tensor, ...]=None) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def kl(q_params: Tuple[Tensor, ...], p_params: Tuple[Tensor, ...]=None) -> Tensor:
        raise NotImplementedError

    def __repr__(self):
        assert (len(self.μ.shape) == 1) and (len(self.log_σ.shape) == 1)
        if (self.μ.shape[0] == 1) and (self.log_σ.shape[0] == 1):
            return self.__class__.__name__ + f' (μ={self.μ[0].item():.3f}, σ={self.log_σ.exp()[0].item():.3f})'

        μ: List[Tensor] = list(self.μ)  # type: ignore
        log_σ: List[Tensor] = list(self.log_σ)  # type: ignore

        μ_ = ', '.join(f'{μi.item():.3f}' for μi in μ)
        σ = ', '.join(f'{math.exp(log_σi.item()):.3f}' for log_σi in log_σ)
        return f'{self.__class__.__name__} (μ=[{μ_}], σ=[{σ}])'


class ZeroSymmetricBimodalNormalTwin(ZeroSymmetricBimodalNormal):
    twin_dist: Tuple[ZeroSymmetricBimodalNormal]

    def set_twin_dist(self, dist: ZeroSymmetricBimodalNormal):
        self.twin_dist = (dist,)

    def μ__log_σ(self) -> Tuple[Tensor, Tensor]:
        twin_dist = self.twin_dist[0]
        return twin_dist.μ * self.μ, twin_dist.log_σ

    def nll(self, params: Tuple[Tensor, ...], sample_params: Tuple[Tensor, ...]=None) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def kl(q_params: Tuple[Tensor, ...], p_params: Tuple[Tensor, ...]=None) -> Tensor:
        raise NotImplementedError


class Laplace(Distrib):
    normalization: Tensor
    μ: Tensor
    logscale: Tensor

    def __init__(self, μ: float=0, scale: float=1, learnable_μ: bool=False,
                 learnable_scale: bool=False):
        """
        RSamples from a Laplace distribution using the reparameterization trick.

        ``params`` for methods should contain ``μ`` and ``logscale``
        """
        super(Laplace, self).__init__()
        self.register_buffer('normalization', tr.tensor([-math.log(2)]))
        _μ, _logscale = tr.tensor([float(μ)]), tr.tensor([math.log(scale)])
        if learnable_μ:
            self.μ = Parameter(_μ, requires_grad=True)
        else:
            self.register_buffer('μ', _μ)
        if learnable_scale:
            self.logscale = Parameter(_logscale, requires_grad=True)
        else:
            self.register_buffer('logscale', _logscale)

    @staticmethod
    def rsample(params: Tuple[Tensor, ...]) -> Tensor:
        """ params should contain ``μ`` and ``logscale`` """
        μ, logscale = params
        # Unif(-0.5, 0.5)
        u = tr.rand(μ.size()).to(dtype=μ.dtype, device=μ.device) - 0.5
        sample = μ - tr.exp(logscale) * tr.sign(u) * tr.log((-2) * tr.abs(u) + 1 + δ)
        return sample

    @staticmethod
    def sample(params: Tuple[Tensor, ...]) -> Tensor:
        """ params should contain ``μ`` and ``logscale`` """
        with tr.no_grad():
            ret = Laplace.rsample(params)
        return ret

    def log_p(self, x: Tensor, p_params: Tuple[Tensor, ...]=None) -> Tensor:
        """
        Computes log p(x).
        Default p is standard Laplace distribution (μ = logscale = 0). """
        if p_params is None:
            return -tr.abs(x) + self.normalization
        μ, logscale = p_params
        return -tr.abs(x - μ) * tr.exp(-logscale) - logscale + self.normalization

    def get_params(self, size: Size) -> Tuple[Tensor, ...]:
        μ = self.μ.expand(size)
        logscale = self.logscale.expand(size)
        return μ, logscale

    def nll(self, params: Tuple[Tensor, ...], sample_params: Tuple[Tensor, ...]=None) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def kl(q_params: Tuple[Tensor, ...], p_params: Tuple[Tensor, ...]=None) -> Tensor:
        """
        Computes KL(q||p). Default p is the standard Laplace distribution
        (μ = logscale = 0). DOES NOT use parameters of the class instance.
        """
        q_μ, q_log_σ = q_params
        if p_params is None:
            return -q_log_σ + q_μ.abs() + tr.exp(q_log_σ - q_μ.abs() / q_log_σ.exp()) - 1

        # https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html
        # @register_kl(Laplace, Laplace)
        # def _kl_laplace_laplace(p, q):
        # mind that here is KL(q||p)
        # From http://www.mast.queensu.ca/~communications/Papers/gil-msc11.pdf
        p_μ, p_log_σ = p_params
        log_scale_ratio = q_log_σ - p_log_σ
        abs_Δμ = (q_μ - p_μ).abs()
        return -log_scale_ratio + abs_Δμ / p_log_σ.exp() + tr.exp(log_scale_ratio - abs_Δμ / q_log_σ.exp()) - 1

    @property
    def nparams(self):
        return 2

    @property
    def is_reparameterizable(self):
        return True

    def __repr__(self):
        return self.__class__.__name__ + ' (μ={:.3f}, scale={:.3f})'.format(
            self.μ[0].item(), self.logscale.exp()[0].item())
