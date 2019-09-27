# noinspection PyPep8Naming
from typing import Tuple, Union, List
import math
import numpy as np
import torch as tr
from torch import Tensor, Size
from .distrib_base import Distrib

δ = 1e-8
SizeType = Union[Size, List[int], Tuple[int, ...]]


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

    def check_inputs(self, params: Tensor=None, size: SizeType=None) -> Tuple[Tensor, Tensor]:
        """ μ_log_σ = params """
        μ_log_σ = params

        if (μ_log_σ is not None) and (size is None):
            μ = μ_log_σ.select(-1, 0)
            log_σ = μ_log_σ.select(-1, 1)
            return μ, log_σ
        if (size is not None) and (μ_log_σ is not None):
            μ = μ_log_σ.select(-1, 0).expand(size)
            log_σ = μ_log_σ.select(-1, 1).expand(size)
            return μ, log_σ
        if size is not None:
            μ = self.μ.expand(size)
            log_σ = self.log_σ.expand(size)
            return μ, log_σ
        if (size is None) and (μ_log_σ is None):
            raise ValueError('Either one of size or params should be provided.')
        raise ValueError(f'Given invalid inputs: (size={size}, μ__log_σ={μ_log_σ})')

    def _rsample(self, params: Tensor=None, size: SizeType=None) -> Tensor:
        """ params should contain ``μ`` and ``log_σ`` """
        μ, log_σ = self.check_inputs(params=params, size=size)
        ε = tr.randn(μ.size()).to(dtype=μ.dtype, device=μ.device)
        return tr.exp(log_σ) * ε + μ

    def _sample(self, params: Tensor=None, size: SizeType=None) -> Tensor:
        """ params should contain ``μ`` and ``log_σ`` """
        with tr.no_grad():
            μ, log_σ = self.check_inputs(params=params, size=size)
            return tr.normal(μ, log_σ.exp())

    def log_prob(self, sample: Tensor, params: Tensor=None) -> Tensor:
        """ returns unreduced log probability without flow density """
        if params is not None:
            μ, log_σ = self.check_inputs(params=params)
        else:
            μ, log_σ = self.check_inputs(size=sample.size())

        x, c = sample, self.normalization
        return (log_σ * 2 + (x - μ)**2 / tr.exp(log_σ * 2) + c) * (-0.5)

    def nll(self, params: Tensor, sample_params: Tensor=None) -> Tensor:
        """
        Analytically computes
            E_N(mu_2,sigma_2^2) [ - log N(mu_1, sigma_1^2) ]
        If mu_2, and sigma_2^2 are not provided, defaults to entropy.
        """
        μ, log_σ = self.check_inputs(params=params)
        if sample_params is not None:
            sample_μ, sample_log_σ = self.check_inputs(params=sample_params)
        else:
            sample_μ, sample_log_σ = μ, log_σ

        c = self.normalization
        nll_ = ((log_σ * 2).exp() * (sample_μ - μ)**2
                + tr.exp(sample_log_σ * 2 - log_σ * 2)
                + 2 * log_σ + c) * 0.5
        return nll_

    @staticmethod
    def kl(q_params: Tuple[Tensor, ...], p_params: Tuple[Tensor, ...]=None):
        """
        Computes KL(q||p). Default p is the standard Normal distribution.
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
        p_μ, p_log_σ = p_params
        log_var_ratio = (q_log_σ - p_log_σ) * 2
        return (log_var_ratio.exp() + (p_μ - q_μ)**2 / tr.exp(q_log_σ * 2) - 1 - log_var_ratio) * 0.5

    def kld(self, q_params: Tensor, p_params: Tensor=None) -> Tensor:
        """
        Computes KL(q||p). Default p is the standard Normal distribution.
        """
        q_params_ = self.check_inputs(params=q_params)
        p_params_ = self.check_inputs(params=p_params) if (p_params is not None) else None
        return self.kl(q_params_, p_params_)

    def get_params(self) -> Tensor:
        return tr.cat([self.μ, self.log_σ])

    @property
    def nparams(self):
        return 2

    @property
    def ndim(self):
        return 1

    @property
    def is_reparameterizable(self):
        return True

    def __repr__(self):
        return self.__class__.__name__ + ' ({:.3f}, {:.3f})'.format(self.μ[0], self.log_σ.exp()[0])

    def _get_default_prior_params(self, z_dim: int) -> Tensor:
        return tr.zeros(z_dim, self.nparams)


class Laplace(Distrib):
    normalization: Tensor
    μ: Tensor
    logscale: Tensor

    def __init__(self, μ: float=0, scale: float=1):
        """
        RSamples from a Laplace distribution using the reparameterization trick.

        ``params`` for methods should contain ``μ`` and ``logscale``
        """
        super(Laplace, self).__init__()
        self.register_buffer('normalization', tr.tensor([-math.log(2)]))
        self.register_buffer('μ', tr.tensor([float(μ)]))
        self.register_buffer('logscale', tr.tensor([math.log(scale)]))

    def check_inputs(self, params: Tensor=None, size: SizeType=None) -> Tuple[Tensor, ...]:
        """ μ_logscale = params """
        μ_logscale = params

        if (μ_logscale is not None) and (size is None):
            μ = μ_logscale.select(-1, 0)
            logscale = μ_logscale.select(-1, 1)
            return μ, logscale
        if (size is not None) and (μ_logscale is not None):
            μ = μ_logscale.select(-1, 0).expand(size)
            logscale = μ_logscale.select(-1, 1).expand(size)
            return μ, logscale
        if size is not None:
            μ = self.μ.expand(size)
            logscale = self.logscale.expand(size)
            return μ, logscale
        if (size is None) and (μ_logscale is None):
            raise ValueError('Either one of size or params should be provided.')
        raise ValueError(f'Given invalid inputs: (size={size}, μ_logscale={μ_logscale})')

    def _rsample(self, params: Tensor=None, size: SizeType=None) -> Tensor:
        """ params should contain ``μ`` and ``logscale`` """
        μ, logscale = self.check_inputs(params=params, size=size)
        scale = tr.exp(logscale)
        # Unif(-0.5, 0.5)
        u = tr.rand(μ.size()).to(dtype=μ.dtype, device=μ.device) - 0.5
        sample = μ - scale * tr.sign(u) * tr.log((-2) * tr.abs(u) + 1 + δ)
        return sample

    def _sample(self, params: Tensor=None, size: SizeType=None) -> Tensor:
        """ params should contain ``μ`` and ``logscale`` """
        with tr.no_grad():
            return self._rsample(params=params, size=size)

    def log_prob(self, sample: Tensor, params: Tensor=None) -> Tensor:
        """ returns unreduced log probability without flow density """
        if params is not None:
            μ, logscale = self.check_inputs(params=params)
        else:
            μ, logscale = self.check_inputs(size=sample.size())

        c = self.normalization
        inv_scale = tr.exp(-logscale)
        ins_exp = - tr.abs(sample - μ) * inv_scale
        return ins_exp + c - logscale

    def get_params(self) -> Tensor:
        return tr.cat([self.μ, self.logscale])

    @property
    def nparams(self):
        return 2

    @property
    def ndim(self):
        return 1

    @property
    def is_reparameterizable(self):
        return True

    def __repr__(self):
        return self.__class__.__name__ + ' ({:.3f}, {:.3f})'.format(self.μ[0], self.logscale.exp()[0])

    def _get_default_prior_params(self, z_dim: int) -> Tensor:
        return tr.zeros(z_dim, self.nparams)
