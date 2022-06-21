from typing import Tuple, Dict
from abc import abstractmethod
import math
import torch as tr
from torch import Tensor
from torch.nn import Module


NORMNORM = math.log(2 * math.pi)  # normal distrib normalization


class Distrib(Module):
    def __init__(self):
        """
        Abstract class implementation that sets API.
        ``forward/__call__`` methods return unreduced log probability.
        """
        super(Distrib, self).__init__()

    @abstractmethod
    def forward(self, x: Tensor, *param: Tensor) -> Tensor:
        """ Computes log p(x). ``param`` can be empty. """
        raise NotImplementedError

    @abstractmethod
    def rsample(self, *param: Tensor) -> Tensor:
        """
        Samples x of the same dimension as the size of the first parameter
        (assumes that all parameters are of the same size).
        """
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, x: Tensor, *param: Tensor) -> Tensor:
        """ Computes log p(x). """
        raise NotImplementedError

    @abstractmethod
    def kld(self, q_params: Tuple[Tensor, ...], p_params: Tuple[Tensor, ...] = None) -> Tensor:
        """ Computes KL(q||p). Returns unreduced. """
        raise NotImplementedError


class StdNormal(Distrib):
    def __init__(self):
        """
        * Standard Normal. Has some methods for Normal distrib not just Standard Normal.
        * ``forward/__call__`` methods return unreduced log probability.
        * ``mu, log_sigma = param``
        """
        super(StdNormal, self).__init__()

    def forward(self, x: Tensor, *param: Tensor) -> Tensor:
        """
        * Computes log p(x). Default params is standard normal distribution.
        * ``mu, log_sigma = param`` if not empty (otherwise uses standard normal).
        """
        return self.log_prob(x, *param)

    def rsample(self, *param: Tensor) -> Tensor:
        """
        * Samples x of the same dimension as the size of the first parameter
          (assumes that all parameters are of the same size).
        * ``mu, log_sigma = param``
        """
        mu, log_sigma = param
        eps = tr.randn(mu.size()).to(dtype=mu.dtype, device=mu.device)
        return tr.exp(log_sigma) * eps + mu

    def log_prob(self, x: Tensor, *param: Tensor) -> Tensor:
        """
        * Computes log p(x). Default params is standard normal distribution.
        * ``mu, log_sigma = param`` if not empty (otherwise uses standard normal).
        """
        if param:
            mu, log_sigma = param
            return (log_sigma * 2 + (x - mu) ** 2 / tr.exp(log_sigma * 2) + NORMNORM) * (-0.5)
        return (x**2 + NORMNORM) * (-0.5)

    def kld(self, q_params: Tuple[Tensor, ...], p_params: Tuple[Tensor, ...] = None) -> Tensor:
        """
        * Computes KL(q||p) where q,p are Normal distribs. Default p is a Standard Normal distrib.
          Returns unreduced.
        * ``q_mu, q_log_sigma = q_params``
        * ``p_mu, p_log_sigma = p_params``
        """
        if p_params is None:
            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            q_mu, q_log_sigma = q_params
            return (q_log_sigma * 2 + 1 - q_mu ** 2 - tr.exp(q_log_sigma * 2)) * (-0.5)
        # https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html
        # @register_kl(Normal, Normal)
        # def _kl_normal_normal(p, q):
        # mind that here is KL(q||p)
        q_mu, q_log_sigma = q_params
        p_mu, p_log_sigma = p_params
        log_var_ratio = (p_log_sigma - q_log_sigma) * 2
        return (log_var_ratio.exp() + (q_mu - p_mu)**2 / tr.exp(p_log_sigma * 2) - 1 - log_var_ratio) * 0.5


def log_importance_weight_matrix(batch_size: int, dataset_size: int) -> Tensor:
    """
    Code was taken from https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py
    """
    n = dataset_size
    m = batch_size - 1
    strat_weight = (n - m) / (n * m)
    w_mat = tr.empty(batch_size, batch_size).fill_(1 / m)
    w_mat.view(-1)[::m+1] = 1 / n
    w_mat.view(-1)[1::m+1] = strat_weight
    w_mat[m-1, 0] = strat_weight
    return w_mat.log()


class BetaTCVAEModKLDLoss(Module):
    """
    If ``dataset_size`` is not set in constructor it should later be set via ``self.set_dataset_size`` method.

    ``forward/__call__`` methods return ``.view(n, -1).(sum=1)``
    reduced modified KLD of shape (batch_size,):

    ``mutual_information*alpha + total_correlation*gamma + dimension_wise_kl*lambda``

    Carefully pick ``prior_dist``, ``q_dist`` and samplers for reparametrization trick.
    Not all combinations are valid.
    """
    alpha: float
    gamma: float
    lambda_: float
    dataset_size: int

    def __init__(self, dataset_size: int, prior_dist: Distrib, q_dist: Distrib,
                 alpha=1, gamma=1, lambda_=1) -> None:
        super(BetaTCVAEModKLDLoss, self).__init__()
        self.prior_dist = prior_dist
        self.q_dist = q_dist
        self.alpha, self.gamma, self.lambda_ = alpha, gamma, lambda_
        self.dataset_size = dataset_size

    def get_logqz__logqz_prod_margs(self, z: Tensor, *qz_params: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute log q(z) ~= log 1/(NM) Σ_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        """
        n: int = z.shape[0]  # batch_size
        m = self.dataset_size
        z_dim: int = z.shape[1]

        pre_logqz = self.q_dist(
            z.view(n, 1, z_dim),
            *(param.view(1, n, z_dim) for param in qz_params)
        )  # shape is (batch_size_samples, batch_size_gauss_basis, z_dim)

        # minibatch weighted sampling (didn't work for me for some reason):
        # log_mn = math.log(n * m)
        # logqz_prod_margs = (tr.logsumexp(pre_logqz, dim=1, keepdim=False) - log_mn).sum(dim=1)
        # logqz = tr.logsumexp(pre_logqz.sum(dim=2), dim=1, keepdim=False) - log_mn

        # minibatch stratified sampling:
        logiw_matrix = log_importance_weight_matrix(n, m).to(dtype=pre_logqz.dtype, device=pre_logqz.device)
        logqz = tr.logsumexp(logiw_matrix + pre_logqz.sum(dim=2), dim=1, keepdim=False)
        logqz_prod_margs = tr.logsumexp(
            logiw_matrix.view(n, n, 1) + pre_logqz,
            dim=1, keepdim=False
        ).sum(dim=1)

        return logqz, logqz_prod_margs

    def forward(self, z: Tensor, *qz_params: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        n = z.shape[0]  # batch_size
        logqz_x = self.q_dist(z, *qz_params).view(n, -1).sum(dim=1)
        logpz = self.prior_dist(z).view(n, -1).sum(dim=1)
        logqz, logqz_prod_margs = self.get_logqz__logqz_prod_margs(z, *qz_params)
        mi = logqz_x - logqz  # mutual_information
        tc = logqz - logqz_prod_margs  # total_correlation
        dwkl = logqz_prod_margs - logpz  # dimension_wise_kl

        return (mi * self.alpha + tc * self.gamma + dwkl * self.lambda_,
                dict(mi=mi.mean().item(), tc=tc.mean().item(), dwkl=dwkl.mean().item()))

    def extra_repr(self) -> str:
        return f'dataset_size={self.dataset_size}, alpha={self.alpha}, gamma={self.gamma}, lambda_={self.lambda_}'


class KLDTCLoss(BetaTCVAEModKLDLoss):
    """
    If ``dataset_size`` is not set in constructor it should later be set via ``self.set_dataset_size`` method.

    Compatible closed form kld should be the same as ``self.q_dist`` == ``self.prior_dist``.

    ``forward/__call__`` methods return ``.view(n, -1).(sum=1)``
    reduced TC (total correlation) of shape (batch_size,).

    Carefully pick ``prior_dist``, ``q_dist`` and samplers for reparametrization trick.
    Not all combinations are valid.
    """
    def forward(self, z: Tensor, *qz_params: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        logqz, logqz_prod_margs = self.get_logqz__logqz_prod_margs(z, *qz_params)
        tc = logqz - logqz_prod_margs  # total_correlation
        return tc, {}

    def extra_repr(self) -> str:
        return f'dataset_size={self.dataset_size}'

    # def kld(self, z: Tensor, *qz_params: Tensor) -> Tensor:
    #     n = z.shape[0]  # batch_size
    #     logqz_x = self.q_dist(z, *qz_params).view(n, -1).sum(dim=1)
    #     logpz = self.prior_dist(z).view(n, -1).sum(dim=1)
    #     return logqz_x - logpz
