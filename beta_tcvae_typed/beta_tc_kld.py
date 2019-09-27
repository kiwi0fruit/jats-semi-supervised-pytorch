from typing import Tuple, Union, Optional as Opt, List, Type, Dict
from abc import abstractmethod
import math
import torch as tr
from torch import Tensor

from kiwi_bugfix_typechecker import nn
from normalizing_flows_typed import Flow, SequentialFlow

from .beta_tc_types import BetaTC, BaseKLDLoss
from .distrib import Normal, Laplace  # , Distrib
from .functions import logsumexp, log_importance_weight_matrix

Flows = Union[Flow, SequentialFlow]


class Verbose:
    _verbose: bool
    stored: List[Tuple[Opt[Tensor], ...]]

    def __init__(self):
        self._verbose = False
        self.stored = []

    def set_verbose(self, verbose: bool=True):
        self._verbose = verbose

    @abstractmethod
    def stored_pop(self) -> None:
        """
        override this method to peek ``self.stored.pop()`` output.
        """
        raise NotImplementedError


class BetaTCKLDLoss(BaseKLDLoss, Verbose):
    γ: float
    λ: float
    kl: BetaTC
    _maybe_unmod_kld: bool
    mss: bool
    Dist: Type[Normal]
    q_dist: Normal
    dataset_size: int
    qz_x_flow: Opt[Flows]
    stored: List[Tuple[Tensor, Tensor, Opt[Tensor], Opt[Tensor], Opt[Tensor]]]  # type: ignore
    q_params_μ_first: bool
    _prior_dict: Dict[int, Normal]
    _prior_dists: nn.ModuleList

    def __init__(self, z_dims: Tuple[int, ...], prior_dists: Tuple[Normal, ...]=None, PriorDist: Type[Normal]=None,
                 q_dist: Normal=Normal(), mss: bool=True, kl: BetaTC=BetaTC(mi__γ_tc__λ_dw=True),
                 dataset_size: int=0):
        """
        1) WARNING: Carefully pick ``prior_dist``, ``q_dist`` and ``BaseSample`` subclass for ``Encoder``
        reparametrization trick (when used with ``semi_supervised_typed``).
        Not all combinations are valid. ``AuxiliaryDeepGenerativeModel`` is of particular caution
        as it uses optional ``pz_params`` arg of the forward method.

        2) ``forward`` and ``__call__`` methods return modified or unmodified KLD
        (depends on ``self.γ``, ``self.λ``, ``self.kl``).

        3) KLD modification is not compatible with adding ``qz_flow`` normalizing flow.
        Experimental support is added for ``kl=BetaTC(kld__γmin1_tc=True)`` but it affects only
        q_0(z) that is before the flows hence I'm not sure about the impact.

        :param z_dims: tuple of dims to switch between
        :param prior_dists: default is Normal
        :param PriorDist: default is Normal
        :param q_dist:
        :param mss: whether to use minibatch stratified sampling. Another option is minibatch weighted sampling
        :param kl: KLD formula
        :param dataset_size: if not set here it should later be set via self.set_dataset_size method
        """
        super(BetaTCKLDLoss, self).__init__()
        Verbose.__init__(self=self)
        self.kl = kl
        self._maybe_unmod_kld = kl.mi__γ_tc__λ_dw or kl.kld__γmin1_tc
        self.γ, self.λ = 1, 1
        self.mss = mss
        self.z_dim = z_dims[0]

        self.q_dist = q_dist
        if isinstance(self.q_dist, (Normal, Laplace)):
            self.q_params_μ_first = True
        else:
            self.q_params_μ_first = False

        # distribution family of p(z)
        if (prior_dists is not None) and (PriorDist is None):
            if (len(z_dims) != len(prior_dists)) or (len(z_dims) != len(set(z_dims))):
                raise ValueError
            self._prior_dict = {z_dim: prior_dist for z_dim, prior_dist in zip(z_dims, prior_dists)}
        elif prior_dists is None:
            PriorDist_ = PriorDist if (PriorDist is not None) else Normal
            self._prior_dict = {dim: PriorDist_() for dim in z_dims}
            for dim, dist in self._prior_dict.items():
                dist.set_prior_params(z_dim=dim)
        else:
            raise ValueError
        prior_dists_ = list(self._prior_dict.values())
        self._prior_dists = nn.ModuleList(prior_dists_)
        if len(set([type(dist) for dist in prior_dists_] + [type(q_dist)])) != 1:
            raise NotImplementedError
        self.Dist = type(q_dist)

        self.dataset_size = dataset_size
        self.qz_x_flow = None

    @property
    def prior_dist(self) -> Normal:
        return self._prior_dict[self.z_dim]

    def set_qz_x_flow(self, flow: Flows):
        self.qz_x_flow = flow
        self.q_params_μ_first = False

    def set_dataset_size(self, dataset_size: int):
        self.dataset_size = dataset_size

    def set_γ(self, γ: float):
        self.γ = γ

    def set_λ(self, λ: float):
        self.λ = λ

    def sample_z(self, batch_size: int=1) -> Tensor:
        """ Samples from the model p(z) """
        # sample from prior (value will be sampled by guide when computing the ELBO)
        prior_params = self.prior_dist.get_prior_params(batch_size)
        zs = self.prior_dist.sample(params=prior_params)
        return zs

    @staticmethod
    def get__logqz__logqz_Π_margs(q_dist: Normal, z: Tensor, z_params: Tensor, batch_size: int,
                                  dataset_size: int, z_dim: int, mss: bool) -> Tuple[Tensor, Tensor]:
        """ compute log q(z) ~= log 1/(NM) Σ_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m)) """

        _logqz = q_dist.__call__(
            z.view(batch_size, 1, z_dim),
            z_params.view(1, batch_size, z_dim, q_dist.nparams)
        )
        # I don't know how to integrate sum_log_abs_det_jacobian into _logqz...
        # print(z.shape, _logqz.shape, tmp.shape   sample.shape, μ.shape, log_σ.shape,   params.shape)
        # (128, 9) (128, 128, 9) (128, 128, 9)   (128, 1, 9) (1, 128, 9) (1, 128, 9)   (1, 128, 9, 2)

        if not mss:
            # minibatch weighted sampling
            logqz_Π_margs = (
                    logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)
            ).sum(dim=1)
            logqz = (logsumexp(_logqz.sum(dim=2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
        else:
            # minibatch stratified sampling
            logiw_matrix = log_importance_weight_matrix(batch_size, dataset_size).to(dtype=_logqz.dtype,
                                                                                     device=_logqz.device)
            logqz = logsumexp(logiw_matrix + _logqz.sum(dim=2), dim=1, keepdim=False)
            logqz_Π_margs = logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + _logqz,
                dim=1, keepdim=False
            ).sum(dim=1)
        return logqz, logqz_Π_margs

    def forward_(self, z: Tensor, qz_params: Tuple[Tensor, ...], pz_params: Tuple[Tensor, ...]=None,
                 unmod_kld: bool=False, try_closed_form: bool=True) -> Tuple[Tensor, Tensor]:
        """
        :param z: sample from q-distribuion
        :param qz_params: params like (μ, log_σ) of the q-distribution
        :param pz_params: params like (μ, log_σ) of the p-distribution
        :param unmod_kld: force unmodified KLD
        :param try_closed_form:
        :return: KL(q||p), flow(z)
            modified or unmodified KLD = log q(z|x) - log p(z) of size (batch_size,)
        """
        γ, λ, kl = self.γ, self.λ, self.kl
        unmod_kld_ = unmod_kld or (self._maybe_unmod_kld and (λ == 1) and (γ == 1))
        batch_size, z_dim = z.shape[0], z.shape[1]
        self.z_dim = z_dim

        if try_closed_form and unmod_kld_ and (self.qz_x_flow is None):
            kld = self.Dist.kl(qz_params, pz_params).view(batch_size, -1).sum(dim=1)
            if self._verbose:
                self.stored.append((kld, kld, None, None, None))
            return kld, z

        qz_params_ = tr.stack(qz_params, dim=-1)
        pz_params_ = (tr.stack(pz_params, dim=-1)
                      if (pz_params is not None) else
                      self.prior_dist.get_prior_params(batch_size))

        # log q(z|x):
        logqz_x = self.q_dist.__call__(z, params=qz_params_).view(batch_size, -1).sum(dim=1)

        sladetj: Union[Tensor, int]
        if self.qz_x_flow is not None:
            fz, sladetj_ = self.qz_x_flow.__call__(z)
            sladetj = sladetj_.view(batch_size, -1).sum(dim=1)  # sladetj := sum_log_abs_det_jacobian
        else:
            fz, sladetj = z, 0

        logpz = self.prior_dist.log_prob_reduced(fz, params=pz_params_)

        if unmod_kld_:
            kld = logqz_x - sladetj - logpz  # sladetj := sum_log_abs_det_jacobian
            if self._verbose:
                self.stored.append((kld, kld, None, None, None))
            return kld, fz

        if self.dataset_size == 0:
            raise ValueError('self.dataset_size should be set via self.set_dataset_size(...)')
        if self.qz_x_flow is not None:
            raise NotImplementedError

        logqz, logqz_Π_margs = self.get__logqz__logqz_Π_margs(
            self.q_dist, z, qz_params_, batch_size, self.dataset_size, z_dim, mss=self.mss)

        if kl.kld__γmin1_tc:
            modified_kld = (
                (logqz_x - logpz)  # kld
                + (logqz - logqz_Π_margs) * (γ - 1)  # total_corr * (γ - 1)
                # + (logqz_Π_margs - logpz) * (λ - 1)  # dim_wise_kl * (λ - 1)
            )
        elif kl.mi__γ_tc__λ_dw:
            modified_kld = (
                (logqz_x - logqz)  # mutual_info
                + (logqz - logqz_Π_margs) * γ  # total_corr * γ
                + (logqz_Π_margs - logpz) * λ  # dim_wise_kl * λ
            )
        elif kl.γ_tc__λ_dw:
            modified_kld = (
                (logqz - logqz_Π_margs) * γ  # total_corr * γ
                + (logqz_Π_margs - logpz) * λ  # dim_wise_kl * λ
            )
        else:
            raise NotImplementedError
        if self._verbose:
            self.stored.append((
                modified_kld,
                logqz_x - logpz,  # KLD
                logqz_x - logqz,  # MI
                logqz - logqz_Π_margs,  # TC
                logqz_Π_margs - logpz  # DW
            ))
        return modified_kld, fz

    def stored_pop(self) -> None:
        """
        ``self.stored.pop()`` returns (kld_mod, kld, mi, tc, dw)
        """
        raise RuntimeError
