from typing import Tuple, Union, Optional as Opt, Dict
from contextlib import contextmanager
import math
from torch import Tensor
import torch as tr
from normalizing_flows_typed import Flow, SequentialFlow
from kiwi_bugfix_typechecker import test_assert

from .beta_tc_types import BetaTC, BaseKLDLoss, XYDictStrZ
from .distrib import Normal, Distrib
from .functions import log_importance_weight_matrix

TRY_CLOSED_FORM = False
Flows = Union[Flow, SequentialFlow]
test_assert()


def closed_form(q_dist: Distrib, prior_dist: Distrib, qz_x_flow: Opt[Flows]) -> bool:
    return (
        TRY_CLOSED_FORM and
        isinstance(q_dist, type(prior_dist)) and
        (prior_dist.inv_pz_flow is None) and
        (qz_x_flow is None)
    )


class PostInit:
    _disabled_post_init: bool = False

    def __final_init__(self):
        """
        Do not override this method.
        Override self.__post_init__() instead.
        """
        if self._disabled_post_init:
            return
        self.__post_init__()

    def __post_init__(self):  # pylint: disable=no-self-use
        """
        Do not call it in the self.__init__().
        Use self.__final_init__() there instead.
        """
        return

    @contextmanager
    def disable_final_init(self):
        disabled_post_init = self._disabled_post_init
        self._disabled_post_init = True
        yield None
        self._disabled_post_init = disabled_post_init


class Verbose:
    _verbose: bool

    def __init__(self):
        self._verbose = False

    def set_verbose(self, verbose: bool=True):
        self._verbose = verbose


class KLDLoss(BaseKLDLoss, PostInit, Verbose):
    _qz_x_flow: Opt[Flows]
    _q_dist: Distrib
    _prior_dist: Distrib
    _closed_form: bool

    def __init__(self, prior_dist: Distrib=None, q_dist: Distrib=None) -> None:
        super(KLDLoss, self).__init__()

        prior_dist_ = prior_dist if (prior_dist is not None) else Normal()
        self._prior_dist = prior_dist_

        q_dist_ = q_dist if (q_dist is not None) else Normal()
        assert q_dist_.inv_pz_flow is None
        self._q_dist = q_dist_
        assert isinstance(self._q_dist, type(self._q_dist))

        self._qz_x_flow = None
        self.set_closed_form()
        self.__final_init__()

    @property
    def prior_dist(self) -> Distrib:
        return self._prior_dist

    @property
    def q_dist(self) -> Distrib:
        return self._q_dist

    @property
    def qz_x_flow(self) -> Opt[Flows]:
        return self._qz_x_flow

    @property
    def pz_inv_flow(self) -> Opt[Flows]:
        return self.prior_dist.inv_pz_flow

    @property
    def closed_form(self) -> bool:
        return self._closed_form

    @staticmethod
    def closed_form_(q_dist: Distrib, prior_dist: Distrib, qz_x_flow: Opt[Flows]) -> bool:
        return (
                TRY_CLOSED_FORM and
                isinstance(q_dist, type(prior_dist)) and
                (prior_dist.inv_pz_flow is None) and
                (qz_x_flow is None)
        )

    def set_closed_form(self):
        self._closed_form = self.closed_form_(self.q_dist, self.prior_dist, self._qz_x_flow)

    def flow_qz_x(self, z: Tensor) -> Tensor:
        if self._qz_x_flow is not None:
            fz, _ = self._qz_x_flow.__call__(z)
            return fz
        return z

    def inv_flow_pz(self, z: Tensor) -> Tensor:
        prior_dist = self.prior_dist
        if prior_dist.inv_pz_flow is not None:
            zk = z
            z0, _ = prior_dist.inv_pz_flow.__call__(zk)
            return z0
        return z

    def flow_pz(self, z: Tensor) -> Tensor:
        prior_dist = self.prior_dist
        if prior_dist.inv_pz_flow is not None:
            z0 = z
            zk, _ = prior_dist.inv_pz_flow.backward(z0)
            return zk
        return z

    def set_q_dist(self, q_dist: Distrib):
        self._q_dist = q_dist
        self.set_closed_form()

    def set_prior_dist(self, prior_dist: Distrib):
        self._prior_dist = prior_dist
        self.set_closed_form()

    def set_qz_x_flow(self, flow: Flows):
        self._qz_x_flow = flow
        self._closed_form = False

    def set_pz_inv_flow(self, flow: Flows=None):
        self._prior_dist.set_inv_pz_flow(flow)
        self._closed_form = False

    def flow_qz_x_sladetj(self, z: Tensor) -> Tuple[Tensor, Union[Tensor, int]]:
        """For VariationalAutoencoder._kld_same_family only."""
        if self._qz_x_flow is not None:
            zk, sladetj = self._qz_x_flow.__call__(z)
            sladetj = sladetj.view(sladetj.size(0), -1).sum(dim=1)
            return zk, sladetj
        return z, 0

    def inv_flow_pz_sladetj_inv(self, z: Tensor) -> Tuple[Tensor, Union[Tensor, int]]:
        """For VariationalAutoencoder._kld_same_family only."""
        prior_dist = self.prior_dist
        if prior_dist.inv_pz_flow is not None:
            zk = z
            z0, sladetj_inv = prior_dist.inv_pz_flow.__call__(zk)
            sladetj_inv = sladetj_inv.view(sladetj_inv.size(0), -1).sum(dim=1)
            return z0, sladetj_inv
        return z, 0

    @contextmanager
    def unmodified_kld(self):  # pylint: disable=no-self-use
        yield None

    def forward_(self, z: Tensor, qz_params: Tuple[Tensor, ...], pz_params: Tuple[Tensor, ...]=None) -> XYDictStrZ:
        if self._closed_form:
            kld = self.q_dist.kld(qz_params, pz_params).view(z.shape[0], -1).sum(dim=1)
            return kld, z, {}

        q_μ, q_log_σ = qz_params

        # sladetj := sum_log_abs_det_jacobian
        # log_qzxk == log_qzx0 - q_sladetj
        # log_pz0 == log_pzk - p_sladetj_inv
        # kld = log_qzxk - log_pzk
        z_q0 = z
        log_qzx0 = self.q_dist.log_prob(z_q0, (q_μ, q_log_σ)).view(z.shape[0], -1).sum(dim=1)

        # sladetj := sum log abs det jacobian
        z_qk_pk, q_sladetj = self.flow_qz_x_sladetj(z_q0)
        z_p0, p_sladetj_inv = self.inv_flow_pz_sladetj_inv(z_qk_pk)

        if pz_params is not None:
            p_μ, p_log_σ = pz_params
            log_pz0 = self.prior_dist.log_prob(z_p0, (p_μ, p_log_σ)).view(z.shape[0], -1).sum(dim=1)
        else:
            log_pz0 = self.prior_dist.log_prob(z_p0).view(z.shape[0], -1).sum(dim=1)

        kld = log_qzx0 - q_sladetj - log_pz0 - p_sladetj_inv
        return kld, z_qk_pk, {}


class BetaTCKLDLoss(KLDLoss):
    γ: float
    λ: float
    mss: bool
    dataset_size: int

    _kl: BetaTC
    unmod_kld_allowed: bool
    _force_unmod_kld: bool = False

    def __init__(self, prior_dist: Distrib=None, q_dist: Distrib=None,
                 mss: bool=True, kl: BetaTC=BetaTC(mi__γ_tc__λ_dw=True), dataset_size: int=0) -> None:
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

        :param prior_dist:
        :param q_dist:
        :param mss: whether to use minibatch stratified sampling. Another option is minibatch weighted sampling
        :param kl: KLD formula
        :param dataset_size: if not set here it should later be set via self.set_dataset_size method
        """
        with self.disable_final_init():
            super(BetaTCKLDLoss, self).__init__(prior_dist=prior_dist, q_dist=q_dist)
        Verbose.__init__(self=self)
        self._kl = kl
        self.unmod_kld_allowed = self._kl.mi__γ_tc__λ_dw or self._kl.λ_kld__γmin1_tc
        self.γ, self.λ = 1, 1
        self.mss = mss
        self.dataset_size = dataset_size
        self.__final_init__()

    def set_dataset_size(self, dataset_size: int):
        self.dataset_size = dataset_size

    def set_γ(self, γ: float):
        self.γ = γ

    def set_λ(self, λ: float):
        self.λ = λ

    @contextmanager
    def unmodified_kld(self):
        self._force_unmod_kld = True
        yield None
        self._force_unmod_kld = False

    def get__logqz__logqz_Π_margs(self, z: Tensor, qz_params: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor]:
        """ compute log q(z) ~= log 1/(NM) Σ_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m)) """
        batch_size: int = z.shape[0]
        z_dim: int = z.shape[1]

        pre_logqz = self.q_dist.__call__(
            z.view(batch_size, 1, z_dim),
            tuple(param.view(1, batch_size, z_dim) for param in qz_params)
        )  # shape is (batch_size_samples, batch_size_gauss_basis, z_dim)

        if not self.mss:
            # minibatch weighted sampling
            log_mn = math.log(batch_size * self.dataset_size)
            logqz_Π_margs = (tr.logsumexp(pre_logqz, dim=1, keepdim=False) - log_mn).sum(dim=1)
            logqz = tr.logsumexp(pre_logqz.sum(dim=2), dim=1, keepdim=False) - log_mn
        else:
            # minibatch stratified sampling
            logiw_matrix = log_importance_weight_matrix(batch_size, self.dataset_size).to(dtype=pre_logqz.dtype,
                                                                                          device=pre_logqz.device)
            logqz = tr.logsumexp(logiw_matrix + pre_logqz.sum(dim=2), dim=1, keepdim=False)
            logqz_Π_margs = tr.logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + pre_logqz,
                dim=1, keepdim=False
            ).sum(dim=1)
        return logqz, logqz_Π_margs

    def forward_(self, z: Tensor, qz_params: Tuple[Tensor, ...], pz_params: Tuple[Tensor, ...]=None,
                 ) -> XYDictStrZ:
        verb: Dict[str, Tensor]
        γ, λ, kl = self.γ, self.λ, self._kl
        unmod_kld = self._force_unmod_kld or (self.unmod_kld_allowed and (λ == 1) and (γ == 1))
        batch_size = z.shape[0]

        if self._closed_form and unmod_kld:
            kld = self.q_dist.kld(qz_params, pz_params).view(batch_size, -1).sum(dim=1)

            verb = {} if not self._verbose else dict(kld_unmod=kld)
            return kld, z, verb

        sladetj: Union[Tensor, int]
        if self._qz_x_flow is not None:
            fz, sladetj_ = self._qz_x_flow.__call__(z)
            # sladetj := sum log abs det jacobian
            sladetj = sladetj_.view(batch_size, -1).sum(dim=1)
        else:
            fz, sladetj = z, 0

        if unmod_kld:
            if self._closed_form:
                kld = self.q_dist.kld(qz_params, pz_params).view(batch_size, -1).sum(dim=1)
            else:
                # log q(z|x):
                logqz_x = self.q_dist.__call__(z, qz_params).view(batch_size, -1).sum(dim=1)
                logpz = self.prior_dist.log_prob_reduced(fz, p_params=pz_params)
                # sladetj := sum log abs det jacobian
                kld = logqz_x - sladetj - logpz

            verb = {} if not self._verbose else dict(kld_unmod=kld)
            return kld, fz, verb

        assert self.dataset_size > 0
        assert self._qz_x_flow is None
        assert self.prior_dist.inv_pz_flow is None

        # log q(z|x):
        logqz_x = self.q_dist.__call__(z, qz_params).view(batch_size, -1).sum(dim=1)
        logpz = self.prior_dist.log_prob_reduced(z, p_params=pz_params)

        logqz, logqz_Π_margs = self.get__logqz__logqz_Π_margs(z, qz_params)

        tc = logqz - logqz_Π_margs

        if kl.λ_kld__γmin1_tc:
            if self._closed_form:
                kld = self.q_dist.kld(qz_params, pz_params).view(batch_size, -1).sum(dim=1)
            else:
                kld = logqz_x - sladetj - logpz
            modified_kld = (
                kld * λ
                + tc * (γ - 1)  # total_corr * (γ - 1)
                # + (logqz_Π_margs - logpz) * (λ - 1)  # dim_wise_kl * (λ - 1)
            )
        elif kl.mi__γ_tc__λ_dw:
            modified_kld = (
                (logqz_x - logqz)  # mutual_info
                + tc * γ  # total_corr * γ
                + (logqz_Π_margs - logpz) * λ  # dim_wise_kl * λ
            )
        elif kl.γ_tc__λ_dw:
            modified_kld = (
                tc * γ  # total_corr * γ
                + (logqz_Π_margs - logpz) * λ  # dim_wise_kl * λ
            )
        else:
            raise NotImplementedError

        verb = dict(tc=tc) if not self._verbose else dict(
                kld_unmod=self.q_dist.kld(
                    qz_params, pz_params).view(batch_size, -1).sum(dim=1) if self._closed_form else (logqz_x - logpz),
                tckld=modified_kld,
                mi=logqz_x - logqz,
                tc=tc,
                dw=logqz_Π_margs - logpz
            )
        return modified_kld, fz, verb
