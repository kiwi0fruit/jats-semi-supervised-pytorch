# noinspection PyPep8Naming
from typing import Union, Tuple, Optional as Opt, List
from abc import abstractmethod
from torch import Tensor, Size
from normalizing_flows_typed import Flow, SequentialFlow
from .distrib_base_type import DistribBase

Flows = Union[Flow, SequentialFlow]
SizeType = Union[Size, List[int], Tuple[int, ...]]


class Distrib(DistribBase):
    inv_pz_flow: Opt[Flows]
    prior_params: Opt[Tensor]
    _flow_not_tested: bool

    def __init__(self):
        super(Distrib, self).__init__()
        self.inv_pz_flow = None
        self.prior_params = None
        self._flow_not_tested = True

    @property
    @abstractmethod
    def nparams(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _sample(self, params: Tensor=None, size: SizeType=None) -> Tensor:
        raise NotImplementedError

    def sample(self, params: Tensor = None, size: SizeType = None) -> Tensor:
        """ see ``self._sample`` docstr """
        sample = self._sample(params=params, size=size)
        if self.inv_pz_flow is not None:
            f_sample, _ = self.inv_pz_flow.backward(sample)
            return f_sample
        return sample

    @abstractmethod
    def _rsample(self, params: Tensor=None, size: SizeType=None) -> Tensor:
        raise NotImplementedError

    def rsample(self, params: Tensor = None, size: SizeType = None) -> Tensor:
        """ see ``self._rsample`` docstr """
        rsample = self._rsample(params=params, size=size)
        if self.inv_pz_flow is not None:
            f_rsample, _ = self.inv_pz_flow.backward(rsample)
            return f_rsample
        return rsample

    @abstractmethod
    def log_prob(self, sample: Tensor, params: Tensor=None) -> Tensor:
        """ returns unreduced log probability without flow density """
        raise NotImplementedError

    @abstractmethod
    def _get_default_prior_params(self, z_dim: int) -> Tensor:
        raise NotImplementedError

    def set_prior_params(self, z_dim: int=1):
        del self.prior_params
        self.register_buffer('prior_params', self._get_default_prior_params(z_dim=z_dim))

    def get_prior_params(self, batch_size: int=1) -> Tensor:
        """ Return prior parameters """
        if self.prior_params is None:
            raise ValueError('use self.set_prior_params(...) first')
        expanded_size = (batch_size,) + self.prior_params.size()
        return self.prior_params.expand(expanded_size)

    def forward_(self, sample: Tensor, params: Tensor=None) -> Tensor:
        """ returns unreduced log probability without flow density """
        return self.log_prob(sample=sample, params=params)

    def log_prob_reduced(self, sample: Tensor, params: Tensor=None) -> Tensor:
        """ returns reduced log probability with flow density added """

        batch_size = sample.shape[0]
        if self.inv_pz_flow is None:
            return self.log_prob(sample=sample, params=params).view(batch_size, -1).sum(dim=1)

        # sladetj := sum_log_abs_det_jacobian
        # log_pz0 == log_pzk - sladetj_inv
        zk = sample
        z0, sladetj_inv = self.inv_pz_flow.__call__(zk)
        sladetj_inv = sladetj_inv.view(batch_size, -1).sum(dim=1)
        log_pz0 = self.log_prob(sample=z0, params=params).view(batch_size, -1).sum(dim=1)

        log_pzk = log_pz0 + sladetj_inv
        return log_pzk

    @abstractmethod
    def check_inputs(self, params: Tensor=None, size: SizeType=None) -> Tuple[Tensor, ...]:
        raise NotImplementedError

    def set_inv_pz_flow(self, flow: Flows):
        self.inv_pz_flow = flow
