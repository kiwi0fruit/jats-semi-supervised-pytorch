from typing import Tuple, Dict, Optional as Opt
from abc import abstractmethod
from torch import Tensor
from torch.nn import Module
from . import beta_tc_arg

XYDictStrZ = Tuple[Tensor, Tensor, Dict[str, Tensor]]


class BetaTC(beta_tc_arg.BetaTC):
    def __init__(
    self,
    mi__γ_tc__λ_dw: bool = False,
    γ_tc__λ_dw: bool = False,
    λ_kld__γmin1_tc: bool = False  # kld__γmin1_tc__λmin1_dw
    ): ...


# noinspection PyAbstractClass
class BaseKLDLoss(Module):
    @abstractmethod
    def forward_(self, z: Tensor, qz_params: Tuple[Tensor, ...], pz_params: Tuple[Tensor, ...]=None,
                 ) -> XYDictStrZ: ...
    def forward(self, z: Tensor, qz_params: Tuple[Tensor, ...], pz_params: Tuple[Tensor, ...]=None,  # type: ignore
                ) -> XYDictStrZ: ...
    def __call__(self, z: Tensor, qz_params: Tuple[Tensor, ...], pz_params: Tuple[Tensor, ...]=None,  # type: ignore
                 ) -> XYDictStrZ:
        """
        Computes the KL-divergence of some element z.

        KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                 = -E[log p(z) - log q(z)]

        :param z: sample from q-distribuion
        :param qz_params: (μ, log_σ) of the q-distribution
        :param pz_params: (μ, log_σ) of the p-distribution
        :return: KL(q||p), flow(z), verbose. Where verbose is {} or dict(...) depending on self._verbose
        """
        ...
