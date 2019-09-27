from typing import Tuple
from abc import abstractmethod
from torch import Tensor
from kiwi_bugfix_typechecker.nn import Module
from . import beta_tc_arg


class BetaTC(beta_tc_arg.BetaTC):
    def __init__(
    self,
    mi__γ_tc__λ_dw: bool = False,
    γ_tc__λ_dw: bool = False,
    kld__γmin1_tc: bool = False  # kld__γmin1_tc__λmin1_dw
    ): ...


# noinspection PyAbstractClass
class BaseKLDLoss(Module):
    @abstractmethod
    def forward_(self, z: Tensor, qz_params: Tuple[Tensor, ...], pz_params: Tuple[Tensor, ...]=None,
                 unmod_kld: bool=False, try_closed_form: bool=True) -> Tuple[Tensor, Tensor]: ...
    def forward(self, z: Tensor, qz_params: Tuple[Tensor, ...], pz_params: Tuple[Tensor, ...]=None,  # type: ignore
                unmod_kld: bool=False, try_closed_form: bool=True) -> Tuple[Tensor, Tensor]: ...
    def __call__(self, z: Tensor, qz_params: Tuple[Tensor, ...], pz_params: Tuple[Tensor, ...]=None,  # type: ignore
                 unmod_kld: bool=False, try_closed_form: bool=True) -> Tuple[Tensor, Tensor]: ...
