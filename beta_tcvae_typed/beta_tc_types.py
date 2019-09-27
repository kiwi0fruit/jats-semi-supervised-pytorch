from typing import Tuple
from abc import abstractmethod
from torch import Tensor
from kiwi_bugfix_typechecker.nn import Module
from .beta_tc_arg import BetaTC


BetaTC = BetaTC


class BaseKLDLoss(Module):
    @abstractmethod
    def forward_(self, z: Tensor, qz_params: Tuple[Tensor, ...], pz_params: Tuple[Tensor, ...]=None,
                 unmod_kld: bool=False, try_closed_form: bool=True) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def forward(  # pylint: disable=arguments-differ
            self, z: Tensor, qz_params: Tuple[Tensor, ...], pz_params: Tuple[Tensor, ...]=None,
            unmod_kld: bool=False, try_closed_form: bool=True) -> Tuple[Tensor, Tensor]:
        return self.forward_(z=z, qz_params=qz_params, pz_params=pz_params, unmod_kld=unmod_kld,
                             try_closed_form=try_closed_form)
