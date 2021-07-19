from typing import Tuple, Dict
from abc import abstractmethod
from torch import Tensor
from torch.nn import Module
from .beta_tc_arg import BetaTC


BetaTC = BetaTC
XYDictStrZ = Tuple[Tensor, Tensor, Dict[str, Tensor]]


class BaseKLDLoss(Module):
    @abstractmethod
    def forward_(self, z: Tensor, qz_params: Tuple[Tensor, ...], pz_params: Tuple[Tensor, ...]=None) -> XYDictStrZ:
        raise NotImplementedError

    def forward(  # pylint: disable=arguments-differ
            self, z: Tensor, qz_params: Tuple[Tensor, ...], pz_params: Tuple[Tensor, ...]=None) -> XYDictStrZ:
        return self.forward_(z=z, qz_params=qz_params, pz_params=pz_params)
