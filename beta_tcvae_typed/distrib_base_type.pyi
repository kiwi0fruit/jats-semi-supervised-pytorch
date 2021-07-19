from typing import Tuple
from abc import abstractmethod
from torch import Tensor
from torch.nn import Module


class DistribBase(Module):
    @abstractmethod
    def forward_(self, x: Tensor, p_params: Tuple[Tensor, ...]=None) -> Tensor: ...
    def forward(self, x: Tensor, p_params: Tuple[Tensor, ...]=None) -> Tensor: ...  # type: ignore
    def __call__(self, x: Tensor, p_params: Tuple[Tensor, ...]=None) -> Tensor:  # type: ignore
        """ returns unreduced log probability without flow density. """
        ...
