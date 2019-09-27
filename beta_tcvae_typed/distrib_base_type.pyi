from abc import abstractmethod
from torch import Tensor
from kiwi_bugfix_typechecker.nn import Module


class DistribBase(Module):
    @abstractmethod
    def forward_(self, sample: Tensor, params: Tensor=None) -> Tensor: ...
    def forward(self, sample: Tensor, params: Tensor=None) -> Tensor: ...  # type: ignore
    def __call__(self, sample: Tensor, params: Tensor=None) -> Tensor:  # type: ignore
        """ returns unreduced log probability without flow density """
        ...
