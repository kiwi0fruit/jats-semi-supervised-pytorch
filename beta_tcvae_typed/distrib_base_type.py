from abc import abstractmethod
from torch import Tensor
from kiwi_bugfix_typechecker.nn import Module


class DistribBase(Module):
    @abstractmethod
    def forward_(self, sample: Tensor, params: Tensor=None) -> Tensor:
        raise NotImplementedError

    def forward(self, sample: Tensor, params: Tensor=None) -> Tensor:  # pylint: disable=arguments-differ
        return self.forward_(sample=sample, params=params)
