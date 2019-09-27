from abc import abstractmethod
from torch import Tensor
from kiwi_bugfix_typechecker.nn import Module


class ModuleZToX(Module):
    @abstractmethod
    def forward_(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, z: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        return self.forward_(z=z)
