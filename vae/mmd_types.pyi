from abc import abstractmethod
from torch import Tensor
from kiwi_bugfix_typechecker.nn import Module


# noinspection PyAbstractClass
class ModuleZToX(Module):
    @abstractmethod
    def forward_(self, z: Tensor) -> Tensor: ...
    def forward(self, z: Tensor) -> Tensor: ...  # type: ignore
    def __call__(self, z: Tensor) -> Tensor: ...  # type: ignore
