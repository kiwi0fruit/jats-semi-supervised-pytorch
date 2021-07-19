from abc import abstractmethod
from typing import Tuple
from torch import Tensor
from torch.nn import Module, Sequential


class SequentialZToXY(Sequential):
    @abstractmethod
    def forward_(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:  # pylint: disable=arguments-differ
        return self.forward_(z=z)


class ModuleXToX(Module):
    @abstractmethod
    def forward_(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        return self.forward_(x=x)


class ModuleXToXYZ(Module):
    @abstractmethod
    def forward_(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:  # pylint: disable=arguments-differ
        return self.forward_(x=x)


class ModuleZToXY(Module):
    @abstractmethod
    def forward_(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:  # pylint: disable=arguments-differ
        return self.forward_(z=z)


class ModuleZOptJToXY(Module):
    @abstractmethod
    def forward_(self, z: Tensor, j: Tensor=None) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def forward(self, z: Tensor, j: Tensor=None) -> Tuple[Tensor, Tensor]:  # pylint: disable=arguments-differ
        return self.forward_(z=z, j=j)
