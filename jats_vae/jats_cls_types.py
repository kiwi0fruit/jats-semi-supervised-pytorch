from abc import abstractmethod
from torch import Tensor
from torch.nn import Module


class ModuleXYToX(Module):
    @abstractmethod
    def forward_(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        return self.forward_(x=x, y=y)


class ModuleXOptYToX(Module):
    @abstractmethod
    def forward_(self, x: Tensor, y: Tensor=None) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor, y: Tensor=None) -> Tensor:  # pylint: disable=arguments-differ
        return self.forward_(x=x, y=y)
