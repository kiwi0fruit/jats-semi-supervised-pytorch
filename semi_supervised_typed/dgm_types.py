from abc import abstractmethod
from typing import Tuple
from torch import Tensor
from torch.nn import Module


class ModuleXOptYToXY(Module):
    @abstractmethod
    def forward_(self, x: Tensor, y: Tensor=None) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def forward(self, x: Tensor, y: Tensor=None) -> Tuple[Tensor, Tensor]:  # pylint: disable=arguments-differ
        return self.forward_(x=x, y=y)
