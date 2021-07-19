from typing import Tuple
from abc import abstractmethod
from torch import Tensor
from torch.nn import Module


class BaseDiscriminator(Module):
    @abstractmethod
    def forward_(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:  # pylint: disable=arguments-differ
        return self.forward_(z=z)
