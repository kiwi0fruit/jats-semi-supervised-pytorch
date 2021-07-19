from typing import Tuple
from abc import abstractmethod
from torch import Tensor
from torch.nn import Module


class DistribBase(Module):
    @abstractmethod
    def forward_(self, x: Tensor, p_params: Tuple[Tensor, ...]=None) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor, p_params: Tuple[Tensor, ...]=None) -> Tensor:  # pylint: disable=arguments-differ
        return self.forward_(x=x, p_params=p_params)
