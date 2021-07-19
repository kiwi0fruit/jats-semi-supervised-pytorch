from typing import Tuple, Dict
from abc import abstractmethod
from torch import Tensor
from torch.nn import Module


RetSVI = Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]
class BaseSVI(Module):
    @abstractmethod
    def forward_(self, x: Tensor, y: Tensor=None, weight: Tensor=None, x_nll: Tensor=None,
                 x_cls: Tensor=None) -> RetSVI:
        raise NotImplementedError

    def forward(  # pylint: disable=arguments-differ
            self, x: Tensor, y: Tensor=None, weight: Tensor=None, x_nll: Tensor=None,
            x_cls: Tensor=None) -> RetSVI:
        return self.forward_(x=x, y=y, weight=weight, x_nll=x_nll, x_cls=x_cls)


class Loss:
    @abstractmethod
    def forward_(self, x_params: Tensor, target: Tensor, weight: Tensor=None) -> Tensor:
        raise NotImplementedError

    def forward(self, x_params: Tensor, target: Tensor, weight: Tensor=None) -> Tensor:
        return self.forward_(x_params=x_params, target=target, weight=weight)
