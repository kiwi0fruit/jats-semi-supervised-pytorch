from typing import Union
from abc import abstractmethod
from torch import Tensor
from torch.nn import Module
from semi_supervised_typed import Loss


# noinspection PyAbstractClass
class BaseBaseWeightedLoss(Loss):  # pylint: disable=abstract-method
    pass


class BaseBasisStripLoss(Module):
    @abstractmethod
    def forward_(self, μ: Tensor, log_σ: Tensor, kld: Tensor=None) -> Union[Tensor, int]:
        raise NotImplementedError

    def forward(  # pylint: disable=arguments-differ
            self, μ: Tensor, log_σ: Tensor, kld: Tensor=None) -> Union[Tensor, int]:
        return self.forward_(μ=μ, log_σ=log_σ, kld=kld)


class BaseTrimLoss(Module):
    @abstractmethod
    def forward_(self, μ: Tensor, log_σ: Tensor) -> Union[Tensor, int]:
        raise NotImplementedError

    def forward(self, μ: Tensor, log_σ: Tensor) -> Union[Tensor, int]:  # pylint: disable=arguments-differ
        return self.forward_(μ=μ, log_σ=log_σ)
