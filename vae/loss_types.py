from typing import Union
from abc import abstractmethod
from torch import Tensor
from semi_supervised_typed import Loss
from kiwi_bugfix_typechecker.nn import Module


# noinspection PyAbstractClass
class BaseBaseWeightedLoss(Loss):  # pylint: disable=abstract-method
    pass


class BaseTrimLoss(Module):
    @abstractmethod
    def forward_(self, μ: Tensor, log_σ: Tensor, kld: Tensor=None) -> Union[Tensor, int]:
        raise NotImplementedError

    def forward(  # pylint: disable=arguments-differ
            self, μ: Tensor, log_σ: Tensor, kld: Tensor=None) -> Union[Tensor, int]:
        return self.forward_(μ=μ, log_σ=log_σ, kld=kld)
