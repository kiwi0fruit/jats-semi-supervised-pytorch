from typing import Union
from abc import abstractmethod, ABCMeta
import torch as tr
from torch import Tensor
from semi_supervised_typed import Loss
from torch.nn import Module


# noinspection PyAbstractClass
class BaseBaseWeightedLoss(Loss, metaclass=ABCMeta):
    def __call__(self, x_params: Tensor, target: Tensor, weight: Tensor=None) -> Tensor:
        """
        See torch.nn.Module.forward documentation.

        >>> from torch.nn import Module
        >>> Module.forward

        "..." in common case would be nothing.

        :param x_params: flat Tensor of size (batch_n, x_params_multiplier * features_size_0 * ...)
            that is the output of the nn.Linear. It would be reshaped to
            (batch_size, x_params_multiplier, features_size_0, ...) if (x_params_multiplier > 1) else
            (batch_size, features_size_0, ...)
        :param target: of the size (batch_size, features_size_0, ...)
        :param weight: of the size (batch_size, features_size_0, ...)
        :return: Tensor of the size (batch_size,)
        """
        ...


# noinspection PyAbstractClass
class BaseBasisStripLoss(Module):
    @abstractmethod
    def forward_(self, μ: Tensor, log_σ: Tensor, kld: Tensor=None) -> Union[Tensor, int]: ...
    def forward(self, μ: Tensor, log_σ: Tensor, kld: Tensor=None) -> Union[Tensor, int]: ...  # type: ignore
    def __call__(self, μ: Tensor, log_σ: Tensor, kld: Tensor=None) -> Union[Tensor, int]: ...  # type: ignore


# noinspection PyAbstractClass
class BaseTrimLoss(Module):
    @abstractmethod
    def forward_(self, μ: Tensor, log_σ: Tensor) -> Union[Tensor, int]: ...
    def forward(self, μ: Tensor, log_σ: Tensor) -> Union[Tensor, int]: ...  # type: ignore
    def __call__(self, μ: Tensor, log_σ: Tensor) -> Union[Tensor, int]: ...  # type: ignore
