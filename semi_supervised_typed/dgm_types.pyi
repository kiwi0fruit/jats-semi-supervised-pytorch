from abc import abstractmethod
from typing import Tuple
import torch as tr
from torch import Tensor
from torch.nn import Module


class ModuleXOptYToXY(Module):
    # noinspection PyMissingConstructor
    def __init__(self): ...
    @abstractmethod
    def forward_(self, x: Tensor, y: Tensor=None) -> Tuple[Tensor, Tensor]: ...
    def forward(self, x: Tensor, y: Tensor=None) -> Tuple[Tensor, Tensor]: ...  # type: ignore
    def __call__(self, x: Tensor, y: Tensor=None) -> Tuple[Tensor, Tensor]:  # type: ignore
        """ :return: (probs, cross_entropy). If y is None then entropy is returned. """
        ...
