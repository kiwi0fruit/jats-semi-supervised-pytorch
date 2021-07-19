from abc import abstractmethod
import torch as tr
from torch import Tensor
from torch.nn import Module


class ModuleXYToX(Module):
    # noinspection PyMissingConstructor
    def __init__(self): ...
    @abstractmethod
    def forward_(self, x: Tensor, y: Tensor) -> Tensor: ...
    def forward(self, x: Tensor, y: Tensor) -> Tensor: ...  # type: ignore
    def __call__(self, x: Tensor, y: Tensor) -> Tensor: ...  # type: ignore


class ModuleXOptYToX(Module):
    # noinspection PyMissingConstructor
    def __init__(self): ...
    @abstractmethod
    def forward_(self, x: Tensor, y: Tensor=None) -> Tensor: ...
    def forward(self, x: Tensor, y: Tensor=None) -> Tensor: ...  # type: ignore
    def __call__(self, x: Tensor, y: Tensor=None) -> Tensor: ...  # type: ignore
