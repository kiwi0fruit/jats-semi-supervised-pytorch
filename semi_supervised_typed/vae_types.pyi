from abc import abstractmethod
from typing import Tuple
import torch as tr
from torch import Tensor
from torch.nn import Module
from beta_tcvae_typed import XYDictStrZ


RetLadderEncoder = Tuple[Tensor, Tuple[Tensor, Tuple[Tensor, ...]]]
class BaseLadderEncoder(Module):
    # noinspection PyMissingConstructor
    def __init__(self): ...
    @abstractmethod
    def forward_(self, x: Tensor) -> RetLadderEncoder: ...
    def forward(self, x: Tensor) -> RetLadderEncoder: ...  # type: ignore
    def __call__(self, x: Tensor) -> RetLadderEncoder: ...  # type: ignore


RetLadderDecoder = Tuple[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]
class BaseLadderDecoder(Module):
    # noinspection PyMissingConstructor
    def __init__(self): ...
    @abstractmethod
    def forward_(self, x: Tensor, l_μ: Tensor, l_log_σ: Tensor) -> RetLadderDecoder: ...
    def forward(self, x: Tensor, l_μ: Tensor, l_log_σ: Tensor) -> RetLadderDecoder: ...  # type: ignore
    def __call__(self, x: Tensor, l_μ: Tensor, l_log_σ: Tensor) -> RetLadderDecoder: ...  # type: ignore


class ModuleXToX(Module):
    # noinspection PyMissingConstructor
    def __init__(self): ...
    @abstractmethod
    def forward_(self, x: Tensor) -> Tensor: ...
    def forward(self, x: Tensor) -> Tensor: ...  # type: ignore
    def __call__(self, x: Tensor) -> Tensor: ...  # type: ignore


class ModuleXYToXYDictStrOptZ(Module):
    # noinspection PyMissingConstructor
    def __init__(self): ...
    @abstractmethod
    def forward_(self, x: Tensor, y: Tensor) -> XYDictStrZ: ...
    def forward(self, x: Tensor, y: Tensor) -> XYDictStrZ: ...  # type: ignore
    def __call__(self, x: Tensor, y: Tensor) -> XYDictStrZ: ...  # type: ignore
