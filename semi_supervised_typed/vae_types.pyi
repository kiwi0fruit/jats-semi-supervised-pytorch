from abc import abstractmethod
from typing import Tuple
import torch as tr
from torch import Tensor
from kiwi_bugfix_typechecker.nn import Module


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


class ModuleXYToXY(Module):
    # noinspection PyMissingConstructor
    def __init__(self): ...
    @abstractmethod
    def forward_(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]: ...
    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]: ...  # type: ignore
    def __call__(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]: ...  # type: ignore
