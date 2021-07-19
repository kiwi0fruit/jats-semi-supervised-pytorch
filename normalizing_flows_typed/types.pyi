from abc import abstractmethod
from typing import Tuple
from torch import Tensor
from torch.nn import Module, Sequential


class SequentialZToXY(Sequential):
    # noinspection PyMissingConstructor
    def __init__(self, *args: Module) -> None: ...
    @abstractmethod
    def forward_(self, z: Tensor) -> Tuple[Tensor, Tensor]: ...
    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]: ...  # type: ignore
    def __call__(self, z: Tensor) -> Tuple[Tensor, Tensor]: ...  # type: ignore


class ModuleXToX(Module):
    # noinspection PyMissingConstructor
    def __init__(self): ...
    @abstractmethod
    def forward_(self, x: Tensor) -> Tensor: ...
    def forward(self, x: Tensor) -> Tensor: ...  # type: ignore
    def __call__(self, x: Tensor) -> Tensor: ...  # type: ignore


class ModuleXToXYZ(Module):
    # noinspection PyMissingConstructor
    def __init__(self): ...
    @abstractmethod
    def forward_(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]: ...
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]: ...  # type: ignore
    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]: ...  # type: ignore


class ModuleZToXY(Module):
    # noinspection PyMissingConstructor
    def __init__(self): ...
    @abstractmethod
    def forward_(self, z: Tensor) -> Tuple[Tensor, Tensor]:  ...
    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:  ...  # type: ignore
    def __call__(self, z: Tensor) -> Tuple[Tensor, Tensor]: ...  # type: ignore


class ModuleZOptJToXY(Module):
    # noinspection PyMissingConstructor
    def __init__(self): ...
    @abstractmethod
    def forward_(self, z: Tensor, j: Tensor=None) -> Tuple[Tensor, Tensor]:  ...
    def forward(self, z: Tensor, j: Tensor=None) -> Tuple[Tensor, Tensor]:  ...  # type: ignore
    def __call__(self, z: Tensor, j: Tensor=None) -> Tuple[Tensor, Tensor]: ...  # type: ignore
