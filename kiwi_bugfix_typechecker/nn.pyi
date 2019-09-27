from typing import Any, List
from torch import Tensor
from torch.nn import (  # type: ignore
    Module as Module_, Tanh as Tanh_, Softplus as Softplus_, Sequential as Sequential_, Linear as Linear_,
    ReLU as ReLU_, SELU as SELU_, BatchNorm1d as BatchNorm1d_, LogSoftmax as LogSoftmax_, Sigmoid as Sigmoid_,
    CrossEntropyLoss as CrossEntropyLoss_, BCEWithLogitsLoss as BCEWithLogitsLoss_, ModuleList as ModuleList_,
    Parameter as Parameter_
)


class Module(Module_, object):
    # noinspection PyMissingConstructor
    def __init__(self) -> None: ...
    # noinspection PyShadowingBuiltins
    def forward(self, *input: Any) -> Any: ...
    # noinspection PyShadowingBuiltins
    def __call__(self, *input: Any, **kwargs: Any) -> Any: ...


# noinspection PyAbstractClass
class ModuleList(ModuleList_, list):
    # noinspection PyMissingConstructor
    def __init__(self, modules: List[Module]=None) -> None: ...


class Parameter(Parameter_, Tensor):
    def __new__(cls, data: Tensor=None, requires_grad: bool=True) -> Any: ...


class Tanh(Tanh_, Module):
    # noinspection PyMissingConstructor
    def __init__(self) -> None: ...
    # noinspection PyShadowingBuiltins
    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore
    # noinspection PyShadowingBuiltins
    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class Softplus(Softplus_, Module):
    # noinspection PyMissingConstructor
    def __init__(self, beta: float=1, threshold: float=20) -> None: ...
    # noinspection PyShadowingBuiltins
    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore
    # noinspection PyShadowingBuiltins
    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class Sequential(Sequential_, Module):
    # noinspection PyMissingConstructor
    def __init__(self, *args: Module) -> None: ...
    # noinspection PyShadowingBuiltins
    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore
    # noinspection PyShadowingBuiltins
    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class Linear(Linear_, Module):
    in_features: int
    # noinspection PyMissingConstructor
    def __init__(self, in_features: int, out_features: int, bias: bool=None) -> None: ...
    # noinspection PyShadowingBuiltins
    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore
    # noinspection PyShadowingBuiltins
    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class ReLU(ReLU_, Module):
    # noinspection PyMissingConstructor
    def __init__(self, inplace: bool=False) -> None: ...
    # noinspection PyShadowingBuiltins
    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore
    # noinspection PyShadowingBuiltins
    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class SELU(SELU_, Module):
    # noinspection PyMissingConstructor
    def __init__(self, inplace: bool=False) -> None: ...
    # noinspection PyShadowingBuiltins
    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore
    # noinspection PyShadowingBuiltins
    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class BatchNorm1d(BatchNorm1d_, Module):
    # noinspection PyMissingConstructor
    def __init__(self, num_features: int, eps: float=1e-5, momentum: float=0.1, affine: bool=True,
                 track_running_stats: bool=True) -> None: ...
    # noinspection PyShadowingBuiltins
    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore
    # noinspection PyShadowingBuiltins
    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class LogSoftmax(LogSoftmax_, Module):
    # noinspection PyMissingConstructor
    def __init__(self, dim: int=None) -> None: ...
    # noinspection PyShadowingBuiltins
    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore
    # noinspection PyShadowingBuiltins
    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class Sigmoid(Sigmoid_, Module):
    # noinspection PyMissingConstructor
    def __init__(self) -> None: ...
    # noinspection PyShadowingBuiltins
    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore
    # noinspection PyShadowingBuiltins
    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class CrossEntropyLoss(CrossEntropyLoss_, Module):
    # noinspection PyMissingConstructor
    def __init__(self, weight: Tensor=None, size_average: bool=None, ignore_index: int=-100,
                 reduce: bool=None, reduction: str='mean') -> None: ...
    # noinspection PyShadowingBuiltins
    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore
    # noinspection PyShadowingBuiltins
    def __call__(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore


class BCEWithLogitsLoss(BCEWithLogitsLoss_, Module):
    # noinspection PyMissingConstructor
    def __init__(self, weight: Tensor=None, size_average: bool=None,
                 reduce: bool=None, reduction: str='mean', pos_weight: Tensor=None) -> None: ...
    # noinspection PyShadowingBuiltins
    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore
    # noinspection PyShadowingBuiltins
    def __call__(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore
