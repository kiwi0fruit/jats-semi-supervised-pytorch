from typing import Tuple
from abc import abstractmethod
from torch import Tensor
from torch.nn import Module


XTupleYi = Tuple[Tensor, Tuple[Tensor, ...]]
class ModuleXToXTupleYi(Module):
    # noinspection PyMissingConstructor
    def __init__(self): ...
    @abstractmethod
    def forward_(self, x: Tensor) -> XTupleYi: ...
    def forward(self, x: Tensor) -> XTupleYi: ...  # type: ignore
    def __call__(self, x: Tensor) -> XTupleYi: ...  # type: ignore


class BaseGaussianMerge(Module):
    # noinspection PyMissingConstructor
    def __init__(self): ...
    @abstractmethod
    def forward_(self, z: Tensor, μ0: Tensor, log_σ0: Tensor) -> XTupleYi: ...
    def forward(self, z: Tensor, μ0: Tensor, log_σ0: Tensor) -> XTupleYi: ...  # type: ignore
    def __call__(self, z: Tensor, μ0: Tensor, log_σ0: Tensor) -> XTupleYi: ...  # type: ignore
