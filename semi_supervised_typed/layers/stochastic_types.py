from typing import Tuple
from abc import abstractmethod
from torch import Tensor
from torch.nn import Module


XTupleYi = Tuple[Tensor, Tuple[Tensor, ...]]
class ModuleXToXTupleYi(Module):
    @abstractmethod
    def forward_(self, x: Tensor) -> XTupleYi:
        raise NotImplementedError

    def forward(self, x: Tensor) -> XTupleYi:  # pylint: disable=arguments-differ
        return self.forward_(x=x)


class BaseGaussianMerge(Module):
    @abstractmethod
    def forward_(self, z: Tensor, μ0: Tensor, log_σ0: Tensor) -> XTupleYi:
        raise NotImplementedError

    def forward(self, z: Tensor, μ0: Tensor, log_σ0: Tensor) -> XTupleYi:  # pylint: disable=arguments-differ
        return self.forward_(z=z, μ0=μ0, log_σ0=log_σ0)
