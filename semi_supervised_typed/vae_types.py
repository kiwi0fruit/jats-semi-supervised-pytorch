from abc import abstractmethod
from typing import Tuple
from torch import Tensor
from torch.nn import Module
from beta_tcvae_typed import XYDictStrZ


RetLadderEncoder = Tuple[Tensor, Tuple[Tensor, Tuple[Tensor, ...]]]
class BaseLadderEncoder(Module):
    @abstractmethod
    def forward_(self, x: Tensor) -> RetLadderEncoder:
        raise NotImplementedError

    def forward(self, x: Tensor) -> RetLadderEncoder:  # pylint: disable=arguments-differ
        return self.forward_(x=x)


RetLadderDecoder = Tuple[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]
class BaseLadderDecoder(Module):
    @abstractmethod
    def forward_(self, x: Tensor, l_μ: Tensor, l_log_σ: Tensor) -> RetLadderDecoder:
        raise NotImplementedError

    def forward(self, x: Tensor, l_μ: Tensor, l_log_σ: Tensor) -> RetLadderDecoder:  # pylint: disable=arguments-differ
        return self.forward_(x=x, l_μ=l_μ, l_log_σ=l_log_σ)


class ModuleXToX(Module):
    @abstractmethod
    def forward_(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        return self.forward_(x=x)


class ModuleXYToXYDictStrOptZ(Module):
    @abstractmethod
    def forward_(self, x: Tensor, y: Tensor) -> XYDictStrZ:
        raise NotImplementedError

    def forward(self, x: Tensor, y: Tensor) -> XYDictStrZ:  # pylint: disable=arguments-differ
        return self.forward_(x=x, y=y)
