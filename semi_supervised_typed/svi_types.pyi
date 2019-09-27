from typing import Tuple
from abc import abstractmethod
from torch import Tensor
from kiwi_bugfix_typechecker.nn import Module


RetSVI = Tuple[Tensor, Tensor, Tensor]
class BaseSVI(Module):
    @abstractmethod
    def forward_(self, x: Tensor, y: Tensor=None, weight: Tensor=None, x_nll: Tensor=None) -> RetSVI: ...
    def forward(self, x: Tensor, y: Tensor=None, weight: Tensor=None, x_nll: Tensor=None) -> RetSVI: ...  # type: ignore
    def __call__(self, x: Tensor, y: Tensor=None, weight: Tensor=None, x_nll: Tensor=None) -> RetSVI:  # type: ignore
        """
        returns ``(nelbo, cross_entropy, probs)``.
        When ``y is None`` ``cross_entropy`` is a dummy zero tensor.
        ``nelbo`` > 0 is of size (batch_size,). ``cross_entropy`` is classification loss.
        """
        ...


class Loss:
    def __init__(self):
        """
        See the second parent class documentation.
        Intended to be the first parent class and __init__ should call the second class constructor.
        The second class should be a child of the torch.nn.Module (and inherit __call__ method).
        """
        ...
    @abstractmethod
    def forward_(self, x_params: Tensor, target: Tensor, weight: Tensor=None) -> Tensor: ...
    def forward(self, x_params: Tensor, target: Tensor, weight: Tensor=None) -> Tensor: ...
    def __call__(self, x_params: Tensor, target: Tensor, weight: Tensor=None) -> Tensor:
        """
        See torch.nn.Module.forward documentation.

        >>> from torch.nn import Module
        >>> Module.forward
        """
        ...
