from typing import Any
from torch import Tensor
from torch.nn import Parameter as Parameter_


class Parameter(Parameter_, Tensor):
    def __init__(self, data: Tensor=None, requires_grad: bool=True) -> None: ...
    def __new__(cls, data: Tensor=None, requires_grad: bool=True) -> Any: ...
