from typing import Sequence as Seq, List, Tuple, Union
from torch import Tensor
import torch as tr


# noinspection PyShadowingBuiltins
def softplus(input: Tensor, beta: float=1, threshold: float=20) -> Tensor: ...


# noinspection PyShadowingBuiltins
def logsigmoid(input: Tensor) -> Tensor: ...


# noinspection PyShadowingBuiltins,PyShadowingNames
def norm(input: Tensor, p: Union[int, float, str]="fro", dim: Union[int, Tuple[int, int], List[int]]=None,
         keepdim: bool=False, out: Tensor=None, dtype: tr.dtype=None) -> Tensor: ...
