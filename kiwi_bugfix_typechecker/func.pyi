from typing import Sequence as Seq, List, Tuple, Union
from torch import Tensor
import torch as tr


# noinspection PyShadowingBuiltins
def binary_cross_entropy(input: Tensor, target: Tensor, weight: Tensor=None, size_average: bool=None,
                         reduce: bool=None, reduction: str='mean') -> Tensor: ...


# noinspection PyShadowingBuiltins
def softplus(input: Tensor, beta: float=1, threshold: float=20) -> Tensor: ...


# noinspection PyShadowingBuiltins
def leaky_relu(input: Tensor, negative_slope: float=0.01, inplace: bool=False) -> Tensor: ...


# noinspection PyShadowingBuiltins
def logsigmoid(input: Tensor) -> Tensor: ...


# noinspection PyShadowingBuiltins,PyShadowingNames
def pad(input: Tensor, pad: Seq[int], mode: str='constant', value: float=0) -> Tensor: ...


# noinspection PyShadowingBuiltins,PyShadowingNames
def norm(input: Tensor, p: Union[int, float, str]="fro", dim: Union[int, Tuple[int, int], List[int]]=None,
         keepdim: bool=False, out: Tensor=None, dtype: tr.dtype=None) -> Tensor: ...


# noinspection PyShadowingBuiltins
def elu(input: Tensor, alpha: float=1., inplace: bool=False) -> Tensor: ...


# noinspection PyShadowingBuiltins
def softmax(input: Tensor, dim: int=None, _stacklevel: int=3, dtype: tr.dtype=None) -> Tensor: ...
