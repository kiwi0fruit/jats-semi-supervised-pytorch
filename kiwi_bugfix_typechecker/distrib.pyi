from typing import Any, Union, Tuple, List
from torch import Tensor, Size
from torch.distributions import Distribution as Distribution_  # type: ignore

SizeType = Union[Size, List[int], Tuple[int, ...]]


# noinspection PyAbstractClass
class Distribution(Distribution_, object):
    def __init__(self, batch_shape: SizeType=Size(), event_shape: SizeType=Size(), validate_args: Any=None) -> None: ...
    def log_prob(self, value: Tensor) -> Tensor: ...
    def sample(self, sample_shape: SizeType=Size()) -> Tensor: ...
    def rsample(self, sample_shape: SizeType=Size()) -> Tensor: ...
