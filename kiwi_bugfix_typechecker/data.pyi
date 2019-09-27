from typing import Sequence as Seq, Union
# noinspection PyPep8Naming
from numpy import ndarray as Array
from torch.utils.data.sampler import WeightedRandomSampler as WeightedRandomSampler_


class WeightedRandomSampler(WeightedRandomSampler_):
    # noinspection PyMissingConstructor
    def __init__(self, weights: Union[Seq[float], Array], num_samples: int, replacement: bool=True) -> None: ...
