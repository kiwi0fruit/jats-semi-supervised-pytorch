from typing import Tuple, Union
import numpy as np
# noinspection PyPep8Naming
from numpy import ndarray as Array
from torch import Tensor


ε = 10**-8


class Transforms:
    possible_int_values: Tuple[int, ...]

    def __init__(self, α: float=1, round_threshold: float=0.20, rep24to15: bool=True):
        """
        Additionally uses global constant ε to adjust mapping to (0, 1): (0, 1) to (0 + ε, 1 - ε)

        :param α: power to adjust mapping to (0, 1):
            (1, 2, 3, 4, 5) to (0, c, 0.5, 1 - c, 1).
            Examples: α=1 (2, 4) to (0.25, 0.75), α=0.51454 (2, 4) to (0.15, 0.85)
        :param round_threshold:
            replacement of x from (0, 1) line segment:
            x from [0.5 - thr, 0.5 + thr] to 0.5.
            Examples: 0.25 (like integer round), 0.16 (3 equal parts), 0.125 (like there are 0.25 & 0.75 values)
        :param rep24to15: replace (2, 4) to (1, 5)
        """
        self.α = α
        self.rep24to15 = rep24to15
        self.round_threshold = round_threshold
        if rep24to15:
            self.possible_int_values = (1, 3, 5)
        else:
            self.possible_int_values = (1, 2, 3, 4, 5)

        self.possible_values = tuple(float(s) for s in self.to_0__1(np.array(self.possible_int_values)))
        self.buckets_n = len(self.possible_int_values)

    def to_0__1(self, profiles: Array, use_ε: bool=True) -> Array:
        """ (1, 2, 3, 4, 5) to (0 + ε, c, 0.5, 1 - c, 1 - ε) """
        profiles = 0.5 * profiles - 1.5  # to  -1, -0.5, 0, 0.5, 1
        profiles = np.sign(profiles) * np.power(np.abs(profiles), self.α)  # to  -1, -b, 0, b, 1
        profiles = np.round(0.5 * (profiles + 1), 4)  # to  0, c, 0.5, 1-c, 1
        if use_ε:
            profiles[profiles > (1 - ε)] = 1 - ε
            profiles[profiles < ε] = ε
        return profiles

    def to_1__5(self, profiles: Array) -> Array:
        """ (0 + ε, c, 0.5, 1 - c, 1 - ε) to (1, 2, 3, 4, 5) """
        if not self.rep24to15:
            profiles = 2 * profiles - 1  # to  -1, -b, 0, b, 1
            profiles = np.sign(profiles) * np.power(np.abs(profiles), 1. / self.α)  # to  -1, -0.5, 0, 0.5, 1
            return np.round(2 * profiles + 3).astype(int)  # to  1, 2, 3, 4, 5

        profiles = np.round(2 * profiles).astype(int)  # to  0, 1, 2
        return 2 * profiles + 1  # to  1, 2, 3, 4, 5

    def _round_x_rec(self, x: Union[Array, Tensor]) -> Union[Array, Tensor]:
        """ thr = self.round_threshold.
            x in [0.5 - thr, 0.5 + thr] to 0.5 """
        if self.rep24to15:
            thr = self.round_threshold
            mask0 = x < (0.5 - thr)
            mask1 = x > (0.5 + thr)
            x[mask0] = ε
            x[mask1] = 1 - ε
            x[~mask0 & ~mask1] = 0.5
            return x
        raise NotImplementedError

    def round_x_rec_tens(self, x_rec: Tensor) -> Tensor:
        ret = self._round_x_rec(x_rec.clone().detach())
        if isinstance(ret, Tensor):
            return ret
        raise RuntimeError

    def round_x_rec_arr(self, x_rec: Array) -> Array:
        ret = self._round_x_rec(np.copy(x_rec))
        if isinstance(ret, Array):
            return ret
        raise RuntimeError

    def to_categorical(self, profiles: Array) -> Array:
        """ (1, 2, 3, 4, 5) to (0, 1, 2, 3, 4).
            Or (1, 2, 3, 4, 5) to (0, 0, 1, 2, 2) if replace 2 to 1 and 4 to 5. """
        if self.rep24to15:
            return (Transforms.rep_2to1_4to5(profiles) - 1) // 2
        return profiles - 1

    @staticmethod
    def rep_2to1_4to5(profiles: Array) -> Array:
        return 2 * np.sign(profiles - 3) + 3


def get_weight(types_tal_sex: Array) -> Array:
    _, counts = np.unique(types_tal_sex, return_counts=True)
    if len(counts) != 32:
        raise ValueError("types_tal_sex doesn't have all 32 types samples")
    return (1. / counts)[types_tal_sex - 1]
