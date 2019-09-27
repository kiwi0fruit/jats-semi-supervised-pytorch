from typing import Union, Callable, Optional as Opt
from dataclasses import dataclass
# noinspection PyPep8Naming
from numpy import ndarray as Array
# noinspection PyPep8Naming
from wpca import WPCA as ClassWPCA  # type: ignore
from factor_analyzer import FactorAnalyzer  # type: ignore


class BaseAnalyzer:
    x_rec: Array
    z_normalized: Array
    z: Array
    _z_norm_refs: Opt[Array]


@dataclass
class LinearAnalyzer(BaseAnalyzer):
    n: int
    analyzer: Union[ClassWPCA, FactorAnalyzer]
    x: Array
    μ_x: Union[Array, int]
    σ_x: Union[Array, int]
    μ_z: Union[Array, int]
    σ_z: Union[Array, int]
    inverse_transform_matrix: Array
    normalize_x: bool
    normalize_z: bool

    def transform(self, x: Array) -> Array:
        return (self.analyzer.transform((x - self.μ_x) / self.σ_x) - self.μ_z) / self.σ_z

    def inverse_transform(self, z_normalized: Array) -> Array:
        """ Only first ``self.passthrough_dim`` elements of ``x`` are used.

        :return: x_rec (of the same shape as x, e.g. with passthrough elements prepended) """
        # (~6000, ~160) = (~6000, ~9) @ (~9, ~160):
        x_rec = (z_normalized @ self.inverse_transform_matrix) * self.σ_x + self.μ_x
        return x_rec

    def set_z_norm_refs(self, get_refs: Callable[[Array], Array]):
        self._z_norm_refs = get_refs(self.z_normalized)

    @property
    def z_norm_refs(self) -> Array:
        if self._z_norm_refs is not None:
            return self._z_norm_refs
        raise ValueError('set_z_norm_refs first')

    def __post_init__(self):
        self.x_rec = self.inverse_transform(self.transform(self.x))
        self.z_normalized = self.transform(self.x)
        self.z = self.z_normalized * self.σ_z + self.μ_z
        self._z_norm_refs = None
