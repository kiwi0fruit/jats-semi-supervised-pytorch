from typing import Union
from dataclasses import dataclass
import numpy as np
# noinspection PyPep8Naming
from numpy import ndarray as Array
# noinspection PyPep8Naming
from wpca import WPCA as ClassWPCA  # type: ignore
from factor_analyzer import FactorAnalyzer  # type: ignore
from . import analyzer_pre


@dataclass
class LinearAnalyzer(analyzer_pre.LinearAnalyzer):
    def __init__(
    self,
    n: int,
    analyzer: Union[ClassWPCA, FactorAnalyzer],
    x: Array,
    μ_x: Union[Array, int],
    σ_x: Union[Array, int],
    μ_z: Union[Array, int],
    σ_z: Union[Array, int],
    inverse_transform_matrix: Array,
    normalize_x: bool,
    normalize_z: bool
    ): ...
