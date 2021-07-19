from typing import Iterable, Callable, Union, Tuple, Optional as Opt
import numpy as np
# noinspection PyPep8Naming
from numpy import ndarray as Array
# noinspection PyPep8Naming
from wpca import WPCA as ClassWPCA
from factor_analyzer import FactorAnalyzer
from sklearn.linear_model import LinearRegression

from .utils import get_x_normalized_μ_σ, weights_matrix
from .analyzer import LinearAnalyzer


def get__inverse_transform_matrix__μ_z__σ_z(
        z: Array, weight_vec: Opt[Array], normalize_z: bool, x_normalized: Array
) -> Tuple[Array, Union[Array, int], Union[Array, int]]:
    z_normalized = z
    μ_z: Union[Array, int] = 0
    σ_z: Union[Array, int] = 1
    if normalize_z:
        z_normalized, μ_z, σ_z = get_x_normalized_μ_σ(z, weight_vec)

    # z_normalized_extended @ inverse_transform == x_normalized_not_extended:
    wlr = LinearRegression()
    wlr.fit(z_normalized, x_normalized, sample_weight=weight_vec)
    inverse_transform_matrix: Array = wlr.coef_.T  # (~8, ~160)
    return inverse_transform_matrix, μ_z, σ_z


def get_pca(input_: Array, learn_input: Array, learn_weight_vec: Opt[Array], n_comp_list: Iterable[int],
            err_printer: Callable[[Array, Array, str], None]=None, normalize_x: bool=True,
            normalize_z: bool=False) -> LinearAnalyzer:
    """ The last from ``n_comp_list`` would be returned. """

    def expl(pca_): return np.round(np.sum(pca_.explained_variance_ratio_), 2)
    n_comp_list = list(n_comp_list)

    x = x_normalized = learn_input  # (~6000, ~162)
    weight_vec = learn_weight_vec
    μ_x: Union[Array, int] = 0
    σ_x: Union[Array, int] = 1
    if normalize_x:
        x_normalized, μ_x, σ_x = get_x_normalized_μ_σ(x, weight_vec)
    weight_vec_as_mat = weights_matrix(weight_vec, x) if (weight_vec is not None) else None

    for j, i in enumerate(n_comp_list):
        pca = ClassWPCA(i)
        pca.fit(x_normalized, weights=weight_vec_as_mat)
        z: Array = pca.transform(x_normalized)

        inverse_transform_matrix, μ_z, σ_z = get__inverse_transform_matrix__μ_z__σ_z(
            z, weight_vec, normalize_z, x_normalized)

        an = LinearAnalyzer(
            n=pca.n_components, analyzer=pca, x=input_, μ_x=μ_x, σ_x=σ_x, μ_z=μ_z, σ_z=σ_z,
            inverse_transform_matrix=inverse_transform_matrix, normalize_x=normalize_x, normalize_z=normalize_z)

        if err_printer is not None:
            pref = f"Expl = {expl(pca)}, PC N = {pca.n_components}, "
            err_printer(input_, an.x_rec, pref)

        if (j + 1) == len(n_comp_list):
            break
    else:
        raise ValueError('Empty n_comp_list')
    return an


def get_fa(input_: Array, learn_input: Array, learn_weight_vec: Opt[Array], n_comp_list: Iterable[int],
           err_printer: Callable[[Array, Array, str], None]=None, normalize_x: bool=True,
           normalize_z: bool=False) -> LinearAnalyzer:
    """ The last from ``n_comp_list`` would be returned. """

    n_comp_list = list(n_comp_list)

    x = x_normalized = learn_input  # (~6000, ~162)
    weight_vec = learn_weight_vec
    μ_x: Union[Array, int] = 0
    σ_x: Union[Array, int] = 1
    if normalize_x:
        x_normalized, μ_x, σ_x = get_x_normalized_μ_σ(x, weight_vec)
    Σ_x = np.cov(x_normalized.T, aweights=weight_vec)  # (~162, ~162)

    for j, i in enumerate(n_comp_list):
        fa = FactorAnalyzer(n_factors=i, is_corr_matrix=True, method='ml',
                            rotation=(None, 'varimax', 'oblimax', 'quartimax', 'equamax')[0]
                            )
        # rotation=('equamax', None)[1]
        fa.fit(Σ_x)
        fa.mean_ = np.zeros(x.shape[1])
        fa.std_ = fa.mean_ + 1.
        z = fa.transform(x_normalized)  # same as:
        # from numpy.linalg import inv
        # (~6000, ~9) = (~6000, ~162) @ ((~162, ~162) @ (~162, ~9))
        # z = ((x_normalized - 0) / 1) @ (inv(Σ_x) @ fa.structure_)

        inverse_transform_matrix, μ_z, σ_z = get__inverse_transform_matrix__μ_z__σ_z(
            z, weight_vec, normalize_z, x_normalized)

        an = LinearAnalyzer(
            n=fa.n_factors, analyzer=fa, x=input_, μ_x=μ_x, σ_x=σ_x, μ_z=μ_z, σ_z=σ_z,
            inverse_transform_matrix=inverse_transform_matrix, normalize_x=normalize_x, normalize_z=normalize_z)

        if err_printer is not None:
            pref = f"Factors N = {fa.n_factors}, "
            err_printer(input_, an.x_rec, pref)

        if (j + 1) == len(n_comp_list):
            break
    else:
        raise ValueError('Empty n_comp_list')
    return an
