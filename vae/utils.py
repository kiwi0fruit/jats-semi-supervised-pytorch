from typing import Tuple, Union, Dict
import numpy as np
# noinspection PyPep8Naming
from numpy import ndarray as Array
from torch import Tensor

ε = 10**-8


def ε_trim_0__1_inplace(x: Union[Array, Tensor]) -> Union[Array, Tensor]:
    x[x > (1 - ε)] = 1 - ε
    x[x < ε] = ε
    return x


def get_x_normalized_μ_σ(x: Array, weights: Array=None) -> Tuple[Array, Array, Array]:
    weights_: Dict[str, Array]
    if weights is None:
        weights_ = dict()
    elif len(x) != len(weights):
        raise ValueError(f'len(A) != len(weights): {len(x)} != {len(weights)}')
    else:
        weights_ = dict(weights=weights)
    μ = np.average(x, axis=0, **weights_)
    σ = np.sqrt(np.average((x - μ) ** 2, axis=0, **weights_))
    return (x - μ) / σ, μ, σ


def weights_matrix(weights: Array, A: Array) -> Array:
    """ ``weights`` is a vector, ``return`` shape would be the same as of ``A``. """
    if len(weights) != len(A):
        raise ValueError(f"len(weights) != len(A): {len(weights)} != {len(A)}")
    weights = np.copy(weights)
    weights = weights[np.newaxis].T
    weights_matrix_ = np.empty_like(A)
    weights_matrix_[:] = weights
    return weights_matrix_


def ndarr(x: Union[Array, Tensor]) -> Array:
    if isinstance(x, Tensor):
        return x.detach().numpy()
    if isinstance(x, Array):
        return x
    raise NotImplementedError


def wmse_wnll_all(x_recon: Union[Array, Tensor], x: Union[Array, Tensor],
                  weight_vec: Array=None, weight_mat: Array=None, nll: bool=False) -> float:
    x_rec, x_ = ndarr(x_recon), ndarr(x)
    weight_vec_kw: Dict[str, Array]
    if weight_vec is None:
        weight_vec_kw = dict()
    elif (len(x_rec) != len(weight_vec)) or (len(x_) != len(weight_vec)):
        raise ValueError(f'len(x_recon) and len(x) should equal len(weights)={len(weight_vec)}')
    else:
        weight_vec_kw = dict(weights=weight_vec)
    weight_mat_ = weight_mat if (weight_mat is not None) else 1
    if not nll:
        d = np.mean((x_rec - x_)**2 * weight_mat_, axis=1)
    else:
        d = np.sum((-x_ * np.log(x_rec + ε) + (x_ - 1) * np.log(1 - x_rec + ε)) * weight_mat_, axis=1)
    return float(np.average(d, axis=0, **weight_vec_kw))


def norm_weights_matrix(weights_vec: Array, shape_ref_weights_mat: Array) -> Array:
    return weights_matrix(weights_vec / np.sum(weights_vec) * len(weights_vec), shape_ref_weights_mat)


def categ_to_one_hot(x: Array) -> Array:
    uni_old = np.unique(x)
    uni_new = np.arange(len(uni_old)).astype(int)
    from_ = np.arange(uni_old[-1] + 1)
    from_[uni_old] = uni_new
    x_normal = from_[x]
    eye = np.eye(len(uni_old))
    return eye[x_normal]


def reals_to_buckets(x: Array, max_buckets: int=100) -> Array:
    """
    :param x: of shape (batch_size, features_size)
    :param max_buckets: used in np.round((x - np.min(x)) / (np.max(x) - np.min(x)) * max_buckets)
    :return: array of shape (batch_size, features_size) of int.
        Where each value is a number of the class.
    """
    min_, max_ = np.min(x), np.max(x)
    x_buckets = np.round((x - min_) / (max_ - min_) * max_buckets).astype(int)
    uni_old = np.unique(x_buckets)
    uni_new = np.arange(len(uni_old)).astype(int)
    from_ = np.arange(uni_old[-1] + 1)
    from_[uni_old] = uni_new
    return from_[x_buckets]
