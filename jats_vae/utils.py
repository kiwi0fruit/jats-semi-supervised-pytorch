from typing import Tuple, Union, Callable, List, Optional as Opt, Dict
import numpy as np
import numpy.linalg as la
# noinspection PyPep8Naming
from numpy import ndarray as Array
from torch import Tensor
import torch as tr

from semi_supervised_typed import VariationalAutoencoder
from vae.utils import (get_x_normalized_μ_σ, wmse_wnll_all, ε_trim_0__1_inplace, ndarr, norm_weights_matrix,
                       density_extremum, OutputChecker)
from vae.tools import get_zμ_ymax
from vae.losses import BernoulliLoss, CategoricalLoss, BaseWeightedLoss
from socionics_db import Data

ε = 10**-8


def print_err_tuple(*x_rec: Union[Array, Tensor],
                    x: Union[Array, Tensor], data: Data, line_printer: Callable[[str], None]) -> None:
    def prepare(tpl: Tuple[Union[Array, Tensor], ...]) -> List[Union[Array, Tensor]]:
        ret = list(tpl)
        if (len(ret) > 2) and (ret[1] is ret[2]):
            ret = ret[:1] + ret[2:]
        return ret

    x_rec_ = prepare(x_rec)
    rand = data.rand_learn_sampler_size_of_test
    for loss_name, isnll in (('MSE', False), ('NLL', True)):
        for name, select in (
            ('all', None), ('learn', data.learn_indxs), ('test', data.test_indxs),
            ('ctrl1', rand()), ('ctrl2', rand())
        ):
            x0_, weight_mat, weight = (
                x if (select is None) else x[select],
                data.get_weight_mat(select),
                data.get_weight_vec(select)
            )
            for weight_name in ('±TW', 'TW ', '¬W '):
                loss = [wmse_wnll_all(
                    ε_trim_0__1_inplace(x0_rec if (select is None) else x0_rec[select]),
                    x0_,
                    weight if weight_name != '¬W ' else None,
                    weight_mat if weight_name == '±TW' else None,
                    nll=isnll
                ) for x0_rec in x_rec_]
                line = f"{loss_name}_{name}_{{{weight_name}}} = (" + ", ".join(f"{e:.4f}" for e in loss) + ")"
                line_printer(line)


def get_refs(z: Array, types_tal: Array, types_self: Array) -> Array:
    if (types_tal is None) or (types_self is None):
        raise NotImplementedError
    return np.stack([density_extremum(z[(types_tal == types_self) & (types_self == j)]) for j in range(1, 17)], axis=0)


def euc_dist_count_types(z: Array=None, z_refs: Array=None, first_type: Array=None, n_types: int=None) -> Array:
    """
    :param z: of shape (~6000, z_dim)
    :param z_refs: of shape (n_types, z_dim)
    :param first_type: of shape (~6000,) can be used instead of z_refs.
        Should be from (1, ..., n_types)
    :param n_types: number of types
    :return: counted types of shape (n_types,)
    """
    if (z is not None) and (z_refs is not None):
        # 16-list of (~6000,) -> stack along axis -1:
        dists = np.stack([la.norm(z - ref, axis=-1) for ref in z_refs], axis=-1)  # (~6000, 16)
        first_type_ = np.argmax(dists, axis=-1) + 1
        n_types_ = len(z_refs)
    elif (first_type is not None) and (n_types is not None):
        first_type_, n_types_ = first_type, n_types
    else:
        raise ValueError('Either (z and z_refs) or (first_type and n_types) should be provided')
    uniqs, counts = np.unique(first_type_, return_counts=True)

    return np.array([
        counts[np.where(uniqs == i)[0][0]] if (i in uniqs) else 0
        for i in range(1, n_types_ + 1)
    ])


def get_basis_kld_vec(log_stats: List[dict], kld_drop: float=None) -> Tuple[Opt[List[int]], Opt[List[float]]]:
    """
    ``log_stats`` should not be empty.

    ``basis`` would be an empty list if every ``kld_vec[i]`` is < ``kld_drop``.

    :return: (basis, kld_vec)
    """
    if not log_stats:
        if kld_drop is None:
            return None, None
        raise RuntimeError('Bug: get_basis_kld_vec is called for model without history (not log_stats).')

    kld_vec: Opt[List[float]]
    kld_vec_ = log_stats[-1].get('KLD_vec')
    if kld_vec_ is None:
        kld_vec = kld_vec_
    elif isinstance(kld_vec_, list) and all(isinstance(i, (float, int)) for i in kld_vec_):
        kld_vec = kld_vec_
    else:
        raise RuntimeError('Bad KLD_vec key value in the last dict from log_stats.')

    if kld_drop is None:
        return None, kld_vec

    if kld_vec is None:
        raise RuntimeError("Last dict from log_stats doesn't have 'KLD_vec' key or it's value is None.")

    basis: List[int] = []
    # Select and sort KDL by which first achieved kld_drop:
    for dic in log_stats:
        for i, kld in enumerate(dic['KLD_vec']):
            if (kld >= kld_drop) and (i not in basis):
                basis.append(i)
    # Select KLD >= kld_drop:
    basis2 = [i for i, kld in enumerate(kld_vec) if kld >= kld_drop]
    basis = [i for i in basis if i in basis2]

    return basis, kld_vec


class OutputChecker1(OutputChecker):
    basis: Opt[List[int]]

    def __init__(self, x: Array, weights: Opt[Array], max_abs_Σ: float=1, kld_trim: Tuple[Opt[float], ...]=(None,),
                 allowed_basis_dims: Tuple[int, ...]=None,
                 line_printers: Tuple[Callable[..., None], ...]=(), dtype: tr.dtype=tr.float,
                 device: tr.device=tr.device('cuda') if tr.cuda.is_available() else tr.device('cpu')):
        """
        If successful check then ``self.basis`` should not be an empty list.
        But it can be an empty list if no dims were selected (hence not success).
        ``self.basis is None`` means that selections wasn't performed in the first place.
        """
        self.max_abs_Σ = max_abs_Σ
        self.x = x
        self.weights = weights
        self.dtype = dtype
        self.device = device
        self.ckeck_Σ = (max_abs_Σ + ε) < 1
        self.kld_trim = kld_trim[-1]
        self.line_printers = line_printers
        self.allowed_basis_dims = allowed_basis_dims
        self.basis = None

    def check(self, model: VariationalAutoencoder, log_stats: List[Dict]) -> bool:
        basis: Opt[List[int]]
        basis_is_good = True
        if self.kld_trim is not None:
            basis, _ = get_basis_kld_vec(log_stats, self.kld_trim)
            if not basis:
                basis_is_good = False
            elif self.allowed_basis_dims:
                basis_is_good = len(basis) in self.allowed_basis_dims
            if not basis_is_good:
                for f in self.line_printers:
                    f(f'Empty basis or bad basis dim.: {len(basis) if (basis is not None) else None}. '
                      + f' Allowed basis dims: {self.allowed_basis_dims}')
        else:
            basis = None
        self.basis = basis

        if not basis_is_good:
            return False

        if (self.max_abs_Σ + ε) >= 1:
            return basis_is_good
        with tr.no_grad():
            zμ, _, _, _, _ = get_zμ_ymax(x=tr.tensor(self.x, dtype=self.dtype, device=self.device), model=model)
        z_normalized, _, _ = get_x_normalized_μ_σ(ndarr(zμ), self.weights)
        basis_ = basis if (basis is not None) else list(range(z_normalized.shape[1]))

        Σ_is_good = True
        if (self.max_abs_Σ + ε) < 1:
            Σ = np.cov(z_normalized[:, basis_].T, aweights=self.weights)
            real_max_abs_Σ = np.max(np.abs(Σ - np.eye(len(basis_))))
            Σ_is_good = real_max_abs_Σ <= self.max_abs_Σ
            if not Σ_is_good:
                for f in self.line_printers:
                    f(f'Bad covariation matrix with max abs value = {real_max_abs_Σ}')

        return Σ_is_good


class TestArraysToTensors:
    separate_nll: bool
    separate_nll_lbl: bool

    _x_test_a: Array
    _x_test_lbl_a: Opt[Array]
    _x_lrn_lbl_a: Opt[Array]

    _x_test_nll_a: Opt[Array]
    _x_test_nll_lbl_a: Opt[Array]

    _weight_vec_test_a: Opt[Array]
    _weight_vec_test_lbl_a: Opt[Array]
    _weight_vec_lrn_lbl_a: Opt[Array]

    _weight_mat_test_a: Opt[Array]
    _weight_mat_test_lbl_a: Opt[Array]

    _one_hot_target_test_lbl_a: Opt[Array]
    _one_hot_target_lrn_lbl_a: Opt[Array]

    data: Data

    _x_test: Opt[Tensor] = None
    _x_test_lbl: Opt[Tensor] = None
    _x_lrn_lbl: Opt[Tensor] = None

    _x_test_nll: Opt[Tensor] = None
    _x_test_nll_lbl: Opt[Tensor] = None

    _weight_vec_test: Opt[Tensor] = None
    _weight_vec_test_lbl: Opt[Tensor] = None
    _weight_vec_lrn_lbl: Opt[Tensor] = None

    _weight_mat_test: Opt[Tensor] = None
    _weight_mat_test_lbl: Opt[Tensor] = None

    _one_hot_target_test_lbl: Opt[Tensor] = None
    _one_hot_target_lrn_lbl: Opt[Tensor] = None

    def __init__(self, data: Data, nll: BaseWeightedLoss):
        self.data = data

        self._x_test_a = data.test_input
        if isinstance(nll, BernoulliLoss):
            self.separate_nll = False
            self._x_test_nll_a = self._x_test_a
        elif isinstance(nll, CategoricalLoss):
            if data.input_bucketed is None:
                raise ValueError('For categorical loss bucketed data should be present.')
            self.separate_nll = True
            self._x_test_nll_a = data.input_bucketed[data.test_indxs]
        else:
            raise NotImplementedError
        self._weight_vec_test_a = (
            norm_weights_matrix(data.test_weight_vec, self._x_test_a)
            if (data.test_weight_vec is not None) else None
        )

        self._weight_mat_test_a = data.test_weight_mat if (data.test_weight_mat is not None) else None

        idxs_tst_lbl = data.test_indxs_labelled
        if (idxs_tst_lbl is not None) and (data.test_one_hot_target_labelled is not None):
            x_test_lbl_a: Array = data.input[idxs_tst_lbl]
            self._x_test_lbl_a = x_test_lbl_a
            if isinstance(nll, BernoulliLoss):
                self.separate_nll_lbl = False
                self._x_test_nll_lbl_a = x_test_lbl_a
            elif isinstance(nll, CategoricalLoss):
                if data.input_bucketed is None:
                    raise ValueError('For categorical loss bucketed data should be present.')
                self.separate_nll_lbl = True
                self._x_test_nll_lbl_a = data.input_bucketed[idxs_tst_lbl]
            else:
                raise NotImplementedError
            self._weight_vec_test_lbl_a = (
                norm_weights_matrix(data.test_weight_vec_labelled, x_test_lbl_a)
                if (data.test_weight_vec_labelled is not None) else None
            )

            self._weight_mat_test_lbl_a = data.weight_mat[idxs_tst_lbl] if (data.weight_mat is not None) else None
            self._one_hot_target_test_lbl_a = data.test_one_hot_target_labelled
        else:
            self.separate_nll_lbl = False
            self._x_test_lbl_a = self._x_test_nll_lbl_a = self._weight_vec_test_lbl_a = None
            self._weight_mat_test_lbl_a = self._one_hot_target_test_lbl_a = None

        idxs_lrn_lbl = data.learn_indxs_labelled
        if (idxs_lrn_lbl is not None) and (data.learn_one_hot_target_labelled is not None):
            x_lrn_lbl_a: Array = data.input[idxs_lrn_lbl]
            self._x_lrn_lbl_a = x_lrn_lbl_a

            self._weight_vec_lrn_lbl_a = (
                norm_weights_matrix(data.learn_weight_vec_labelled, x_lrn_lbl_a)
                if (data.learn_weight_vec_labelled is not None) else None
            )
            self._one_hot_target_lrn_lbl_a = data.learn_one_hot_target_labelled
        else:
            self._x_lrn_lbl_a = self._weight_vec_lrn_lbl_a = self._one_hot_target_lrn_lbl_a = None

    def set_test_tensors(self, transform_float: Callable[[Array], Tensor], transform_int: Callable[[Array], Tensor]):
        def float_(x: Opt[Array]) -> Opt[Tensor]:
            return transform_float(x) if (x is not None) else None
        def int_(x: Opt[Array]) -> Opt[Tensor]:
            return transform_int(x) if (x is not None) else None

        self._x_test = float_(self._x_test_a)
        self._x_test_nll = int_(self._x_test_nll_a) if self.separate_nll else float_(self._x_test_nll_a)
        self._weight_vec_test = float_(self._weight_vec_test_a)
        self._weight_mat_test = float_(self._weight_mat_test_a)

        self._x_test_lbl = float_(self._x_test_lbl_a)
        self._x_lrn_lbl = float_(self._x_lrn_lbl_a)
        self._x_test_nll_lbl = int_(self._x_test_nll_lbl_a) if self.separate_nll_lbl else float_(self._x_test_nll_lbl_a)
        self._weight_vec_test_lbl = float_(self._weight_vec_test_lbl_a)
        self._weight_vec_lrn_lbl = float_(self._weight_vec_lrn_lbl_a)
        self._weight_mat_test_lbl = float_(self._weight_mat_test_lbl_a)

        self._one_hot_target_test_lbl = float_(self._one_hot_target_test_lbl_a)
        self._one_hot_target_lrn_lbl = float_(self._one_hot_target_lrn_lbl_a)

    def get_test_tensors(self) -> Tuple[Tensor, Tensor, Opt[Tensor], Opt[Tensor]]:
        """ returns (x_test, x_for_nll_test, weight_vec_test, weight_mat_test) """
        if (self._x_test is None) or (self._x_test_nll is None):
            raise ValueError
        return self._x_test, self._x_test_nll, self._weight_vec_test, self._weight_mat_test

    def get_test_labelled_tensors(self) -> Tuple[Tensor, Tensor, Opt[Tensor], Opt[Tensor], Tensor]:
        """
        returns tuple (x_test_label, x_for_nll_test_label, weight_vec_test_label,
            weight_mat_test_label, one_hot_target_test_label)
        """
        if (self._x_test_lbl is None) or (self._x_test_nll_lbl is None) or (self._one_hot_target_test_lbl is None):
            raise ValueError
        return (self._x_test_lbl, self._x_test_nll_lbl, self._weight_vec_test_lbl, self._weight_mat_test_lbl,
                self._one_hot_target_test_lbl)

    def get_learn_labelled_tensors(self) -> Tuple[Tensor, Opt[Tensor], Tensor]:
        """
        This method turned out to be useless.

        :returns: (x_learn_label, weight_vec_learn_label, one_hot_target_learn_label)
        """
        if (self._x_lrn_lbl is None) or (self._one_hot_target_lrn_lbl is None):
            raise ValueError
        return self._x_lrn_lbl, self._weight_vec_lrn_lbl, self._one_hot_target_lrn_lbl

    def get_rand_learn_labelled_tensors(self) -> Tuple[Tensor, Opt[Tensor], Tensor]:
        """
        This method turned out to be useless.

        :returns: (x_learn_label_random, weight_vec_learn_label_random, one_hot_target_learn_label_random).
            With number of samples the same as in ``self.get_test_labelled_tensors``.
        """
        idxs = self.data.learn_indxs_labelled
        if idxs is None:
            raise ValueError
        idxs_rand = self.data.rand_learn_lbl_sampler_size_of_test_lbl()
        idxs_at_lrn_lbl = [i for i, idx in enumerate(idxs) if idx in idxs_rand]

        x_lrn_lbl, w, one_hot_target_lrn_lbl = self.get_learn_labelled_tensors()
        w_rand = self.data.get_weight_vec(idxs_rand)
        return (
            x_lrn_lbl[idxs_at_lrn_lbl],
            tr.tensor(norm_weights_matrix(w_rand, self.data.input[idxs_rand]), dtype=w.dtype, device=w.device)
            if (w is not None) and (w_rand is not None) else None,
            one_hot_target_lrn_lbl[idxs_at_lrn_lbl])
