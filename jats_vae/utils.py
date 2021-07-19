from typing import Tuple, Union, Callable, List, Optional as Opt
import numpy as np
import numpy.linalg as la
# noinspection PyPep8Naming
from numpy import ndarray as Array
from torch import Tensor
import torch as tr

from vae.utils import wmse_wnll_all, ε_trim_0__1_inplace, norm_weights_matrix
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


class TestArraysToTensors:
    separate_nll: bool

    _x_tst_a: Array
    _x_tst_lbl_a: Opt[Array]
    _x_lrn_a: Array
    _x_lrn_lbl_a: Opt[Array]

    _x_tst_nll_a: Opt[Array]
    _x_lrn_nll_a: Opt[Array]
    _x_tst_nll_lbl_a: Opt[Array]
    _x_lrn_nll_lbl_a: Opt[Array]

    _w_vec_tst_a: Opt[Array]
    _w_vec_lrn_a: Opt[Array]
    _w_vec_tst_lbl_a: Opt[Array]
    _w_vec_lrn_lbl_a: Opt[Array]

    _w_mat_tst_a: Opt[Array]
    _w_mat_lrn_a: Opt[Array]
    _w_mat_tst_lbl_a: Opt[Array]
    _w_mat_lrn_lbl_a: Opt[Array]

    _target_tst_lbl_a: Opt[Array]
    _target_lrn_lbl_a: Opt[Array]
    _one_hot_y_tst_lbl_a: Opt[Array]
    _one_hot_y_lrn_lbl_a: Opt[Array]

    data: Data

    _x_tst: Opt[Tensor] = None
    _x_lrn: Opt[Tensor] = None
    _x_tst_lbl: Opt[Tensor] = None
    _x_lrn_lbl: Opt[Tensor] = None

    _x_tst_nll: Opt[Tensor] = None
    _x_lrn_nll: Opt[Tensor] = None
    _x_tst_nll_lbl: Opt[Tensor] = None
    _x_lrn_nll_lbl: Opt[Tensor] = None

    _w_vec_tst: Opt[Tensor] = None
    _w_vec_lrn: Opt[Tensor] = None
    _w_vec_tst_lbl: Opt[Tensor] = None
    _w_vec_lrn_lbl: Opt[Tensor] = None

    _w_mat_tst: Opt[Tensor] = None
    _w_mat_lrn: Opt[Tensor] = None
    _w_mat_tst_lbl: Opt[Tensor] = None
    _w_mat_lrn_lbl: Opt[Tensor] = None

    _target_tst_lbl: Opt[Tensor] = None
    _target_lrn_lbl: Opt[Tensor] = None
    _one_hot_y_tst_lbl: Opt[Tensor] = None
    _one_hot_y_lrn_lbl: Opt[Tensor] = None

    def __init__(self, data: Data, nll: BaseWeightedLoss, dtype: tr.dtype=tr.float,
                 device: tr.device=tr.device('cuda') if tr.cuda.is_available() else tr.device('cpu')):
        self.data = data
        self.dtype = dtype
        self.device = device

        # x_test -------------------------------
        self._x_tst_a = data.test_input

        if isinstance(nll, BernoulliLoss):
            self.separate_nll = False
            self._x_tst_nll_a = self._x_tst_a
        elif isinstance(nll, CategoricalLoss):
            if data.input_bucketed is None:
                raise ValueError('For categorical loss bucketed data should be present.')
            self.separate_nll = True
            self._x_tst_nll_a = data.input_bucketed[data.test_indxs]
        else:
            raise NotImplementedError

        self._w_vec_tst_a = (
            norm_weights_matrix(data.test_weight_vec, self._x_tst_a)
            if (data.test_weight_vec is not None) else None
        )
        self._w_mat_tst_a = data.test_weight_mat if (data.test_weight_mat is not None) else None

        # x_learn -------------------------------
        self._x_lrn_a = data.learn_input

        if isinstance(nll, BernoulliLoss):
            self._x_lrn_nll_a = self._x_lrn_a
        elif isinstance(nll, CategoricalLoss):
            if data.input_bucketed is None:
                raise ValueError('For categorical loss bucketed data should be present.')
            self._x_lrn_nll_a = data.input_bucketed[data.learn_indxs]
        else:
            raise NotImplementedError

        self._w_vec_lrn_a = (
            norm_weights_matrix(data.learn_weight_vec, self._x_lrn_a)
            if (data.learn_weight_vec is not None) else None
        )
        self._w_mat_lrn_a = data.learn_weight_mat if (data.learn_weight_mat is not None) else None

        # x_test_labelled ----------------------
        idxs_tst_lbl = data.test_indxs_labelled
        if (idxs_tst_lbl is not None) and (data.test_one_hot_target_labelled is not None) and (data.target is not None):
            x_tst_lbl_a: Array = data.input[idxs_tst_lbl]
            self._x_tst_lbl_a = x_tst_lbl_a

            if isinstance(nll, BernoulliLoss):
                self._x_tst_nll_lbl_a = x_tst_lbl_a
            elif isinstance(nll, CategoricalLoss):
                if data.input_bucketed is None:
                    raise ValueError('For categorical loss bucketed data should be present.')
                self._x_tst_nll_lbl_a = data.input_bucketed[idxs_tst_lbl]
            else:
                raise NotImplementedError

            self._w_vec_tst_lbl_a = (
                norm_weights_matrix(data.test_weight_vec_labelled, x_tst_lbl_a)
                if (data.test_weight_vec_labelled is not None) else None
            )

            self._w_mat_tst_lbl_a = data.weight_mat[idxs_tst_lbl] if (data.weight_mat is not None) else None
            self._one_hot_y_tst_lbl_a = data.test_one_hot_target_labelled
            self._target_tst_lbl_a = data.target[idxs_tst_lbl]
        else:
            self._x_tst_lbl_a = self._x_tst_nll_lbl_a = self._w_vec_tst_lbl_a = None
            self._w_mat_tst_lbl_a = self._one_hot_y_tst_lbl_a = self._target_tst_lbl_a = None

        # x_learn_labelled ----------------------
        idxs_lrn_lbl = data.learn_indxs_labelled
        if ((idxs_lrn_lbl is not None) and (data.learn_one_hot_target_labelled is not None)
                and (data.target is not None)):
            x_lrn_lbl_a: Array = data.input[idxs_lrn_lbl]
            self._x_lrn_lbl_a = x_lrn_lbl_a

            if isinstance(nll, BernoulliLoss):
                self._x_lrn_nll_lbl_a = x_lrn_lbl_a
            elif isinstance(nll, CategoricalLoss):
                if data.input_bucketed is None:
                    raise ValueError('For categorical loss bucketed data should be present.')
                self._x_lrn_nll_lbl_a = data.input_bucketed[idxs_lrn_lbl]
            else:
                raise NotImplementedError

            self._w_vec_lrn_lbl_a = (
                norm_weights_matrix(data.learn_weight_vec_labelled, x_lrn_lbl_a)
                if (data.learn_weight_vec_labelled is not None) else None
            )

            self._w_mat_lrn_lbl_a = data.weight_mat[idxs_lrn_lbl] if (data.weight_mat is not None) else None
            self._one_hot_y_lrn_lbl_a = data.learn_one_hot_target_labelled
            self._target_lrn_lbl_a = data.target[idxs_lrn_lbl]
        else:
            self._x_lrn_lbl_a = self._x_lrn_nll_lbl_a = self._w_vec_lrn_lbl_a = None
            self._w_mat_lrn_lbl_a = self._one_hot_y_lrn_lbl_a = self._target_lrn_lbl_a = None

    def set_test_tensors(self):
        def float_(x: Opt[Array]) -> Opt[Tensor]:
            return tr.tensor(x, dtype=self.dtype, device=self.device) if (x is not None) else None
        def int_(x: Opt[Array]) -> Opt[Tensor]:
            return tr.tensor(x, dtype=tr.long, device=self.device) if (x is not None) else None

        self._x_tst = float_(self._x_tst_a)
        self._x_tst_nll = int_(self._x_tst_nll_a) if self.separate_nll else float_(self._x_tst_nll_a)
        self._w_vec_tst = float_(self._w_vec_tst_a)
        self._w_mat_tst = float_(self._w_mat_tst_a)

        self._x_lrn = float_(self._x_lrn_a)
        self._x_lrn_nll = int_(self._x_lrn_nll_a) if self.separate_nll else float_(self._x_lrn_nll_a)
        self._w_vec_lrn = float_(self._w_vec_lrn_a)
        self._w_mat_lrn = float_(self._w_mat_lrn_a)

        self._x_tst_lbl = float_(self._x_tst_lbl_a)
        self._x_tst_nll_lbl = int_(self._x_tst_nll_lbl_a) if self.separate_nll else float_(self._x_tst_nll_lbl_a)
        self._w_vec_tst_lbl = float_(self._w_vec_tst_lbl_a)
        self._w_mat_tst_lbl = float_(self._w_mat_tst_lbl_a)
        self._target_tst_lbl = int_(self._target_tst_lbl_a)
        self._one_hot_y_tst_lbl = float_(self._one_hot_y_tst_lbl_a)

        self._x_lrn_lbl = float_(self._x_lrn_lbl_a)
        self._x_lrn_nll_lbl = int_(self._x_lrn_nll_lbl_a) if self.separate_nll else float_(self._x_lrn_nll_lbl_a)
        self._w_vec_lrn_lbl = float_(self._w_vec_lrn_lbl_a)
        self._w_mat_lrn_lbl = float_(self._w_mat_lrn_lbl_a)
        self._target_lrn_lbl = int_(self._target_lrn_lbl_a)
        self._one_hot_y_lrn_lbl = float_(self._one_hot_y_lrn_lbl_a)

    def get_test_tensors(self) -> Tuple[Tensor, Tensor, Opt[Tensor], Opt[Tensor]]:
        """ :return: (x_tst, x_for_nll_tst, weight_vec_tst, weight_mat_tst) """
        if (self._x_tst is None) or (self._x_tst_nll is None):
            raise ValueError
        return self._x_tst, self._x_tst_nll, self._w_vec_tst, self._w_mat_tst

    def get_learn_tensors(self) -> Tuple[Tensor, Tensor, Opt[Tensor], Opt[Tensor]]:
        """ :return: (x_lrn, x_for_nll_lrn, weight_vec_lrn, weight_mat_lrn) """
        if (self._x_lrn is None) or (self._x_lrn_nll is None):
            raise ValueError
        return self._x_lrn, self._x_lrn_nll, self._w_vec_lrn, self._w_mat_lrn

    def get_random_learn_tensors(self) -> Tuple[Tensor, Tensor, Opt[Tensor], Opt[Tensor]]:
        """
        :return: (x_lrn_rnd, x_for_nll_lrn_rnd, weight_vec_lrn_rnd, weight_mat_lrn_rnd).
            With number of samples the same as in self.get_test_tensors.
        """
        idxs = self.data.learn_indxs
        idxs_rand = self.data.rand_learn_sampler_size_of_test()
        idxs_at_lrn = [i for i, idx in enumerate(idxs) if idx in idxs_rand]

        x, x_nll, w_vec, w_mat = self.get_learn_tensors()
        w_vec_rand = self.data.get_weight_vec(idxs_rand)
        return (
            x[idxs_at_lrn],
            x_nll[idxs_at_lrn],
            tr.tensor(norm_weights_matrix(w_vec_rand, self.data.input[idxs_rand]),
                      dtype=w_vec.dtype, device=w_vec.device)
            if (w_vec is not None) and (w_vec_rand is not None) else None,
            w_mat[idxs_at_lrn] if (w_mat is not None) else None)

    def get_test_labelled_tensors(self) -> Tuple[Tensor, Tensor, Opt[Tensor], Opt[Tensor], Tensor, Tensor]:
        """ :return: (x_tst_lbl, x_for_nll_tst_lbl, weight_vec_tst_lbl, weight_mat_tst_lbl,
            one_hot_y_tst_lbl, target_tst_lbl) """
        if ((self._x_tst_lbl is None) or (self._x_tst_nll_lbl is None) or (self._one_hot_y_tst_lbl is None)
                or (self._target_tst_lbl is None)):
            raise ValueError
        return (self._x_tst_lbl, self._x_tst_nll_lbl, self._w_vec_tst_lbl, self._w_mat_tst_lbl,
                self._one_hot_y_tst_lbl, self._target_tst_lbl)

    def get_learn_labelled_tensors(self) -> Tuple[Tensor, Tensor, Opt[Tensor], Opt[Tensor], Tensor, Tensor]:
        """ :return: (x_lrn_lbl, x_for_nll_lrn_lbl, weight_vec_lrn_lbl,
            weight_mat_lrn_lbl, one_hot_y_lrn_lbl, target_lrn_lbl) """
        if ((self._x_lrn_lbl is None) or (self._x_lrn_nll_lbl is None) or (self._one_hot_y_lrn_lbl is None)
                or (self._target_lrn_lbl is None)):
            raise ValueError
        return (self._x_lrn_lbl, self._x_lrn_nll_lbl, self._w_vec_lrn_lbl, self._w_mat_lrn_lbl,
                self._one_hot_y_lrn_lbl, self._target_lrn_lbl)

    def get_random_learn_labelled_tensors(self) -> Tuple[Tensor, Tensor, Opt[Tensor], Opt[Tensor], Tensor, Tensor]:
        """
        :return: (x_lrn_lbl_rnd, x_for_nll_lrn_rnd, weight_vec_lrn_lbl_rnd, weight_mat_lrn,
             one_hot_y_lrn_lbl_rnd, target_lrn_lbl_rnd).
            With number of samples the same as in self.get_test_labelled_tensors.
        """
        idxs = self.data.learn_indxs_labelled
        if idxs is None:
            raise ValueError
        idxs_rand = self.data.rand_learn_lbl_sampler_size_of_test_lbl()
        idxs_at_lrn = [i for i, idx in enumerate(idxs) if idx in idxs_rand]

        x, x_nll, w_vec, w_mat, one_hot_y, target = self.get_learn_labelled_tensors()
        w_vec_rand = self.data.get_weight_vec(idxs_rand)
        return (
            x[idxs_at_lrn],
            x_nll[idxs_at_lrn],
            tr.tensor(norm_weights_matrix(w_vec_rand, self.data.input[idxs_rand]),
                      dtype=w_vec.dtype, device=w_vec.device)
            if (w_vec is not None) and (w_vec_rand is not None) else None,
            w_mat[idxs_at_lrn] if (w_mat is not None) else None,
            one_hot_y[idxs_at_lrn],
            target[idxs_at_lrn])
