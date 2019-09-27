from typing import Tuple, Optional as Opt, Iterable, Union
from dataclasses import dataclass
from itertools import cycle
import numpy as np
# noinspection PyPep8Naming
from numpy import ndarray as Array
import torch as tr
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from kiwi_bugfix_typechecker.data import WeightedRandomSampler

from .losses import BaseWeightedLoss, BernoulliLoss, CategoricalLoss
from .utils import reals_to_buckets

cuda = tr.cuda.is_available()


@dataclass
class Dim:
    """
    Attributes:

    ``x`` : int
      input dim
    ``h`` : Tuple[int, ...]
      hidden layers dims
    ``z`` : int
      latent variables layer dim
    ``buckets`` : int
      number of possible discrete values in input x.
    ``y`` : int
      classes dim for Deep Generative Models
    """
    x: int
    h: Tuple[int, ...]
    z: int
    buckets: int
    y: int = 0

    def set_z(self, n: int):
        self.z = n

    def set_y(self, n: int):
        self.y = n


VAELoaderReturnType = Tuple[Tensor, Tensor, Tensor, Tensor]
DGMLoaderReturnType = Tuple[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]


class TrainLoader:
    separate_nll: bool
    target: bool
    weight_mat: bool
    _train_dataset: TensorDataset
    _labelled_train_dataset: Opt[TensorDataset]
    _sampler: Opt[WeightedRandomSampler]
    _labelled_sampler: Opt[WeightedRandomSampler]
    unlabelled: DataLoader
    labelled: Opt[DataLoader]
    unlabelled_dataset_size: int
    labelled_dataset_size: int

    # noinspection PyShadowingBuiltins
    def __init__(self, input: Array, input_bucketed: Union[Array, bool]=None,  # pylint: disable=redefined-builtin
                 weight_vec: Array=None, weight_mat: Union[Array, bool]=None, target: Array=None,
                 input_lbl: Array=None, input_lbl_bucketed: Union[Array, bool]=None,
                 weight_vec_lbl: Array=None, weight_mat_lbl: Union[Array, bool]=None, target_lbl: Array=None,
                 one_hot_target_lbl: Array=None, dtype: tr.dtype=tr.float, batch_size: int=128) -> None:
        """ ``idxs_label`` should contain indexes to select from ``input``. """
        if ((input_bucketed is False) or (weight_mat is False) or (input_lbl_bucketed is False)
                or (weight_mat_lbl is False)):
            raise ValueError
        if (input_lbl is None) and (
            (input_lbl_bucketed is not None)
            or (weight_vec_lbl is not None)
            or (weight_mat_lbl is not None)
            or (target_lbl is not None)
            or (one_hot_target_lbl is not None)
        ):
            raise ValueError
        if input_lbl is not None:
            if ((input_bucketed is not None) and (
                    input_lbl_bucketed is None)) or ((input_bucketed is None) and (input_lbl_bucketed is not None)):
                raise ValueError
            if ((weight_vec is not None) and (
                    weight_vec_lbl is None)) or ((weight_vec is None) and (weight_vec_lbl is not None)):
                raise ValueError
            if ((weight_mat is not None) and (
                    weight_mat_lbl is None)) or ((weight_mat is None) and (weight_mat_lbl is not None)):
                raise ValueError
            if ((target is not None) and (target_lbl is None)) or ((target is None) and (target_lbl is not None)):
                raise ValueError

        # ------------------------------
        _dummy = tr.tensor(np.zeros(shape=(input.shape[0],), dtype=np.uint8), dtype=tr.uint8)
        _dummy_lbl = (tr.tensor(np.zeros(shape=(input_lbl.shape[0],), dtype=np.uint8), dtype=tr.uint8)
                      if (input_lbl is not None) else _dummy)

        _input = tr.tensor(input, dtype=dtype)
        _input_lbl = tr.tensor(input_lbl, dtype=dtype) if (input_lbl is not None) else _dummy_lbl

        if target is not None:
            self.target = True
            _target = tr.tensor(target).long()
            _target_lbl = tr.tensor(target_lbl).long() if (target_lbl is not None) else _dummy_lbl
        elif (target is None) and (target_lbl is None):
            self.target = False
            _target, _target_lbl = _dummy, _dummy_lbl
        else:
            raise ValueError

        if isinstance(input_bucketed, Array) and (
                isinstance(input_lbl_bucketed, Array) or (input_lbl_bucketed is None)):
            self.separate_nll = True
            _input_bucketed = tr.tensor(input_bucketed).long()
            _input_lbl_bucketed = (tr.tensor(input_lbl_bucketed).long()
                                   if (input_lbl_bucketed is not None) else _dummy_lbl)
        elif (input_bucketed is True) and ((input_lbl_bucketed is True) or (input_lbl is None)):
            self.separate_nll = True
            _input_bucketed = tr.tensor(reals_to_buckets(input)).long()
            _input_lbl_bucketed = (tr.tensor(reals_to_buckets(input_lbl)).long()
                                   if (input_lbl is not None) else _dummy_lbl)
        elif (input_bucketed is None) and (input_lbl_bucketed is None):
            self.separate_nll = False
            _input_bucketed, _input_lbl_bucketed = _dummy, _dummy_lbl
        else:
            raise ValueError

        if isinstance(weight_mat, Array) and (isinstance(weight_mat_lbl, Array) or (weight_mat_lbl is None)):
            self.weight_mat = True
            _weight_mat = tr.tensor(weight_mat, dtype=dtype)
            _weight_mat_lbl = tr.tensor(weight_mat_lbl, dtype=dtype) if (weight_mat_lbl is not None) else _dummy_lbl
        elif (weight_mat is True) and ((weight_mat_lbl is True) or (input_lbl is None)):
            self.weight_mat = True
            _weight_mat = tr.tensor(np.ones(shape=input.shape, dtype=float), dtype=dtype)
            _weight_mat_lbl = (
                tr.tensor(np.ones(shape=input_lbl.shape, dtype=float), dtype=dtype)
                if (input_lbl is not None) else _dummy_lbl)
        elif (weight_mat is None) and (weight_mat_lbl is None):
            self.weight_mat = False
            _weight_mat, _weight_mat_lbl = _dummy, _dummy_lbl
        else:
            raise ValueError

        self._train_dataset = TensorDataset(_input, _input_bucketed, _target, _weight_mat)
        self._sampler = WeightedRandomSampler(
            weights=weight_vec.astype(np.float64),
            num_samples=len(weight_vec)) if (weight_vec is not None) else None

        if (input_lbl is not None) and (one_hot_target_lbl is not None):
            self._labelled_train_dataset = TensorDataset(_input_lbl, _input_lbl_bucketed, _target_lbl, _weight_mat_lbl,
                                                         tr.tensor(one_hot_target_lbl, dtype=dtype))
            self._labelled_sampler = WeightedRandomSampler(
                weights=weight_vec_lbl.astype(np.float64),
                num_samples=len(weight_vec_lbl)) if (weight_vec_lbl is not None) else None
        else:
            self._labelled_train_dataset = self._labelled_sampler = None

        self.regenerate_loaders(batch_size=batch_size)

    def upd_nll_state(self, nll: BaseWeightedLoss):
        if isinstance(nll, BernoulliLoss):
            self.separate_nll = False
        elif isinstance(nll, CategoricalLoss):
            if not self.separate_nll:
                raise ValueError('For categorical loss bucketed data should be present.')
        else:
            raise NotImplementedError

    def _get_loader(self, batch_size: int=128, force_not_weighted: bool=False, labelled: bool=False) -> Opt[DataLoader]:
        if labelled:
            sampler, train_dataset = self._labelled_sampler, self._labelled_train_dataset
        else:
            sampler, train_dataset = self._sampler, self._train_dataset

        if train_dataset is None:
            return None
        if (sampler is not None) and not force_not_weighted:
            return DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=3, pin_memory=cuda)
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=cuda)

    def regenerate_loaders(self, batch_size: int=128, force_not_weighted: bool=False):
        unlabelled = self._get_loader(batch_size=batch_size, force_not_weighted=force_not_weighted,
                                      labelled=False)
        if unlabelled is not None:
            self.unlabelled = unlabelled
        else:
            raise RuntimeError
        self.labelled = self._get_loader(batch_size=batch_size, force_not_weighted=force_not_weighted,
                                         labelled=True)
        self.unlabelled_dataset_size = len(self.unlabelled.dataset)
        self.labelled_dataset_size = (len(self.labelled.dataset) if (self.labelled is not None) else
                                      self.unlabelled_dataset_size)

    def get_vae_loader(self) -> Iterable:
        """
        Iterable returns (x, x_for_nll, target, weight_mat).
        """
        return self.unlabelled

    def get_dgm_loader(self) -> Iterable:
        """
        Iterable returns ((x, x_for_nll, target, weight_mat, one_hot_target), (x, x_for_nll, target, weight_mat)).
        where first is labelled and the second one is not labelled.
        """
        if self.labelled is None:
            raise ValueError('self.labelled is None. Presumably one_hot_target_label was not provided to constructor.')
        return zip(cycle(self.labelled), self.unlabelled)
