from typing import Tuple, List, Union, Callable, Optional as Opt
from dataclasses import dataclass
import numpy as np
# noinspection PyPep8Naming
from numpy import ndarray as Array
import torch as tr

from vae.data import Dim, TrainLoader
from vae.utils import categ_to_one_hot

from .read_db import DBSpec, DB
from .generate_control_groups import read_test_samples, random_control_samples
from .utils import Transforms, get_weight


def weight_neg_pos(row: Array, db_name: str) -> Array:
    if not (db_name in ('solti', 'bolti', 'solti_eng')):
        raise NotImplementedError

    def vec(*items): return np.array([float(item) for item in items]).astype(float)

    row = np.copy(row[2:])  # do not additionally weight sex and age questions
    for i, j in ((2, 1), (4, 5), (1, 0), (3, 1), (5, 2)):
        row[row == i] = j  # {1,2}->0  3->1  {4,5}->2
    uniques, N = np.unique(row, return_counts=True)
    for i in uniques:
        if i not in (0, 1, 2):
            raise ValueError
    if (uniques[0] == 0) and (uniques[-1] == 2):
        N_no, N_yes = N[0], N[-1]
        weights = vec(
            vec(0.4, 0.6) @ vec(1, 0.5 * (N_no + N_yes) / N_no),  # no weight
            1,  # don't know weight
            vec(0.4, 0.6) @ vec(1, 0.5 * (N_no + N_yes) / N_yes)  # yes weight
        )
    else:
        weights = vec(1, 1, 1)
    return np.concatenate((np.array([1., 1.]), weights[row]), axis=0)  # prepend sex and age back


def debug_loader(loader: TrainLoader):
    for force_not_weighted, str_ in ((False, ''), (True, ' NOT')):
        batch_size = loader.unlabelled.batch_size
        loader.regenerate_loaders(batch_size=4 * 16 * 2, force_not_weighted=force_not_weighted)
        stats_list = [
            [
                len(np.where((target == i) & (sex == j) & (sex == j))[0])
                for i in range(1, 17) for j in (0, 1)]
            for x, _, target_, _ in loader.get_loader()
            for sex, target in ([np.round(x.numpy()[:, 0]).astype(int), np.abs(target_.numpy())],)
        ]
        batch_list = [len(x) for x, _, target_, _ in loader.get_loader()]
        delta_dbl_batch_list = [len(u1) - len(u2) for _, (u1, _, _, _), (u2, _, _, _) in loader.get_double_lbl_loader()]
        stats = np.array(stats_list)
        stats = stats.reshape((stats.shape[0], 2, stats.shape[1] // 2))
        print(f'Average number of types should{str_} be around the same:\n{np.round(np.mean(stats, axis=0), 1)}')
        print(f'Last 3 lines:\n{stats[-3:].astype(float)}')
        print(f'x batch size list: {batch_list}')
        print(f'Delta double batch size list: {delta_dbl_batch_list}')
        loader.regenerate_loaders(batch_size=batch_size)


def debug_lbl_loader(loader: TrainLoader):
    for force_not_weighted, str_ in ((False, ''), (True, ' NOT')):
        batch_size = loader.unlabelled.batch_size
        loader.regenerate_loaders(batch_size=4 * 16 * 2, force_not_weighted=force_not_weighted)
        stats_list = [
            [
                len(np.where((target == i) & (sex == j) & (sex == j))[0])
                for i in range(1, 17) for j in (0, 1)]
            for (x, _, target_, _, _), _ in loader.get_lbl_loader()
            for sex, target in ([np.round(x.numpy()[:, 0]).astype(int), np.abs(target_.numpy())],)
        ]
        batch_list = [len(x) for (x, _, target_, _, _), _ in loader.get_lbl_loader()]
        stats = np.array(stats_list)
        stats = stats.reshape((stats.shape[0], 2, stats.shape[1] // 2))
        print(f'Average number of types should{str_} be around the same:\n{np.round(np.mean(stats, axis=0), 1)}')
        print(f'Last 3 lines:\n{stats[-3:].astype(float)}')
        print(f'x batch size list: {batch_list}')
        loader.regenerate_loaders(batch_size=batch_size)


@dataclass
class JATSModelOutput:
    x: Array
    z: Array
    zμ: Array  # deterministic μ if available else z
    fzμ: Array  # inverse flow applied to z in SOME cases of priors with flows
    μ: Array  # first from params
    σ: Array  # second from params
    x_rec_cont_zμ: Array
    x_rec_disc_zμ: Tuple[Array, ...]
    x_rec_sample_zμ: Array
    μz: Array  # mean(z)
    σz: Array  # std(z)
    a: Opt[Array]  # alt. latents
    b: Opt[Array]  # alt. latents #2
    c: Opt[Array]  # alt. latents #3
    y_probs: Opt[Array] = None
    y_probs2: Opt[Array] = None
    y_probs32: Opt[Array] = None
    zμ_new: Opt[Array] = None
    zμ_rec: Opt[Array] = None
    log: str = ''


WeightGetter = Callable[[Opt[List[int]]], Opt[Array]]
LearnSampleGetter = Callable[[], List[int]]


@dataclass
class Data:
    """
    ``*weight_mat`` should not incorporate per dim 0 ``*weights_vec``.
    """

    input: Array
    input_bucketed: Opt[Array]
    weight_vec: Opt[Array]
    weight_mat: Opt[Array]
    target: Opt[Array]
    interesting: Array

    indxs_labelled: Opt[List[int]]
    weight_vec_labelled: Opt[Array]

    test_indxs: List[int]
    test_input: Array
    test_weight_vec: Opt[Array]

    test_indxs_labelled: Opt[List[int]]
    test_weight_vec_labelled: Opt[Array]
    test_one_hot_target_labelled: Opt[Array]

    learn_indxs: List[int]
    learn_input: Array
    learn_weight_vec: Opt[Array]

    get_weight_vec_: Tuple[WeightGetter]
    rand_learn_sampler_size_of_test_: Tuple[LearnSampleGetter]
    rand_learn_lbl_sampler_size_of_test_lbl_: Opt[Tuple[LearnSampleGetter]] = None

    test_weight_mat: Opt[Array] = None
    test_weight_mat_labelled: Opt[Array] = None
    learn_weight_mat: Opt[Array] = None

    learn_indxs_labelled: Opt[List[int]] = None
    learn_weight_vec_labelled: Opt[Array] = None
    learn_one_hot_target_labelled: Opt[Array] = None

    @property
    def get_weight_vec(self) -> WeightGetter:
        return self.get_weight_vec_[0]

    @property
    def rand_learn_sampler_size_of_test(self) -> LearnSampleGetter:
        return self.rand_learn_sampler_size_of_test_[0]

    @property
    def rand_learn_lbl_sampler_size_of_test_lbl(self) -> LearnSampleGetter:
        att = self.rand_learn_lbl_sampler_size_of_test_lbl_
        if att is None:
            raise ValueError
        return att[0]

    def __post_init__(self):
        self.test_weight_mat = self.get_weight_mat(self.test_indxs)
        self.test_weight_mat_labelled = self.get_weight_mat(self.test_indxs_labelled)
        self.learn_weight_mat = self.get_weight_mat(self.learn_indxs)
        self._get_weight_vec = (self.get_weight_vec_,)

    def get_weight_mat(self, lst: List[int] = None) -> Opt[Array]:
        """ This function uses the fact that ``self.weight_mat`` doesn't incorporate sample weights. """
        if self.weight_mat is None:
            return None
        return self.weight_mat if (lst is None) else self.weight_mat[lst]


def load(trans: Transforms, db_spec: DBSpec, samples_per_class_32: int, dtype: tr.dtype=tr.float,
         use_weight: bool=True, use_weight_mat: bool=True, z_dim: int=12,
         labels: str='type') -> Tuple[TrainLoader, Dim, Data, DB]:
    """
    ``target`` is ``self_type`` when ``talanov_s_type`` ~= ``self_type`` or ``-talanov_s_type`` otherwise.
    """
    db = db_spec.reader()
    interesting: Array = db.interesting[:, 1]
    x = trans.to_0__1(trans.rep_2to1_4to5(db.profiles) if trans.rep24to15 else db.profiles, use_ε=False)
    dims = Dim(x=x.shape[1], buckets=trans.buckets_n, h=db_spec.h_dims, z=z_dim)
    x_bucketed = trans.to_categorical(db.profiles)
    def f(row: Array) -> Array: return weight_neg_pos(row, db_name=db_spec.name)
    weight_mat: Array = np.apply_along_axis(f, axis=-1, arr=db.profiles)
    # weight_mat = np.array([f(Q) for Q in db.profiles])

    test_indxs, learn_indxs, stats = read_test_samples(db_spec.name)
    _types_tal_sex = tuple(int(s) for s in db.types_tal_sex)

    def get_weight_vec(lst: Union[List[int], Array]=None, use_weight_: bool=use_weight,
                       types_tal_sex_: Tuple[int, ...]=_types_tal_sex) -> Opt[Array]:
        if not use_weight_:
            return None
        types_tal_sex = np.array(types_tal_sex_)
        return get_weight(types_tal_sex if (lst is None) else types_tal_sex[lst])

    labels_ = db.type_smart_coincide
    y_dim = 16
    mask = labels_ > 0

    if labels != 'type':
        labels_ = db.other_smart_coincide[labels]
        y_dim = db.other_y_dim[labels]
        mask = mask & (labels_ > 0)

    dims.set_y(y_dim)
    assert len(np.unique(labels_[mask])) == y_dim

    idxs_lbl = [int(s) for s in np.where(mask)[0]]
    #TODO: this creates random labelled rearn dataset
    # If Guide is intended to be used for learning of multiple models then this part should be moved somewhere
    test_lbl_indxs = [i for i in test_indxs if i in idxs_lbl]
    learn_lbl_indxs = [i for i in learn_indxs if i in idxs_lbl]
    # l means used in labelled sampler:
    learn_l_lbl_indxs = random_control_samples(stats, idxs_subset=idxs_lbl, take=samples_per_class_32)

    if len(x) != (len(test_indxs) + len(learn_indxs)):
        raise RuntimeError(f'{len(x)} != ({len(test_indxs)} + {len(learn_indxs)})')
    if len(idxs_lbl) != (len(test_lbl_indxs) + len(learn_lbl_indxs)):
        raise RuntimeError(
            f'{len(idxs_lbl)} != ({len(test_lbl_indxs)} + {len(learn_lbl_indxs)})')

    test_lbl_weight_vec = get_weight_vec(test_lbl_indxs)
    learn_l_lbl_weight_vec = get_weight_vec(learn_l_lbl_indxs)
    learn_lbl_weight_vec = get_weight_vec(learn_lbl_indxs)
    test_lbl_one_hot_target = categ_to_one_hot(labels_[test_lbl_indxs])
    learn_l_lbl_one_hot_target = categ_to_one_hot(labels_[learn_l_lbl_indxs])
    learn_lbl_one_hot_target = categ_to_one_hot(labels_[learn_lbl_indxs])

    learn_weight_vec = get_weight_vec(learn_indxs)

    train_loader = TrainLoader(
        input=x[learn_indxs],
        input_bucketed=x_bucketed[learn_indxs],
        target=db.type_smart_coincide[learn_indxs],
        weight_vec=learn_weight_vec,
        weight_mat=weight_mat[learn_indxs] if use_weight_mat else None,

        input_lbl=x[learn_l_lbl_indxs],
        input_lbl_bucketed=x_bucketed[learn_l_lbl_indxs],
        target_lbl=db.type_smart_coincide[learn_l_lbl_indxs],
        weight_vec_lbl=learn_l_lbl_weight_vec,
        weight_mat_lbl=weight_mat[learn_l_lbl_indxs] if use_weight_mat else None,
        one_hot_target_lbl=learn_l_lbl_one_hot_target,

        dtype=dtype)

    def rand_learn_sampler_size_of_test_() -> List[int]:
        return random_control_samples(stats)

    def rand_learn_lbl_sampler_size_of_test_lbl_() -> List[int]:
        return random_control_samples(stats, idxs_subset=idxs_lbl)

    data = Data(
        input=x, input_bucketed=x_bucketed, weight_vec=get_weight_vec(),
        weight_mat=weight_mat if use_weight_mat else None,
        target=db.type_smart_coincide, interesting=interesting,

        indxs_labelled=idxs_lbl,
        weight_vec_labelled=get_weight_vec(idxs_lbl),

        test_indxs=test_indxs,
        test_input=x[test_indxs],
        test_weight_vec=get_weight_vec(test_indxs),

        learn_indxs=learn_indxs,
        learn_input=x[learn_indxs],
        learn_weight_vec=get_weight_vec(learn_indxs),

        get_weight_vec_=(get_weight_vec,),

        rand_learn_sampler_size_of_test_=(rand_learn_sampler_size_of_test_,),
        rand_learn_lbl_sampler_size_of_test_lbl_=(rand_learn_lbl_sampler_size_of_test_lbl_,),

        test_indxs_labelled=test_lbl_indxs,
        test_weight_vec_labelled=test_lbl_weight_vec,
        test_one_hot_target_labelled=test_lbl_one_hot_target,

        learn_indxs_labelled=learn_lbl_indxs,
        learn_weight_vec_labelled=learn_lbl_weight_vec,
        learn_one_hot_target_labelled=learn_lbl_one_hot_target,
    )
    return train_loader, dims, data, db
