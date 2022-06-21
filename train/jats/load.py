from typing import Tuple
import numpy as np
from numpy.typing import NDArray as Array
from pandas import DataFrame
import torch as tr
from torch import Tensor
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader, TensorDataset

MAIN_QUEST_N = 160
EXTRA_QUEST = ['sex', 'age', 'education', 'language']


def replace_2to1_4to5(profiles: Array) -> Array:
    return 2 * np.sign(profiles - 3) + 3


def map_to_0__1(profiles: Array, use_eps: bool = True, eps=10 ** -8, alpha=1.) -> Array:
    """ Maps (1, 2, 3, 4, 5) to (0 + eps, f(alpha), 0.5, 1 - f(alpha), 1 - eps) """
    profiles = 0.5 * profiles - 1.5  # to  -1, -0.5, 0, 0.5, 1
    profiles = np.sign(profiles) * np.power(np.abs(profiles), alpha)  # to  -1, -b, 0, b, 1
    profiles = np.round(0.5 * (profiles + 1), 4)  # to  0, c, 0.5, 1-c, 1
    if use_eps:
        profiles[profiles > (1 - eps)] = 1 - eps
        profiles[profiles < eps] = eps
    return profiles


def get_weight_32(types_sex: Array) -> Array:
    """
    Returns weights suitable both for ``torch.utils.data.sampler.WeightedRandomSampler``
    and for weighted losses via simple (x * weight).sum().
    Input should be from ``range(1, 32 + 1)``.
    ``MALE_LABEL_SHIFT`` from ``preprocess_db`` should guarantee this.
    """
    uni, counts = np.unique(types_sex, return_counts=True)
    if len(counts) != 32:
        raise ValueError(
            f"types_tal_sex doesn't have all 32 types samples: {len(counts)} ({np.sum(counts)}, {uni}, {counts})")
    return (1. / counts)[types_sex - 1] / 32


def get_weight_16(types: Array) -> Array:
    """
    Returns weights suitable both for ``torch.utils.data.sampler.WeightedRandomSampler``
    and for weighted losses via simple (x * weight).sum().
    Input should be from ``range(1, 16 + 1)``.
    """
    uni, counts = np.unique(types, return_counts=True)
    if len(counts) != 16:
        raise ValueError(
            f"types_tal_sex doesn't have all 16 types samples: {len(counts)} ({np.sum(counts)}, {uni}, {counts})")
    return (1. / counts)[types - 1] / 16


def get_data(df: DataFrame) -> Tuple[Array, Array, Array, Array, Array, Array, Array]:
    """
    Returns tuple of:

    * IDs array (integer).
    * passthrough part of the X (binary).
    * X without passthrough part (in pseudo Bernoulli format like in Kingma, Welling, Auto-Encoding Variational Bayes).
    * Extra X - same as X but doesn't needed in the decoder output (same format as X).
    * Target (class indices in the range [0,... 15] when sure labels and in the range [-16,... -1] otherwise).
    * Weights suitable both for ``torch.utils.data.sampler.WeightedRandomSampler``
      and for weighted losses via simple (x * weight).sum().
    * target_type_sex (non-smart)
    """
    ids = df['id'].values.reshape(-1, 1)
    passthr = map_to_0__1(df['sex'].values.reshape(-1, 1))
    x = map_to_0__1(replace_2to1_4to5(df[[str(i) for i in range(1, MAIN_QUEST_N + 1)]].values))
    x_extra = map_to_0__1(replace_2to1_4to5(df[EXTRA_QUEST].values))
    smart_type_sex = df['smart_type_sex'].values

    target = df['smart_coincide'].values.copy()
    mask = smart_type_sex > 0
    target[mask] = target[mask] - 1
    if (tuple(int(i) for i in np.unique(target[mask])) != tuple(range(0, 16))
        or tuple(int(i) for i in np.unique(target[~mask])) not in [tuple(range(-16, 0)), ()]
        ): raise RuntimeError('It was not in vain that I was not completely sure of the code above.')
    return ids, passthr, x, x_extra, target, get_weight_32(np.abs(smart_type_sex)), np.abs(smart_type_sex)


def get_target_stratify(df: DataFrame) -> Array:
    return df['smart_type_sex'].values


def get_labeled_mask(df: DataFrame) -> Array:
    return df['smart_coincide'].values > 0


def get_loader(df: DataFrame, mode: str, batch_size: int, num_workers: int = 0) -> DataLoader:
    """
    Returns a loader of the output format:

    * ``(x, x_ext)`` if ``mode='unlbl'``,
    * ``(x, x_ext, target, passth)`` if ``mode='lbl'``,
    * ``(x, x_ext, target, passth, weights, weights_lbl, mask_lbl)`` if ``mode='both'``,
    * ``(x, x_ext, passth)`` if ``mode='plot'``,
    """
    if mode not in ('unlbl', 'lbl', 'both', 'plot'): raise ValueError

    if mode is 'unlbl':
        _, passthr, x, e_ext, _, weights, _ = get_data(df)
        dataset = TensorDataset(Tensor(x), Tensor(e_ext), Tensor(passthr))
        sampler = WeightedRandomSampler(weights=weights.astype(np.float64),
                                        num_samples=len(x))
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    if mode == 'lbl':
        _, passthr, x, e_ext, target, weights, _ = get_data(df[get_labeled_mask(df)])
        dataset = TensorDataset(Tensor(x), Tensor(e_ext),
                                Tensor(target).to(dtype=tr.long), Tensor(passthr))
        sampler = WeightedRandomSampler(weights=weights.astype(np.float64),
                                        num_samples=len(x))
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    if mode == 'both':
        _, passthr, x, e_ext, target, weights, _ = get_data(df)
        weights_lbl = np.copy(weights)
        mask_lbl = get_labeled_mask(df)
        weights_lbl[mask_lbl] = get_data(df[mask_lbl])[5]

        dataset = TensorDataset(Tensor(x), Tensor(e_ext),
                                Tensor(target).to(dtype=tr.long), Tensor(passthr),
                                Tensor(weights.astype(np.float64)),
                                Tensor(weights_lbl.astype(np.float64)),
                                Tensor(mask_lbl).to(dtype=tr.bool))
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    if mode == 'plot':
        _, passthr, x, e_ext, _, weights, _ = get_data(df[get_labeled_mask(df)])
        dataset = TensorDataset(Tensor(x), Tensor(e_ext), Tensor(passthr))
        sampler = WeightedRandomSampler(weights=weights.astype(np.float64),
                                        num_samples=len(x))
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
