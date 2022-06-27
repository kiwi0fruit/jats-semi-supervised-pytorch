from typing import List, Dict, Tuple
import numpy as np
from numpy.typing import NDArray as Array
import pandas as pd
from pandas import DataFrame

MISSING_TYPES: Tuple[int, ...] = tuple(range(1, 17))
EXTRA_NO_SELF_TYPES = False
MALE_LABEL_SHIFT = 16  # `get_weight` function assumes this


def check_32_sex_types(df: DataFrame):
    """
    Check sex-type balance for supervised part.
    """
    types_self = df['self'].values
    types_tal = df['diagnosis'].values
    sex = df['sex'].values
    typed_mask = types_self == types_tal
    men_mask = sex > np.min(sex)
    out = []
    for type_ in range(1, 17):
        type_i_mask = types_tal == type_
        out.append([type_, 0, len(types_tal[type_i_mask & typed_mask & ~men_mask])])
        out.append([type_, 1, len(types_tal[type_i_mask & typed_mask & men_mask])])
    print(out)


def smart_coincide_0(
        tal_profs: Array, types_self: Array, types_tal: Array, threshold: int = -90,
        thresholds: Tuple[Tuple[int, Tuple[int, ...]], ...] = ()) -> Array:
    """
    Old: threshold=90, thresholds_plus=((81, (4, 8, 16)),)

    :param tal_profs: of shape (~30000, 16) of float
    :param types_self: of shape (~30000,) from {-1, 1, ..., 16}
    :param types_tal: of shape (~30000,) from {1, ..., 16}
    :param threshold: default threshold for smart Talanov's types.
        Positive: self type can be from a set of Talanov's types formed by threshold percent from max type scale.
        Zero: self type should coincide with Talanov's.
        Negative: self type should coincide with Talanov's and additionally the next type
        should be not closer than threshold percent from max type scale.
    :param thresholds: custom thresholds per type like ((81, (4, 16)),) that would turn into {81: (4, 16)}
    :return: of shape (~30000,) in {-16, ..., -1, 1, ..., 16} positive when smart coincided
    """
    if len(tal_profs) != len(types_self): raise ValueError

    tal_profs = np.round(np.einsum('ij,i->ij', tal_profs, np.max(tal_profs, axis=1)**-1) * 100).astype(int)
    types_tal_one_hot = np.eye(16).astype(int)[types_tal - 1]

    def trimmed_tal_profs(thr: int) -> Array:
        if thr == 0:
            return 100 * types_tal_one_hot
        ret = np.copy(tal_profs)
        ret[ret < abs(thr)] = 0
        return ret

    thr_types = dict(thresholds)
    defined = [i for types in thr_types.values() for i in types]
    if (len(defined) != len(set(defined))) or (threshold in thr_types): raise AssertionError
    thr_types[threshold] = tuple(i for i in range(1, 17) if i not in defined)
    absthr_arr: Dict[int, Array] = {thr: trimmed_tal_profs(thr) for thr in set(abs(t) for t in thr_types) | {0}}

    def get_thr(type_: int) -> int:
        if type_ == -1:
            return 0
        for thr, types in thr_types.items():
            if type_ in types:
                return thr
        raise AssertionError

    type_thr: Dict[int, int] = {i: get_thr(i) for i in range(-1, 17) if i != 0}

    tal_profs_ = np.array([absthr_arr[abs(type_thr[n])][i] for i, n in enumerate(types_self)])

    map_: Dict[int, int] = {i: i for i in range(1, 17)}

    def kernel(bests: List[int], self: int, tal: int, thr_pos: bool) -> int:
        if not (tal >= 1): raise AssertionError
        if self < 1:
            return -tal
        if thr_pos:
            return self if (self in bests) else -tal
        return self if (len(bests) == 1) and (bests[0] == self) else -tal

    smart_coin = np.array([kernel(
        bests=[map_[int(s)] for s in list(np.where(row > abs(type_thr[self]))[0] + 1)],
        self=map_[int(self)] if (self >= 1) else -1,
        tal=map_[int(tal)],
        thr_pos=type_thr[self] > 0
    ) for row, self, tal in zip(tal_profs_, types_self, types_tal)])

    return smart_coin


def smart_coincide_1(tal_profs: Array, types_self: Array, types_tal: Array, males: Array,
                     threshold: int,
                     thresholds_males: Tuple[Tuple[int, Tuple[int, ...]], ...] = (),
                     thresholds_females: Tuple[Tuple[int, Tuple[int, ...]], ...] = ()) -> Array:
    smart_coin_males = smart_coincide_0(
        tal_profs=tal_profs, types_self=types_self, types_tal=types_tal, threshold=threshold,
        thresholds=thresholds_males
    )
    smart_coin_females = smart_coincide_0(
        tal_profs=tal_profs, types_self=types_self, types_tal=types_tal, threshold=threshold,
        thresholds=thresholds_females
    )
    smart_coin = smart_coin_females
    smart_coin[males] = smart_coin_males[males]
    return smart_coin


def types_tal_good_mask(df: DataFrame,
                        second_type_gap_pc_thr: int = 67,
                        k_the_sigma_thr: float = -2,
                        k_halves_correl_thr: float = -2) -> Array:
    """
    :param df: of shape (~30000, K)
    :param second_type_gap_pc_thr: threshold for second type.
        Default: the 2nd type should be <= 67% of the 1st type.
    :param k_the_sigma_thr: threshold is mean(the_sigma) + k_the_sigma_thr * std(the_sigma)
    :param k_halves_correl_thr: threshold is mean(halves_correl) + k_halves_correl_thr * std(halves_correl)
    :return: bool mask of shape (~30000,) that is True when profile is "good".
    """
    tal_profs = df[[f't{i}' for i in range(1, 17)]].values
    the_sigma = df['sigma'].values
    halves_correl = df['halves_correl'].values

    sort = np.sort(tal_profs, axis=-1)
    sort[sort <= 0] = 0
    sort = np.round(np.einsum('ij,i->ij', sort, np.max(tal_profs, axis=-1)**-1) * 100).astype(int)
    second_type_gap_mask = (sort[:, -1] == 100) & (sort[:, -2] <= second_type_gap_pc_thr)

    the_sigma_mask = the_sigma >= (np.mean(the_sigma) + k_the_sigma_thr * np.std(the_sigma))
    halves_correl_mask = halves_correl >= (np.mean(halves_correl) + k_halves_correl_thr * np.std(halves_correl))

    return second_type_gap_mask & the_sigma_mask & halves_correl_mask


def smart_coincide_2(
        tal_profs: Array, types_self: Array, types_tal: Array, males: Array,
        threshold: int = 95,  # was 90,
        thresholds_males: Tuple[Tuple[int, Tuple[int, ...]], ...] = (
            # (85, (4,)), (95, (3, 10, 16)), (-90, (12, 13))
            (85, (4,)),
        ),
        thresholds_females: Tuple[Tuple[int, Tuple[int, ...]], ...] = (
            # (95, (16,)),
        )) -> Array:
    return smart_coincide_1(tal_profs=tal_profs, types_self=types_self, types_tal=types_tal, males=males,
                            threshold=threshold,
                            thresholds_males=thresholds_males, thresholds_females=thresholds_females)


def preprocess_profiles(df: DataFrame) -> DataFrame:
    """ Writes DF on disk and returns it."""
    tal_profs_columns = [f't{i}' for i in range(1, 17)]
    tal_profs = df[tal_profs_columns].values
    types_self = df['self'].values
    types_tal = df['diagnosis'].values
    sex: Array = df['sex'].values
    males: Array = sex == 5

    if tuple(np.unique(types_tal)) != tuple(range(1, 17)): raise ValueError
    if tuple(np.unique(types_self)) != ((-1,) + tuple(range(1, 17))): raise ValueError
    if tuple(np.unique(sex)) != (1, 5): raise ValueError

    def _types_self_extra() -> Array:
        good = types_tal_good_mask(df=df)
        good_no_self = good & (types_self == -1)

        types_ = types_tal == MISSING_TYPES[0]
        for type_ in MISSING_TYPES[1:]:
            types_ = types_ | (types_tal == type_)
        good_no_self_types = good_no_self & types_

        types_self_extra_ = np.copy(types_self)
        types_self_extra_[good_no_self_types] = types_tal[good_no_self_types]
        if EXTRA_NO_SELF_TYPES:
            def add_extra(special_type: int, second_type_gap_pc_thr: int, mask: Array = None):
                good_special = types_tal_good_mask(df=df, second_type_gap_pc_thr=second_type_gap_pc_thr)
                good_no_self_special = good_special & (types_self == -1) & (types_tal == special_type)
                if mask is not None:
                    good_no_self_special = good_no_self_special & mask
                types_self_extra_[good_no_self_special] = types_tal[good_no_self_special]

            add_extra(3, 80, males)
            add_extra(3, 70, ~males)
            add_extra(4, 85, males)
            add_extra(9, 70, males)
            add_extra(10, 80, males)
            add_extra(12, 70, males)
            add_extra(12, 75, ~males)
            add_extra(13, 70, males)
            add_extra(16, 85)
        return types_self_extra_

    types_self_extra = _types_self_extra()
    types_tal_ = np.copy(types_tal)

    type_smart_coin = smart_coincide_2(tal_profs=tal_profs, types_self=types_self_extra, types_tal=types_tal_,
                                       males=males)

    types_mod_tal = np.copy(types_tal)
    types_mod_tal[type_smart_coin > 0] = type_smart_coin[type_smart_coin > 0]

    types_mod_tal_sex = np.copy(types_mod_tal)
    types_mod_tal_sex[males] += MALE_LABEL_SHIFT
    types_mod_tal_sex = types_mod_tal_sex * np.sign(type_smart_coin)

    if len({len(df), len(type_smart_coin), len(types_mod_tal)}) != 1:
        raise ValueError('Data has inconsistent dimensions.')
    df.insert(len(df.columns), "smart_type_sex", types_mod_tal_sex)
    df.insert(len(df.columns), "smart_coincide", type_smart_coin)

    df = df.drop(columns=['self', 'diagnosis', 'sigma', 'halves_correl'] + tal_profs_columns)
    df.to_csv('db_final.csv', index=False)
    return df


def check_32_sex_types2(df: DataFrame):
    """
    Check sex-type balance for supervised part.
    Min result is [4, 1, 35].
    """
    smart_coincide = df['smart_coincide'].values
    sex = df['sex'].values
    typed_mask = smart_coincide > 0
    men_mask = sex > np.min(sex)
    out = []
    for type_ in range(1, 17):
        type_i_mask = smart_coincide == type_
        out.append([type_, 0, len(smart_coincide[type_i_mask & typed_mask & ~men_mask])])
        out.append([type_, 1, len(smart_coincide[type_i_mask & typed_mask & men_mask])])
    print(out)


if __name__ == '__main__':
    check_32_sex_types(pd.read_csv(f'db.csv'))
    check_32_sex_types2(preprocess_profiles(pd.read_csv(f'db.csv')))
