"""
[Optional] When using db_out.csv check consistency:
  should be the same ID column in raw and out.
  With this I won't need the ID column at all.
"""
import os.path as p
from typing import NamedTuple, List, Tuple, Callable, Dict, Any
import ast
import pandas as pd
from pandas import DataFrame
import numpy as np
# noinspection PyPep8Naming
from numpy import ndarray as Array
from kiwi_bugfix_typechecker import test_assert

BETTER: bool = False  # use good enough types or better types

test_assert()
TO_DICT: Dict[str, Dict[int, int]] = dict(
    dom={
        1: 1, 2: 2, 3: 3, 4: 4,
        5: 5, 6: 2, 7: 6, 8: 4,
        9: 5, 10: 7, 11: 6, 12: 8,
        13: 1, 14: 7, 15: 3, 16: 8
    },
    temper={
        1: 1, 2: 2, 3: 3, 4: 4,
        5: 1, 6: 2, 7: 3, 8: 4,
        9: 1, 10: 2, 11: 3, 12: 4,
        13: 1, 14: 2, 15: 3, 16: 4
    },
    quadra_club={
        1: 1, 2: 1, 3: 2, 4: 2,
        5: 3, 6: 3, 7: 4, 8: 4,
        9: 5, 10: 5, 11: 6, 12: 6,
        13: 7, 14: 7, 15: 8, 16: 8
    },
    quadra={
        1: 1, 2: 1, 3: 1, 4: 1,
        5: 2, 6: 2, 7: 2, 8: 2,
        9: 3, 10: 3, 11: 3, 12: 3,
        13: 4, 14: 4, 15: 4, 16: 4
    },
    club={
        1: 1, 2: 1, 3: 2, 4: 2,
        5: 3, 6: 3, 7: 4, 8: 4,
        9: 2, 10: 2, 11: 1, 12: 1,
        13: 4, 14: 4, 15: 3, 16: 3
    },
    mr={
        1: 1, 2: 2, 3: 1, 4: 2,
        5: 1, 6: 2, 7: 1, 8: 2,
        9: 1, 10: 2, 11: 1, 12: 2,
        13: 1, 14: 2, 15: 1, 16: 2
    },
    ei={
        1: 1, 2: 2, 3: 2, 4: 1,
        5: 1, 6: 2, 7: 2, 8: 1,
        9: 1, 10: 2, 11: 2, 12: 1,
        13: 1, 14: 2, 15: 2, 16: 1
    },
    mr_tf={
        1: 1, 2: 2, 3: 3, 4: 4,
        5: 1, 6: 2, 7: 3, 8: 4,
        9: 3, 10: 4, 11: 1, 12: 2,
        13: 3, 14: 4, 15: 1, 16: 2
    },
    xir_xer={
        1: 1, 2: 1, 3: 2, 4: 2,
        5: 1, 6: 1, 7: 2, 8: 2,
        9: 1, 10: 1, 11: 2, 12: 2,
        13: 1, 14: 1, 15: 2, 16: 2
    },
    lc={
        1: 1, 2: 1, 3: 1, 4: 1,
        5: 2, 6: 2, 7: 2, 8: 2,
        9: 2, 10: 2, 11: 2, 12: 2,
        13: 1, 14: 1, 15: 1, 16: 1
    },
)
BOLTI = 'BOLTI_434__2017_08_18__N3197'
SOLTI = 'SOLTI_160__2016_03_20__N6406'
SOLTI_ENG = 'SOLTI_160_ENG__2019_08_07__NXXXX'

EXTRA_QUESTIONS = ['sex in (female, male)', 'age in (0-20, 21-25, 26-30, 31-40, 41-100)']
EXTRA_COLUMNS_IDS = ['sex', 'age']
SOLTI_ENG_ID_Δ = 20000
# MISSING_TYPES = (3, 4, 9, 10, 16)  # (3, 4, 10, 12, 16)
MISSING_TYPES = tuple(range(1, 17))


class DB(NamedTuple):
    """
    6000 is an example number of completed questionnaires.
    160 is an example number of questions in the questionnaire.

    Attributes:
    -----------
    profiles : np.ndarray
      2D like (~6000, ~2 + ~160) shape
    df : pd.DataFrame
      2D like (~6000, ~8 + ~160) shape
    types_tal : np.ndarray
      1D like (~6000,) shape; in {1, ..., 16}
    types_self : np.ndarray
      1D like (~6000,) shape; in {-1, 1, ..., 16}
    types_tal_sex : np.ndarray
      1D like (~6000,) shape; in {1, ..., 16, 17, ..., 32} females first
    types_smart_coincide : np.ndarray
      1D like (~6000,) shape; in {-16, ..., -1, 1, ..., 16}
    questions : List[Tuple[str, str]]
      2D like (~160, ~2) shape. First one is original Russian question.
      Second one is autotranslated English question.
    interesting : np.ndarray
      2D like (~10, ~2) shape. Interesting indexes (not IDs!).
      First is a real type. Second is index.
    """
    profiles: Array
    df: DataFrame
    types_tal: Array
    types_self: Array
    types_tal_sex: Array
    questions: List[Tuple[str, str]]
    interesting: Array
    type_smart_coincide: Array
    other_smart_coincide: Dict[str, Array]
    other_y_dim: Dict[str, int]


class DBSpec:
    name: str
    _reader: Tuple[Callable[[], DB]]
    h_dims: Tuple[int, ...]

    def __init__(self, name: str, reader: Callable[[], DB], h_dims: Tuple[int, ...]):
        self.name = name
        self.h_dims = h_dims
        self._reader = (reader,)

    @property
    def reader(self) -> Callable[[], DB]:
        return self._reader[0]

    def dump(self) -> Dict[str, Any]:
        return dict(name=self.name, h_dims=self.h_dims)


def read_profiles(profile_name: str) -> Tuple[DataFrame, DataFrame, List[str], List[str], List[List[int]]]:
    """
    * Questions lists are prepended with 'sex' and 'age' descriptions.
    * ``sex`` is mapped: 0 to 1, 1 to 5.
    * ``self`` empty string is mapped to 0.
    * ``confidence`` empty string and ``'None'`` is mapped to ``-1``.

    :param profile_name:
    :return:
        (raw_data_frame, out_data_frame, questions, questions_eng, interesting_ids)
    """
    _dir = p.join(p.dirname(p.abspath(__file__)), profile_name)

    def _questions(file_name: str):
        with open(p.join(_dir, file_name), 'r', encoding='utf-8') as f_:
            return EXTRA_QUESTIONS + f_.read().strip().splitlines()

    df_raw = pd.read_csv(p.join(_dir, 'db_raw.csv'), converters=dict(
        sex=lambda s: 5 if (int(s) == 1) else 1,
        self=lambda s: int(s) if s else -1,
        confidence=lambda s: int(s) if s not in ('None', '') else -1,
    ))
    df_out = pd.read_csv(p.join(_dir, 'db_out.csv'))

    if len(df_out) != len(df_raw):
        raise ValueError('Inconsistent db_raw.csv and db_out.csv length.')

    questions = _questions('questions.txt')
    questions_eng = _questions('questions_autotranslated.txt')

    columns = df_raw.columns.values.tolist()
    if ('sex' not in columns) or ('age' not in columns):
        raise ValueError("Either 'sex' or 'age' is not in the columns names.")
    quest_n = len([name for name in columns if name.isdigit()]) + len(EXTRA_QUESTIONS)
    if (quest_n != len(questions)) or (quest_n != len(questions_eng)):
        raise ValueError("Inconsistent number of questions.")

    with open(p.join(_dir, 'interesting_ids.ast'), 'r', encoding='utf-8') as f:
        interesting_ids = ast.literal_eval(f.read())

    # patch ids
    add_file = p.join(_dir, 'dbo_ids_add.ast')
    remove_file = p.join(_dir, 'dbo_ids_remove.ast')
    add_ids: List[int] = []
    remove_ids: List[int] = []
    if p.isfile(add_file):
        with open(add_file, 'r', encoding='utf-8') as f:
            add_ids = ast.literal_eval(f.read())
    if p.isfile(remove_file):
        with open(remove_file, 'r', encoding='utf-8') as f:
            remove_ids = ast.literal_eval(f.read())

    Δ = SOLTI_ENG_ID_Δ if (profile_name == SOLTI_ENG) else 0
    remove_ids = [i + Δ for i in remove_ids if i not in add_ids]

    ids = df_raw['id'].values
    if remove_ids:
        idxs = [idx for idx, id_ in enumerate(ids) if id_ in remove_ids]
        df_raw = df_raw.drop(df_raw.index[idxs])
        df_out = df_out.drop(df_out.index[idxs])
        ids = df_raw['id'].values
    elif profile_name == SOLTI_ENG:
        raise ValueError

    main_quest_n = quest_n - len(EXTRA_QUESTIONS)
    profs = df_raw[[str(i) for i in range(1, main_quest_n + 1)]].values

    max_same_quest = np.array([np.max(np.unique(row, return_counts=True)[1]) for row in profs])
    del_mask = max_same_quest > np.mean(max_same_quest) + 4 * np.std(max_same_quest)
    if len(ids[del_mask]) > 0:
        del_ids = ids[del_mask] - Δ
        raise ValueError(f'These IDs have too many same questions: (for {profile_name}): {list(del_ids)}'
                         + f' (counts: {list(max_same_quest[del_mask])})')

    corr = np.corrcoef(profs) - np.eye(len(profs))
    if np.isnan(corr).any():
        del_ids = ids[list(set(i for i, row in enumerate(corr) if all(np.isnan(row))))] - Δ
        raise ValueError(f'These IDs give NaN correlations with other IDs (for {profile_name}): {list(del_ids)}')

    mask = np.max(corr, axis=1) >= 0.99
    if len(corr[mask]) > 0:
        idxs = np.arange(0, len(profs))
        has_equals = [int(i) for i in idxs[mask]]
        del_idxs: List[int] = []
        for i in has_equals.copy():
            for j, c in enumerate(corr[i]):
                if (c >= 0.99) and (i in has_equals):
                    del_idxs.append(j)
                    has_equals = [s for s in has_equals if s != j]
        assert len(has_equals) == len(set(has_equals))
        del_ids = ids[del_idxs] - Δ
        raise ValueError(f'Duplicate profiles. Recommended to delete IDs (for {profile_name}): {list(del_ids)}')
    return df_raw, df_out, questions, questions_eng, interesting_ids


def types_tal_good_mask(df_out: DataFrame, tal_profs: Array,
                        second_type_gap_pc_thr: int=70,
                        k_the_sigma_thr: float=-2,
                        k_halves_correl_thr: float=-2) -> Array:
    """
    :param df_out: of shape (~6000, K)
    :param tal_profs: of shape (~6000, 16)
    :param second_type_gap_pc_thr: threshold for second type.
        Default: the 2nd type should be <= 67% of the 1st type.
    :param k_the_sigma_thr: threshold is mean(the_sigma) + k_the_sigma_thr * std(the_sigma)
    :param k_halves_correl_thr: threshold is mean(halves_correl) + k_halves_correl_thr * std(halves_correl)
    :return: bool mask of shape (~6000,) that is True when profile is "good".
    """
    sort = np.sort(tal_profs, axis=-1)
    sort[sort <= 0] = 0
    sort = np.round(np.einsum('ij,i->ij', sort, np.max(tal_profs, axis=-1)**-1) * 100).astype(int)
    second_type_gap_mask = (sort[:, -1] == 100) & (sort[:, -2] <= second_type_gap_pc_thr)

    the_sigma = df_out['sigma_of_the_profile'].values
    the_sigma_mask = the_sigma >= (np.mean(the_sigma) + k_the_sigma_thr * np.std(the_sigma))

    halves_correl = df_out['correl_of_the_halves'].values
    halves_correl_mask = halves_correl >= (np.mean(halves_correl) + k_halves_correl_thr * np.std(halves_correl))

    return second_type_gap_mask & the_sigma_mask & halves_correl_mask


def _smart_coincide(
        tal_profs: Array, types_self: Array, types_tal: Array, threshold: int=-80,
        thresholds: Tuple[Tuple[int, Tuple[int, ...]], ...]=(), labels: str='type') -> Array:
    """
    >>> TO_DICT

    Old: threshold=90, thresholds_plus=((81, (4, 8, 16)),)

    :param tal_profs: of shape (~6000, 16) of float
    :param types_self: of shape (~6000,) from {-1, 1, ..., 16}
    :param types_tal: of shape (~6000,) from {1, ..., 16}
    :param threshold: default threshold for smart Talanov's types.
        Positive: self type can be from a set of Talanov's types formed by threshold percent from max type scale.
        Zero: self type should coincide with Talanov's.
        Negative: self type should coincide with Talanov's and additionally the next type
        should be not closer than threshold percent from max type scale.
    :param thresholds: custom thresholds per type like ((81, (4, 16)),) that would turn into {81: (4, 16)}
    :param labels: for other values see keys of TO_DICT const
    :return: of shape (~6000,) in {-16, ..., -1, 1, ..., 16} positive when smart coincided
    """
    if len(tal_profs) != len(types_self):
        raise ValueError('Inconsistent tal_profs and types_self length.')

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
    assert (len(defined) == len(set(defined))) and (threshold not in thr_types)
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

    if labels == 'type':
        map_: Dict[int, int] = {i: i for i in range(1, 17)}
    else:
        map_ = TO_DICT[labels]

    def kernel(bests: List[int], self: int, tal: int, thr_pos: bool) -> int:
        assert tal >= 1
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


def smart_coincide_db(tal_profs: Array, types_self: Array, types_tal: Array, males: Array,
                      threshold: int=90,
                      labels: str='type',
                      thresholds_males: Tuple[Tuple[int, Tuple[int, ...]], ...]=(),
                      thresholds_females: Tuple[Tuple[int, Tuple[int, ...]], ...]=()) -> Array:
    smart_coin_males = _smart_coincide(
        tal_profs=tal_profs, types_self=types_self, types_tal=types_tal, labels=labels, threshold=threshold,
        thresholds=thresholds_males
    )
    smart_coin_females = _smart_coincide(
        tal_profs=tal_profs, types_self=types_self, types_tal=types_tal, labels=labels, threshold=threshold,
        thresholds=thresholds_females
    )
    smart_coin = smart_coin_females
    smart_coin[males] = smart_coin_males[males]
    return smart_coin


def smart_coincide_solti_good(
        tal_profs: Array, types_self: Array, types_tal: Array, males: Array,
        threshold: int=90,
        labels: str='type',
        thresholds_males: Tuple[Tuple[int, Tuple[int, ...]], ...]=(
            (81, (4,)),
        ),
        thresholds_females: Tuple[Tuple[int, Tuple[int, ...]], ...]=()) -> Array:
    return smart_coincide_db(tal_profs=tal_profs, types_self=types_self, types_tal=types_tal, males=males,
                             threshold=threshold, labels=labels,
                             thresholds_males=thresholds_males, thresholds_females=thresholds_females)


def smart_coincide_solti_better(
        tal_profs: Array, types_self: Array, types_tal: Array, males: Array,
        threshold: int=-80,
        labels: str='type',
        thresholds_males: Tuple[Tuple[int, Tuple[int, ...]], ...]=(
            (-99, (16,)), (-90, (3, 4)),  # old: (81, (3,)), (-97, (3, 16)),
        ),
        thresholds_females: Tuple[Tuple[int, Tuple[int, ...]], ...]=(
            (-90, (16,)),  # old: (-95, (16,)),
        )) -> Array:
    return smart_coincide_db(tal_profs=tal_profs, types_self=types_self, types_tal=types_tal, males=males,
                             threshold=threshold, labels=labels,
                             thresholds_males=thresholds_males, thresholds_females=thresholds_females)


def smart_coincide_bolti(
        tal_profs: Array, types_self: Array, types_tal: Array, males: Array,
        threshold: int=90,
        labels: str='type',
        thresholds_males: Tuple[Tuple[int, Tuple[int, ...]], ...]=(
            (81, (3, 4, 16)),
        ),
        thresholds_females: Tuple[Tuple[int, Tuple[int, ...]], ...]=()) -> Array:
    return smart_coincide_db(tal_profs=tal_profs, types_self=types_self, types_tal=types_tal, males=males,
                             threshold=threshold, labels=labels,
                             thresholds_males=thresholds_males, thresholds_females=thresholds_females)


def preprocess_profiles(df_raw: DataFrame,
                        df_out: DataFrame,
                        questions: List[str],
                        questions_eng: List[str],
                        select_columns: List[str],
                        interesting_indexes: Array,
                        db_name: str=None) -> DB:
    tal_profs = df_out.loc[:, [str(i) for i in range(1, 17)]].values

    types_self = df_raw['self'].values
    types_tal = df_raw['diagnosis'].values
    profiles = df_raw.loc[:, select_columns].values
    sex = profiles[:, 0]

    if tuple(np.unique(types_tal)) != tuple(range(1, 17)):
        raise ValueError
    if tuple(np.unique(types_self)) != ((-1,) + tuple(range(1, 17))):
        raise ValueError
    if tuple(np.unique(sex)) != (1, 5):
        raise ValueError

    def _types_self_extra() -> Array:
        good = types_tal_good_mask(df_out=df_out, tal_profs=tal_profs)
        good_no_self = good & (types_self == -1)  # & (sex == 5)

        types_ = types_tal == MISSING_TYPES[0]
        for type_ in MISSING_TYPES[1:]:
            types_ = types_ | (types_tal == type_)
        good_no_self_types = good_no_self & types_

        types_self_extra_ = np.copy(types_self)
        types_self_extra_[good_no_self_types] = types_tal[good_no_self_types]
        if db_name == 'bolti':
            special_type = 4
            good_special = types_tal_good_mask(
                df_out=df_out, tal_profs=tal_profs,
                second_type_gap_pc_thr=90, k_the_sigma_thr=-2, k_halves_correl_thr=-2)
            good_no_self_special = good_special & (types_self == -1) & (types_tal == special_type)
            # print(len(types_self[good_no_self_special]))
            # raise
            types_self_extra_[good_no_self_special] = types_tal[good_no_self_special]

        return types_self_extra_

    types_self_extra = _types_self_extra()
    types_tal_ = np.copy(types_tal)
    males = profiles[:, 0] == 5

    if db_name == 'bolti':
        smart_coincide = smart_coincide_bolti
    elif BETTER:
        smart_coincide = smart_coincide_solti_better
    else:
        smart_coincide = smart_coincide_solti_good

    def smart_coincide_(labels: str) -> Array:
        return smart_coincide(tal_profs=tal_profs, types_self=types_self_extra, types_tal=types_tal_, males=males,
                              labels=labels)

    type_smart_coin = smart_coincide_('type')

    type_smart_mask = type_smart_coin > 0
    types_tal[type_smart_mask] = type_smart_coin[type_smart_mask]

    # print(df, '\n-----------------------\n', profiles, '\n\n', profiles.shape)
    if len({len(profiles), len(df_raw), len(types_self), len(types_tal)}) != 1:
        raise ValueError('Data has inconsistent dimensions.')
    types_tal_sex = np.copy(types_tal)
    types_tal_sex[males] += 16
    return DB(profiles=profiles, df=df_raw, types_tal=types_tal, types_self=types_self, types_tal_sex=types_tal_sex,
              questions=list(zip(questions, questions_eng)), interesting=interesting_indexes,
              type_smart_coincide=type_smart_coin,
              other_smart_coincide={tp: smart_coincide_(tp) for tp in TO_DICT},
              other_y_dim={tp: len(set(TO_DICT[tp].values())) for tp in TO_DICT})


def ids_to_idx(interesting_ids: List[List[int]], ids: Array) -> Array:
    inters_idxs = [np.argmin(np.abs(ids - id_)) for id_ in [s[-1] for s in interesting_ids]]
    # inters_idxs = [idx for idx, id_ in enumerate(ids) if id_ in [s[-1] for s in interesting_ids]]
    if len(interesting_ids) != len(inters_idxs):
        raise AssertionError(interesting_ids, inters_idxs)
    interesting_indexes = np.array([[lst[0], idx] for lst, idx in zip(interesting_ids, inters_idxs)])
    return interesting_indexes


def read_bolti_434() -> DB:
    _df_raw, _df_out, questions, questions_eng, interesting_ids = read_profiles(BOLTI)
    mask = _df_raw['goal'] != 3
    df_raw: DataFrame = _df_raw.loc[mask]
    df_out: DataFrame = _df_out.loc[mask]
    if len(_df_raw) == len(df_raw):
        raise ValueError('Selecting (goal != 3) failed.')

    interesting_indexes = ids_to_idx(interesting_ids, df_raw['id'].values)
    columns = EXTRA_COLUMNS_IDS + [str(i) for i in range(1, 430 + 1)]
    return preprocess_profiles(df_raw=df_raw, df_out=df_out, questions=questions, questions_eng=questions_eng,
                               select_columns=columns, interesting_indexes=interesting_indexes, db_name='bolti')


def read_solti_160() -> DB:
    df_ru, df_out_ru, questions, questions_eng, interesting_ids_ru = read_profiles(SOLTI)
    df_en, df_out_en, _, _, interesting_ids_en = read_profiles(SOLTI_ENG)

    out_columns = ['sigma_of_the_profile', 'correl_of_the_halves'] + [str(i) for i in range(1, 17)]
    df_raw: DataFrame = pd.concat([df_ru, df_en], ignore_index=True)
    df_out: DataFrame = pd.concat([df_out_ru[out_columns], df_out_en[out_columns]], ignore_index=True)
    interesting_ids_ru_1 = [id_ for type_, id_ in interesting_ids_ru]
    interesting_ids = interesting_ids_ru + [lst for lst in interesting_ids_en if lst[-1] not in interesting_ids_ru_1]

    interesting_indexes = ids_to_idx(interesting_ids, df_raw['id'].values)
    raw_columns = EXTRA_COLUMNS_IDS + [str(i) for i in range(1, 160 + 1)]
    return preprocess_profiles(df_raw=df_raw, df_out=df_out, questions=questions, questions_eng=questions_eng,
                               select_columns=raw_columns, interesting_indexes=interesting_indexes, db_name='solti')


DBs: Dict[str, DBSpec] = dict(
    solti=DBSpec(name='solti', reader=read_solti_160, h_dims=(128, 128, 128)),  # It is OVERRIDDEN in ready.py!
    bolti=DBSpec(name='bolti', reader=read_bolti_434, h_dims=(256, 192, 128))
    # I removed bolti as it needs manual picking of some rare sex+types
)

# read_bolti_434()
# read_solti_160()
