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


BOLTI = 'BOLTI_434__2017_08_18__N3197'
SOLTI = 'SOLTI_160__2016_03_20__N6406'

EXTRA_QUESTIONS = ['sex in (female, male)', 'age in (0-20, 21-25, 26-30, 31-40, 41-100)']
EXTRA_COLUMNS_IDS = ['sex', 'age']


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
    types_smart_coincide: Array
    questions: List[Tuple[str, str]]
    interesting: Array
    dominant_smart_coincide: Array
    temperament_smart_coincide: Array


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


def read_profiles(profile_name: str) -> Tuple[DataFrame, DataFrame, List[str], List[str], Array]:
    """
    * Questions lists are prepended with 'sex' and 'age' descriptions.
    * ``sex`` is mapped: 0 to 1, 1 to 5.
    * ``self`` empty string is mapped to 0.
    * ``confidence`` empty string and ``'None'`` is mapped to ``-1``.

    :param profile_name:
    :return:
        (raw_data_frame, tal_profiles_data_frame, questions, questions_eng, interesting_indexes)
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
    df_tal_profiles = df_out.loc[:, [str(i) for i in range(1, 17)]]

    if len(df_out) != len(df_tal_profiles):
        raise ValueError('Inconsistent db_raw.csv and db_out.csv length.')

    questions = _questions('questions.txt')
    questions_eng = _questions('questions_autotranslated.txt')

    columns = df_raw.columns.values.tolist()
    if ('sex' not in columns) or ('age' not in columns):
        raise ValueError("Either 'sex' or 'age' is not in the columns names.")
    quest_n = len([name for name in columns if name.isdigit()]) + len(EXTRA_QUESTIONS)
    if (quest_n != len(questions)) or (quest_n != len(questions_eng)):
        raise ValueError("Inconsistent number of questions.")

    with open(p.join(_dir, 'interesting_indexes.py'), 'r', encoding='utf-8') as f:
        interesting_indexes = np.array(ast.literal_eval(f.read()))

    return df_raw, df_tal_profiles, questions, questions_eng, interesting_indexes


def to_dominant(type_: int) -> int:
    return {
        -1: -1,
        1: 1, 2: 2, 3: 3, 4: 4,
        5: 5, 6: 2, 7: 6, 8: 4,
        9: 5, 10: 7, 11: 6, 12: 8,
        13: 1, 14: 7, 15: 3, 16: 8
    }[type_]


def to_temperament(type_: int) -> int:
    return {
        -1: -1,
        1: 1, 2: 2, 3: 3, 4: 4,
        5: 1, 6: 2, 7: 3, 8: 4,
        9: 1, 10: 2, 11: 3, 12: 4,
        13: 1, 14: 2, 15: 3, 16: 4
    }[type_]


def to_type(type_: int) -> int:
    return type_


def smart_coincide(tal_profs: Array, types_self: Array, types_tal: Array, thr: int=81,
                   labels: str=('type', 'dominant', 'temperament')[0]) -> Array:
    """
    :param tal_profs: of shape (~6000, 16) of float
    :param types_self: of shape (~6000,) from {-1, 1, ..., 16}
    :param types_tal: of shape (~6000,) from {1, ..., 16}
    :param thr: threshold for smart Talanov's types.
    :param labels:
    :return: of shape (~6000,) in {-16, ..., -1, 1, ..., 16} positive when smart coincided
    """
    if len(tal_profs) != len(types_self):
        raise ValueError('Inconsistent tal_profs and types_self length.')

    tal_profs = np.round(np.einsum('ij,i->ij', tal_profs, np.max(tal_profs, axis=1)**-1) * 100).astype(int)
    tal_profs[tal_profs < thr] = 0

    if labels == 'type':
        f = to_type
    elif labels == 'dominant':
        f = to_dominant
    elif labels == 'temperament':
        f = to_temperament
    else:
        raise ValueError(f'Bad labels value: {labels}')

    def kernel(bests: List[int], self: int, tal: int) -> int:
        return self if self in bests else -tal

    return np.array([kernel(
        bests=[f(int(s)) for s in list(np.where(row != 0)[0] + 1)],
        self=f(int(self)), tal=f(int(tal))
    ) for row, self, tal in zip(tal_profs, types_self, types_tal)])


def preprocess_profiles(df_raw: DataFrame,
                        df_tal_profs: DataFrame,
                        questions: List[str],
                        questions_eng: List[str],
                        select_columns: List[str],
                        interesting_indexes: Array) -> DB:
    types_self = df_raw.loc[:, ['self']].values.T[0]
    types_tal = df_raw.loc[:, ['diagnosis']].values.T[0]
    types_smart_coincide = smart_coincide(tal_profs=df_tal_profs.values, types_self=types_self, types_tal=types_tal)
    dominant_smart_coincide = smart_coincide(tal_profs=df_tal_profs.values, types_self=types_self, types_tal=types_tal,
                                             labels='dominant')
    temperament_smart_coincide = smart_coincide(tal_profs=df_tal_profs.values, types_self=types_self,
                                                types_tal=types_tal, labels='temperament')
    mask = types_smart_coincide > 0
    types_tal[mask] = types_smart_coincide[mask]
    profiles = df_raw.loc[:, select_columns].values
    # print(df, '\n-----------------------\n', profiles, '\n\n', profiles.shape)

    if len({len(profiles), len(df_raw), len(types_self), len(types_tal)}) != 1:
        raise ValueError('Data has inconsistent dimensions.')
    types_tal_sex = np.copy(types_tal)
    types_tal_sex[profiles[:, 0] == 5] += 16
    return DB(profiles=profiles, df=df_raw, types_tal=types_tal, types_self=types_self, types_tal_sex=types_tal_sex,
              questions=list(zip(questions, questions_eng)), interesting=interesting_indexes,
              types_smart_coincide=types_smart_coincide,
              dominant_smart_coincide=dominant_smart_coincide,
              temperament_smart_coincide=temperament_smart_coincide)


def read_bolti_434() -> DB:
    _df, _df_tal_profs, questions, questions_eng, interesting_indexes = read_profiles(BOLTI)
    mask = _df['goal'] != 3
    df: DataFrame = _df.loc[mask]
    df_tal_profs: DataFrame = _df_tal_profs.loc[mask]
    if len(_df) == len(df):
        raise ValueError('Selecting (goal != 3) failed.')
    columns = EXTRA_COLUMNS_IDS + [str(i) for i in range(1, 430 + 1)]
    return preprocess_profiles(df_raw=df, df_tal_profs=df_tal_profs, questions=questions, questions_eng=questions_eng,
                               select_columns=columns, interesting_indexes=interesting_indexes)  # not IDs!


def read_solti_160() -> DB:
    df, df_tal_profs, questions, questions_eng, interesting_indexes = read_profiles(SOLTI)
    columns = EXTRA_COLUMNS_IDS + [str(i) for i in range(1, 160 + 1)]
    return preprocess_profiles(df_raw=df, df_tal_profs=df_tal_profs, questions=questions, questions_eng=questions_eng,
                               select_columns=columns, interesting_indexes=interesting_indexes)  # not IDs!


DBs: Dict[str, DBSpec] = dict(
    solti=DBSpec(name='solti', reader=read_solti_160, h_dims=(128, 128)),  # latest was (128,)
    bolti=DBSpec(name='bolti', reader=read_bolti_434, h_dims=(256, 128))
)

# read_bolti_434()
read_solti_160()
