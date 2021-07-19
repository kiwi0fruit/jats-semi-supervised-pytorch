import os.path as p
from typing import NamedTuple, List, Tuple
import random
import ast
import numpy as np

from .read_db import DBs

TEST_PER_MILLE = 300  # *OLTI: SOLTI=300 BOLTI=334
ALL_MIN_PER_CLASS_32 = 20  # *OLTI: SOLTI=20 BOLTI=3
MIN_TEST_PER_CLASS_32 = 6  # *OLTI: SOLTI=6 BOLTI=1

SAMPLES_PER_CLASS_32 = ALL_MIN_PER_CLASS_32 - MIN_TEST_PER_CLASS_32
dir_ = p.dirname(__file__)
if (TEST_PER_MILLE * ALL_MIN_PER_CLASS_32) // 1000 != MIN_TEST_PER_CLASS_32:
    raise AssertionError


def take_n(count: int, debug: str) -> int:
    if count >= ALL_MIN_PER_CLASS_32:
        test_count = (TEST_PER_MILLE * count) // 1000
        if test_count >= MIN_TEST_PER_CLASS_32:
            return test_count
        raise ValueError(f'@{debug}: test_count = {test_count} < {MIN_TEST_PER_CLASS_32}')
    raise ValueError(f'@{debug}: all_count = {count} < {ALL_MIN_PER_CLASS_32}')


class Stat(NamedTuple):
    sex: int
    type: int
    match: int
    count_: int
    take: int
    all: List[int]
    taken: List[int]
    left: List[int]


def write_test_samples(print_: bool = False) -> None:
    """ Writes tuple to files per DB:

    >>> '''
    >>> (
    >>>     test_samples: List[int],
    >>>     learn_samples: List[int],
    >>>     stats: List[dict ~ Stat]
    >>> )
    >>> '''
    """
    for db_spec in DBs.values():
        db = db_spec.reader()
        interesting = list(db.interesting[:, 1])
        sex = db.df['sex'].values
        sex = (sex - np.min(sex)) // (np.max(sex) - np.min(sex))

        uni = np.unique(db.type_smart_coincide)
        missing = [i for i in range(-16, 17) if (i != 0) and (i not in uni)]
        if missing: raise AssertionError(missing)

        coincide = db.type_smart_coincide
        if len(sex) != len(coincide):
            raise ValueError('Bad DB')
        if tuple(np.unique(sex)) != (0, 1):
            raise NotImplementedError

        stats: List[Stat] = []
        test_samples: List[int] = []
        learn_samples: List[int] = []
        for _type in range(1, 17):
            for _sex in (0, 1):
                for _match in (False, True):
                    mask = (coincide > 0) if _match else (coincide < 0)
                    mask = mask & (abs(coincide) == _type) & (sex == _sex)
                    count = len(sex[mask])
                    take = take_n(count, debug=f'sex={_sex},type={_type},match={int(_match)}')
                    all_ = [int(i) for i in np.where(mask)[0]]
                    taken = [i for i in interesting if i in all_]
                    if len(taken) < take:
                        taken += random.sample([i for i in all_ if (i not in taken)], take - len(taken))
                    else:
                        take = len(taken)
                    stats.append(Stat(
                        sex=_sex, type=_type, match=int(_match), count_=count,
                        take=take, all=all_, taken=taken,
                        left=[i for i in all_ if i not in taken]
                    ))
                    left = [i for i in all_ if (i not in taken)]
                    test_samples += taken
                    learn_samples += left

        test_samples = list(sorted(test_samples))
        learn_samples = list(sorted(learn_samples))
        print(repr((test_samples, learn_samples, [dict(s._asdict()) for s in stats])),
              file=open(p.join(dir_, '__' + db_spec.name + '.ast'), 'w', encoding='utf-8'))

        if print_:
            for item in sorted(stats, key=lambda st: st.count):
                print(item)
            print(f'Test DB takes {sum(s.take for s in stats)} of {sum(s.count_ for s in stats)} profiles.')


def random_control_samples(stats: List[Stat], idxs_subset: List[int] = None, take: int = None) -> List[int]:
    """ default ``take`` is ``None``: take the same number as in test data. """
    ret: List[int] = []
    if len(stats) != 64:
        raise RuntimeError
    for stat in stats:
        left = stat.left
        if idxs_subset:
            left = [i for i in stat.left if i in idxs_subset]
            if not left:
                if stat.match:
                    raise ValueError
                continue

        L = len(left)
        if take is not None:
            take_ = min(take, L)
        elif idxs_subset:
            take_ = len([i for i in stat.taken if i in idxs_subset])
        else:
            take_ = len(stat.taken)

        if not stat.match:
            take_ = min(take_, L)
        ret += random.sample(left, take_)
    return list(sorted(ret))


def read_test_samples(db_name: str, debug: bool = False) -> Tuple[List[int], List[int], List[Stat]]:
    """
    ``(db_name in ('solti', 'bolti')) is True``

    Reads from file DB .ast spec file (without "__" prefix) and

    returns ``test_samples, learn_samples, stats``
    """
    pref = '' if not debug else '__'
    with open(p.join(dir_, pref + db_name + '.ast'), 'r', encoding='utf-8') as f:
        test_samples, learn_samples, stats = ast.literal_eval(f.read())

    return test_samples, learn_samples, [Stat(**dic) for dic in stats]


def test_test_samples(debug: bool = False):
    for db_spec in DBs.values():
        test_, _, stats_ = read_test_samples(db_spec.name, debug=debug)
        for stat in stats_:
            stat_ = dict(stat._asdict())
            del stat_['all'], stat_['taken'], stat_['left']
            print(stat_)
        print(f'\n{db_spec.name}:')
        print(np.array([test_, random_control_samples(stats_), random_control_samples(stats_)]))


# TODO pylint: disable=fixme
#  [copy and run me in the cell]
r'''
# %%
from socionics_db.generate_control_groups import write_test_samples, test_test_samples
write_test_samples()

# %% Run after exporting spec file, renaming and removing "__" prefix:
test_test_samples(debug=True)
'''
