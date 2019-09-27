import os.path as p
from typing import NamedTuple, List, Tuple
import random
import ast
import numpy as np

from .read_db import DBs

dir_ = p.dirname(__file__)


def take_n(count: int) -> int:
    if count >= 2:
        return max((10 * count) // 100, 1)
    raise ValueError('count < 2')


class Stat(NamedTuple):
    sex: int
    type: int
    match: int
    count_: int
    take: int
    all: List[int]
    taken: List[int]
    left: List[int]


def write_test_samples(print_: bool=False) -> None:
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
        self, tal = db.types_self, db.types_tal
        if len(sex) != len(tal):
            raise ValueError('Bad DB')
        if tuple(np.unique(sex)) != (0, 1):
            raise NotImplementedError
        if tuple(np.unique(tal)) != tuple(range(1, 17)):
            raise NotImplementedError

        stats: List[Stat] = []
        test_samples: List[int] = []
        learn_samples: List[int] = []
        for _type in range(1, 17):
            for _sex in (0, 1):
                for _match in (False, True):
                    mask = (self == tal) if _match else (self != tal)
                    mask = mask & (tal == _type) & (sex == _sex)
                    count = len(sex[mask])
                    take = take_n(count)
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


def random_control_samples(stats: List[Stat], idxs_lbl: List[int]=None) -> List[int]:
    ret: List[int] = []
    if len(stats) != 64:
        raise RuntimeError
    for stat in stats:
        if idxs_lbl is not None:
            take = len([i for i in stat.taken if i in idxs_lbl])
            left = [i for i in stat.left if i in idxs_lbl]
        else:
            take, left = stat.take, stat.left
        ret += random.sample(left, take)
    return list(sorted(ret))


def read_test_samples(db_name: str) -> Tuple[List[int], List[int], List[Stat]]:
    """
    ``(db_name in ('solti', 'bolti')) is True``

    Reads from file DB .ast spec file (without "__" prefix) and

    returns ``test_samples, learn_samples, stats``
    """
    with open(p.join(dir_, db_name + '.ast'), 'r', encoding='utf-8') as f:
        test_samples, learn_samples, stats = ast.literal_eval(f.read())

    return test_samples, learn_samples, [Stat(**dic) for dic in stats]


def test_test_samples():
    for db_spec in DBs.values():
        test_, _, stats_ = read_test_samples(db_spec.name)
        print(f'\n{db_spec.name}:')
        print(np.array([test_, random_control_samples(stats_), random_control_samples(stats_)]))


# TODO pylint: disable=fixme
#  [copy and run me in the cell]
r'''
# %%
from socionics_db.generate_control_groups import write_test_samples, test_test_samples
write_test_samples()

# %% Run after exporting spec file, renaming and removing "__" prefix:
test_test_samples()
'''
