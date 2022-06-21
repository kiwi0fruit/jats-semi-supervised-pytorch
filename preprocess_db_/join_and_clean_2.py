from typing import List
from numpy.typing import NDArray as Array
from os import path
import ast
import numpy as np
import pandas as pd

RUS_ID_INC = 200000
MAIN_QUEST_N = 160
CORR_CUT_L = 5000
CORR_THR = 0.67


def check_format_and_join() -> pd.DataFrame:
    """
    Reads 3 "db_raw_{...}.csv" databases.
    Checks format.
    Increments RUS questionnaires IDs with INC.
    Joins.
    Returns DF.
    """
    # Read:
    dfs = tuple(pd.read_csv(f'db_{s}.csv', converters=dict(
        sex=lambda s: 5 if (int(s) == 1) else 1,
        self=lambda s: int(s) if s else -1,
        language=lambda s: 5 if (int(s) == 1) else 1,
    )) for s in ('rus', 'eng_1', 'eng_2', 'eng_1b', 'eng_2b'))

    # Checks format:
    columns = tuple(tuple(sorted(df.columns.values.tolist())) for df in dfs)
    for col in columns[1:]:
        if not (columns[0] == col): raise AssertionError

    # Increment RUS questionnaires IDs with INC:
    dfs[0]['id'] = dfs[0]['id'] + RUS_ID_INC

    # Join:
    df = pd.concat(dfs).drop_duplicates(subset=['id']).sort_values(by=['id'])
    return df


def clean(df: pd.DataFrame):
    """
    Patches IDs.
    Removes duplicate or bad questionnaires.
    Writes.
    """
    # Patch IDs:
    add_file = 'ids_add.ast'
    remove_file = 'ids_remove.ast'
    add_ids: List[int] = []
    remove_ids: List[int] = []
    if path.isfile(add_file):
        with open(add_file, 'r', encoding='utf-8') as f:
            add_ids = ast.literal_eval(f.read())
    if path.isfile(remove_file):
        with open(remove_file, 'r', encoding='utf-8') as f:
            remove_ids = ast.literal_eval(f.read())

    remove_ids = [i for i in remove_ids if i not in add_ids]

    print(df.shape)
    ids = df['id'].values
    if remove_ids:
        indexes_to_keep = [idx for idx, id_ in enumerate(ids) if id_ not in remove_ids]
        indexes_to_drop = set(range(df.shape[0])) - set(indexes_to_keep)

        df.take(sorted(indexes_to_drop)).sort_values(by=['id']).to_csv('db_drop.csv', index=False)
        df = df.take(indexes_to_keep)
        ids = df['id'].values
    print(df.shape)

    # Remove duplicate or bad questionnaires:
    profs = df[[str(i) for i in range(1, MAIN_QUEST_N + 1)]].values

    max_same_quest = np.array([np.max(np.unique(row, return_counts=True)[1]) for row in profs])
    del_mask = max_same_quest > np.mean(max_same_quest) + 4 * np.std(max_same_quest)
    if len(ids[del_mask]) > 0:
        del_ids = ids[del_mask]
        raise ValueError(f'These IDs have too many same questions: {list(del_ids)}'
                         + f' (counts: {list(max_same_quest[del_mask])})')

    def corr_cut(profs_cut: Array, ids_cut: Array) -> List[int]:
        has_equals = []

        corr = np.corrcoef(profs_cut) - np.eye(len(profs_cut))
        if np.isnan(corr).any():
            del_ids_ = ids_cut[list(set(i for i, row in enumerate(corr) if all(np.isnan(row))))]
            raise ValueError(f'These IDs give NaN correlations with other IDs: {list(del_ids_)}')

        mask = np.max(corr, axis=1) > CORR_THR

        if len(corr[mask]) > 0:
            idxs_ = np.arange(0, len(profs_cut))
            has_equals = ids_cut[idxs_[mask]]

        return [int(i) for i in has_equals]

    def corr_fin(profs_cut: Array, ids_cut: Array) -> List[int]:
        duplicates = []
        if len(ids_cut) != len(set(ids_cut)): raise AssertionError
        if len(profs_cut) != len(ids_cut): raise AssertionError

        corr = np.corrcoef(profs_cut) - np.eye(len(profs_cut))
        # (correlation with itself were excluded)
        mask = np.max(corr, axis=1) > CORR_THR
        if len(corr[mask]) > 0:
            idxs_ = np.arange(0, len(profs_cut))
            del_idxs: List[int] = []
            # all possible indexes of candidates for deletion:
            has_equals = [int(i) for i in idxs_[mask]]
            for i in has_equals.copy():
                for j, c in enumerate(corr[i]):  # go through scalars of the corr. row
                    if (c > CORR_THR) and (i in has_equals):
                        del_idxs.append(j)
                        # because correlations with itself were excluded it would
                        # prevent the first item i from deletion.
                        has_equals = [s for s in has_equals if s != j]

            del_ids_ = ids_cut[del_idxs]
            duplicates = sorted(set(del_ids_))

        print(len(ids_cut), len(duplicates))
        return duplicates

    n = profs.shape[0]
    chunk_l = CORR_CUT_L
    chunks_n = n // chunk_l
    if n > chunks_n * chunk_l:
        chunks_n += 1

    has_equals_ = []
    for i_ in range(chunks_n):
        for j_ in range(chunks_n):
            if i_ >= j_:
                continue
            profs_cut_1 = profs[i_ * chunk_l:(i_ + 1) * chunk_l]
            profs_cut_2 = profs[j_ * chunk_l:(j_ + 1) * chunk_l]
            ids_1 = ids[i_ * chunk_l:(i_ + 1) * chunk_l]
            ids_2 = ids[j_ * chunk_l:(j_ + 1) * chunk_l]
            has_equals_ += corr_cut(
                np.concatenate((profs_cut_1, profs_cut_2), axis=0),
                np.concatenate((ids_1, ids_2), axis=0),
            )
    has_equals_ = [i for i in has_equals_ if i not in add_ids]

    if has_equals_:
        has_equals_ = sorted(set(has_equals_))
        idxs_has_equals = [i for i in range(0, len(ids)) if int(ids[i]) in has_equals_]

        duplicates_ = corr_fin(profs[idxs_has_equals], np.array(has_equals_))
        raise ValueError(f'Duplicate profiles. Recommended to delete IDs: {duplicates_}')

    print(df.shape)
    # Write DF:
    df.sort_values(by=['id']).to_csv('db.csv', index=False)


if __name__ == '__main__':
    clean(check_format_and_join())
