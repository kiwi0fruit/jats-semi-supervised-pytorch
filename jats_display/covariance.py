from typing import Tuple, Dict, List, Optional as Opt
from itertools import combinations
import numpy as np
# noinspection PyPep8Naming
from numpy import ndarray as Array
from vae.utils import get_x_normalized_μ_σ
from socionics_db import JATSModelOutput
from .display import plot_z_cov


def compare_cov(run_id1: str, run_id2: str, models: Dict[str, JATSModelOutput], weights: Opt[Array],
                corr_threshold: float, path_prefix: str, plot: bool=True) -> Tuple[float, float]:
    profs1 = models[run_id1]
    profs2 = models[run_id2]

    len1 = profs1.z_normalized.shape[1]
    z_cat = np.concatenate((profs1.z_normalized, profs2.z_normalized), axis=1)
    z_cat_normalized, _, _ = get_x_normalized_μ_σ(z_cat, weights)
    sub_abs_Σ = np.abs(np.cov(z_cat_normalized.T, aweights=weights))[:len1, len1:]
    sub_abs_Σ[sub_abs_Σ < corr_threshold] = 0.
    if plot:
        plot_z_cov(z_cat, weights, file=path_prefix + f'-{run_id1}-vs-{run_id2}',
                   sub=lambda m: np.abs(m[:len1, len1:]))
    ret = [float(np.sum(np.max(sub_abs_Σ, axis=i))) for i in (1, 0)]
    return ret[0], ret[1]


def check_cov(models: Dict[str, JATSModelOutput], weight_vec: Opt[Array], path_prefix: str,
              allowed_basis_dims: Tuple[int, ...]):
    for mod_id, mod_out in models.items():
        if len(mod_out.basis) not in allowed_basis_dims:
            print(f'WARNING: {mod_id} model has basis of length not from {allowed_basis_dims}: {mod_out.basis}')

    if len(models.keys()) == 1:
        print('Nothing to compare.')
        return

    corr_threshold = 0.85
    n_models_to_find = 4
    score_to_reach = 8
    plot_cov = False

    selections = []
    plotted: List[Tuple[str, str]] = []
    fixed_combinations = tuple(set(tuple(sorted(com))
                                   for com in combinations(tuple(models.keys()), n_models_to_find)))
    for selection in fixed_combinations:
        scores: Dict[str, Dict[str, float]] = {id_: dict() for id_ in selection}
        for i, j in set(tuple(sorted((i, j)))
                        for i in selection for j in selection
                        if i != j):
            scores[i][j], scores[j][i] = compare_cov(i, j, models=models, weights=weight_vec,
                                                     corr_threshold=corr_threshold, path_prefix=path_prefix,
                                                     plot=((i, j) not in plotted) and plot_cov)
            plotted.append((i, j))
        sum_scores = [(id0, sum(score for id1, score in id0_vs_other_ids.items()) / len(id0_vs_other_ids))
                      for id0, id0_vs_other_ids in scores.items()]
        sum_scores = list(sorted(sum_scores, key=lambda s: s[1]))
        selections.append((tuple(selection), (sum_scores, scores)))

    selections2 = list(sorted(selections, key=lambda s: s[1][0][0][1]))
    # [1][0]: tuple(selection), {({sum_scores}, scores)}; [0][1]: sum_scores[0] == (id, {sum})
    range_ = [-2, -1] if len(selections2) > 1 else [-1]
    for _k in range_:
        _, (sum_scores, _) = selections2[_k]
        for _id, _score in sum_scores:
            print(_id, _score)
        print()
        for i, j in set(tuple(sorted((i, j)))
                        for i in selections2[_k][0] for j in selections2[_k][0]
                        if i != j):
            compare_cov(i, j, models=models, weights=weight_vec, corr_threshold=corr_threshold,
                        path_prefix=path_prefix, plot=True)

    best_sum_scores = selections2[-1][1][0]
    if len([1 for key, score in best_sum_scores if score >= score_to_reach]) == n_models_to_find:
        print('Success!')
    else:
        print('Failure!')
