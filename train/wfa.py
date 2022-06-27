from typing import Iterable, Union, Optional as Opt, Tuple
from dataclasses import dataclass
from os import path
from os.path import join
import numpy as np
from numpy.typing import NDArray as Array
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from factor_analyzer import FactorAnalyzer
from jats.load import get_data, get_target_stratify, get_weight_16
from jats.plot import get_x_normed_mu_sigma

TEST_SIZE = 0.25
RANDOM_STATE = 1


parent_dir = path.dirname(path.dirname(path.abspath(__file__)))
preprocess_db = join(parent_dir, 'preprocess_db')
jats_df = pd.read_csv(join(preprocess_db, 'db_final.csv'))


train_set, test_set = train_test_split(
    jats_df.values,
    test_size=TEST_SIZE,
    shuffle=True,
    stratify=get_target_stratify(jats_df),
    random_state=RANDOM_STATE
)
df_train = pd.DataFrame(train_set, columns=jats_df.columns)
df_test = pd.DataFrame(test_set, columns=jats_df.columns)

_, _, x_l, x_ext_l, _, weights_l, type_sex_l = get_data(df_train)
_, _, x_v, x_ext_v, _, weights_v, type_sex_v = get_data(df_test)


def sex_split(sex: int, type_sex: Array) -> Tuple[Array, Array]:
    mask = type_sex > 16
    if sex == 0:
        mask = ~mask
        types = type_sex[mask]
    else:
        types = type_sex[mask] - 16
    return mask, get_weight_16(types)


@dataclass
class LinearAnalyzer:
    n: int
    analyzer: FactorAnalyzer
    mu_x: Union[Array, int]
    sigma_x: Union[Array, int]
    inv_transform_matrix: Array
    normalize_x: bool

    def transform(self, x: Array) -> Array:
        return self.analyzer.transform((x - self.mu_x) / self.sigma_x)

    def inv_transform(self, z: Array, z_ext: Opt[Array] = None) -> Array:
        """ Only first ``self.passthrough_dim`` elements of ``x`` are used.

        :return: x_rec (of the same shape as x, e.g. with passthrough elements prepended) """
        # (~30000, ~160) = (~30000, ~9) @ (~9, ~160):
        n = self.inv_transform_matrix.shape[1]
        if z_ext is not None:
            z = np.concatenate((z_ext, z), axis=1)
        return (z @ self.inv_transform_matrix) * self.sigma_x[:n] + self.mu_x[:n]


def get_inv_transform_matrix(z: Array, w: Opt[Array], x_normed: Array) -> Array:
    # z_extended @ inverse_transform == x_normalized_not_extended:
    lr = LinearRegression()
    lr.fit(z, x_normed, sample_weight=w)
    inverse_transform_matrix: Array = lr.coef_.T  # (~9, ~160)
    return inverse_transform_matrix


def get_fa(x_val: Array, x_ext_val: Array, w_val: Opt[Array],
           x_lrn: Array, x_ext_lrn: Array, w_lrn: Opt[Array],
           n_factors_list: Iterable[int], normalize_x: bool = True,
           filepath: str = None, pss=False, eps=0.04) -> Tuple[LinearAnalyzer, ...]:
    """ The last from ``n_comp_list`` would be returned. """

    n_factors_list = list(n_factors_list)
    pss_val: Tuple[Array, ...] = ()
    pss_lrn: Tuple[Array, ...] = ()
    if pss:
        pss_val = ((x_ext_val[:, :1] - 0.5) / 0.5,)
        pss_lrn = ((x_ext_lrn[:, :1] - 0.5) / 0.5,)

    x_ext_val = np.concatenate((x_val, x_ext_val), axis=1)  # (~5000, ~161)
    x_ext_lrn = x_ext_normed_lrn = np.concatenate((x_lrn, x_ext_lrn), axis=1)  # (~25000, ~161)
    mu_x: Union[Array, int] = 0
    sigma_x: Union[Array, int] = 1
    if normalize_x:
        x_ext_normed_lrn, mu_x, sigma_x = get_x_normed_mu_sigma(x_ext_lrn, w_lrn)
    x_normed_lrn = x_ext_normed_lrn[:, :x_lrn.shape[1]]
    w_sigma_mat = np.cov(x_ext_normed_lrn.T, aweights=w_lrn)  # (~161, ~161)

    def prnt(s: str):
        if filepath is not None:
            print(s, file=open(filepath, 'a'))
        print(s)

    def get_bce_mse(x, x_rec, w) -> Tuple[float, float]:
        x_prob = np.copy(x_rec)
        x_prob[x_rec > 1. - eps] = 1. - eps
        x_prob[x_rec < 0. + eps] = 0. + eps
        bce = np.sum(np.sum(x * np.log(x_prob) + (1 - x) * np.log(1 - x_prob), axis=1) * w) * (-1)
        mse = np.sum(np.mean((x - x_prob) ** 2, axis=1) * w)
        return float(bce), float(mse)

    for j, i in enumerate(n_factors_list):
        fa = FactorAnalyzer(n_factors=i, is_corr_matrix=True, method='ml',
                            rotation=(None, 'varimax', 'oblimax', 'quartimax', 'equamax')[0])
        fa.fit(w_sigma_mat)
        fa.mean_ = np.zeros(x_ext_lrn.shape[1])
        fa.std_ = fa.mean_ + 1.
        z = fa.transform(x_ext_normed_lrn)  # same as:
        # from numpy.linalg import inv
        # (~30000, ~9) = (~30000, ~161) @ ((~161, ~161) @ (~161, ~9))
        # z = ((x_normalized - 0) / 1) @ (inv(Σ_x) @ fa.structure_)

        inv_transform_matrix = get_inv_transform_matrix(
            np.concatenate((pss_lrn[0], z), axis=1) if pss_lrn else z,
            w_lrn, x_normed_lrn
        )

        an = LinearAnalyzer(
            n=fa.n_factors, analyzer=fa, mu_x=mu_x, sigma_x=sigma_x,
            inv_transform_matrix=inv_transform_matrix, normalize_x=normalize_x)

        x_rec_val = an.inv_transform(an.transform(x_ext_val), *pss_val)
        x_rec_lrn = an.inv_transform(an.transform(x_ext_lrn), *pss_lrn)
        bce_val, mse_val = get_bce_mse(x_val, x_rec_val, w_val)
        bce_lrn, mse_lrn = get_bce_mse(x_lrn, x_rec_lrn, w_lrn)

        prnt(f"n_factors={i}, eps={eps}, pss={pss}")
        prnt(f'mse_val={mse_val}')
        prnt(f'mse_lrn={mse_lrn}')
        prnt(f'bce_val={bce_val}')
        prnt(f'bce_lrn={bce_lrn}')
        prnt('')

        if (j + 1) == len(n_factors_list):
            break
    else:
        raise ValueError('Empty n_comp_list')
    return (an,)


@dataclass
class Sex:
    x_ext_normed_lrn: Array
    x_normed_lrn: Array
    mu_x: Array
    sigma_x: Array
    w_sigma_mat: Array
    w_lrn: Array
    w_val: Array
    mask_lrn: Array
    mask_val: Array


def get_fa_pss(x_val: Array, x_ext_val: Array, type_sex_val: Array,
               x_lrn: Array, x_ext_lrn: Array, type_sex_lrn: Array,
               n_factors_list: Iterable[int], normalize_x: bool = True,
               filepath: str = None, eps=0.04) -> Tuple[LinearAnalyzer, ...]:
    """ The last from ``n_comp_list`` would be returned. """
    n_factors_list = list(n_factors_list)
    x_ext_val = np.concatenate((x_val, x_ext_val[:, 1:]), axis=1)  # (~5000, ~161)
    x_ext_lrn = np.concatenate((x_lrn, x_ext_lrn[:, 1:]), axis=1)  # (~25000, ~161)

    def prnt(s: str, filepath_=filepath):
        if filepath_ is not None:
            print(s, file=open(filepath_, 'a'))
        print(s)

    def get_bce_mse(x, x_rec, w, ep=eps) -> Tuple[float, float]:
        x_prob = np.copy(x_rec)
        x_prob[x_rec > 1. - ep] = 1. - ep
        x_prob[x_rec < 0. + ep] = 0. + ep
        bce = np.sum(np.sum(x * np.log(x_prob) + (1 - x) * np.log(1 - x_prob), axis=1) * w) * (-1)
        mse = np.sum(np.mean((x - x_prob) ** 2, axis=1) * w)
        return float(bce), float(mse)

    def get_sex_dc(sex_: int, x_ext_lrn_: Array, type_sex_val_: Array, type_sex_lrn_: Array,
                   x_lrn_shape_1: int = x_lrn.shape[1], normalize_x_=normalize_x) -> Sex:
        mask_lrn, w_lrn = sex_split(sex_, type_sex_lrn_)
        mask_val, w_val = sex_split(sex_, type_sex_val_)
        x_ext_lrn_sex_ = x_ext_lrn_[mask_lrn]

        x_ext_normed_lrn = x_ext_lrn_sex_
        mu_x: Union[Array, int] = 0
        sigma_x: Union[Array, int] = 1
        if normalize_x_:
            x_ext_normed_lrn, mu_x, sigma_x = get_x_normed_mu_sigma(x_ext_lrn_sex_, w_lrn)
        x_normed_lrn = x_ext_normed_lrn[:, :x_lrn_shape_1]
        w_sigma_mat = np.cov(x_ext_normed_lrn.T, aweights=w_lrn)  # (~161, ~161)
        return Sex(x_ext_normed_lrn=x_ext_normed_lrn, x_normed_lrn=x_normed_lrn, mu_x=mu_x, sigma_x=sigma_x,
                   w_sigma_mat=w_sigma_mat, w_lrn=w_lrn, w_val=w_val, mask_lrn=mask_lrn, mask_val=mask_val)

    sexdc = [get_sex_dc(k, x_ext_lrn, type_sex_val, type_sex_lrn) for k in (0, 1)]

    for j, i in enumerate(n_factors_list):
        bce_val = []
        mse_val = []
        bce_lrn = []
        mse_lrn = []
        an = []
        for sex in sexdc:
            fa = FactorAnalyzer(n_factors=i, is_corr_matrix=True, method='ml',
                                rotation=(None, 'varimax', 'oblimax', 'quartimax', 'equamax')[0])
            x_ext_lrn_sex = x_ext_lrn[sex.mask_lrn]
            x_ext_val_sex = x_ext_val[sex.mask_val]

            fa.fit(sex.w_sigma_mat)
            fa.mean_ = np.zeros(x_ext_lrn_sex.shape[1])
            fa.std_ = fa.mean_ + 1.
            z = fa.transform(sex.x_ext_normed_lrn)  # same as:
            # from numpy.linalg import inv
            # (~30000, ~9) = (~30000, ~161) @ ((~161, ~161) @ (~161, ~9))
            # z = ((x_normalized - 0) / 1) @ (inv(Σ_x) @ fa.structure_)

            inv_transform_matrix = get_inv_transform_matrix(
                z, sex.w_lrn, sex.x_normed_lrn)

            an_ = LinearAnalyzer(
                n=fa.n_factors, analyzer=fa, mu_x=sex.mu_x, sigma_x=sex.sigma_x,
                inv_transform_matrix=inv_transform_matrix, normalize_x=normalize_x)

            x_rec_val = an_.inv_transform(an_.transform(x_ext_val_sex))
            x_rec_lrn = an_.inv_transform(an_.transform(x_ext_lrn_sex))
            bce_val_, mse_val_ = get_bce_mse(x_val[sex.mask_val], x_rec_val, sex.w_val)
            bce_lrn_, mse_lrn_ = get_bce_mse(x_lrn[sex.mask_lrn], x_rec_lrn, sex.w_lrn)

            an.append(an_)
            bce_val.append(bce_val_)
            mse_val.append(mse_val_)
            bce_lrn.append(bce_lrn_)
            mse_lrn.append(mse_lrn_)

        prnt(f"n_factors={i}, eps={eps}, pss='smart'")
        prnt(f'mse_val={sum(mse_val) / 2}')
        prnt(f'mse_lrn={sum(mse_lrn) / 2}')
        prnt(f'bce_val={sum(bce_val) / 2}')
        prnt(f'bce_lrn={sum(bce_lrn) / 2}')
        prnt('')

        if (j + 1) == len(n_factors_list):
            break
    else:
        raise ValueError('Empty n_comp_list')
    return tuple(an)


file_path = join(parent_dir, 'fa_log.txt')
print('', file=open(file_path, 'w'))

get_fa_pss(x_v, x_ext_v, type_sex_v, x_l, x_ext_l, type_sex_l,
           (7, 8, 9, 10), filepath=file_path)
print('\n\n', file=open(file_path, 'w'))
get_fa(x_v, x_ext_v, weights_v, x_l, x_ext_l, weights_l,
       (7, 8, 9, 10), filepath=file_path, pss=True)
print('\n\n', file=open(file_path, 'w'))
get_fa(x_v, x_ext_v, weights_v, x_l, x_ext_l, weights_l,
       (7, 8, 9, 10), filepath=file_path)
