from typing import Callable, List, Union, Tuple, Dict, Optional as Opt, Any, NamedTuple
from dataclasses import asdict
# from types import ModuleType
from os import path as p
import torch as tr
from torch import Tensor
import numpy as np
# noinspection PyPep8Naming
from numpy import ndarray as Array
import numpy.linalg as la
import pandas as pd
import seaborn as sns
import IPython.display as ds
from kiwi_bugfix_typechecker import ipython, test_assert

from beta_tcvae_typed import Normal, BetaTCKLDLoss, KLDLoss
from semi_supervised_typed import (VariationalAutoencoder, SVI, DeepGenerativeModel, AuxiliaryDeepGenerativeModel,
                                   VAEClassifyMeta, Passer)
from vae import BaseWeightedLoss
from vae.utils import get_x_normalized_μ_σ, ndarr
from vae.data import TrainLoader, LoaderRetType
from vae.display import Log
from vae.linear_component_analyzer import LinearAnalyzer
from vae.tools import get_z_ymax
from socionics_db import JATSModelOutput, Transforms, Data
from jats_vae.utils import print_err_tuple, euc_dist_count_types, TestArraysToTensors
from jats_vae.semi_supervised import ClassifierPassthrJATS24KhCFBase
from .colors import types_colors  # pylint: disable=relative-beyond-top-level


test_assert()
DEBUG = False
DISPLAY = False
LstTpl = List[Tuple[str, Union[int, Tensor]]]
LstTplOpt = List[Tuple[str, Union[int, Tensor, None]]]
ε = 10**-8


class Batch(NamedTuple):
    x: Opt[Tensor]
    x_nll: Opt[Tensor]
    w_mat: Opt[Tensor]
    y: Opt[Tensor]
    u: Tensor
    u_nll: Opt[Tensor]
    w_mat_u: Opt[Tensor]


def plot_z_cov(z: Array, weights: Opt[Array], file: str, sub: Callable[[Array], Array]=lambda m: m,
               basis_strip: Tuple[Tuple[int, ...], Tuple[int, ...]]=None) -> None:
    import matplotlibhelper as mh
    import matplotlib.pyplot as plt
    z_normalized, _, _ = get_x_normalized_μ_σ(z, weights)
    Σ = np.cov(z_normalized.T, aweights=weights)
    E = np.eye(len(Σ)) if Σ.shape else np.eye(1)
    fig = plt.figure(figsize=mh.figsize(w=6))
    corr_mat = sub(Σ - E)
    mask = sub(E > (1 - ε))
    if basis_strip is not None:
        strip_0, strip_1 = basis_strip
        if strip_0:
            mask[strip_0, :] = True
            corr_mat[strip_0, :] = 0.
        if strip_1:
            mask[:, strip_1] = True
            corr_mat[:, strip_1] = 0.
    sns.heatmap(corr_mat, center=0, square=True, mask=mask)
    plt.title(p.basename(file) + f'_MaxAbs={np.max(np.abs(corr_mat)):.3f}')
    fig.tight_layout()
    img = mh.img(name=file)
    if DISPLAY:
        ipython.display(ds.HTML(f'<img src="{img}" width="900">'))


def plot_dist(z_batch: Opt[Array], z: Array, types: Array,  # pylint: disable=too-many-branches
              types_self: Array, file: str, μ_z: Array=None, σ_z: Array=None,
              plt_types: Tuple[int, ...]=tuple(range(1, 17)), questions: bool=False) -> None:
    z_dims = z.shape[1]
    if z_dims == 1:
        return  # presumably matplotlib is buggy in this case
    import matplotlibhelper as mh
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(z_dims, 4, figsize=(16, 3 * z_dims))
    for i in range(z_dims):
        j = i
        ax_title = mh.stex(f'ˎs_˱{i}˲ˎ (ˎz_˱{j}˲ˎ)')

        # 1st plot (all database):
        ax = axs[i, 0]
        if z_batch is not None:
            z1, z2 = z_batch[:, j], z[:, i]
            sns.distplot(z1, bins=30, ax=ax)
            sns.distplot(z2, bins=30, ax=ax)
            if (μ_z is None) and (σ_z is None):
                μz_σz = ''
            elif isinstance(μ_z, Array) and isinstance(σ_z, Array):
                μz_σz = f';{μ_z[j]:.2f},{σ_z[j]:.2f}'
            else:
                raise ValueError(f'Bad μ_z and/or σ_z:\n{μ_z};\n{σ_z}')
            ax.set_title(ax_title + f', μ,σ={np.mean(z1.T):.2f},{np.std(z1.T):.2f}' + μz_σz)
        else:
            sns.distplot(z[:, i], bins=30, ax=ax)
            ax.set_title(ax_title)

        # 2nd plot (per-type density for Talanov's diags.):
        for type_ in plt_types:
            sns.distplot(z[types == type_, i], bins=20,
                         ax=axs[i, 1], color=types_colors[type_ - 1])
        axs[i, 1].set_title(ax_title + ' axis (Tal.)')

        # 3rd plot (discrete per-type density for Talanov's diags.):
        ax = axs[i, 2]
        if questions:
            for type_ in plt_types:
                Z, color = z[types == type_, i], types_colors[type_ - 1]
                sns.kdeplot(Z, kernel='gau', ax=ax, color=color, bw=0.075)  # label=str(type_)
        else:
            for kwargs in (dict(), dict(marker='o', linestyle='')):
                for type_ in plt_types:
                    Z, color = z[types == type_, i], types_colors[type_ - 1]
                    Y, X = np.histogram(Z, bins=10)
                    X = np.array([(X[i] + X[i - 1]) / 2 for i in range(1, len(X))])
                    ax.plot(X, Y / np.sum(Y), color=color, **kwargs)

        ax.set_title(ax_title + ' axis (Tal.)')
        # ax.legend()
        # ax.set_xlim(-4, 4)

        # 4th plot (per-type density for Talanov's == self diags.):
        ax = axs[i, 3]
        for type_ in plt_types:
            Z = z[(types == type_) & (types_self == type_), i]
            sns.kdeplot(Z, kernel='gau', ax=ax, color=types_colors[type_ - 1], bw=0.075 if questions else "scott")
        ax.set_title(ax_title + ' axis (Tal. = self)')

    fig.tight_layout()
    img = mh.img(name=file)
    if DISPLAY:
        ipython.display(ds.HTML(f'<img src="{img}" width="900">'))


def get_x_plot_batch(data_loader: TrainLoader) -> Array:
    batch_size = data_loader.unlabelled.batch_size
    data_loader.regenerate_loaders(batch_size=1600)
    train_loader = data_loader.get_loader()
    item: LoaderRetType = next(iter(train_loader))
    x_batch, _, _, _ = item
    data_loader.regenerate_loaders(batch_size=batch_size)
    return ndarr(x_batch)


def get__x__z__plot_batch(model: VariationalAutoencoder, data_loader: TrainLoader) -> Tuple[Array, Array]:
    batch_size = data_loader.unlabelled.batch_size
    data_loader.regenerate_loaders(batch_size=1600)
    train_loader = data_loader.get_loader()
    item: LoaderRetType = next(iter(train_loader))
    x_batch, _, _, _ = item
    with tr.no_grad():
        z_batch, _, _, _ = get_z_ymax(x=x_batch, y=None, model=model)
    data_loader.regenerate_loaders(batch_size=batch_size)
    return ndarr(x_batch), ndarr(z_batch)


def plot_jats(weight: Opt[Array], prefix_path_db_nn: str, prefix_path_db: str, types_tal: Array, types_self: Array,
              x_batch: Array, lrn_idxs: List[int]=None, tst_idxs: List[int]=None,
              z_batch: Array=None, profiles: JATSModelOutput=None,
              pca: LinearAnalyzer=None, fa: LinearAnalyzer=None, plot_pca: bool=False,
              plt_types: Tuple[int, ...]=tuple(range(1, 17)),
              basis_strip: Tuple[int, ...]=()) -> None:
    """
    Plot latent dimensions.
    """
    def int2none(inp: Union[int, Array]) -> Opt[Array]:
        if isinstance(inp, int):
            return None
        return inp

    if plot_pca and pca is not None:
        plot_dist(z_batch=pca.transform(x_batch), z=pca.z_normalized, μ_z=int2none(pca.μ_z), σ_z=int2none(pca.σ_z),
                  file=prefix_path_db + f'-WPCA{pca.n}', types=types_tal, types_self=types_self, plt_types=plt_types)
        plot_z_cov(pca.z_normalized, weight, file=prefix_path_db + f'-WPCA{pca.n}-z-cov')

    if plot_pca and fa is not None:
        plot_dist(z_batch=fa.transform(x_batch), z=fa.z_normalized, μ_z=int2none(fa.μ_z), σ_z=int2none(fa.σ_z),
                  file=prefix_path_db + f'-WFA{fa.n}', types=types_tal, types_self=types_self, plt_types=plt_types)
        plot_z_cov(fa.z_normalized, weight, file=prefix_path_db + f'-WFA{fa.n}-z-cov')

    if plot_pca and (pca is not None) and (fa is not None):
        len_ = fa.z_normalized.shape[1]
        plot_z_cov(np.concatenate((fa.z_normalized, pca.z_normalized), axis=1),
                   weight, file=prefix_path_db_nn + f'-fa{fa.n}-vs-pca{pca.n}', sub=lambda m: np.abs(m[:len_, len_:]))

    if profiles is not None:
        if (z_batch is None) or (pca is None) or (fa is None):
            raise ValueError('(zμ_batch is None) or (pca is None) or (fa is None)')
        plot_dist(z_batch=z_batch, z=profiles.z, μ_z=profiles.μz, σ_z=profiles.σz,
                  file=prefix_path_db_nn, types=types_tal, types_self=types_self,
                  plt_types=plt_types)

        # male_idxs, male_idxs_batch = profiles.x[:, 0] > 0.5, x_batch[:, 0] > 0.5
        # plot_dist(z_batch=z_batch[male_idxs_batch], z=profiles.z[male_idxs],
        #           file=prefix_path_db_nn + "-male", types=types_tal[male_idxs], types_self=types_self[male_idxs],
        #           plt_types=plt_types)
        # plot_dist(z_batch=z_batch[~male_idxs_batch], z=profiles.z[~male_idxs],
        #           file=prefix_path_db_nn + "-female", types=types_tal[~male_idxs], types_self=types_self[~male_idxs],
        #           plt_types=plt_types)

        if lrn_idxs:
            plot_dist(z_batch=None, z=profiles.z[lrn_idxs], μ_z=profiles.μz, σ_z=profiles.σz,
                      file=prefix_path_db_nn + "-lrn", types=types_tal[lrn_idxs], types_self=types_self[lrn_idxs],
                      plt_types=plt_types)
        if tst_idxs:
            plot_dist(z_batch=None, z=profiles.z[tst_idxs], μ_z=profiles.μz, σ_z=profiles.σz,
                      file=prefix_path_db_nn + "-tst", types=types_tal[tst_idxs], types_self=types_self[tst_idxs],
                      plt_types=plt_types)

        plot_dist(z_batch=None, z=profiles.fzμ, file=prefix_path_db_nn + '-fzmu',
                  types=types_tal, types_self=types_self, plt_types=plt_types)
        if profiles.a is not None:
            plot_dist(z_batch=None, z=profiles.a, file=prefix_path_db_nn + '-a',
                      types=types_tal, types_self=types_self, plt_types=plt_types)
            # if lrn_idxs:
            #     plot_dist(z_batch=None, z=profiles.a[lrn_idxs], file=prefix_path_db_nn + '-a-lrn',
            #               types=types_tal[lrn_idxs], types_self=types_self[lrn_idxs], plt_types=plt_types)
            # if tst_idxs:
            #     plot_dist(z_batch=None, z=profiles.a[tst_idxs], file=prefix_path_db_nn + '-a-tst',
            #               types=types_tal[tst_idxs], types_self=types_self[tst_idxs], plt_types=plt_types)
        if profiles.b is not None:
            plot_dist(z_batch=None, z=profiles.b, file=prefix_path_db_nn + '-b',
                      types=types_tal, types_self=types_self, plt_types=plt_types)
        if profiles.c is not None:
            plot_dist(z_batch=None, z=profiles.c, file=prefix_path_db_nn + '-c',
                      types=types_tal, types_self=types_self, plt_types=plt_types)

        plot_z_cov(profiles.zμ, weight, file=prefix_path_db_nn + '-z-cov', basis_strip=(basis_strip, basis_strip))
        plot_z_cov(profiles.fzμ, weight, file=prefix_path_db_nn + '-fz-cov', basis_strip=(basis_strip, basis_strip))

        len_ = pca.z_normalized.shape[1]
        plot_z_cov(np.concatenate((pca.z_normalized, profiles.zμ), axis=1),
                   weight, file=prefix_path_db_nn + f'-pca{pca.n}-vs-z', sub=lambda m: np.abs(m[:len_, len_:]))
        len_ = fa.z_normalized.shape[1]
        plot_z_cov(np.concatenate((fa.z_normalized, profiles.zμ), axis=1),
                   weight, file=prefix_path_db_nn + f'-fa{fa.n}-vs-z', sub=lambda m: np.abs(m[:len_, len_:]))


def explore_jats(profiles: JATSModelOutput,
                 dat: Data,
                 trans: Transforms,
                 types: Array,
                 types_sex: Array,
                 types_self: Array,
                 logger: Log,
                 pca: LinearAnalyzer,
                 fa: LinearAnalyzer,
                 ids: Array,
                 inspect: Tuple[int, ...]=None) -> None:
    lg = logger
    if inspect is None:
        inspect = tuple(range(len(dat.interesting)))

    intrs_kwargs = asdict(profiles)
    L = len(profiles.x)
    for k, v in intrs_kwargs.items():
        if isinstance(v, Array) and (len(v) == L):
            v_: Array = v
            intrs_kwargs[k] = v_[dat.interesting]
    intrs_kwargs['x_rec_disc_zμ'] = tuple(s[dat.interesting] for s in profiles.x_rec_disc_zμ)
    intrs = JATSModelOutput(**intrs_kwargs)

    # Peek questions of selected profiles:
    # -------------------------------------------------------
    out = np.stack(
        [trans.to_1__5(intrs.x[i]) - 3 for i in inspect] +
        [trans.to_1__5(intrs.x_rec_disc_zμ[0][i]) - 3 for i in inspect],
        axis=-1)
    # lg.print(f'All questions ({len(out)}):\n', out)
    # pd.DataFrame(out).to_csv(p.join(TMP, '__view.csv'))
    mask10, mask20 = out[:, 0] != 100, out[:, 0] != 100
    for j, i in enumerate(inspect):
        mask1 = out[:, j] < 0
        mask11 = out[:, j + len(inspect)] < 0
        mask2 = out[:, j] > 0
        mask22 = out[:, j + len(inspect)] > 0
        a, b = len(out[mask1 & mask11]), len(out[mask1])
        lg.print(f'Negative questions only ({a}/{b} ~ {round(100 * a / b)}%) for `interesting[{i}]`:\n', out[mask1])
        a, b = len(out[mask2 & mask22]), len(out[mask2])
        lg.print(f'Positive questions only ({a}/{b} ~ {round(100 * a / b)}%) for `interesting[{i}]`:\n', out[mask2])
        mask10 = mask10 & mask1
        mask20 = mask20 & mask2
    # lg.print(f'2-negative questions only ({len(out[mask10])}) for intersection of `interesting`:\n', out[mask10])
    # lg.print(f'2-positive questions only ({len(out[mask20])}) for intersection of `interesting`:\n', out[mask20])

    # Print overall error:
    # -------------------------------------------------------
    def line_printer(s: str) -> None:
        lg.print(s)
        lg.print_i(s)

    print_err_tuple(profiles.x_rec_cont_zμ, *profiles.x_rec_disc_zμ, profiles.x_rec_sample_zμ,
                    x=profiles.x, data=dat, line_printer=line_printer)

    # Inspect selected profiles:
    # -------------------------------------------------------
    def mse_(x, y): return np.mean(np.power(x - y, 2))

    lg.print('First counted y-NN, then TT, then TT=self, then TT_sex.')

    if profiles.y_probs is not None:
        try:
            n = profiles.y_probs.shape[1]
            n_row = n // 4
            if n_row * 4 != n:
                raise ValueError('n_row * 4 != n')
            lg.display(pd.DataFrame(euc_dist_count_types(
                first_type=np.argmax(profiles.y_probs, axis=-1) + 1,
                n_types=n
            ).reshape((n_row, 4))))
            lg.print('y-NN.\n\n')
        except ValueError:
            lg.print('y-NN count failed')
    else:
        lg.print('y-NN count failed')

    try:
        lg.display(pd.DataFrame(euc_dist_count_types(first_type=types, n_types=16).reshape((4, 4))))
        lg.print('TT.\n\n')
    except ValueError:
        lg.print('TT count failed')

    try:
        lg.display(pd.DataFrame(euc_dist_count_types(
            first_type=types[types == types_self],
            n_types=16
        ).reshape((4, 4))))
        lg.print('TT=self.\n\n')
    except ValueError:
        lg.print('TT=self count failed')

    try:
        lg.display(pd.DataFrame(euc_dist_count_types(first_type=types_sex, n_types=32).reshape((8, 4))))
        lg.print('TT_sex.\n\n')
    except ValueError:
        lg.print('TT_sex count failed')

    for i in inspect:
        lg.print(
            f'Inspect `interesting[{i}]`:\n' +
            f'MSE #{i}:nn(#{i})            = {mse_(intrs.x[i], intrs.x_rec_cont_zμ[i]):.3f}\n' +
            f'MSE discr. #{i}:nn(#{i})     = {mse_(intrs.x[i], intrs.x_rec_disc_zμ[0][i]):.3f}\n' +
            ''.join([
                f'MSE discr. #{i}:#{j}         = {mse_(intrs.x[i], intrs.x[j]):.3f}\n' +
                f'MSE nn(#{i}):nn(#{j})        = {mse_(intrs.x_rec_cont_zμ[i], intrs.x_rec_cont_zμ[j]):.3f}\n' +
                f'MSE discr. nn(#{i}):nn(#{j}) = {mse_(intrs.x_rec_disc_zμ[0][i], intrs.x_rec_disc_zμ[0][j]):.3f}\n'
                for j in inspect if i != j]))

        def euc_dist(x: Array, A: Array) -> Array:
            return la.norm(A - x, axis=-1)
            # return np.sqrt(np.sum(np.power(A - x, 2), axis=-1))

        explr_x = euc_dist(intrs.x[i], profiles.x)
        sortd_x = np.argsort(explr_x)

        explr_x_recon = euc_dist(intrs.x_rec_cont_zμ[i], profiles.x_rec_cont_zμ)
        sortd_x_recon = np.argsort(explr_x_recon)

        explr_zμ = euc_dist(intrs.zμ[i], profiles.zμ)
        sortd_zμ = np.argsort(explr_zμ)

        explr_fzμ = euc_dist(intrs.fzμ[i], profiles.fzμ)
        sortd_fzμ = np.argsort(explr_fzμ)

        intrs_fa_z = fa.transform(intrs.x)
        intrs_fa_x_rec = fa.inverse_transform(intrs_fa_z)

        intrs_pca_z = pca.transform(intrs.x)
        intrs_pca_x_rec = pca.inverse_transform(intrs_pca_z)

        explr_fa_z = euc_dist(intrs_fa_z[i], fa.z_normalized)
        sortd_fa_z = np.argsort(explr_fa_z)

        explr_pca_z = euc_dist(intrs_pca_z[i], pca.z_normalized)
        sortd_pca_z = np.argsort(explr_pca_z)

        explr_fa_x_rec = euc_dist(intrs_fa_x_rec[i], fa.x_rec)
        sortd_fa_x_rec = np.argsort(explr_fa_x_rec)

        explr_pca_x_rec = euc_dist(intrs_pca_x_rec[i], pca.x_rec)
        sortd_pca_x_rec = np.argsort(explr_pca_x_rec)

        def x100(x: Array) -> Array:
            return np.round(100 * x).astype(int)

        def get_types(intrs_y_probs: Opt[Array], postfix: str='',
                      transform: Callable[[Array], Array]=lambda x: x + 1) -> None:
            assert intrs_y_probs is not None
            n_ = intrs_y_probs.shape[1]
            sorted_y = np.array(list(reversed(np.argsort(intrs_y_probs[i]))))
            dict_ = {
                f'type y-nn (of {n_})': transform(sorted_y[:n_]),
                f'prob{postfix} y': x100(intrs_y_probs[i][sorted_y])[:n_],
            }
            try:
                lg.display(pd.DataFrame(dict_))
            except ValueError:
                lg.print(dict_)

        if profiles.y_probs is not None:
            get_types(intrs.y_probs)
        if profiles.y_probs2 is not None:
            get_types(intrs.y_probs2, '2')

        def new_types(x: Array) -> Array: return np.array([i + 1 if (i < 16) else -(i - 15) for i in x])
        if profiles.y_probs32 is not None:
            get_types(intrs.y_probs32, '32', new_types)

        lg.display(pd.DataFrame({
            'idx1': ids[sortd_x[:10]],
            'x 100euc': x100(explr_x[sortd_x])[:10],
            'tal1': types[sortd_x][:10],
            'self1': types_self[sortd_x][:10],
            'idx2': ids[sortd_x_recon[:10]],
            'x_recon 100euc': x100(explr_x_recon[sortd_x_recon])[:10],
            'tal2': types[sortd_x_recon][:10],
            'self2': types_self[sortd_x_recon][:10],
        }))
        lg.display(pd.DataFrame({
            'idx3': ids[sortd_zμ[:10]],
            'zμ 100euc': x100(explr_zμ[sortd_zμ])[:10],
            'tal3': types[sortd_zμ][:10],
            'self3': types_self[sortd_zμ][:10],
            'idx3a': ids[sortd_fzμ[:10]],
            'fzμ 100euc': x100(explr_fzμ[sortd_fzμ])[:10],
            'tal3a': types[sortd_fzμ][:10],
            'self3a': types_self[sortd_fzμ][:10],
        }))
        lg.display(pd.DataFrame({
            'idx4': ids[sortd_pca_z[:10]],
            'pca z 100euc': x100(explr_pca_z[sortd_pca_z])[:10],
            'tal4': types[sortd_pca_z][:10],
            'self4': types_self[sortd_pca_z][:10],
            'idx5': ids[sortd_pca_x_rec[:10]],
            'pca x_rec 100euc': x100(explr_pca_x_rec[sortd_pca_x_rec])[:10],
            'tal5': types[sortd_pca_x_rec][:10],
            'self5': types_self[sortd_pca_x_rec][:10],
        }))
        lg.display(pd.DataFrame({
            'idx6': ids[sortd_fa_z[:10]],
            'fa z 100euc': x100(explr_fa_z[sortd_fa_z])[:10],
            'tal6': types[sortd_fa_z][:10],
            'self6': types_self[sortd_fa_z][:10],
            'idx7': ids[sortd_fa_x_rec[:10]],
            'fa x_rec 100euc': x100(explr_fa_x_rec[sortd_fa_x_rec])[:10],
            'tal7': types[sortd_fa_x_rec][:10],
            'self7': types_self[sortd_fa_x_rec][:10],
        }))

    float_format = pd.options.display.float_format
    pd.options.display.float_format = '{:,.2f}'.format
    lg.print('fp(zμ):')
    lg.display(pd.DataFrame(intrs.fzμ))
    lg.print('zμ:')
    lg.display(pd.DataFrame(intrs.zμ))
    lg.print('σ:')
    lg.display(pd.DataFrame(intrs.σ))
    if intrs.zμ_new is not None:
        lg.print('zμ_new:')
        pd.set_option('display.max_columns', None)
        lg.display(pd.DataFrame(intrs.zμ_new))
    if intrs.zμ_rec is not None:
        lg.print('zμ_rec:')
        lg.display(pd.DataFrame(intrs.zμ_rec))
    lg.print(profiles.log)
    pd.options.display.float_format = float_format


def log_iter(model: Union[VariationalAutoencoder, SVI], nll: BaseWeightedLoss,
             trans: Transforms, test_data: TestArraysToTensors, loader: TrainLoader,
             batch: Batch) -> Tuple[str, Dict[str, Any]]:
    def prepare(tpl: Tuple[Tensor, Tensor]) -> LstTpl:
        x_rec_cont, x_rec_sample = tpl
        return [('', x_rec_cont), ('', trans.round_x_rec_tens(x_rec_cont)), ('', x_rec_sample)]

    b = batch
    u, u_nll, w_vec_u, w_mat_u = test_data.get_test_tensors()
    x, x_nll, w_vec, _, y, type_ = test_data.get_test_labelled_tensors()
    assert (type_ > 0).all() and (len(type_.unique()) == 16)
    u_r1, u_nll_r1, w_vec_u_r1, _ = test_data.get_random_learn_tensors()
    u_r2, u_nll_r2, w_vec_u_r2, _ = test_data.get_random_learn_tensors()
    x_r1, _, w_vec_r1, _, y_r1, _ = test_data.get_random_learn_labelled_tensors()
    x_r2, _, w_vec_r2, _, y_r2, _ = test_data.get_random_learn_labelled_tensors()

    def wμ(x_: Tensor, w_vec_: Opt[Tensor]=None) -> float:
        if w_vec_ is None:
            return x_.mean().item()
        return (x_ * w_vec_[:, 0]).mean().item()

    svi: Opt[SVI] = None
    vae: VariationalAutoencoder
    kld: KLDLoss
    tckld: Opt[BetaTCKLDLoss] = None
    if isinstance(model, SVI):
        svi = model
        kld = svi.model.kld
        if not isinstance(svi.model, DeepGenerativeModel):
            raise ValueError('SVI().model should be DeepGenerativeModel')
        vae = svi.model
    elif isinstance(model, VariationalAutoencoder):
        kld = model.kld
        vae = model
    else:
        raise NotImplementedError
    if isinstance(kld, BetaTCKLDLoss):
        tckld = kld
    adgm: Opt[AuxiliaryDeepGenerativeModel] = None
    vae_cls: Opt[VAEClassifyMeta] = None
    if isinstance(svi, SVI):
        if isinstance(svi.model, AuxiliaryDeepGenerativeModel):
            adgm = svi.model
    else:
        if isinstance(vae, VAEClassifyMeta):
            vae_cls = vae

    kld.set_verbose(True)
    if adgm is not None:
        adgm.set_verbose(True)

    assert isinstance(vae, VariationalAutoencoder)
    pass_sex = Passer(vae.decoder)

    # Losses ------------------------------------------------------------------
    losses: Dict[str, float] = dict()

    def sampler(x_: Tensor) -> Tensor:
        if svi is not None:
            return svi.sampler.__call__(x_)
        return x_

    def tc_kld_to_losses(verbose: Dict[str, Tensor], losses_: Dict[str, float],
                         postfix: str = '') -> Dict[str, float]:
        v = verbose
        if tckld is not None:
            kld_0 = verbose['kld_unmod']
            assert kld_0 is not None
            losses_ = {f'KLD_unmod{postfix}': wμ(sampler(kld_0)), **losses_}

            tckld_, mi, tc, dw = v.get('tckld'), v.get('mi'), v.get('tc'), v.get('dw')
            if (tckld_ is not None) and (mi is not None) and (tc is not None) and (dw is not None):
                losses_ = {f'TCKLD{postfix}': wμ(sampler(tckld_)),
                           f'MI{postfix}': wμ(sampler(mi)),
                           f'TC{postfix}': wμ(sampler(tc)),
                           f'DW{postfix}': wμ(sampler(dw)), **losses_}
        return losses_

    def adgm_to_losses(verbose: Dict[str, Tensor], losses_: Dict[str, float],
                       postfix: str = '') -> Dict[str, float]:
        if adgm is not None:
            assert svi is not None
            kld_z = verbose['adgm_kld_z']
            kld_a = verbose['adgm_kld_a']
            assert (kld_z is not None) and (kld_a is not None)
            losses_ = {f'KLDz{postfix}': wμ(svi.sampler.__call__(kld_z)),
                       f'KLDa{postfix}': wμ(svi.sampler.__call__(kld_a)), **losses_}
        return losses_

    def u_to_losses(losses_: Dict[str, float], u_: Tensor, u_nll_: Opt[Tensor], w_vec_: Opt[Tensor],
                    w_mat_u_: Opt[Tensor], postfix: str='_u', unmod_kld: bool=True) -> Tuple[
        Dict[str, float], Tuple[Tensor, ...], Tensor, Dict[str, Tensor]
    ]:
        """
        I understand that this is not a proper NELBO for unlabelled data.
        But for now it will do.

        It's a correct NELBO for simple VAE though.
        """
        _, y_u_, qz_params_u_, _ = get_z_ymax(x=u_, y=None, model=vae)

        if svi is not None:
            with kld.unmodified_kld():
                nelbo_u_, _, _, verb_ = svi.__call__(x=u_, y=None, weight=w_mat_u_, x_nll=u_nll_)
                u_rec_params_, kld_u_, _ = svi.model.__call__(u_, y_u_)
            nelbo_ = wμ(nelbo_u_)
        else:
            if unmod_kld:
                with kld.unmodified_kld():
                    u_rec_params_, kld_u_, verb_ = vae.__call__(u_, u_)
            else:
                u_rec_params_, kld_u_, verb_ = vae.__call__(u_, u_)
            nelbo_ = 0.

        nlps_ = pass_sex.neg_log_p_passthr_x
        neg_log_p_sex_ = nlps_ if isinstance(nlps_, int) else wμ(nlps_)

        kld_u = wμ(kld_u_, w_vec_)
        nll_u = wμ(
            nll.__call__(x_params=u_rec_params_, target=u_nll_ if (u_nll_ is not None) else u_, weight=w_mat_u_),
            w_vec_)

        if svi is None:
            nelbo_ = nll_u + kld_u

        args = 'x,s' if (neg_log_p_sex_ != 0) else 'x'
        losses_ = {f'NELBO({args}){postfix}': nelbo_ + neg_log_p_sex_, f'NLL{postfix}': nll_u,
                   f'KLD{postfix}': kld_u, **losses_}

        return losses_, qz_params_u_, u_rec_params_, verb_

    def classify_(x_: Tensor, use_zμ: bool=False) -> Tensor:
        cls_: Union[DeepGenerativeModel, VAEClassifyMeta]
        if svi is not None:
            cls_ = svi.model
        elif vae_cls is not None:
            cls_ = vae_cls
        else:
            raise RuntimeError

        probs_, _ = cls_.classify(x_) if not use_zμ else cls_.classify_deterministic(x_)
        return probs_

    def accuracy_(logits_or_probs: Tensor, target: Tensor) -> Tensor:
        if svi is not None:
            return svi.accuracy(logits_or_probs, target)
        if vae_cls is not None:
            return vae_cls.classifier.accuracy(logits_or_probs, target)
        raise RuntimeError

    if svi is not None:
        dgm: DeepGenerativeModel = svi.model
        svi.set_consts(β=1)

        # Labelled x, y and L
        if tckld is not None:
            tckld.set_dataset_size(loader.labelled_dataset_size)
        assert (b.x is not None) and (b.y is not None)
        nelbo, cross_entropy_, _, verb = svi.__call__(x=b.x, y=b.y, weight=b.w_mat, x_nll=b.x_nll)
        L, cross_entropy = wμ(nelbo), wμ(cross_entropy_)

        losses = tc_kld_to_losses(verb, losses)
        losses = adgm_to_losses(verb, losses)

        def lbl_t_dict(svi_: SVI, y_: Tensor, postf: str='') -> Dict[str, float]:
            with dgm.kld.unmodified_kld():
                nelbo_x_, _, _, _ = svi_.__call__(x=x, y=y_, weight=None, x_nll=x_nll)
                nelbo_x = wμ(nelbo_x_)
                x_rec_params, kld_t_, _ = dgm.__call__(x, dgm.classifier.transform_y(y_))
                kld_t = wμ(kld_t_, w_vec)
            nlps = pass_sex.neg_log_p_passthr_x
            neg_log_p_sex = nlps if isinstance(nlps, int) else wμ(nlps)

            nll_t = wμ(nll.__call__(x_params=x_rec_params, target=x_nll, weight=None), w_vec)
            return {f'NELBO_t{postf}': nelbo_x + neg_log_p_sex, f'NLL_t{postf}': nll_t, f'KLD_t{postf}': kld_t}

        x_y_t_losses = lbl_t_dict(svi, y, postf='_l')
        _, yx, _, _ = get_z_ymax(x=x, y=None, model=dgm, verbose=True)
        x_yx_t_losses = lbl_t_dict(svi, yx, postf='_l_nn')

        losses = dict(**x_y_t_losses, **x_yx_t_losses, L=L, **losses)

        # Unlabelled u and U:
        if tckld is not None:
            tckld.set_dataset_size(loader.unlabelled_dataset_size)

        nelbo_u, _, _, verb = svi.__call__(x=b.u, weight=b.w_mat_u, x_nll=b.u_nll)
        U = wμ(nelbo_u)

        losses = tc_kld_to_losses(verb, losses, '_u')
        losses = adgm_to_losses(verb, losses, '_u')

        losses = dict(U=U, **losses)
        losses, qz_params_u, u_rec_params, _ = u_to_losses(losses, u, u_nll, w_vec_u, None, postfix='_t_u')

        # J_α:
        J_α = L + cross_entropy * svi.α + U

        losses, _, _, _ = u_to_losses(losses, u_r1, u_nll_r1, w_vec_u_r1, None, postfix='_r1_u')
        losses, _, _, _ = u_to_losses(losses, u_r2, u_nll_r2, w_vec_u_r2, None, postfix='_r2_u')

        losses = dict(J_α=J_α, CE=cross_entropy, **losses)
    else:
        losses, _, _, verb = u_to_losses(losses, b.u, b.u_nll, None, b.w_mat_u, postfix='',
                                         unmod_kld=False)
        losses = tc_kld_to_losses(verb, losses)
        losses, _, _, _ = u_to_losses(losses, u_r1, u_nll_r1, w_vec_u_r1, None, postfix='_r1_u')
        losses, _, _, _ = u_to_losses(losses, u_r2, u_nll_r2, w_vec_u_r2, None, postfix='_r2_u')
        losses, qz_params_u, u_rec_params, _ = u_to_losses(losses, u, u_nll, w_vec_u, None, postfix='_t')

    def get__acc__acc_per_type(probs: Tensor, postf: str=''):
        acc_ = accuracy_(probs, y)

        acc_per_type_ = {f'{s}{j}{postf}': accuracy_(probs[maskj & sex], y[maskj & sex]).mean().item()
                         for j, maskj in [(i, type_ == i) for i in range(1, 17)]
                         for s, sex in (('f', x[:, 0] < 0.5), ('m', x[:, 0] > 0.5))}
        acc_per_type_[f'Av{postf}'] = sum(acc_per_type_.values()) / len(acc_per_type_)
        return acc_, acc_per_type_

    if (svi is not None) or (vae_cls is not None):
        # ------------------------------------------------------------------------
        cog_funcs_mse_u: float = 0
        cog_funcs_mse_u_locality: float = 0
        if vae_cls is not None:
            if isinstance(vae_cls.classifier, ClassifierPassthrJATS24KhCFBase):
                _cf_cls: ClassifierPassthrJATS24KhCFBase = vae_cls.classifier
                _z_u_ext, _μ_u_ext = vae_cls.get_z_μ(u)
                _cog_funcs_mse_u, _ = _cf_cls.get__mse__reg_loss(_μ_u_ext)
                _cog_funcs_mse_u_locality, _ = _cf_cls.get__mse__reg_loss(_z_u_ext, _μ_u_ext)
                cog_funcs_mse_u = wμ(_cog_funcs_mse_u, w_vec_u)
                cog_funcs_mse_u_locality = wμ(_cog_funcs_mse_u_locality, w_vec_u)
        # ------------------------------------------------------------------------

        acc, acc_per_type = get__acc__acc_per_type(classify_(x))
        acc_μ, acc_μ_per_type = get__acc__acc_per_type(classify_(x, use_zμ=True), postf='μ')

        losses = dict(
            CF_MSE_t=cog_funcs_mse_u,
            CF_MSE_t_loc=cog_funcs_mse_u_locality,
            # wAcc_t_m4=acc_per_type['m4'], minWAcc_t_no_m4=min(v for k, v in acc_per_type.items() if k != 'm4'),
            minWAcc_t=min(acc_per_type.values()),
            wAcc_t=wμ(acc, w_vec), Acc_t=wμ(acc),
            wAcc_r1=wμ(accuracy_(classify_(x_r1), y_r1), w_vec_r1),
            wAcc_μ_r2=wμ(accuracy_(classify_(x_r2, use_zμ=True), y_r2), w_vec_r2),
            minWAcc_μ_t=min(acc_μ_per_type.values()),
            wAcc_μ_t=wμ(acc_μ, w_vec),
            **acc_per_type, **acc_μ_per_type, **losses)

    μ: Opt[Tensor] = None
    log_σ: Opt[Tensor] = None
    if (len(qz_params_u) == 2) or ((len(qz_params_u) == 4) and (adgm is not None)):
        μ, log_σ = qz_params_u[:2]

    # Test ----------------------------------------------------------------
    u_rec_det = prepare(nll.x_recon(u_rec_params))

    u_rec_list: LstTpl = [('μ', 0)]
    u_rec_list += u_rec_det
    if w_vec_u is None:
        weight_vec_x_mat_u = w_mat_u
    elif w_mat_u is None:
        weight_vec_x_mat_u = w_vec_u
    else:
        weight_vec_x_mat_u = w_vec_u * w_mat_u

    u_weight_list_: LstTplOpt = [('±TW:', weight_vec_x_mat_u), ('TW:', w_vec_u), ('¬W:', 1)]
    u_weight_list: LstTpl = [s for s in u_weight_list_ if s[1] is not None]  # type: ignore

    # MSE and BCE:
    mse_bce_test: List[List[Union[float, str]]] = [[
            ((u - u_rec)**2 * weight).mean().item(),
            ((-u * tr.log(u_rec + ε) + (u - 1) * tr.log(-u_rec + 1 + ε)) * weight).view(
                u.size(0), -1).sum(dim=1).mean().item()
        ] if isinstance(u_rec, Tensor)
        else [(z_kind + weight_name)]
        for weight_name, weight in u_weight_list for z_kind, u_rec in u_rec_list]

    # KLD vector --------------------------------------------------------------
    kld_vec: Opt[List[float]]
    if (μ is not None) and (log_σ is not None):
        kld_mat = ndarr(Normal.kl((μ, log_σ)))
        kld_vec = [float(s) for s in np.mean(kld_mat, axis=0)]
    else:
        kld_vec = None

    # Log ---------------------------------------------------------------------
    string = (
        ', '.join(f'{k}={v:.2f}' for k, v in losses.items()) +
        ", BCE_test=(" + ", ".join(e if isinstance(e, str) else f"{e:.2f}" for e in [item[-1] for item in mse_bce_test])
        + ")" +
        ", MSE_test=(" + ", ".join(e if isinstance(e, str) else f"{e:.4f}" for e in [item[0] for item in mse_bce_test])
        + ")" + ' KLD_test_gauss: ' +
        ('  '.join([f'{kld_i:.2f}' for kld_i in kld_vec]) if (kld_vec is not None) else '')
    )
    dic = dict(test_losses=losses, KLD_vec=kld_vec, mse_bce_test=mse_bce_test)

    kld.set_verbose(False)
    if adgm is not None:
        adgm.set_verbose(False)
    return string, dic
