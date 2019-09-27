from typing import Callable, List, Union, Tuple, Dict, Optional as Opt, Any
from dataclasses import asdict
# from types import ModuleType
from os import path as p
import numbers
import torch as tr
from torch import Tensor
import numpy as np
# noinspection PyPep8Naming
from numpy import ndarray as Array
import numpy.linalg as la
import pandas as pd
import seaborn as sns
import IPython.display as ds
from kiwi_bugfix_typechecker import ipython

from beta_tcvae_typed import Normal
from semi_supervised_typed import VariationalAutoencoder, SVI, DeepGenerativeModel, AuxiliaryDeepGenerativeModel
from vae import BaseWeightedLoss, TrimLoss
from vae.utils import get_x_normalized_μ_σ, ndarr
from vae.data import Dim, TrainLoader, VAELoaderReturnType
from vae.display import Log
from vae.linear_component_analyzer import LinearAnalyzer
from socionics_db import JATSModelOutput, Transforms, Data
from jats_vae.utils import print_err_tuple, euc_dist_count_types, TestArraysToTensors
from jats_vae.data import get_zμ_ymax
from .colors import types_colors  # pylint: disable=relative-beyond-top-level

DEBUG = False
LstTpl = List[Tuple[str, Union[int, Tensor]]]
LstTplOpt = List[Tuple[str, Union[int, Tensor, None]]]
ε = 10**-8


def plot_z_cov(z: Array, weights: Opt[Array], file: str, sub: Callable[[Array], Array]=lambda m: m) -> None:
    import matplotlibhelper as mh
    import matplotlib.pyplot as plt
    z_normalized, _, _ = get_x_normalized_μ_σ(z, weights)
    Σ = np.cov(z_normalized.T, aweights=weights)
    E = np.eye(len(Σ)) if Σ.shape else np.eye(1)
    fig = plt.figure(figsize=mh.figsize(w=6))
    sns.heatmap(sub(Σ - E), center=0, square=True, mask=sub(E > (1 - ε)))
    plt.title(p.basename(file))
    fig.tight_layout()
    ipython.display(ds.HTML(f'<img src="{mh.img(plt, name=file)}" width="900">'))


def plot_dist(basis: List[int], z_batch: Array, z: Array, types: Array,  # pylint: disable=too-many-branches
              types_self: Array, file: str, μ_z: Union[Array, float]=None, σ_z: Union[Array, float]=None,
              plt_types: Tuple[int, ...]=tuple(range(1, 17)), questions: bool=False, refs: Array=None) -> None:
    if not basis or (len(basis) == 1):
        return  # presumably matplotlib is buggy in this case
    import matplotlibhelper as mh
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(len(basis), 4, figsize=(16, 3 * len(basis)))
    for i, j in enumerate(basis):
        ax_title = mh.stex(f'ˎs_˱{i}˲ˎ (ˎz_˱{j}˲ˎ)')

        # 1st plot (all database):
        ax = axs[i, 0]
        z1, z2 = z_batch[:, j], z[:, i]
        sns.distplot(z1, bins=30, ax=ax)
        sns.distplot(z2, bins=30, ax=ax)
        if (μ_z is None) and (σ_z is None):
            μz_σz = ''
        elif isinstance(μ_z, numbers.Real) and isinstance(σ_z, numbers.Real):
            μz_σz = f';{μ_z:.2f},{σ_z:.2f}'
        elif isinstance(μ_z, Array) and isinstance(σ_z, Array):
            μz_σz = f';{μ_z[j]:.2f},{σ_z[j]:.2f}'
        else:
            raise ValueError(f'Bad μ_z and/or σ_z:\n{μ_z};\n{σ_z}')
        ax.set_title(ax_title + f', μ,σ={np.mean(z1.T):.2f},{np.std(z1.T):.2f}' + μz_σz)

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
        if refs is not None:
            for type_ in plt_types:
                ax.plot(refs[type_ - 1, i], 0, color=types_colors[type_ - 1], marker='o', linestyle='')
        ax.set_title(ax_title + ' axis (Tal. = self)')

    fig.tight_layout()
    ipython.display(ds.HTML(f'<img src="{mh.img(plt, name=file)}" width="900">'))


def get_x_plot_batch(data_loader: TrainLoader) -> Array:
    data_loader.regenerate_loaders(batch_size=1600)
    train_loader = data_loader.get_vae_loader()
    item: VAELoaderReturnType = next(iter(train_loader))
    x_batch, _, _, _ = item
    return ndarr(x_batch)


def get__x__z__plot_batch(model: VariationalAutoencoder, data_loader: TrainLoader) -> Tuple[Array, Array]:
    data_loader.regenerate_loaders(batch_size=1600)
    train_loader = data_loader.get_vae_loader()
    item: VAELoaderReturnType = next(iter(train_loader))
    x_batch, _, _, _ = item
    with tr.no_grad():
        z_batch, _, _, _, _ = get_zμ_ymax(x=x_batch, model=model)
    return ndarr(x_batch), ndarr(z_batch)


def plot_jats(weight: Opt[Array], prefix_path_db_nn: str, prefix_path_db: str, types_tal: Array, types_self: Array,
              x_batch: Array=None, z_batch: Array=None, profiles: JATSModelOutput=None,
              pca: LinearAnalyzer=None, fa: LinearAnalyzer=None,
              plot_questions: bool=False, plot_pca: bool=False, plt_types: Tuple[int, ...]=tuple(range(1, 17))) -> None:
    """
    Plot latent dimensions.
    """
    if plot_pca and pca is not None:
        if x_batch is None:
            raise ValueError('x_batch is None')
        plot_dist(list(range(pca.n)), z_batch=pca.transform(x_batch),
                  z=pca.z_normalized, μ_z=pca.μ_z, σ_z=pca.σ_z, file=prefix_path_db + f'-WPCA{pca.n}',
                  refs=pca.z_norm_refs, types=types_tal, types_self=types_self, plt_types=plt_types)
        plot_z_cov(pca.z_normalized, weight, file=prefix_path_db + f'-WPCA{pca.n}-z-cov')

    if plot_pca and fa is not None:
        if x_batch is None:
            raise ValueError('x_batch is None')
        plot_dist(list(range(fa.n)), z_batch=fa.transform(x_batch),
                  z=fa.z_normalized, μ_z=fa.μ_z, σ_z=fa.σ_z, file=prefix_path_db + f'-WFA{fa.n}',
                  refs=fa.z_norm_refs, types=types_tal, types_self=types_self, plt_types=plt_types)
        plot_z_cov(fa.z_normalized, weight, file=prefix_path_db + f'-WFA{fa.n}-z-cov')

    if plot_pca and (pca is not None) and (fa is not None):
        len_ = fa.z_normalized.shape[1]
        plot_z_cov(np.concatenate((fa.z_normalized, pca.z_normalized), axis=1),
                   weight, file=prefix_path_db_nn + f'-fa{fa.n}-vs-pca{pca.n}', sub=lambda m: np.abs(m[:len_, len_:]))

    if profiles is not None:
        if (z_batch is None) or (pca is None) or (fa is None):
            raise ValueError('(zμ_batch is None) or (pca is None) or (fa is None)')
        μ, σ = profiles.μ_z, profiles.σ_z
        plot_dist(profiles.basis, z_batch=(z_batch - μ) / σ, z=profiles.z_normalized, μ_z=μ, σ_z=σ,
                  file=prefix_path_db_nn, refs=profiles.z_norm_refs, types=types_tal, types_self=types_self,
                  plt_types=plt_types)
        plot_z_cov(profiles.z_normalized, weight, file=prefix_path_db_nn + '-z-cov')
        len_ = pca.z_normalized.shape[1]
        plot_z_cov(np.concatenate((pca.z_normalized, profiles.z_normalized), axis=1),
                   weight, file=prefix_path_db_nn + f'-pca{pca.n}-vs-z', sub=lambda m: np.abs(m[:len_, len_:]))
        len_ = fa.z_normalized.shape[1]
        plot_z_cov(np.concatenate((fa.z_normalized, profiles.z_normalized), axis=1),
                   weight, file=prefix_path_db_nn + f'-fa{fa.n}-vs-z', sub=lambda m: np.abs(m[:len_, len_:]))

    if plot_questions:
        if (x_batch is None) or (profiles is None):
            raise ValueError('(x_batch is None) or (profiles is None)')
        n = len(x_batch[0])
        buckets = [
            [i * 16 + j for j in range(16)] for i in range(n // 16)
        ] + [[(n // 16) * 16 + i for i in range(n % 16)]]
        for i, bucket in enumerate(buckets):
            plot_dist(bucket, z_batch=x_batch, z=profiles.x, file=prefix_path_db + f'-NormQuestions{i}',
                      questions=True, types=types_tal, types_self=types_self, plt_types=plt_types)


def explore_jats(profiles: JATSModelOutput,
                 dat: Data,
                 dims: Dim,
                 trans: Transforms,
                 types: Array,
                 types_sex: Array,
                 types_self: Array,
                 logger: Log,
                 pca: LinearAnalyzer,
                 fa: LinearAnalyzer,
                 mmd_dists: Dict[int, str]=None,
                 inspect: Tuple[int, ...]=None) -> None:
    lg = logger
    if inspect is None:
        inspect = tuple(range(len(dat.interesting)))

    dropped_kld = list(np.round(np.array(profiles.dropped_kld), 2)) if (profiles.dropped_kld is not None) else ''
    lg.print(f'''Used {len(profiles.basis)}/{dims.z} dimensions.
        Used dims: {profiles.basis},
        KLD of dropped dims: {dropped_kld}.
        '''.replace('\n        ', '\n'))

    intrs_kwargs = asdict(profiles)
    L = len(profiles.x)
    for k, v in intrs_kwargs.items():
        if isinstance(v, Array) and (len(v) == L):
            v_: Array = v
            intrs_kwargs[k] = v_[dat.interesting]
    intrs_kwargs['x_rec_disc'] = tuple(s[dat.interesting] for s in profiles.x_rec_disc)
    intrs = JATSModelOutput(**intrs_kwargs)

    # Peek questions of selected profiles:
    # -------------------------------------------------------
    out = np.stack(
        [trans.to_1__5(intrs.x[i]) - 3 for i in inspect] +
        [trans.to_1__5(intrs.x_rec_disc[0][i]) - 3 for i in inspect],
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

    print_err_tuple(profiles.x_rec_cont, *profiles.x_rec_disc, profiles.x_rec_sample,
                    x=profiles.x, data=dat, line_printer=line_printer)

    lg.print(mmd_dists)
    lg.print_i(mmd_dists)

    # Inspect selected profiles:
    # -------------------------------------------------------
    def mse_(x, y): return np.mean(np.power(x - y, 2))

    lg.print('First counted refs-FA types, then refs-PCA, then refs-NN, then y-NN, then TT, then TT=self, then TT_sex.')

    try:
        lg.display(pd.DataFrame(euc_dist_count_types(fa.z_normalized, fa.z_norm_refs).reshape((4, 4))))
        lg.print('refs-FA.\n\n')
    except ValueError:
        lg.print('refs-FA count failed')

    try:
        lg.display(pd.DataFrame(euc_dist_count_types(pca.z_normalized, pca.z_norm_refs).reshape((4, 4))))
        lg.print('refs-PCA.\n\n')
    except ValueError:
        lg.print('refs-PCA count failed')

    try:
        lg.display(pd.DataFrame(euc_dist_count_types(profiles.z_normalized,
                                                     profiles.z_norm_refs).reshape((4, 4))))
        lg.print('refs-NN.\n\n')
    except ValueError:
        lg.print('refs-NN count failed')

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
            f'MSE #{i}:nn(#{i})            = {mse_(intrs.x[i], intrs.x_rec_cont[i]):.3f}\n' +
            f'MSE discr. #{i}:nn(#{i})     = {mse_(intrs.x[i], intrs.x_rec_disc[0][i]):.3f}\n' +
            ''.join([
                f'MSE discr. #{i}:#{j}         = {mse_(intrs.x[i], intrs.x[j]):.3f}\n' +
                f'MSE nn(#{i}):nn(#{j})        = {mse_(intrs.x_rec_cont[i], intrs.x_rec_cont[j]):.3f}\n' +
                f'MSE discr. nn(#{i}):nn(#{j}) = {mse_(intrs.x_rec_disc[0][i], intrs.x_rec_disc[0][j]):.3f}\n'
                for j in inspect if i != j]))

        def euc_dist(x: Array, A: Array) -> Array:
            return la.norm(A - x, axis=-1)
            # return np.sqrt(np.sum(np.power(A - x, 2), axis=-1))

        explr_x = euc_dist(intrs.x[i], profiles.x)
        sortd_x = np.argsort(explr_x)

        explr_x_recon = euc_dist(intrs.x_rec_cont[i], profiles.x_rec_cont)
        sortd_x_recon = np.argsort(explr_x_recon)

        explr_z = euc_dist(intrs.z_normalized[i], profiles.z_normalized)
        sortd_z = np.argsort(explr_z)

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

        explr_type_z = euc_dist(intrs.z_normalized[i], profiles.z_norm_refs)
        sortd_type_z = np.argsort(explr_type_z)
        explr_type_fa_z = euc_dist(intrs_fa_z[i], fa.z_norm_refs)
        sortd_type_fa_z = np.argsort(explr_type_fa_z)
        explr_type_pca_z = euc_dist(intrs_pca_z[i], pca.z_norm_refs)
        sortd_type_pca_z = np.argsort(explr_type_pca_z)

        if profiles.y_probs is not None:
            if intrs.y_probs is None:
                raise RuntimeError
            n = intrs.y_probs.shape[1]
            sorted_y = np.array(list(reversed(np.argsort(intrs.y_probs[i]))))
            lg.display(pd.DataFrame({
                f'type y-nn (of {n})': sorted_y[:n] + 1,
                'prob y': x100(intrs.y_probs[i][sorted_y])[:n],
            }))
        lg.display(pd.DataFrame({
            'type ref-nn': sortd_type_z[:10] + 1,
            'z 100euc': x100(explr_type_z[sortd_type_z])[:10],
            'type fa': sortd_type_fa_z[:10] + 1,
            'fa z 100euc': x100(explr_type_fa_z[sortd_type_fa_z])[:10],
            'type pca': sortd_type_pca_z[:10] + 1,
            'pca z 100euc': x100(explr_type_pca_z[sortd_type_pca_z])[:10],
        }))
        lg.display(pd.DataFrame({
            'idx1': sortd_x[:10],
            'x 100euc': x100(explr_x[sortd_x])[:10],
            'tal1': types[sortd_x][:10],
            'self1': types_self[sortd_x][:10],
        }))
        lg.display(pd.DataFrame({
            'idx2': sortd_x_recon[:10],
            'x_recon 100euc': x100(explr_x_recon[sortd_x_recon])[:10],
            'tal2': types[sortd_x_recon][:10],
            'self2': types_self[sortd_x_recon][:10],
            'idx3': sortd_z[:10],
            'z 100euc': x100(explr_z[sortd_z])[:10],
            'tal3': types[sortd_z][:10],
            'self3': types_self[sortd_z][:10],
        }))
        lg.display(pd.DataFrame({
            'idx4': sortd_pca_z[:10],
            'pca z 100euc': x100(explr_pca_z[sortd_pca_z])[:10],
            'tal4': types[sortd_pca_z][:10],
            'self4': types_self[sortd_pca_z][:10],
            'idx5': sortd_pca_x_rec[:10],
            'pca x_rec 100euc': x100(explr_pca_x_rec[sortd_pca_x_rec])[:10],
            'tal5': types[sortd_pca_x_rec][:10],
            'self5': types_self[sortd_pca_x_rec][:10],
        }))
        lg.display(pd.DataFrame({
            'idx6': sortd_fa_z[:10],
            'fa z 100euc': x100(explr_fa_z[sortd_fa_z])[:10],
            'tal6': types[sortd_fa_z][:10],
            'self6': types_self[sortd_fa_z][:10],
            'idx7': sortd_fa_x_rec[:10],
            'fa x_rec 100euc': x100(explr_fa_x_rec[sortd_fa_x_rec])[:10],
            'tal7': types[sortd_fa_x_rec][:10],
            'self7': types_self[sortd_fa_x_rec][:10],
        }))

    lg.display(pd.DataFrame(intrs.z_normalized))


def log_iter(model: Union[VariationalAutoencoder, SVI], nll: BaseWeightedLoss, trans: Transforms,
             test_data: TestArraysToTensors, trim: Opt[TrimLoss], loader: TrainLoader) -> Tuple[str, Dict[str, Any]]:
    def prepare(tpl: Tuple[Tensor, Tensor]) -> LstTpl:
        x_rec_cont, x_rec_sample = tpl
        return [('', x_rec_cont), ('', trans.round_x_rec_tens(x_rec_cont)), ('', x_rec_sample)]

    u, u_nll, weight_vec_u, weight_mat_u = test_data.get_test_tensors()
    x, x_nll, weight_vec, weight_mat, y = test_data.get_test_labelled_tensors()
    # x_lrn, weight_vec_lrn, y_lrn = test_data.get_learn_labelled_tensors()
    # x_lrn_rand1, weight_vec_lrn_rand1, y_lrn_rand1 = test_data.get_rand_learn_labelled_tensors()
    # x_lrn_rand2, weight_vec_lrn_rand2, y_lrn_rand2 = test_data.get_rand_learn_labelled_tensors()

    def wμ(x_: Tensor, weight_vec_: Opt[Tensor]) -> float:
        if weight_vec_ is None:
            return x_.mean().item()
        return (x_ * weight_vec_[:, 0]).mean().item()

    weight_vec_u_s = weight_vec_u.repeat(y.shape[1], 1) if (weight_vec_u is not None) else None

    def wμ_x(x_: Tensor) -> float: return wμ(x_, weight_vec)
    def wμ_u(u_: Tensor) -> float: return wμ(u_, weight_vec_u)
    def wμ_u_s(u_: Tensor) -> float: return wμ(u_, weight_vec_u_s)

    svi: Opt[SVI] = None
    if isinstance(model, SVI):
        svi = model
        tc_kld = svi.model.tc_kld
        if not isinstance(svi.model, DeepGenerativeModel):
            raise ValueError('SVI().model should be DeepGenerativeModel')
    elif isinstance(model, VariationalAutoencoder):
        tc_kld = model.tc_kld
    else:
        raise NotImplementedError
    adgm: Opt[AuxiliaryDeepGenerativeModel] = None
    if isinstance(svi, SVI):
        if isinstance(svi.model, AuxiliaryDeepGenerativeModel):
            adgm = svi.model

    if tc_kld is not None:
        tc_kld.set_verbose(True)
    if adgm is not None:
        adgm.set_verbose(True)

    # Losses ------------------------------------------------------------------
    losses: Dict[str, float] = dict()

    def pop_tc_kld_to_losses(losses_: Dict[str, float], wμ_: Callable[[Tensor], float],
                             postfix: str = '') -> Dict[str, float]:
        if tc_kld is not None:
            if svi is None:
                raise RuntimeError
            kld_1, kld_0, mi, tc, dw = tc_kld.stored.pop()
            losses_ = {f'KLD1{postfix}': wμ_(svi.sampler.__call__(kld_1)),
                       f'KLD0{postfix}': wμ_(svi.sampler.__call__(kld_0)), **losses_}
            if (mi is not None) and (tc is not None) and (dw is not None):
                losses_ = {f'MI{postfix}': wμ_(svi.sampler.__call__(mi)),
                           f'TC{postfix}': wμ_(svi.sampler.__call__(tc)),
                           f'DW{postfix}': wμ_(svi.sampler.__call__(dw)), **losses_}
        return losses_

    def pop_adgm_to_losses(losses_: Dict[str, float], wμ_: Callable[[Tensor], float],
                           postfix: str = '') -> Dict[str, float]:
        if adgm is not None:
            if svi is None:
                raise RuntimeError
            kld_z, kld_a = adgm.stored.pop()
            losses_ = {f'KLDz{postfix}': wμ_(svi.sampler.__call__(kld_z)),
                       f'KLDa{postfix}': wμ_(svi.sampler.__call__(kld_a)), **losses_}
        return losses_

    def assert_tc_kld_empty() -> None:
        if tc_kld is not None:
            if tc_kld.stored:
                raise AssertionError(f'len(tc_kld.stored) == {len(tc_kld.stored)}')

    def assert_adgm_empty() -> None:
        if adgm is not None:
            if adgm.stored:
                raise AssertionError(f'len(adgm.stored) == {len(adgm.stored)}')

    assert_tc_kld_empty()
    assert_adgm_empty()

    if svi is not None:
        dgm: DeepGenerativeModel = svi.model
        N_u, N_l = u.shape[0], x.shape[0]
        _N_l, _N_u = svi.N_l, svi.N_u
        svi.set_consts(N_l=N_l, N_u=N_u)

        # Labelled x, y and L
        if tc_kld is not None:
            tc_kld.set_dataset_size(loader.labelled_dataset_size)
        nelbo, cross_entropy_, probs = svi.__call__(x=x, y=y, weight=weight_mat, x_nll=x_nll)
        L, cross_entropy = wμ_x(nelbo), wμ_x(cross_entropy_)

        if (adgm is not None) and (tc_kld is not None) and (len(tc_kld.stored) > 1):
            losses = pop_tc_kld_to_losses(losses, wμ_x, '_a')
        losses = pop_tc_kld_to_losses(losses, wμ_x)
        losses = pop_adgm_to_losses(losses, wμ_x)
        assert_tc_kld_empty()
        assert_adgm_empty()

        z_x, y_x, _, kld_, _ = get_zμ_ymax(x=x, model=dgm)
        x_rec_params = dgm.sample(z_x, y_x)
        kld = wμ_x(kld_)
        nll_ = wμ_x(nll.__call__(x_params=x_rec_params, target=x_nll, weight=weight_mat))
        losses = dict(NLL=nll_, NELBO=L, KLD_mod=kld, **losses)

        # Unlabelled u and U:
        if tc_kld is not None:
            tc_kld.set_dataset_size(loader.unlabelled_dataset_size)
            while tc_kld.stored:
                tc_kld.stored.pop()
        if adgm is not None:
            while adgm.stored:
                adgm.stored.pop()
        assert_tc_kld_empty()
        assert_adgm_empty()

        nelbo_u, _, _ = svi.__call__(x=u, weight=weight_mat_u, x_nll=u_nll)
        U = wμ_u(nelbo_u)

        if (adgm is not None) and (tc_kld is not None) and (len(tc_kld.stored) > 1):
            losses = pop_tc_kld_to_losses(losses, wμ_u_s, '_ua')
        losses = pop_tc_kld_to_losses(losses, wμ_u_s, '_u')
        losses = pop_adgm_to_losses(losses, wμ_u_s, '_u')
        assert_tc_kld_empty()
        assert_adgm_empty()

        z_u, y_u, qz_params_u, kld_u_, _ = get_zμ_ymax(x=u, model=dgm)
        kld_u = wμ_u(kld_u_)
        u_rec_params = dgm.sample(z_u, y_u)
        nll_u = wμ_u(nll.__call__(x_params=u_rec_params, target=u_nll, weight=weight_mat_u))
        losses = dict(NLL_u=nll_u, NELBO_u=U, KLD_mod_u=kld_u, **losses)

        if tc_kld is not None:
            while tc_kld.stored:
                tc_kld.stored.pop()
        if adgm is not None:
            while adgm.stored:
                adgm.stored.pop()
        assert_tc_kld_empty()
        assert_adgm_empty()

        # J_α:
        J_α = L + cross_entropy * svi.α + U
        acc = svi.accuracy(probs, y)

        losses = dict(WAcc=wμ_x(acc), Acc=acc.mean().item(), J_α=J_α, CE=cross_entropy,
                      # WAccLrn=wμ(svi.accuracy(svi.model.classify(x_lrn), y_lrn), weight_vec_lrn),
                      # WAccLrnR1=wμ(svi.accuracy(svi.model.classify(x_lrn_rand1), y_lrn_rand1), weight_vec_lrn_rand1),
                      # WAccLrnR2=wμ(svi.accuracy(svi.model.classify(x_lrn_rand2), y_lrn_rand2), weight_vec_lrn_rand2),
                      **losses)
        svi.set_consts(N_l=_N_l, N_u=_N_u)

    else:
        vae: VariationalAutoencoder = model
        z_u, y_u, qz_params_u, kld_, _ = get_zμ_ymax(x=u, model=vae)
        u_rec_params = vae.sample(z_u, y_u)
        kld = wμ_u(kld_)
        nll_ = wμ_u(nll.__call__(x_params=u_rec_params, target=u_nll, weight=weight_mat_u))
        nelbo_ = nll_ + kld

        losses = pop_tc_kld_to_losses(losses, wμ_u)
        assert_tc_kld_empty()

        losses = dict(NLL=nll_, NELBO=nelbo_,  KLD_mod=kld, **losses)

    μ: Opt[Tensor] = None
    log_σ: Opt[Tensor] = None
    if len(qz_params_u) == 2:
        μ, log_σ = qz_params_u
        if trim is not None:
            trim__ = trim.__call__(μ=μ, log_σ=log_σ)
            trim_ = trim__ if isinstance(trim__, int) else trim__.item()
            losses['Trim'] = trim_

    # Test ----------------------------------------------------------------
    u_rec_det = prepare(nll.x_recon(u_rec_params))

    u_rec_list: LstTpl = [('μ', 0)]
    u_rec_list += u_rec_det
    if weight_vec_u is None:
        weight_vec_x_mat_u = weight_mat_u
    elif weight_mat_u is None:
        weight_vec_x_mat_u = weight_vec_u
    else:
        weight_vec_x_mat_u = weight_vec_u * weight_mat_u

    u_weight_list_: LstTplOpt = [('±TW:', weight_vec_x_mat_u), ('TW:', weight_vec_u), ('¬W:', 1)]
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

    assert_tc_kld_empty()
    assert_adgm_empty()
    if tc_kld is not None:
        tc_kld.set_verbose(False)
    if adgm is not None:
        adgm.set_verbose(False)
    return string, dic
