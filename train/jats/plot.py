from typing import Tuple, List, Optional as Opt, Callable, Dict, Iterable
from os.path import basename
import numpy as np
import numpy.linalg as la
from numpy.typing import NDArray as Array
import pandas as pd
from pandas import DataFrame
import torch as tr
import seaborn as sns
from matplotlib.colors import to_rgb
from .load import get_loader, get_data

PLOT_BATCH_SIZE = 72 * 32
GR = (1 + 5 ** 0.5) / 2  # (w, w / GR) or (h * GR, h)
TYPES_COLORS = (
    'bright red', 'tangerine', 'dandelion', 'apple green',  # 'bright yellow'
    'magenta', 'purple', 'blue', 'teal blue',
    'grass green', 'gold', 'dark orange', 'blood red',
    'aqua blue', 'azure', 'bright violet', 'bright pink',  # 'bright purple'
)
TYPES_COLOR_NAMES = (
    'red', 'orange', 'yellow', 'lime',
    'magenta', 'purple', 'blue', 'teal',
    'green', 'mustard', 'brown', 'crimson',
    'cyan', 'azure', 'violet', 'pink',
)
if len(TYPES_COLORS) != len(TYPES_COLOR_NAMES): raise RuntimeError
types_colors = tuple(to_rgb(f'xkcd:{s}') for s in TYPES_COLORS)


def chunk_forward(module: tr.nn.Module, *x: Array, chunk=1000) -> Tuple[Array, ...]:
    n = x[0].shape[0]
    chunks_n = n // chunk
    if n > chunks_n * chunk:
        chunks_n += 1

    with tr.no_grad():
        args = [tr.Tensor(arr[0 * chunk:(0 + 1) * chunk]) for arr in x]
        outputs: List[List[Array]] = [[tensor.numpy()] for tensor in module(*args)]
        for i in range(1, chunks_n):
            args = [tr.Tensor(arr[i * chunk:(i + 1) * chunk]) for arr in x]
            outputs = [old + [new.numpy()] for old, new in zip(outputs, module(*args))]

    return tuple(np.concatenate(arr_list, axis=0) for arr_list in outputs)


def get_plot_vae_args_weighted_batch(df: DataFrame) -> Tuple[Array, Array, Array]:
    with tr.no_grad():
        x, x_ext, passth = next(iter(get_loader(df, 'plot', PLOT_BATCH_SIZE)))
        return x.numpy(), x_ext.numpy(), passth.numpy()


def get_plot_vae_args__y__w(df: DataFrame) -> Tuple[Tuple[Array, Array, Array], Array, Array]:
    _, passthr, x, e_ext, y, w, _ = get_data(df)
    return (x, e_ext, passthr), y, w


def get_x_normed_mu_sigma(x: Array, weights: Array = None) -> Tuple[Array, Array, Array]:
    weights_: Dict[str, Array]
    if weights is None:
        weights_ = dict()
    elif len(x) != len(weights): raise ValueError
    else:
        weights_ = dict(weights=weights)
    mu = np.average(x, axis=0, **weights_)
    sigma = np.sqrt(np.average((x - mu) ** 2, axis=0, **weights_))
    return (x - mu) / sigma, mu, sigma


def plot_z_correl(z: Array, weights: Opt[Array], file: str,
                  sub: Callable[[Array], Array] = lambda m: m, dpi=75) -> Array:
    import matplotlib.pyplot as plt
    z_normalized, _, _ = get_x_normed_mu_sigma(z, weights)
    cov = np.cov(z_normalized.T, aweights=weights)  # Σ
    eye = np.eye(len(cov)) if cov.shape else np.eye(1)  # I
    fig = plt.figure(figsize=(6, 6 / GR))
    corr_mat = sub(cov - eye)
    mask = sub(eye > (1 - 10**-8))
    sns.heatmap(corr_mat, center=0, square=True, mask=mask)
    max_correl = np.max(np.abs(corr_mat))
    plt.title(basename(file) + f'_MaxAbs={max_correl:.3f}')
    fig.tight_layout()
    plt.savefig(file + ".png", dpi=dpi)
    plt.close(fig)
    return max_correl


def plot_dist(z_weighted: Opt[Array], z_weighted_beta: Opt[Array],  # pylint: disable=too-many-branches
              z: Array, y: Array, file: str, dpi=50, axis_name='z') -> None:
    z_dims = z.shape[1]
    if z_dims == 1:
        return  # presumably matplotlib is buggy in this case

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(z_dims, 3, figsize=(4 * 3, 3 * z_dims))

    for i in range(z_dims):
        ax_title = f'${{{axis_name}}}_{{{i}}}$'
        w_ax_title = f'$w{{{axis_name}}}_{{{i}}}$'

        # 1st plot (all database):
        ax = axs[i, 0]
        if z_weighted is not None:
            zwi, zi = z_weighted[:, i], z[:, i]
            sns.kdeplot(zwi, ax=ax, bw_method="scott")
            sns.kdeplot(zi, ax=ax, bw_method="scott")
            zb = ''
            if z_weighted_beta is not None:
                sns.kdeplot(z_weighted_beta[:, i], ax=ax, bw_method="scott")
                zb = ', $wz(β)$'
            ax.set_title(f'{w_ax_title}, {ax_title}{zb}; μ,σ={np.mean(zwi.T):.2f},{np.std(zwi.T):.2f}')
        else:
            sns.kdeplot(z[:, i], ax=ax, bw_method="scott")
            zb = ''
            if z_weighted_beta is not None:
                sns.kdeplot(z_weighted_beta[:, i], ax=ax, bw_method="scott")
                zb = ', $wz(β)$'
            ax.set_title(f'{ax_title}{zb}')

        smart_coin = (y + 1)
        other_tal_diags = -y
        types_masks = [(t, (t == smart_coin) | (t == other_tal_diags)) for t in tuple(range(1, 17))]

        # 2nd plot (per-type density for Talanov's diagnosis):
        ax = axs[i, 1]
        for t, m in types_masks:
            sns.kdeplot(z[m, i], ax=ax, color=types_colors[t - 1], bw_method="scott")
        ax.set_title(ax_title + ' (TT diagnosis)')

        # 3rd plot (per-type density for Smart coincide types):
        ax = axs[i, 2]
        for t, _ in types_masks:
            sns.kdeplot(z[smart_coin == t, i], ax=ax, color=types_colors[t - 1], bw_method="scott")
        ax.set_title(ax_title + ' (smart coincide)')

    fig.tight_layout()
    plt.savefig(file + ".png", dpi=dpi)
    plt.close(fig)


def barplot(x_: Iterable, hue_: Iterable, y: Array,
            x_leg: str, hue_leg: str, y_leg: str, ax=None):

    x, hue = list(x_), list(hue_)
    if len(hue) != y.shape[0]: raise ValueError

    s = hue[0]
    tmp = [str(s)] * len(x)
    for s in hue[1:]:
        tmp += [str(s)] * len(x)

    df = pd.DataFrame(list(zip(
        x * len(hue), tmp, y.reshape(-1)
    )), columns=[x_leg, hue_leg, y_leg])
    sns.barplot(x=x_leg, hue=hue_leg, y=y_leg, data=df, ax=ax)


def plot_interesting(mu: Array, mu_rot: Array, subdec_mu: Array, probs: Array, df: DataFrame,
                     file_prefix: str, interesting_ids: Tuple[int, ], dpi=75):
    ids = df['id'].values
    sex = df['sex'].values
    smart_coin = df['smart_coincide'].values

    inter_idx_id = [(i, id_) for i, id_ in enumerate(ids) if id_ in interesting_ids]
    inter_idxs = [[i for i, id2 in inter_idx_id if (id1 == id2)][0] for id1 in interesting_ids]

    import matplotlib.pyplot as plt
    n_plots = 5
    fig, axs = plt.subplots(n_plots, 1, figsize=(8, 3 * n_plots))

    # interesting FZA4 axes:
    barplot(range(mu.shape[1]), interesting_ids, mu[inter_idxs],
            'μ axes (FZA4 axes)', 'Questionnaire', 'Axes val.', ax=axs[0])
    sns.move_legend(axs[0], "upper left", bbox_to_anchor=(1, 1))

    # interesting rotated FZA4 axes:
    barplot(range(mu_rot.shape[1]), interesting_ids, mu_rot[inter_idxs],
            'μ_rot axes (rot FZA4 axes)', 'Questionnaire', 'Axes val.', ax=axs[1])
    sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 1))

    # interesting Khizhnyak axes:
    barplot(range(subdec_mu.shape[1]), interesting_ids, subdec_mu[inter_idxs],
            's(μ) axes (Khizhnyak axes)', 'Questionnaire', 'Axes val.', ax=axs[2])
    sns.move_legend(axs[2], "upper left", bbox_to_anchor=(1, 1))

    # interesting type probabilities estimation:
    barplot(range(1, 17), interesting_ids, probs[inter_idxs],
            'JATS types', 'Questionnaire', 'Types prob. guess', ax=axs[3])
    sns.move_legend(axs[3], "upper left", bbox_to_anchor=(1, 1))

    # Number of types via ML classifier:
    predict_type = np.argmax(probs, axis=1) + 1
    males = sex > 3
    counts = [[np.sum(msk & ~males), np.sum(msk & males)]
              for t in range(1, 17) for msk in (predict_type == t,)]

    barplot(range(1, 17), ['female', 'male'], np.array(counts).T,
            'JATS types', 'Sex', 'Number of typed via ML', ax=axs[4])

    fig.tight_layout()
    plt.savefig(file_prefix + 'interest-and-classify-counts.png', dpi=dpi)
    plt.close(fig)

    # Print distances from certain types to CSV:
    for idx in (0, 1):
        delta_mu = la.norm(mu - mu[inter_idxs][idx], axis=-1)
        sortd_mu = np.argsort(delta_mu)[:20]

        df_ = pd.DataFrame({
            'id': ids[sortd_mu],
            'd_mu_euc x 100': np.round(100 * delta_mu[sortd_mu]).astype(int),
            'smart coin': smart_coin[sortd_mu],
            'predict type': predict_type[sortd_mu],
        })
        probs_ = np.round(100 * probs[sortd_mu]).astype(int)
        for i in range(1, 17):
            df_.insert(len(df_.columns), f"p{i}", probs_[:, i - 1])

        df_.to_csv(file_prefix + f'interest-distance-{idx}.csv', index=False)
