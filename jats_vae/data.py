from typing import List, Union
# noinspection PyPep8Naming
from numpy import ndarray as Array
from torch import Tensor
from vae.utils import ndarr, get_x_normalized_μ_σ
from vae.losses import BaseWeightedLoss

from semi_supervised_typed import VariationalAutoencoder
from socionics_db import JATSModelOutput, Transforms, Data
from .utils import get_basis_kld_vec, get_zμ_ymax


def model_output(x: Tensor, model: VariationalAutoencoder, rec_loss: BaseWeightedLoss,
                 dat: Data, trans: Transforms, log_stats: List[dict],
                 kld_drop: float=None, normalize_z: bool=False) -> JATSModelOutput:
    basis, kld_vec = get_basis_kld_vec(log_stats, kld_drop)
    if (basis is not None) and (kld_vec is not None):
        dropped_kld = [kld for i, kld in enumerate(kld_vec) if i not in basis] if (kld_vec is not None) else None
    else:
        dropped_kld = []

    z_μ, y_max, _, _, probs = get_zμ_ymax(x=x, model=model)
    x_preparams = model.sample(z=z_μ, y=y_max)
    z = ndarr(z_μ)

    x_rec_cont_, x_rec_sample_ = rec_loss.x_recon(x_preparams)
    x_rec_cont, x_rec_sample = ndarr(x_rec_cont_), ndarr(x_rec_sample_)
    x_rec_disc = trans.round_x_rec_arr(x_rec_cont)

    μ_learn: Union[Array, int]
    σ_learn: Union[Array, int]
    if normalize_z:
        _, μ_learn, σ_learn = get_x_normalized_μ_σ(z[dat.learn_indxs], dat.learn_weight_vec)
        z_normalized = (z - μ_learn) / σ_learn
    else:
        z_normalized, μ_learn, σ_learn = z, 0, 1

    basis = basis if basis else list(range(z_normalized.shape[1]))
    z_normalized = z_normalized[:, basis]
    y_probs = ndarr(probs) if (probs is not None) else None

    return JATSModelOutput(x=ndarr(x), z=z, z_normalized=z_normalized, x_rec_cont=x_rec_cont,
                           x_rec_disc=(x_rec_disc,), x_rec_sample=x_rec_sample, y_probs=y_probs,
                           basis=basis, dropped_kld=dropped_kld, μ_z=μ_learn, σ_z=σ_learn)
