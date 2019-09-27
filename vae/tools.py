from typing import Tuple, List, Optional as Opt
from os import path
import torch as tr
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.distributions.categorical import Categorical  # type: ignore

from kiwi_bugfix_typechecker.nn import Module
from semi_supervised_typed import VariationalAutoencoder, DeepGenerativeModel


def sample_categ(probs: Tensor, determ: bool=False) -> Tensor:
    """
    Samples categ. distrib.

    :param probs: tensor of shape (batch_size, classes_n)
    :param determ: sample via max value instead of randomly
    :return: one-hot tensor of shape (batch_size, classes_n) of the same type as probs
    """
    idxs: Tensor
    if determ:
        _, idxs = tr.max(probs, dim=-1)
    else:
        idxs = Categorical(probs=probs).sample()
    return tr.eye(probs.size(-1))[idxs].to(dtype=probs.dtype, device=probs.device)


def load_model(checkpoint: str, model: Module, optimizer: Optimizer) -> Tuple[int, int, List[dict]]:
    """
    If fresh model then ``iteration == -1``.

    :return: (epoch, iteration, log_stats)
    """
    log_stats: List[dict]
    epoch, iteration, log_stats = 0, -1, []
    if path.isfile(checkpoint):
        _checkpoint = tr.load(checkpoint)
        model.load_state_dict(_checkpoint['model_state_dict'])
        optimizer.load_state_dict(_checkpoint['optimizer_state_dict'])
        model.eval()
        epoch = _checkpoint.get('epoch', epoch)
        iteration = _checkpoint.get('iteration', iteration)
        log_stats = _checkpoint.get('log_stats', log_stats)
        if not isinstance(log_stats, list) or not isinstance(epoch, int) or not isinstance(iteration, int):
            raise RuntimeError('Loaded log_stats should be a list, epoch and iteration should be int.')

    return epoch, iteration, log_stats


def get_zμ_ymax(x: Tensor, model: VariationalAutoencoder, random: bool=False) -> Tuple[
        Tensor, Tensor, Tuple[Tensor, ...], Tensor, Opt[Tensor]]:
    """ returns ``(z, y, z_params, kld, probs)``. If ``random=False`` attempts to return deterministically. """
    model.set_save_qz_params(True)
    probs: Opt[Tensor]
    if isinstance(model, DeepGenerativeModel):
        model_: DeepGenerativeModel = model
        probs = model_.classify(x)
        y = sample_categ(probs=probs, determ=not random)
    else:
        probs = None
        y = x
    _, kld = model.__call__(x=x, y=y)
    z, qz_params = model.take_away_qz_params()
    if (z is None) or not qz_params:
        raise RuntimeError("VariationalAutoencoder or it's subclasses bug")
    zμ = qz_params[0] if not random and model.q_params_μ_first else z

    model.set_save_qz_params(False)
    return zμ, y, qz_params, kld, probs
