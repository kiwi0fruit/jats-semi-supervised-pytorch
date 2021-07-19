from typing import Tuple, Optional as Opt, Union
import torch as tr
from torch import Tensor
from torch.distributions.categorical import Categorical  # type: ignore

from semi_supervised_typed import VariationalAutoencoder, DeepGenerativeModel, VAEClassifyMeta


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


def get_z_ymax(x: Tensor, y: Opt[Tensor], model: VariationalAutoencoder,
               random: bool=True, verbose: bool=False) -> Tuple[Tensor, Tensor, Tuple[Tensor, ...], Opt[Tensor]]:
    """ returns ``(z, y, z_params, probs)``. If ``random=False`` attempts to return deterministically.
        y is x if no classifier. """
    probs: Opt[Tensor] = None

    cls: Union[DeepGenerativeModel, VAEClassifyMeta, None] = None
    if isinstance(model, DeepGenerativeModel):
        cls = model
    elif isinstance(model, VAEClassifyMeta):
        cls = model

    if cls is not None:
        probs, _ = cls.classify(x) if random else cls.classify_deterministic(x)

    if y is not None:
        y_out = y
    elif probs is not None:
        y_out = sample_categ(probs=probs, determ=not random)
    else:
        y_out = x

    if isinstance(model, DeepGenerativeModel):
        dgm = model
        y_ = dgm.classifier.transform_y(y_out) if verbose else y_out
    else:
        y_ = x

    z, qz_params = model.encode(x, y_)
    if (z is None) or not qz_params:
        raise RuntimeError("VariationalAutoencoder or it's subclasses bug")

    return z, y_out, qz_params, probs
