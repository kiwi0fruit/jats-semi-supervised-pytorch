from typing import Tuple, Union, Sequence
from torch import Tensor
import torch as tr
from torch.distributions.bernoulli import Bernoulli  # type: ignore
from torch.distributions.categorical import Categorical  # type: ignore
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from kiwi_bugfix_typechecker import torch as tr_

from .loss import BaseWeightedLoss

δ = 1e-8


class BernoulliLoss(BaseWeightedLoss, BCEWithLogitsLoss):  # type: ignore
    def __init__(self, features_size: Tuple[int, ...]=None, weight: Tensor=None, pos_weight: Tensor=None):
        BCEWithLogitsLoss.__init__(self=self, weight=weight, reduction='none', pos_weight=pos_weight)
        BaseWeightedLoss.__init__(self=self, features_size=features_size)

    def forward_(self, x_params: Tensor, target: Tensor, weight: Tensor=None) -> Tensor:
        bce = BCEWithLogitsLoss.forward(self=self, input=self.reshape(x_params), target=target)
        if weight is not None:
            bce = bce * weight
        return bce.view(bce.size(0), -1).sum(dim=1)

    def x_prob_params(self, x_params: Tensor) -> Tensor:
        return tr.sigmoid(self.reshape(x_params))

    def x_recon(self, x_params: Tensor) -> Tuple[Tensor, Tensor]:
        p = self.x_prob_params(self.reshape(x_params))
        x_rec_sample = Bernoulli(probs=p).sample()
        return p, x_rec_sample


class CategoricalLoss(BaseWeightedLoss, CrossEntropyLoss):  # type: ignore  # pylint: disable=too-many-ancestors
    classes: Union[int, Sequence[float]]

    def __init__(self, classes: Union[int, Sequence[float]], features_size: Tuple[int, ...]=None,
                 weight: Tensor=None, ignore_index: int=-100):
        if classes is None:
            raise ValueError(f'Bad classes value: {classes}')
        CrossEntropyLoss.__init__(self=self, weight=weight, ignore_index=ignore_index, reduction='none')
        BaseWeightedLoss.__init__(self=self, classes=classes, features_size=features_size)

    def forward_(self, x_params: Tensor, target: Tensor, weight: Tensor=None) -> Tensor:
        ce = CrossEntropyLoss.forward(self=self, input=self.reshape(x_params), target=target)
        if weight is not None:
            ce = ce * weight
        return ce.view(ce.size(0), -1).sum(dim=1)

    def x_prob_params(self, x_params: Tensor) -> Tensor:
        return tr.softmax(self.reshape(x_params), dim=1)

    def x_recon(self, x_params: Tensor) -> Tuple[Tensor, Tensor]:
        basis_: Tuple[float, ...] = tuple(range(self.classes)) if isinstance(self.classes, int) else tuple(self.classes)
        basis = tr.tensor(basis_).to(dtype=x_params.dtype, device=x_params.device)
        π = self.x_prob_params(x_params)

        inner_dims = tuple(range(2, len(π.size())))
        π = π.permute(0, *inner_dims, 1)
        letters = ('k', 'l', 'm', 'n', 'p', 'q', 'r', 's')
        if len(letters) < len(inner_dims):
            raise NotImplementedError
        ein = ''.join(l for _, l in zip(inner_dims, letters))

        x_rec_continuous = tr_.einsum(f"i{ein}j,j->i{ein}", π, basis)
        x_rec_sample = basis[Categorical(probs=π).sample()]
        return x_rec_continuous, x_rec_sample
