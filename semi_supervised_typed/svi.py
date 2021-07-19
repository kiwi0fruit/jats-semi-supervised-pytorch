import torch as tr
from torch import Tensor
from torch.nn import Module, functional as func

from .utils import enumerate_discrete
from .inference import ImportanceWeightedSampler
from .dgm import DeepGenerativeModel
from .svi_types import Loss, BaseSVI, RetSVI

δ = 1e-8


class BCELoss(Loss, Module):  # type: ignore
    def __init__(self):
        Module.__init__(self)
        Loss.__init__(self)

    def forward_(self, x_params: Tensor, target: Tensor, weight: Tensor=None) -> Tensor:
        return func.binary_cross_entropy(
            input=x_params, target=target, weight=weight, reduction='none').view(x_params.size(0), -1).sum(dim=1)


class SVI(BaseSVI):
    """
    Stochastic variational inference (SVI).
    """
    model: DeepGenerativeModel
    nll: Loss
    sampler: ImportanceWeightedSampler
    base_sampler = ImportanceWeightedSampler(mc=1, iw=1)
    α0: float
    N_u: int
    N_l: int
    α: float
    β: float

    def __init__(self, model: DeepGenerativeModel, N_l: int, N_u: int, nll: Loss=BCELoss(), α0: float=0.1, β: float=1,
                 sampler: ImportanceWeightedSampler=base_sampler):
        """
        Initialises a new SVI optimizer for semi-supervised learning.

        α = α0 * (N_u + N_l) / N_l

        :param model: semi-supervised model to evaluate
        :param N_l: len(loader.labelled)
        :param N_u: len(loader.unlabelled)
        :param nll: p(x|y,z) for example BCE, MSE or CE.
            Order of the args: x_rec, x, weight. weight can be None.
        :param α0: used in coefficient α
        :param β: warm-up/scaling of KL-term
        :param sampler: sampler for x and y, e.g. for Monte Carlo.
            Default is ImportanceWeightedSampler(mc=1, iw=1)
        """
        super(SVI, self).__init__()
        self.model = model
        self.nll = nll
        self.sampler = sampler
        self.α0 = α0
        self.β = β
        self.N_l = N_l
        self.N_u = N_u
        self.α = self.α0 * (self.N_u + self.N_l) / self.N_l

    def set_consts(self, α0: float=None, β: float=None, N_l: int=None, N_u: int=None):
        """
        α = α0 * (N_u + N_l) / N_l

        :param α0: used in coefficient α
        :param β: warm-up/scaling of KL-term
        :param N_l: len(loader.labelled)
        :param N_u: len(loader.unlabelled)
        """
        if α0 is not None:
            self.α0 = α0
        if β is not None:
            self.β = β
        if N_l is not None:
            self.N_l = N_l
        if N_u is not None:
            self.N_u = N_u
        self.α = self.α0 * (self.N_u + self.N_l) / self.N_l

    def forward_(self, x: Tensor, y: Tensor=None, weight: Tensor=None, x_nll: Tensor=None,
                 x_cls: Tensor=None) -> RetSVI:
        """
        returns ``(nelbo, cross_entropy, probs_qy_x, verbose)``.
        When ``y is None`` ``cross_entropy`` is entropy.
        ``nelbo`` > 0 is of size (batch_size,). ``cross_entropy`` is classification loss.

        See https://arxiv.org/abs/1406.5298 3.1.2 Generative Semi-supervised Model Objective
        for details.
        """
        # is labelled when y is not None
        labels_n = self.model.y_dim

        # Prepare for sampling
        x_s = x
        w_s = weight if (weight is not None) else None
        x_nll_s = x_nll if (x_nll is not None) else None
        x_cls_ = x_cls if (x_cls is not None) else x

        # Enumerate choices of label
        if y is None:
            y_s = enumerate_discrete(x_s, labels_n)
            x_s = x_s.repeat(labels_n, 1)
            w_s = w_s.repeat(labels_n, 1) if (w_s is not None) else None
            x_nll_s = x_nll_s.repeat(labels_n, 1) if (x_nll_s is not None) else None
        else:
            y_s = self.model.classifier.transform_y(y)

        # Increase sampling dimension
        y_s = self.sampler.resample(y_s)
        x_s = self.sampler.resample(x_s)
        w_s = self.sampler.resample(w_s) if (w_s is not None) else None
        x_nll_s_ = self.sampler.resample(x_nll_s) if (x_nll_s is not None) else x_s

        x_recon_s, kld, verb = self.model.__call__(x_s, y_s)

        # -log p_θ(x|y,z) > 0
        nlog_px_yz = self.nll.__call__(x_recon_s, x_nll_s_, w_s)
        # -log p_θ(y) > 0
        nlog_py = self.model.classifier.neg_log_py(y_s)  # return > 0

        # nelbo > 0; kld = log q_φ(z|x,y) - log p_θ(z) > 0
        nelbo = nlog_px_yz + nlog_py + kld * self.β

        # Equivalent to L(x, y); L > 0; L of the size (batch_size,)
        L = -self.sampler.__call__(-nelbo)

        if y is not None:
            # Probabilities q_φ(y|x) > 0
            # Add auxiliary classification loss E_{p_data(x, y)}[-log q_φ(y|x)]:
            qy_x, cross_entropy = self.model.classify(x_cls_, y)

            if tr.isnan(L).any() or tr.isnan(cross_entropy).any():
                raise ValueError('NaN spotted in objective.')
            return L, cross_entropy, qy_x, verb

        # Probabilities q_φ(y|x) > 0
        # Calculate entropy H(q(y|x)) and sum over all labels; H > 0
        qy_x, H = self.model.classify(x_cls_)

        # qyx_L > 0
        qyx_L = (qy_x * L.view_as(qy_x.t()).t()).sum(dim=-1)

        # Equivalent to U(x); U > 0
        U = qyx_L - H

        if tr.isnan(U).any():
            raise ValueError('NaN spotted in objective.')
        return U, H, qy_x, verb

    def accuracy(self, logits_or_probs: Tensor, target: Tensor) -> Tensor:
        return self.model.classifier.accuracy(logits_or_probs, target)
