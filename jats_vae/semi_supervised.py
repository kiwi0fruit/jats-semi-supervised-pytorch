from typing import Callable, Tuple, Union, List, Optional as Opt, NamedTuple, Type, Sequence as Seq
from itertools import cycle
import math
import torch as tr
from torch import Tensor, nn
from torch.nn import Parameter, functional as func
from semi_supervised_typed import Classifier, VAEPassthroughClassify, DecoderPassthrough, Perceptron, PostInit
from semi_supervised_typed.inference import neg_log_standard_bernoulli
from semi_supervised_typed.vae_types import ModuleXToX
from semi_supervised_typed.vae import Encoder
from semi_supervised_typed.layers.stochastic import XTupleYi
from semi_supervised_typed.layers import GaussianSample
from kiwi_bugfix_typechecker import test_assert
from vae.semi_supervised import (EncoderCustom, DecoderPassthroughTwin, VAEPassthroughClassifyTwin, EncoderTwin,
                                 EncoderTwinSplit0, GaussianSampleTwin, MetaTrimmer)
from beta_tcvae_typed import Normal, Distrib, ZeroSymmetricBimodalNormal, ZeroSymmetricBimodalNormalTwin
from .jats_cls_types import ModuleXYToX, ModuleXOptYToX
from .jats_cls import (
    JatsProbs as Jp, missing_types,
    inter, inv, cat, tpl, cond as cond_,  # swap,
    CF1, CF2, CF3, CF4, CF5, CF6, CF7, CF8, V15, V26, V37, V48,
    SIR, NIR, SER, NER, FIR, TIR, FER, TER, DIR, QIR, DER, QER,
    ENT, INT, INF, ENF, ETQ, ITQ, IFQ, EFQ, ENQ, INQ, ESQ, ISQ,
    NTR, SFR, STR, NFR, NTBG, SFBG, STBG, NFBG,
    E, R, STATIC, N, AD, NRRSR, NESI, T, AB, TRRFR, TEFI, Q, AG, QRRDR, QEDI, AD12,
    MAP_12KHAX_X_8RNKHAX, MAP_ABS12KHAX_X_7COMPLRNKHAX,
    MAP_8AKHAX_X_8RNKHAX, MAP_ABS8AKHAX_X_STATAX, MAP_12KHAX_X_8AKHAX,
    MAP_8KHAX20EXTRAPAIRS_X_6RNKHAX, MAP_ABS8KHAX20EXTRAPAIRS_X_STATAX,
    MAP_8KHAX4EXTRAPAIRS_X_6RNKHAX, MAP_ABS8KHAX4EXTRAPAIRS_X_5COMPLRNKHAX2QAG,
    MAP_8KHAX4EXTRAPAIRS2_X_6RNKHAX, MAP_ABS8KHAX4EXTRAPAIRS2_X_5COMPLRNKHAX2QAG,
    IFBG, EFBG, ETBG, ITBG, FRBG, TRAD, FRAD, TRBG, ETAD, ITAD, EFAD, IFAD,
    TYPE8, MAP_8TYPEAX_X_6RNKHAX, MAP_ABS8TYPEAX_X_5COMPLRNKHAX2QAG, MAP_RELU8TYPEAXTO16TYPEAX_X_2QAG7COMPLRNKHAX,
    PSI, MAP_16PSIAX_X_15RNAX, MAP_3_X_ROT8FZA4_X_16TYPES, TYPE16, MAP_16TYPES_X_8KHAX
)

_ = (CF1, CF2, CF3, CF4, CF5, CF6, CF7, CF8, V15, V26, V37, V48,
     E, STATIC, N, AD, NRRSR, NESI, T, AB, TRRFR, TEFI, Q, AG, QRRDR, QEDI, AD12, EncoderTwin, EncoderTwinSplit0)
del _


test_assert()
δ = 1e-8
sr = math.sqrt


class Inits(NamedTuple):
    ax2fun: Tuple[float, ...] = (
        1, 1, 1,
        sr(sr(2)), sr(sr(2)),
        sr(sr(2)), sr(sr(2)),
        sr(sr(2)), sr(sr(2)),
    )
    w: Tuple[float, ...] = (
        1, sr(.5), -sr(.2), -sr(.5),
        -sr(.5), -sr(.2), sr(.2), sr(.5),
    )
    v: Tuple[float, float] = (
        sr(.5), sr(.5),
    )
    # (non valuable pronal => valuable pronal, non valuable antinal => valuable antinal)
    softmax_scl: float = 1
    softmax_groups_scl: float = 1


class ClassifierJATSCustom(Classifier):
    @staticmethod
    def cross_entropy(probs: Tensor, target: Tensor=None) -> Tensor:
        cross_entropy = Classifier.cross_entropy

        y_dom = y_4th = None
        if target is not None:
            y_dom, y_4th = Jp.p_dom(target), Jp.p_4th(target)
        return (
            cross_entropy(probs=probs, target=target)
            + cross_entropy(probs=Jp.p_dom(probs), target=y_dom)
            + cross_entropy(probs=Jp.p_4th(probs), target=y_4th)
        ) * 0.3333


class ClassifierJATSAxesAlignENTRr(Classifier):
    axs_ce_dim: int = 4
    sigmoid_scl: float = 2
    q_types: Tuple[Tuple[int, ...], ...] = (
        (1, 13), (7, 11), (5, 9), (3, 15), (12, 16), (2, 6), (4, 8), (10, 14),
        (1, 12), (2, 11), (5, 16), (6, 15),
        (6, 10), (4, 16), (2, 14), (8, 12), (3, 7), (9, 13), (11, 15), (1, 5),
        (3, 10), (4, 9), (7, 14), (8, 13),
    )
    q_subsets: Tuple[Tuple[int, ...], ...]

    axes_cross_entropy_mult: float = 1
    types_cross_entropy_mult: float = 1
    zero: Tensor  # buffer
    transform: Tensor  # buffer
    axs_ce_dims_sel: Tuple[int, ...]  # without sex
    passthr_dim: int = 1

    def __init__(self, dims: Tuple[int, Tuple[int, ...], int], activation_fn: Callable[[Tensor], Tensor]=tr.selu):
        self.axs_ce_dims_sel = tuple(range(self.passthr_dim, self.passthr_dim + self.axs_ce_dim))
        with self.disable_final_init():
            super(ClassifierJATSAxesAlignENTRr, self).__init__(dims, activation_fn=activation_fn)

        self.register_buffer('transform', self.get_transform())
        self.register_buffer('zero', tr.tensor([0.]).mean())

        self.q_subsets = self.get_q_subsets()
        assert len(self.q_subsets) == 16

        self.__final_init__()

    def set_ρ(self, ρ: int) -> None:
        pass

    @staticmethod
    def get_transform() -> Tensor:
        """
        0. I=>E
        1. S=>N
        2. F=>T
        3. R=>Rr
        """
        return tr.tensor([
            # Jung cognitive functions (1 vs 4):
            [+1,  1,  0,  1],  # 6+10=>1+13; ISxR=>ENxRr
            [-1,  1,  0,  1],  # 4+16=>7+11; ESxR=>INxRr
            [+1, -1,  0,  1],  # 2+14=>5+9;  INxR=>ESxRr
            [-1, -1,  0,  1],  # 8+12=>3+15; ENxR=>ISxRr
            [+1,  0,  1, -1],  # 3+7=>12+16; IxFRr=>ExTR
            [-1,  0,  1, -1],  # 9+13=>2+6;  ExFRr=>IxTR
            [+1,  0, -1, -1],  # 11+15=>4+8; IxTRr=>ExFR
            [-1,  0, -1, -1],  # 1+5=>10+14; ExTRr=>IxFR
            # Quasi-types functions (1+8 vs 4+5):
            [+1,  1,  1,  0],  # 3+10=>1+12; ISFx=>ENTx
            [-1,  1,  1,  0],  # 4+9=>2+11;  ESFx=>INTx
            [+1, -1,  1,  0],  # 7+14=>5+16; INFx=>ESTx
            [-1, -1,  1,  0],  # 8+13=>6+15; ENFx=>ISTx
        ], dtype=tr.float).transpose(0, 1) / math.sqrt(3)

    def get_q_types(self) -> Tuple[Tuple[int, ...], ...]:
        return self.q_types

    def get_q_subsets(self, verbose: bool=False) -> Tuple[Tuple[int, ...], ...]:
        """
        Classes in ``self.p_types`` should be from 1..16.

        :returns p_subsets spec for each of 16 types.
        """
        q_subsets_: List[List[int]] = [[] for _ in range(1, 17)]
        for axis_i, axis_types in enumerate(self.get_q_types()):
            for type_ in axis_types:
                q_subsets_[type_ - 1].append(axis_i)
        q_subsets = tuple(tuple(sorted(subset)) for subset in q_subsets_)
        if verbose:
            print('types specs: ', [[i + 1, spec] for i, spec in enumerate(q_subsets)])
        return q_subsets

    def get_q(self, z_trans2: Tensor) -> Tensor:
        """
        Returns array of quasi-probabilities vectors (batch_size, len(q_types)).

        Elements of the quasi-probability vectors correspond with ``self.q_types`` (same length).
        """
        q = tr.sigmoid(z_trans2 * self.sigmoid_scl)
        return tr.cat([q, -q + 1], dim=1)

    def transform2(self, z_trans1: Tensor) -> Tensor:
        return z_trans1 @ self.transform

    def axes_cross_entropy(self, z_trans1: Tensor, z_trans2: Tensor, y: Tensor) -> Tensor:
        if self.axes_cross_entropy_mult <= 0:
            return self.zero
        q = self.get_q(z_trans2)
        y_int = y.max(dim=1)[1]  # y_one_hot => y_int
        axes_neg_cross_entropy_ = [
            tr.log(q[mask][:, subset] + δ).sum()
            for subset, mask in zip(self.q_subsets, (y_int == i for i in range(16)))
            if z_trans1[mask].shape[0] > 0
        ]  # list of tensors of shape (batch_subset_size,)
        return sum(axes_neg_cross_entropy_) * (-self.axes_cross_entropy_mult / z_trans1.shape[0])

    def forward_(self, x: Tensor, y: Tensor=None) -> Tuple[Tensor, Tensor]:
        probs = tr.softmax(self.get_logits(x), dim=-1)
        cross_entropy = (
            self.cross_entropy(probs, y).mean() * self.types_cross_entropy_mult
            if (self.types_cross_entropy_mult > 0)
            else self.zero
        )
        if y is None:
            return probs, cross_entropy

        z_trans1 = x[:, self.axs_ce_dims_sel]
        return probs, self.axes_cross_entropy(z_trans1, self.transform2(z_trans1), y) + cross_entropy

    def set_thr(self, idx: int) -> None:
        raise RuntimeError

    def get_z_new(self, z_ax: Tensor) -> Tensor:
        """ z_ax is not extended. """
        return z_ax

    # noinspection PyMethodMayBeStatic
    def repr_learnables(self) -> str:
        return ''


class ClassifierJATSAxesAlign5RnKhAx(ClassifierJATSAxesAlignENTRr):
    axs_ce_dim: int = 5
    thrs: Tuple[Tuple[float, ...], ...] = (
        tuple(0.5 for _ in range(axs_ce_dim)),
        tuple(1. for _ in range(axs_ce_dim)),
        tuple(1. for _ in range(axs_ce_dim)),
    )
    # noinspection PyTypeChecker
    q_types: Tuple[Tuple[int, ...], ...] = (
        inv(R)[1], E[1], N[1], T[1], AD[1],  # MAP POS
        inv(R)[0], E[0], N[0], T[0], AD[0],  # MAP NEG
    )
    thr: Tuple[float, ...]

    def __post_init__(self):
        super(ClassifierJATSAxesAlign5RnKhAx, self).__post_init__()
        self.thr = self.thrs[0]

    def set_thr(self, idx: int) -> None:
        self.thr = self.thrs[min(idx, len(self.thrs) - 1)]

    @staticmethod
    def get_transform() -> Tensor:
        return tr.tensor([1.]).mean()

    def transform2(self, z_trans1: Tensor) -> Tensor:
        return z_trans1

    def get_q(self, z_trans2: Tensor) -> Tensor:
        thr = tr.tensor(self.thr, device=z_trans2.device, dtype=z_trans2.dtype)
        q_pos = tr.sigmoid((z_trans2 + thr) * self.sigmoid_scl)
        q_neg = tr.sigmoid((-z_trans2 + thr) * self.sigmoid_scl)
        return tr.cat([q_pos, q_neg], dim=1)


class ClassifierJATSAxesAlignProjectionNTRrEAd(ClassifierJATSAxesAlign5RnKhAx):
    axs_dim: int = 8
    axs_ce_dim: int = 5
    thrs: Tuple[Tuple[float, ...], ...] = (
        tuple(0. for _ in range(8)) + tuple(2. for _ in range(8)),
    )

    const_khax_dim_x_socax_dim: Tensor
    k0: Tensor
    fixed_dim: int = 4
    k_fixed_dim: Tensor
    k_socax_dim: Parameter
    _k0: float = 0.1
    _k_fixed_dim: float = math.sqrt(0.9)

    passthr_dim: int = 1
    q_types: Tuple[Tuple[int, ...], ...] = ()

    def get_q_types(self) -> Tuple[Tuple[int, ...], ...]:
        o = missing_types  # o - other
        return (
            inv(SIR)[1], NIR[1], inv(SER)[1], NER[1], inv(FIR)[1], TIR[1], inv(FER)[1], TER[1],
            o(inv(SIR)), o(NIR), o(inv(SER)), o(NER), o(inv(FIR)), o(TIR), o(inv(FER)), o(TER),  # MAP POS
            inv(SIR)[0], NIR[0], inv(SER)[0], NER[0], inv(FIR)[0], TIR[0], inv(FER)[0], TER[0],
            o(inv(SIR)), o(NIR), o(inv(SER)), o(NER), o(inv(FIR)), o(TIR), o(inv(FER)), o(TER),  # MAP NEG
        )

    def __post_init__(self):
        super(ClassifierJATSAxesAlignProjectionNTRrEAd, self).__post_init__()
        self.register_buffer('const_khax_dim_x_socax_dim', self.get_const())
        self.k_socax_dim = Parameter(tr.tensor([self._k_fixed_dim
                                                for _ in range(self.axs_ce_dim - self.fixed_dim)],
                                               dtype=tr.float), requires_grad=True)
        self.register_buffer('k_fixed_dim', tr.tensor(
            [self._k_fixed_dim for _ in range(self.fixed_dim)], dtype=tr.float))
        self.register_buffer('k0', tr.tensor([self._k0], dtype=tr.float))

    @staticmethod
    def get_const() -> Tensor:
        #     Rr  E  N  T   Ad
        return tr.tensor([
            [+1,  1, 1, 0,  1],  # 6+10=>1+13; ISxR=>ENxRr
            [-1, -1, 1, 0,  1],  # 5+9=>2+14;  ESxRr=>INxR
            [+1, -1, 1, 0, -1],  # 4+16=>7+11; ESxR=>INxRr
            [-1,  1, 1, 0, -1],  # 3+15=>8+12; ISxRr=>ENxR
            [+1,  1, 0, 1,  0],  # 10+14=>1+5; IxFR=>ExTRr
            [-1, -1, 0, 1,  0],  # 9+13=>2+6;  ExFRr=>IxTR
            [+1, -1, 0, 1,  0],  # 4+8=>11+15; ExFR=>IxTRr
            [-1,  1, 0, 1,  0],  # 3+7=>12+16; IxFRr=>ExTR
        ], dtype=tr.float)

    # noinspection PyMethodMayBeStatic
    def repr_learnables(self) -> str:
        return f'\ntrans_8khax_dim_x_{self.axs_ce_dim}ax_dim:\n{self.trans_khax_dim_x_socax_dim()}\n'

    def transform2(self, z_trans1: Tensor) -> Tensor:
        mat = self.trans_khax_dim_x_socax_dim()
        ret = z_trans1 @ (mat.transpose(0, 1) / mat.norm(dim=1))
        return tr.cat([ret, ret], dim=1)

    def trans_khax_dim_x_socax_dim(self) -> Tensor:
        k_ax_dim = tr.cat([self.k_fixed_dim, self.k_socax_dim], dim=0)
        return self.const_khax_dim_x_socax_dim * (k_ax_dim**2 + self.k0)

    def get_z_new(self, z_ax: Tensor) -> Tensor:
        ret = self.transform2(z_ax[:, :self.axs_ce_dim])
        return ret[:, :ret.shape[1] // 2]

    def get_regul_loss_reduced(self, zμ: Tensor) -> Tensor:  # pylint: disable=unused-argument
        return self.zero


class ClassifierJATSAxesAlignProjectionNTRrEAdAb(ClassifierJATSAxesAlignProjectionNTRrEAd):
    axs_ce_dim: int = 6

    @staticmethod
    def get_const() -> Tensor:
        #     Rr  E  N  T   Ad  Ab
        return tr.tensor([
            [+1,  1, 1, 0,  1,  0],  # 6+10=>1+13; ISxR=>ENxRr
            [-1, -1, 1, 0,  1,  0],  # 5+9=>2+14;  ESxRr=>INxR
            [+1, -1, 1, 0, -1,  0],  # 4+16=>7+11; ESxR=>INxRr
            [-1,  1, 1, 0, -1,  0],  # 3+15=>8+12; ISxRr=>ENxR
            [+1,  1, 0, 1,  0,  1],  # 10+14=>1+5; IxFR=>ExTRr
            [-1, -1, 0, 1,  0,  1],  # 9+13=>2+6;  ExFRr=>IxTR
            [+1, -1, 0, 1,  0, -1],  # 4+8=>11+15; ExFR=>IxTRr
            [-1,  1, 0, 1,  0, -1],  # 3+7=>12+16; IxFRr=>ExTR
        ], dtype=tr.float)


class ClassifierJATSAxesAlignProjectionQNTPEAgAdAb(ClassifierJATSAxesAlignProjectionNTRrEAd):
    fixed_dim: int = 5
    axs_dim: int = 12
    axs_ce_dim: int = 8
    thrs: Tuple[Tuple[float, ...], ...] = (
        tuple(0. for _ in range(12)),
    )
    # noinspection PyTypeChecker
    q_types: Tuple[Tuple[int, ...], ...] = (  # MAP POS:
        inv(SIR)[1], NIR[1], inv(SER)[1], NER[1], inv(FIR)[1], TIR[1], inv(FER)[1], TER[1],
        inv(DIR)[1], QIR[1], inv(DER)[1], QER[1],
        inv(SIR)[0], NIR[0], inv(SER)[0], NER[0], inv(FIR)[0], TIR[0], inv(FER)[0], TER[0],
        inv(DIR)[0], QIR[0], inv(DER)[0], QER[0],
        # ^ MAP NEG
    )

    @staticmethod
    def get_const() -> Tensor:
        #     Rr  E  N  T  Q   Ad  Ab  Ag
        return tr.tensor([
            [+1,  1, 1, 0, 0,  1,  0,  0],  # 6+10=>1+13; ISxR=>ENxRr
            [-1, -1, 1, 0, 0,  1,  0,  0],  # 5+9=>2+14;  ESxRr=>INxR
            [+1, -1, 1, 0, 0, -1,  0,  0],  # 4+16=>7+11; ESxR=>INxRr
            [-1,  1, 1, 0, 0, -1,  0,  0],  # 3+15=>8+12; ISxRr=>ENxR
            [+1,  1, 0, 1, 0,  0,  1,  0],  # 10+14=>1+5; IxFR=>ExTRr
            [-1, -1, 0, 1, 0,  0,  1,  0],  # 9+13=>2+6;  ExFRr=>IxTR
            [+1, -1, 0, 1, 0,  0, -1,  0],  # 4+8=>11+15; ExFR=>IxTRr
            [-1,  1, 0, 1, 0,  0, -1,  0],  # 3+7=>12+16; IxFRr=>ExTR
            [+1,  1, 0, 0, 1,  0,  0,  1],  # 6+14=>1+9;  IxxRD=>ExxRrQ
            [-1, -1, 0, 0, 1,  0,  0,  1],  # 5+13=>2+10; ExxRrD=>IxxRQ
            [+1, -1, 0, 0, 1,  0,  0, -1],  # 4+12=>7+15; ExxRD=>IxxRrQ
            [-1,  1, 0, 0, 1,  0,  0, -1],  # 3+11=>8+16; IxxRrD=>ExxRQ
        ], dtype=tr.float)


class ClassifierJATSAxesAlignProjectionNTRrAdeAdiAbeAbi(ClassifierJATSAxesAlignProjectionNTRrEAd):
    fixed_dim: int = 3
    axs_ce_dim: int = 7

    @staticmethod
    def get_const() -> Tensor:
        #     Rr N  T   EAd IAd EAb IAb
        return tr.tensor([
            [+1, 1, 0,  1,  0,  0,  0],  # 6+10=>1+13; ISxR=>ENxRr
            [-1, 1, 0,  0,  1,  0,  0],  # 5+9=>2+14;  ESxRr=>INxR
            [+1, 1, 0, -1,  0,  0,  0],  # 4+16=>7+11; ESxR=>INxRr
            [-1, 1, 0,  0, -1,  0,  0],  # 3+15=>8+12; ISxRr=>ENxR
            [+1, 0, 1,  0,  0,  1,  0],  # 10+14=>1+5; IxFR=>ExTRr
            [-1, 0, 1,  0,  0,  0,  1],  # 9+13=>2+6;  ExFRr=>IxTR
            [+1, 0, 1,  0,  0, -1,  0],  # 4+8=>11+15; ExFR=>IxTRr
            [-1, 0, 1,  0,  0,  0, -1],  # 3+7=>12+16; IxFRr=>ExTR
        ], dtype=tr.float)


class ClassifierJATSAxesAlignProjectionNTRrEAdeAdiAbeAbi(ClassifierJATSAxesAlignProjectionNTRrEAd):
    axs_ce_dim: int = 8

    @staticmethod
    def get_const() -> Tensor:
        #     Rr  E  N  T   EAd IAd EAb IAb
        return tr.tensor([
            [+1,  1, 1, 0,  1,  0,  0,  0],  # 6+10=>1+13; ISxR=>ENxRr
            [-1, -1, 1, 0,  0,  1,  0,  0],  # 5+9=>2+14;  ESxRr=>INxR
            [+1, -1, 1, 0, -1,  0,  0,  0],  # 4+16=>7+11; ESxR=>INxRr
            [-1,  1, 1, 0,  0, -1,  0,  0],  # 3+15=>8+12; ISxRr=>ENxR
            [+1,  1, 0, 1,  0,  0,  1,  0],  # 10+14=>1+5; IxFR=>ExTRr
            [-1, -1, 0, 1,  0,  0,  0,  1],  # 9+13=>2+6;  ExFRr=>IxTR
            [+1, -1, 0, 1,  0,  0, -1,  0],  # 4+8=>11+15; ExFR=>IxTRr
            [-1,  1, 0, 1,  0,  0,  0, -1],  # 3+7=>12+16; IxFRr=>ExTR
        ], dtype=tr.float)


class ClassifierJATSAxesAlignProjectionNTRrEAdeAdi(ClassifierJATSAxesAlignProjectionNTRrEAd):
    axs_ce_dim: int = 6

    @staticmethod
    def get_const() -> Tensor:
        #     Rr  E  N  T   EAd IAd
        return tr.tensor([
            [+1,  1, 1, 0,  1,  0],  # 6+10=>1+13; ISxR=>ENxRr
            [-1, -1, 1, 0,  0,  1],  # 5+9=>2+14;  ESxRr=>INxR
            [+1, -1, 1, 0, -1,  0],  # 4+16=>7+11; ESxR=>INxRr
            [-1,  1, 1, 0,  0, -1],  # 3+15=>8+12; ISxRr=>ENxR
            [+1,  1, 0, 1,  0,  0],  # 10+14=>1+5; IxFR=>ExTRr
            [-1, -1, 0, 1,  0,  0],  # 9+13=>2+6;  ExFRr=>IxTR
            [+1, -1, 0, 1,  0,  0],  # 4+8=>11+15; ExFR=>IxTRr
            [-1,  1, 0, 1,  0,  0],  # 3+7=>12+16; IxFRr=>ExTR
        ], dtype=tr.float)


class ClassifierJATSAxesAlignCustom1(ClassifierJATSAxesAlign5RnKhAx):
    axs_ce_dim: int = 6
    thrs: Tuple[Tuple[float, ...], ...] = (
        (1.0, 1.0, 0.5, 0.5, 1.0, 0.5),
        # (1.5, 1.7, 0.5, 0.5, 1.7, 0.5),
        (1.5, 1.5, 1.0, 1.0, 1.5, 1.0),
    )
    # noinspection PyTypeChecker
    q_types: Tuple[Tuple[int, ...], ...] = (
        inv(R)[1], E[1], N[1], T[1], AD[1], inter(inv(E), AD)[1],  # MAP POS
        inv(R)[0], E[0], N[0], T[0], AD[0], inter(inv(E), AD)[0],  # MAP NEG
    )


class ClassifierJATSAxesAlignCustom2(ClassifierJATSAxesAlign5RnKhAx):
    axs_ce_dim: int = 6
    thrs: Tuple[Tuple[float, ...], ...] = (
        (1.0, 1.0, 0.5, 0.5, 1.0, 1.0),
        (1.5, 1.5, 1.0, 1.0, 1.5, 1.5),
    )
    # noinspection PyTypeChecker
    q_types: Tuple[Tuple[int, ...], ...] = (
        inv(R)[1], E[1], N[1], T[1], E[1], AD[1],  # MAP POS
        inv(R)[0], E[0], N[0], T[0], E[0], AD[0],  # MAP NEG
    )


class ClassifierJATSAxesAlignCustom3(ClassifierJATSAxesAlign5RnKhAx):
    axs_ce_dim: int = 6
    thrs: Tuple[Tuple[float, ...], ...] = (
        (0.5, 1.0, 0.5, 0.5, 0.5, 0.5),
        (1.0, 1.5, 1.0, 1.0, 1.0, 1.5),
    )
    # noinspection PyTypeChecker
    q_types: Tuple[Tuple[int, ...], ...] = (
        inv(R)[1], E[1], N[1], T[1], AD[1], inter(inv(E), AD)[1],  # MAP POS
        inv(R)[0], E[0], N[0], T[0], AD[0], inter(inv(E), AD)[0],  # MAP NEG
    )


class ClassifierJATSAxesAlignCustom4(ClassifierJATSAxesAlign5RnKhAx):
    axs_ce_dim: int = 6
    thrs: Tuple[Tuple[float, ...], ...] = (
        (1.0, 1.0, 0.5, 0.5, 1.0, 0.5),
        (1.0, 1.5, 1.0, 1.0, 1.5, 1.0),
    )
    # noinspection PyTypeChecker
    q_types: Tuple[Tuple[int, ...], ...] = (
        inv(R)[1], E[1], N[1], T[1], AD[1], inter(inv(E), AD)[1],  # MAP POS
        inv(R)[0], E[0], N[0], T[0], AD[0], inter(inv(E), AD)[0],  # MAP NEG
    )


class ClassifierJATSAxesAlignCustom5(ClassifierJATSAxesAlign5RnKhAx):
    axs_ce_dim: int = 5
    thrs: Tuple[Tuple[float, ...], ...] = (
        tuple(0.1 for _ in range(5)),
        tuple(3.0 for _ in range(5)),  # was 2.0
    )
    # noinspection PyTypeChecker
    q_types: Tuple[Tuple[int, ...], ...] = (
        inv(R)[1], E[1], N[1], T[1], AD[1],  # MAP POS
        inv(R)[0], E[0], N[0], T[0], AD[0],  # MAP NEG
    )


class ClassifierJATSAxesAlignCustom6(ClassifierJATSAxesAlign5RnKhAx):
    axs_ce_dim: int = 7
    thrs: Tuple[Tuple[float, ...], ...] = (
        (1.0, 1.0, 1.0, 1.8, 1.8, 1.8, 1.8),
        (1.5, 1.5, 1.5, 1.8, 1.8, 1.8, 1.8),
    )
    # noinspection PyTypeChecker
    q_types: Tuple[Tuple[int, ...], ...] = (
        inv(R)[1], N[1], T[1], inter(E, AD)[1], inter(inv(E), AD)[1], inter(E, AB)[1], inter(inv(E), AB)[1],  # MAP POS
        inv(R)[0], N[0], T[0], inter(E, AD)[0], inter(inv(E), AD)[0], inter(E, AB)[0], inter(inv(E), AB)[0],  # MAP NEG
    )


class ClassifierJATSAxesAlign8RnKhAx(ClassifierJATSAxesAlign5RnKhAx):
    axs_ce_dim: int = 8
    thrs: Tuple[Tuple[float, ...], ...] = (
        tuple(1. for _ in range(axs_ce_dim)),
        tuple(1. for _ in range(axs_ce_dim)),
        tuple(1.5 for _ in range(axs_ce_dim)),
    )
    # noinspection PyTypeChecker
    q_types: Tuple[Tuple[int, ...], ...] = (
        inv(R)[1], E[1], N[1], T[1], Q[1], AD[1], AB[1], AG[1],  # MAP POS
        inv(R)[0], E[0], N[0], T[0], Q[0], AD[0], AB[0], AG[0],  # MAP NEG
    )


class ClassifierJATSAxesAlign6RnKhAx(ClassifierJATSAxesAlign8RnKhAx):
    axs_ce_dim: int = 6
    thrs: Tuple[Tuple[float, ...], ...] = (
        tuple(1. for _ in range(axs_ce_dim)),
        tuple(1. for _ in range(axs_ce_dim)),
        tuple(1.5 for _ in range(axs_ce_dim)),
    )
    # noinspection PyTypeChecker
    q_types: Tuple[Tuple[int, ...], ...] = (
        inv(R)[1], E[1], N[1], T[1], AD[1], AB[1],  # MAP POS
        inv(R)[0], E[0], N[0], T[0], AD[0], AB[0],  # MAP NEG
    )


class ClassifierJATSAxesAlign7RnKhAx(ClassifierJATSAxesAlign8RnKhAx):
    axs_ce_dim: int = 7
    thrs: Tuple[Tuple[float, ...], ...] = (
        tuple(1. for _ in range(axs_ce_dim)),
        tuple(1. for _ in range(axs_ce_dim)),
        tuple(1.5 for _ in range(axs_ce_dim)),
    )
    # noinspection PyTypeChecker
    q_types: Tuple[Tuple[int, ...], ...] = (
        inv(R)[1], E[1], N[1], T[1], Q[1], AD[1], AG[1],  # MAP POS
        inv(R)[0], E[0], N[0], T[0], Q[0], AD[0], AG[0],  # MAP NEG
    )


class Map12KhAxTo15RnAx(ModuleXToX, PostInit):
    map = (MAP_ABS12KHAX_X_7COMPLRNKHAX, MAP_12KHAX_X_8RNKHAX)
    _khax_dim: int = 12
    abskhax2complrnkh_12kh_x_7crnkh: Tensor
    khax2rnkhax_12kh_x_8rnkh: Tensor
    k_kh2crnkh_7pos: Parameter
    k_kh2rnkh_8pos: Parameter
    zero: Tensor

    def __init__(self):
        super(Map12KhAxTo15RnAx, self).__init__()
        n1, n2, m, k = self.get_n1_n2_m_k()
        self.register_buffer('khax2rnkhax_12kh_x_8rnkh',
                             tr.tensor(self.map[1], dtype=tr.float)[:n1, :2 + k])
        self.register_buffer('abskhax2complrnkh_12kh_x_7crnkh',
                             tr.tensor(self.map[0], dtype=tr.float)[:n2, :1 + m])
        self.k_kh2crnkh_7pos = Parameter(tr.tensor([1. for _ in range(1 + m)], dtype=tr.float), requires_grad=True)
        self.k_kh2rnkh_8pos = Parameter(tr.tensor([1. for _ in range(2 + k)], dtype=tr.float), requires_grad=True)
        self.register_buffer('zero', tr.tensor([0.]).mean())
        self.__final_init__()

    def get_n1_n2_m_k(self) -> Tuple[int, int, int, int]:
        n, m = self._khax_dim, (self._khax_dim // 4) * 2
        return n, n, m, m

    def khax2rnkh(self) -> Tensor:
        """ return shape is (12, 8) """
        return self.khax2rnkhax_12kh_x_8rnkh  # * self.k_kh2rnkh_8pos**2

    def abskhax2complrnkh(self) -> Tensor:
        """ return shape is (12, 7) """
        return self.abskhax2complrnkh_12kh_x_7crnkh  # * self.k_kh2crnkh_7pos**2

    def abs(self, x: Tensor) -> Tensor:
        """ huber loss pseudo δ = 0.1 """
        return func.smooth_l1_loss(x * 4, self.zero, reduction='none') * 0.25

    def z_rnax(self, z_khax: Tensor) -> Tensor:
        z_rnkhax = z_khax @ self.khax2rnkh()
        z_complrnkh = self.abs(z_khax) @ self.abskhax2complrnkh()
        return tr.cat([z_rnkhax, z_complrnkh], dim=1)

    def forward_(self, x: Tensor) -> Tensor:
        return self.z_rnax(z_khax=x)


class Map12KhAxToO(Map12KhAxTo15RnAx):
    def forward_(self, x: Tensor) -> Tensor:
        z_khax = x
        return self.abs(z_khax) @ self.abskhax2complrnkh()[:, (0,)]


class Map8KhAxTo11RnAx(Map12KhAxTo15RnAx):
    _khax_dim: int = 8


class Map8AKhAxTo9RnAx(Map12KhAxTo15RnAx):
    _khax_dim: int = 12
    map = (MAP_ABS8AKHAX_X_STATAX, MAP_8AKHAX_X_8RNKHAX)

    def get_n1_n2_m_k(self) -> Tuple[int, int, int, int]:
        m = (self._khax_dim // 4) * 2
        return 2 + m, 2 + m, 0, m


class Map8KhAx20ExtraPairsTo7RnAx(Map12KhAxTo15RnAx):
    map = (MAP_ABS8KHAX20EXTRAPAIRS_X_STATAX, MAP_8KHAX20EXTRAPAIRS_X_6RNKHAX)
    _khax_dim: int = 8

    def get_n1_n2_m_k(self) -> Tuple[int, int, int, int]:
        return 28, 28, 0, 4


class Map16PsiAxTo15RnAx(Map12KhAxTo15RnAx):
    map = (MAP_ABS12KHAX_X_7COMPLRNKHAX, MAP_16PSIAX_X_15RNAX)
    _khax_dim: int = 16

    def get_n1_n2_m_k(self) -> Tuple[int, int, int, int]:
        return 16, 16, 0, 15 - 2

    def z_rnax(self, z_khax: Tensor) -> Tensor:
        return z_khax @ self.khax2rnkh()


class Map8TypeAxTo13RnAx(Map12KhAxTo15RnAx):
    map = (MAP_ABS8TYPEAX_X_5COMPLRNKHAX2QAG, MAP_8TYPEAX_X_6RNKHAX)
    _khax_dim: int = 8

    def get_n1_n2_m_k(self) -> Tuple[int, int, int, int]:
        return 8, 8, 7 - 1, 6 - 2


class Map8TypeAxTo15RnAx(Map12KhAxTo15RnAx):
    map = (MAP_RELU8TYPEAXTO16TYPEAX_X_2QAG7COMPLRNKHAX, MAP_8TYPEAX_X_6RNKHAX)
    _khax_dim: int = 8

    def abs(self, x: Tensor) -> Tensor:
        """ Softplus with beta = 10 and double dim. """
        return tr.cat([func.softplus(x, beta=10),
                       func.softplus(-x, beta=10)[:, (1, 0, 3, 2, 5, 4, 7, 6)]], dim=1)

    def get_n1_n2_m_k(self) -> Tuple[int, int, int, int]:
        return 8, 16, 9 - 1, 6 - 2


class Map8KhAx4ExtraPairsTo13RnAx(Map8KhAx20ExtraPairsTo7RnAx):
    map = (MAP_ABS8KHAX4EXTRAPAIRS_X_5COMPLRNKHAX2QAG, MAP_8KHAX4EXTRAPAIRS_X_6RNKHAX)

    def get_n1_n2_m_k(self) -> Tuple[int, int, int, int]:
        return 12, 12, 0, 4


class Map8KhAx4ExtraPairs2To12RnAx(Map8KhAx20ExtraPairsTo7RnAx):
    map = (MAP_ABS8KHAX4EXTRAPAIRS2_X_5COMPLRNKHAX2QAG, MAP_8KHAX4EXTRAPAIRS2_X_6RNKHAX)

    def get_n1_n2_m_k(self) -> Tuple[int, int, int, int]:
        return 12, 12, 0, 4


class Map6AKhAxTo7RnAx(Map8AKhAxTo9RnAx):
    _khax_dim: int = 8


class Map12KhAxToStatCEReduced(ModuleXYToX, PostInit):
    """
    >>> ''' 0,    1,   2,    3,    4,    5,   6,    7,    8,    9,   10,   11
    >>>     NERr, NIR, NIRr, NER,  TERr, TIR, TIRr, TER,  QERr, QIR, QIRr, QER '''
    """
    sel_stat: Tuple[int, ...] = (0, 1, 4, 5, 8, 9)
    sel_dyn: Tuple[int, ...] = (2, 3, 6, 7, 10, 11)

    softmax_scl: Tensor
    zero: Tensor
    # sel_stat: Tuple[int, ...]
    # sel_dyn: Tuple[int, ...]

    def __init__(self):
        super(Map12KhAxToStatCEReduced, self).__init__()
        self.softmax_scl = Parameter(tr.tensor([5.], dtype=tr.float), requires_grad=True)
        self.register_buffer('zero', tr.tensor([0.]).mean())
        self.__final_init__()

    def abs(self, x: Tensor) -> Tensor:
        """ huber loss pseudo δ = 0.1 """
        return func.smooth_l1_loss(x * 4, self.zero, reduction='none') * 0.25

    def prob_stat_dyn__1(self, z_khax: Tensor) -> Tensor:
        z_abskhax = self.abs(z_khax)
        stat = z_abskhax[:, self.sel_stat].sum(dim=1)
        dyn = z_abskhax[:, self.sel_dyn].sum(dim=1)
        return tr.softmax(tr.stack([stat, dyn], dim=1) * self.softmax_scl**2, dim=1)

    def prob_stat_dyn(self, z_khax: Tensor) -> Tensor:
        prob = tr.softmax(self.abs(z_khax) * self.softmax_scl**2, dim=1)
        prob_stat = prob[:, self.sel_stat].sum(dim=1)
        prob_dyn = prob[:, self.sel_dyn].sum(dim=1)
        return tr.stack([prob_stat, prob_dyn], dim=1)

    def prob_stat(self, z_khax: Tensor) -> Tensor:
        return self.prob_stat_dyn(z_khax)[:, (0,)]

    def forward_(self, x: Tensor, y: Tensor) -> Tensor:
        return Classifier.cross_entropy(probs=self.prob_stat_dyn(z_khax=x), target=Jp.p_stat_dyn(y))


class Map8KhAxToStatCEReduced20ExtraPairs(Map12KhAxToStatCEReduced):
    sel_stat: Tuple[int, ...] = (0, 1, 4, 5, 16, 18)
    sel_dyn: Tuple[int, ...] = (2, 3, 6, 7, 17, 19)


class Map8KhAxToStatCEReduced4ExtraPairs(Map12KhAxToStatCEReduced):
    sel_stat: Tuple[int, ...] = (0, 1, 4, 5, 8, 10)
    sel_dyn: Tuple[int, ...] = (2, 3, 6, 7, 9, 11)


class Map8KhAxToStatCEReduced4ExtraPairs2(Map12KhAxToStatCEReduced):
    sel_stat: Tuple[int, ...] = (0, 1, 4, 5)
    sel_dyn: Tuple[int, ...] = (2, 3, 6, 7)


class Map16PsiAxToStatCEReduced(Map12KhAxToStatCEReduced):
    sel_stat: Tuple[int, ...] = (0, 1, 4, 5, 8, 9, 12, 13)
    sel_dyn: Tuple[int, ...] = (2, 3, 6, 7, 10, 11, 14, 15)

    def abs(self, x: Tensor) -> Tensor:
        """ dummy """
        return x


class Map8KhAxToStatCEReduced(Map12KhAxToStatCEReduced):
    """
    >>> ''' 0,    1,   2,    3,    4,    5,   6,    7
    >>>     NERr, NIR, NIRr, NER,  TERr, TIR, TIRr, TER '''
    """
    sel_stat: Tuple[int, ...] = (0, 1, 4, 5)
    sel_dyn: Tuple[int, ...] = (2, 3, 6, 7)


class Map8TypeAxToStatCEReduced(Map8KhAxToStatCEReduced):
    """
    >>> ''' 0,     1,    2,     3,      4,     5,     6,     7
    >>>     10=>1, 9=>2, 12=>3, 11=>4,  14=>5, 13=>6, 16=>7, 15=>8 '''
    """
    pass


class Map12KhAxToStatCEReducedPer3(Map12KhAxToStatCEReduced):
    """
    >>> ''' 0,    1,   2,    3,    4,    5,   6,    7,    8,    9,   10,   11
    >>>     NERr, NIR, NIRr, NER,  TERr, TIR, TIRr, TER,  QERr, QIR, QIRr, QER '''
    """

    def prob_stat_dyn(self, z_khax: Tensor) -> Tensor:
        batch_size = z_khax.shape[0]
        return tr.softmax(
            (self.abs(z_khax) * self.softmax_scl**2).view(batch_size, 3, 4), dim=2
        ).view(batch_size, 3, 2, 2).sum(dim=3)  # (batch_size, 3, 2)

    def prob_stat(self, z_khax: Tensor) -> Tensor:
        batch_size = z_khax.shape[0]
        return self.prob_stat_dyn(z_khax)[:, :, 0].view(batch_size, 3)

    def forward_(self, x: Tensor, y: Tensor) -> Tensor:
        batch_size = x.shape[0]
        target, p = Jp.p_stat_dyn(y), self.prob_stat_dyn(z_khax=x)
        return (Classifier.cross_entropy(p[:, 0, :].view(batch_size, 2), target)
                + Classifier.cross_entropy(p[:, 1, :].view(batch_size, 2), target)
                + Classifier.cross_entropy(p[:, 2, :].view(batch_size, 2), target)) * 0.33


class Map8KhZgAxToStatCEReduced(Map12KhAxToStatCEReduced):
    """
    >>> ''' 0,    1,  2,    3,    4,    5,    6,    7
    >>>    +ERr, +ER, N±IR, N±ER, T±IR, T±ER, Q±IR, Q±ER '''
    """
    sel_stat: Tuple[int, ...] = (0, 2, 4, 6)
    sel_dyn: Tuple[int, ...] = (1, 3, 5, 7)


class Map6KhZgAxToStatCEReduced(Map12KhAxToStatCEReduced):
    """
    >>> ''' 0,    1,  2,    3,    4,    5
    >>>    +ERr, +ER, N±IR, N±ER, T±IR, T±ER '''
    """
    sel_stat: Tuple[int, ...] = (0, 2, 4)
    sel_dyn: Tuple[int, ...] = (1, 3, 5)


class Perceptron2(Perceptron):
    def __init__(self, dims: Seq[int], activation_fn: Callable[[Tensor], Tensor]=tr.selu,
                 output_activation: Opt[Callable[[Tensor], Tensor]]=None):
        dims_ = tuple(dims)
        assert (dims_[0] == 1 + 6) and (dims_[-1] == 8) and (len(dims_) > 2)
        dims_ = (1 + 4,) + dims_[1:-1] + (4,)

        super(Perceptron2, self).__init__(dims=dims_, activation_fn=activation_fn, output_activation=output_activation)
        self.dims = dims
        self.layers1 = self.get_layers(dims_)

    def forward_(self, x: Tensor) -> Tensor:
        return tr.cat([self.forward_i(x[:, (0, 1, 2, 3, 4)], self.layers),
                       self.forward_i(x[:, (0, 1, 2, 5, 6)], self.layers1)], dim=1)


class Perceptron3(Perceptron):
    def __init__(self, dims: Seq[int], activation_fn: Callable[[Tensor], Tensor]=tr.selu,
                 output_activation: Opt[Callable[[Tensor], Tensor]]=None):
        dims_ = tuple(dims)
        assert (dims_[0] == 1 + 5) and (dims_[-1] == 8) and (len(dims_) > 2)
        dims_n = (1 + 4,) + dims_[1:-1] + (4,)
        dims_t = (1 + 3,) + dims_[1:-1] + (4,)

        super(Perceptron3, self).__init__(dims=dims_n, activation_fn=activation_fn, output_activation=output_activation)
        self.dims = dims
        self.layers1 = self.get_layers(dims_t)

    def forward_(self, x: Tensor) -> Tensor:
        return tr.cat([self.forward_i(x[:, (0, 1, 2, 3, 4)], self.layers),
                       self.forward_i(x[:, (0, 1, 2, 5)], self.layers1)], dim=1)


SEMI = 4
# 0: S=>N F=>T IR=>ERr IRr=>ER
# 1: S=>N Bg=>Ad Bg=>Ad F=>T IR=>ERr IRr=>ER Bg-12=>Ad+12
# 2: SBg=>NAd SAd=>NBg F=>T IR=>ERr IRr=>ER
# 3: S=>N F=>T IRr=>ER
# 4: S=>N F=>T IR=>ERr IRr=>ER IBg=>EAd IAd=>EBg
def cond(condition: Union[Tuple[int, ...], int], *ax: Tuple[int, ...]):
    return cond_(condition, *ax, const=SEMI)


class SubDecoderPassthrSocAxTo12KhAx(ModuleXToX, PostInit):
    socax_dim: int = 8
    socpass_dim: int = 0
    khax_dim: int = 12
    khax_real_dim: int = 0
    khax_cls_dim: int = 0
    khax_trans_dim: int = 12
    for_cls_dim: int = 12

    passthr_dim: int = 1
    socextra_dim: int = 0
    axs_ce_dim_modify: int = 0
    # without passthr:
    ind_axs_to_cls: Tuple[int, ...] = tuple(range(7))
    ind_axs_to_cls__ext: Tuple[int, ...]  # with passthr

    mmd_mult: float = 0  # 750
    trim_mult: float = 200
    trim_thr: float = 3
    h_dims: Tuple[int, ...] = (32, 32)
    dist: Distrib
    zero: Tensor  # buffer
    map: Perceptron
    map_khax2rnax: Map12KhAxTo15RnAx
    map_khax_to_oy_ce_reduced: Map12KhAxToStatCEReduced
    Map: Type[Perceptron] = Perceptron

    def __init__(self, passthrough_dim: int=1, activation_fn: Callable[[Tensor], Tensor]=tr.selu,
                 output_activation: Opt[Callable[[Tensor], Tensor]]=None):
        """
        In case of 12 Khizhnyak axes:

        z_ext_ind: (sex, Rr, E, N, T, ?Q', ??Ad, ... ??Ag, extra1, ...)
        z_ext_dep: (sex, NERr, ..., QER,  extra1, ...)
        """
        super(SubDecoderPassthrSocAxTo12KhAx, self).__init__()
        assert self.passthr_dim == passthrough_dim
        self.map = self.Map(dims=(self.passthr_dim + self.socax_dim, *self.h_dims, self.khax_dim + self.socextra_dim),
                            activation_fn=activation_fn,
                            output_activation=output_activation)
        self.register_buffer('zero', tr.tensor([0.]).mean())
        self.dist = self.get_dist()
        self.ind_axs_to_cls__ext = tuple(i + self.passthr_dim for i in self.ind_axs_to_cls)
        self.map_khax2rnax, self.map_khax_to_oy_ce_reduced = self.get_maps()
        if self.khax_real_dim == 0:
            self.khax_real_dim = self.khax_dim
        self.__final_init__()

    @staticmethod
    def get_dist() -> Distrib:
        return Normal()

    @staticmethod
    def get_maps() -> Tuple[Map12KhAxTo15RnAx, Map12KhAxToStatCEReduced]:
        return Map12KhAxTo15RnAx(), Map12KhAxToStatCEReduced()

    # noinspection PyMethodMayBeStatic
    def z_ext_trans_dep(self, z_ext_dep: Tensor) -> Tensor:
        """ z := z_ext_dep;
        return should always be of the form: cat(sex(z), khax(z), f(z), extraend(z)) """
        return z_ext_dep

    def z_ext_trans_dep_(self, z_ext_dep: Tensor, f: Callable[[Tensor], Tensor], replace: bool=False) -> Tensor:
        """
        z := z_ext_dep;
        return is of the form: cat(sex(z), khax(z), f(khax(z)), extraend(z))
        """
        s = z_ext_dep[:, :self.passthr_dim]
        n = self.passthr_dim + self.khax_dim
        z_khax = z_ext_dep[:, self.passthr_dim:n]

        fz = f(z_khax)
        cat_ = [s, fz] if replace else [s, z_khax, fz]

        if z_ext_dep.shape[1] > n:
            z_extra = z_ext_dep[:, n:]
            cat_.append(z_extra)
        return tr.cat(cat_, dim=1)

    def z_dep_and_trans_dep(self, z_ext_ind: Tensor, z_ext_dep: Tensor) -> Tensor:  # pylint: disable=unused-argument
        return z_ext_dep[:, self.passthr_dim:]

    def get_z_khax_plus(self, z_ext_ind: Tensor) -> Tensor:
        n = self.passthr_dim + self.socax_dim
        return self.map.__call__(z_ext_ind[:, :n])

    def z_ext_dep(self, z_ext_ind: Tensor) -> Tensor:
        z_khax_plus = self.get_z_khax_plus(z_ext_ind)  # == cat(..., z_khax, z_socextra)
        s = z_ext_ind[:, :self.passthr_dim]

        n = self.passthr_dim + self.socax_dim
        if z_ext_ind.shape[1] > n:
            return tr.cat([s, z_khax_plus, z_ext_ind[:, n:]], dim=1)  # == cat(..., z_khax_plus, z_nonsocextra)

        return tr.cat([s, z_khax_plus], dim=1)

    def z_ext_for_cls(self, z_ext_ind: Tensor) -> Tensor:
        z_ext_for_cls = self.z_ext_dep(z_ext_ind)[:, :self.passthr_dim + self.for_cls_dim]
        if self.ind_axs_to_cls__ext:
            return tr.cat([z_ext_for_cls, z_ext_ind[:, self.ind_axs_to_cls__ext]], dim=1)
        return z_ext_for_cls

    def z_for_plot(self, z_ext_ind: Tensor) -> Tensor:
        z_ext_dep = self.z_ext_dep(z_ext_ind)  # == cat(..., z_socextra, z_nonsocextra)

        z_khax = z_ext_dep[:, self.passthr_dim:self.passthr_dim + self.khax_real_dim]
        return tr.cat([self.z_dep_and_trans_dep(z_ext_ind, z_ext_dep), self.map_khax_to_oy_ce_reduced.prob_stat(z_khax),
                       self.map_khax2rnax.z_rnax(z_khax)], dim=1)

    def forward_(self, x: Tensor) -> Tensor:
        z_ext_ind = x
        return self.z_ext_trans_dep(self.z_ext_dep(z_ext_ind))

    # noinspection PyMethodMayBeStatic
    def repr_learnables(self) -> str:  # API
        return (f'abskhax2complrnkh:\n{self.map_khax2rnax.abskhax2complrnkh()}\n'
                + f'khax2rnkh:\n{self.map_khax2rnax.khax2rnkh()}')

    def regul_loss_reduced(self, z_ext_ind: Tensor, z_ext_dep: Tensor) -> Tensor:  # API
        z_dep = z_ext_dep[:, self.passthr_dim:]
        trim = (tr.relu(z_dep - self.trim_thr)  # not neg. thr. at right
                + tr.relu(-z_dep - self.trim_thr)  # not pos. thr. at left
                ).view(z_dep.shape[0], -1).sum(dim=1).mean() * self.trim_mult
        mmd = self.dist.mmd(z_dep) * self.mmd_mult if (self.mmd_mult > 0) else self.zero
        return trim + mmd


class SubDecoderPassthrSocAxTo12KhAx2(SubDecoderPassthrSocAxTo12KhAx):  # TODO
    ind_axs_to_cls: Tuple[int, ...] = tuple(range(7))
    # was: (0, 1, 2, 3, 4); (0, 1, 2, 3, 3, 4, 4); (0, 1, 2, 2, 3, 3, 4, 4, 5, 5)
    h_dims: Tuple[int, ...] = (32, 32)  # part 1 - (32, 32); part 2.0+ - (64,)
    # was: !(64, 64); (96, 96, 96); (64,)
    # mmd_mult: float = 750 * 5  # 750 * 6.67  # this should be overrided only in part 2 - 37+ (but not 35, 36)


class SubDecoderPassthrSocAxTo12KhAx6(SubDecoderPassthrSocAxTo12KhAx):
    ind_axs_to_cls: Tuple[int, ...] = tuple(range(7))
    h_dims: Tuple[int, ...] = (64,)
    # socax_dim: int = 7


# noinspection PyPep8Naming
class SubDecoderPassthrSocAxTo12KhAx3(SubDecoderPassthrSocAxTo12KhAx):
    ind_axs_to_cls: Tuple[int, ...] = tuple(range(7))  # classifier .extra_dim is len(ind_axs_to_cls)
    for_cls_dim: int = 12
    socax_dim: int = 8

    socextra_dim: int = 0  # should be incorporated into for_cls_dim TOO if to be used with axes align
    semi_type_indep: Tuple[int, ...] = (7,)  # semi-type-indep. axes
    # WARNING: semi-type-indep. axes CANNOT BE combined with number of socax_dim less than all axes.

    socpass_dim: int = len(semi_type_indep)
    axs_ce_dim_modify: int = socextra_dim + socpass_dim
    semi_type_indep__ext: Tuple[int, ...]  # with passthr

    def __post_init__(self):
        super(SubDecoderPassthrSocAxTo12KhAx3, self).__post_init__()
        self.semi_type_indep__ext = tuple(i + self.passthr_dim for i in self.semi_type_indep)

    def z_ext_dep(self, z_ext_ind: Tensor) -> Tensor:
        if self.semi_type_indep__ext:
            z_khax_plus = self.get_z_khax_plus(z_ext_ind)  # == cat(..., z_khax, z_socextra)
            s = z_ext_ind[:, :self.passthr_dim]
            return tr.cat([s, z_khax_plus, z_ext_ind[:, self.semi_type_indep__ext]], dim=1)
        return super(SubDecoderPassthrSocAxTo12KhAx3, self).z_ext_dep(z_ext_ind)

    def z_ext_for_cls(self, z_ext_ind: Tensor) -> Tensor:
        z_ext_dep = self.z_ext_dep(z_ext_ind)
        if self.ind_axs_to_cls__ext:
            n = self.passthr_dim + self.for_cls_dim
            z_extra = z_ext_ind[:, self.ind_axs_to_cls__ext]
            if z_ext_dep.shape[1] > n:
                return tr.cat([z_ext_dep[:, :n], z_extra, z_ext_dep[:, n:]], dim=1)
            return tr.cat([z_ext_dep, z_extra], dim=1)
        return z_ext_dep


class SubDecoderPassthrSocAxToKhAxPlus(SubDecoderPassthrSocAxTo12KhAx):
    khax_dim: int = 12
    khax_trans_dim: int = 12
    socextra_dim: int = 15
    for_cls_dim: int = 12 + 15


class SubDecoderPassthrSocAxTo8KhAx(SubDecoderPassthrSocAxTo12KhAx):
    khax_dim: int = 8
    khax_trans_dim: int = 8
    for_cls_dim: int = 8

    @staticmethod
    def get_maps() -> Tuple[Map12KhAxTo15RnAx, Map12KhAxToStatCEReduced]:
        return Map8KhAxTo11RnAx(), Map8KhAxToStatCEReduced()


class SubDecoderPassthrSocAxTo12KhAx4(SubDecoderPassthrSocAxTo12KhAx):  # TODO
    ind_axs_to_cls: Tuple[int, ...] = (0, 1, 2)
    h_dims: Tuple[int, ...] = (64,)
    for_cls_dim: int = 12
    qd_exclusive_dim: int = 1

    h_dims_qd: Tuple[int, ...] = (32,)
    map_qd: Perceptron

    def __post_init__(self):
        super(SubDecoderPassthrSocAxTo12KhAx4, self).__post_init__()
        activation_fn = self.map.activation_fn.a
        output_activation = self.map.output_activation.a if (self.map.output_activation is not None) else None
        self.map = self.Map(dims=(self.passthr_dim + self.socax_dim - self.qd_exclusive_dim, *self.h_dims, 8),
                            activation_fn=activation_fn, output_activation=output_activation)
        self.map_qd = self.Map(dims=(self.passthr_dim + self.socax_dim, *self.h_dims_qd, 4),
                               activation_fn=activation_fn, output_activation=output_activation)

    def get_z_khax_plus(self, z_ext_ind: Tensor) -> Tensor:
        n = self.passthr_dim + self.socax_dim
        z_khax_nstf = self.map.__call__(z_ext_ind[:, :n - self.qd_exclusive_dim])
        z_khax_qd = self.map_qd.__call__(z_ext_ind[:, :n])
        return tr.cat([z_khax_nstf, z_khax_qd], dim=1)

    def z_ext_for_cls(self, z_ext_ind: Tensor) -> Tensor:
        z_ext_dep = self.z_ext_dep(z_ext_ind)
        if self.ind_axs_to_cls__ext:
            n = self.passthr_dim + self.for_cls_dim
            if z_ext_dep.shape[1] > n:
                return tr.cat([z_ext_dep[:, :n], z_ext_ind[:, self.ind_axs_to_cls__ext], z_ext_ind[:, n:]], dim=1)
            return tr.cat([z_ext_dep[:, :n], z_ext_ind[:, self.ind_axs_to_cls__ext]], dim=1)
        return z_ext_dep


class SubDecoderPassthrSocAxTo12KhAx5(SubDecoderPassthrSocAxTo12KhAx4):  # TODO
    socax_dim: int = 8
    ind_axs_to_cls: Tuple[int, ...] = tuple(range(7))  # classifier .extra_dim is len(ind_axs_to_cls)
    h_dims: Tuple[int, ...] = (64,)
    h_dims_tf: Tuple[int, ...] = (32,)
    h_dims_qd: Tuple[int, ...] = (32,)

    for_cls_dim: int = 12

    map_tf: Perceptron
    map_qd: Perceptron

    transform: Tensor
    ns_dim: int = 7  # 5 + 2 + 2
    tf_dim: int = 7  # 5 + 2
    qd_dim: int = 8  # 5 + 2

    def __post_init__(self):
        super(SubDecoderPassthrSocAxTo12KhAx5, self).__post_init__()
        self.register_buffer('transform', self.get_transform())

        activation_fn = self.map.activation_fn.a
        output_activation = self.map.output_activation.a if (self.map.output_activation is not None) else None
        self.map = self.Map(dims=(self.passthr_dim + self.ns_dim, *self.h_dims, 8),
                            activation_fn=activation_fn, output_activation=output_activation)
        self.map_tf = self.Map(dims=(self.passthr_dim + self.tf_dim, *self.h_dims_tf, 4),
                               activation_fn=activation_fn, output_activation=output_activation)
        self.map_qd = self.Map(dims=(self.passthr_dim + self.qd_dim, *self.h_dims_qd, 4),
                               activation_fn=activation_fn, output_activation=output_activation)

    def z_ext_dep(self, z_ext_ind: Tensor) -> Tensor:
        z_khax_plus = self.get_z_khax_plus(z_ext_ind)  # == cat(..., z_khax, z_socextra)
        s = z_ext_ind[:, :self.passthr_dim]

        n = self.passthr_dim + self.socax_dim
        if z_ext_ind.shape[1] > n:
            return tr.cat([s, z_khax_plus, z_ext_ind[:, n:]], dim=1)  # == cat(..., z_khax_plus, z_nonsocextra)

        return tr.cat([s, z_khax_plus], dim=1)

    @staticmethod
    def get_transform() -> Tensor:
        cos, sin, π = math.cos, math.sin, math.pi
        rot45 = tr.tensor([[cos(π / 4), -sin(π / 4)],
                           [sin(π / 4), +cos(π / 4)]])
        rot_inv_45 = rot45.transpose(0, 1)
        return rot_inv_45

    def __ort_transform__(self, z_ext_ind: Tensor) -> Tuple[Tensor, Tensor, Tensor]:  # OFF
        common = z_ext_ind[:, (0, 4, 5)]  # sex, N, (TAd, TBg), (ERr, Er), (EAd, EBg), AV
        # .                                   0  1     2    3      4   5      6    7   8
        fad_t = z_ext_ind[:, (2, 3)] @ self.transform
        rr_e = z_ext_ind[:, (4, 5)] @ self.transform
        ad_e2 = z_ext_ind[:, (6, 7)] @ self.transform
        fad, t, e2 = fad_t[:, (0,)], fad_t[:, (1,)], ad_e2[:, (1,)]
        # .                                     N EAd EBg
        ns = tr.cat([common, rr_e, z_ext_ind[:, (1, 6, 7)], ad_e2], dim=1)  # ERr,Er,(N,Ad),E2,{EAd,EBg}
        # .                                      AV
        tf = tr.cat([common, rr_e, z_ext_ind[:, (8,)], t, e2], dim=1)  # ERr,Er,(AV,T),E2
        # .                                     TAd
        qd = tr.cat([common, rr_e, z_ext_ind[:, (2,)], fad, e2], dim=1)  # ERr,Er,(TAd,FAd),E2,{TBg is absent}
        return ns, tf, qd

    def ort_transform(self, z_ext_ind: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # sex, N, (TAd, TBg), (ERr, Er), (EAd, EBg), AV
        # 0    1   2    3      4   5      6    7     8
        fad_t = z_ext_ind[:, (2, 3)] @ self.transform
        nstf = tr.cat([z_ext_ind[:, (0, 1, 4, 5, 6, 7, 8)], fad_t[:, (1,)]], dim=1)
        qd = z_ext_ind
        return nstf, nstf, qd

    def get_z_khax_plus(self, z_ext_ind: Tensor) -> Tensor:
        z_ext_ns, z_ext_tf, z_ext_qd = self.ort_transform(z_ext_ind[:, :self.passthr_dim + self.socax_dim])
        z_khax_ns = self.map.__call__(z_ext_ns)
        # z_khax_tf = self.map_tf.__call__(z_ext_tf)
        z_khax_qd = self.map_qd.__call__(z_ext_qd)
        return tr.cat([z_khax_ns, z_khax_qd], dim=1)

    def z_ext_for_cls(self, z_ext_ind: Tensor) -> Tensor:
        z_ext_dep = self.z_ext_dep(z_ext_ind)
        if self.ind_axs_to_cls__ext:
            n = self.passthr_dim + self.for_cls_dim
            if z_ext_dep.shape[1] > n:
                return tr.cat([z_ext_dep[:, :n], z_ext_ind[:, self.ind_axs_to_cls__ext], z_ext_ind[:, n:]], dim=1)
            return tr.cat([z_ext_dep[:, :n], z_ext_ind[:, self.ind_axs_to_cls__ext]], dim=1)
        return z_ext_dep


class SubDecoderPassthrSocAxTo8KhAx20ExtraPairs(SubDecoderPassthrSocAxTo8KhAx):
    khax_dim: int = 8
    khax_trans_dim: int = 8
    socextra_dim: int = 20
    for_cls_dim: int = 8 + 20
    khax_real_dim: int = 8 + 20

    @staticmethod
    def get_maps() -> Tuple[Map12KhAxTo15RnAx, Map12KhAxToStatCEReduced]:
        return Map8KhAx20ExtraPairsTo7RnAx(), Map8KhAxToStatCEReduced20ExtraPairs()


class SubDecoderPassthrSocAxTo8KhAx4ExtraPairs(SubDecoderPassthrSocAxTo8KhAx):
    khax_dim: int = 8
    khax_trans_dim: int = 8
    socextra_dim: int = 4
    for_cls_dim: int = 8 + 4
    khax_real_dim: int = 8 + 4

    @staticmethod
    def get_maps() -> Tuple[Map12KhAxTo15RnAx, Map12KhAxToStatCEReduced]:
        return Map8KhAx4ExtraPairsTo13RnAx(), Map8KhAxToStatCEReduced4ExtraPairs()


class SubDecoderPassthrSocAxTo8KhAx4ExtraPairs2(SubDecoderPassthrSocAxTo8KhAx4ExtraPairs):
    @staticmethod
    def get_maps() -> Tuple[Map12KhAxTo15RnAx, Map12KhAxToStatCEReduced]:
        return Map8KhAx4ExtraPairs2To12RnAx(), Map8KhAxToStatCEReduced4ExtraPairs2()


class SubDecoderPassthrSocAxTo16PsiAx(SubDecoderPassthrSocAxTo12KhAx):
    khax_dim: int = 16
    khax_trans_dim: int = 16
    socextra_dim: int = 0
    for_cls_dim: int = 16
    khax_real_dim: int = 16

    @staticmethod
    def get_maps() -> Tuple[Map12KhAxTo15RnAx, Map12KhAxToStatCEReduced]:
        return Map16PsiAxTo15RnAx(), Map16PsiAxToStatCEReduced()


class SubDecoderPassthrSocAxLinTo8KhAx(SubDecoderPassthrSocAxTo8KhAx):
    socax_dim: int = 7
    fza_dim: int
    rotfza2types_3x8x16: Tensor
    types2khax8_16x8: Tensor
    pi: Tensor
    k_fza2types_3x8: Parameter
    rots: Parameter

    def __post_init__(self):
        super(SubDecoderPassthrSocAxLinTo8KhAx, self).__post_init__()
        self.fza_dim = self.socax_dim + 1
        self.register_buffer('rotfza2types_3x8x16', tr.tensor(MAP_3_X_ROT8FZA4_X_16TYPES, dtype=tr.float))
        self.register_buffer('types2khax8_16x8', tr.tensor(MAP_16TYPES_X_8KHAX, dtype=tr.float))
        self.k_fza2types_3x8 = Parameter(
            tr.tensor([[0.7 for __ in range(self.fza_dim)] for _ in range(3)], dtype=tr.float),
            requires_grad=True)
        self.rots = Parameter(tr.tensor([0. for _ in range(3)], dtype=tr.float), requires_grad=True)
        self.register_buffer('pi', tr.tensor([math.pi], dtype=tr.float))

    def get_rots(self) -> Tuple[Tensor, ...]:
        # ...(...) * π/6 + π/4 == ...(...) * 30° + 45°
        rots = tuple(self.pi * ((func.hardtanh(self.rots[i]) + self.rots[i] * 0.01) * 0.1667 + 0.25)
                     for i in range(3))
        return tuple(tr.tensor([[tr.cos(rot), -tr.sin(rot)],
                                [tr.sin(rot), tr.cos(rot)]]).transpose(0, 1)
                     for rot in rots)

    def rot_transform(self, z_fza: Tensor) -> Tensor:
        rot1, rot2, rot3 = self.get_rots()
        return tr.cat([
            z_fza[:, (0,)],
            z_fza[:, (1, 2)] @ rot1,
            z_fza[:, (3, 4)] @ rot2,
            z_fza[:, (5, 6)] @ rot3,
            z_fza[:, (7,)],
        ], dim=1)

    def rotfza2types(self) -> Tensor:
        """ return shape is (8, 16) """
        k, A = self.k_fza2types_3x8**2, self.rotfza2types_3x8x16
        n = k.shape[1]
        return k[0].reshape(n, -1) * A[0] + k[1].reshape(n, -1) * A[1] + k[2].reshape(n, -1) * A[2]

    def get_z_khax_plus(self, z_ext_ind: Tensor) -> Tensor:
        z_fza = z_ext_ind[:, self.passthr_dim:self.passthr_dim + self.fza_dim]
        return (self.rot_transform(z_fza) @ self.rotfza2types()) @ self.types2khax8_16x8

    def z_for_plot(self, z_ext_ind: Tensor) -> Tensor:
        z_ext_dep = self.z_ext_dep(z_ext_ind)  # == cat(..., z_socextra, z_nonsocextra)
        z_fza = z_ext_ind[:, self.passthr_dim:self.passthr_dim + self.fza_dim]
        rot1, rot2, rot3 = self.get_rots()
        z_khax = z_ext_dep[:, self.passthr_dim:self.passthr_dim + self.khax_real_dim]
        return tr.cat([self.z_dep_and_trans_dep(z_ext_ind, z_ext_dep),
                       z_fza[:, (1, 2)] @ rot1,
                       z_fza[:, (3, 4)] @ rot2,
                       z_fza[:, (5, 6)] @ rot3,
                       self.map_khax_to_oy_ce_reduced.prob_stat(z_khax),
                       self.map_khax2rnax.z_rnax(z_khax)], dim=1)

    # noinspection PyMethodMayBeStatic
    def repr_learnables(self) -> str:  # API
        rots = tuple(((func.hardtanh(self.rots[i]) + self.rots[i] * 0.01) * 0.1667 + 0.25)**(-1)
                     for i in range(3))
        return (super(SubDecoderPassthrSocAxLinTo8KhAx, self).repr_learnables()
                + f'\nfza2types:\n{self.rotfza2types()}'
                + f'\n(rots/π)^(-1): \n{rots}'
                )


class SubDecoderPassthrSocAxLinTo16PsiAx(SubDecoderPassthrSocAxLinTo8KhAx):
    khax_dim: int = 16
    khax_trans_dim: int = 16
    socextra_dim: int = 0
    for_cls_dim: int = 16
    khax_real_dim: int = 16

    @staticmethod
    def get_maps() -> Tuple[Map12KhAxTo15RnAx, Map12KhAxToStatCEReduced]:
        return Map16PsiAxTo15RnAx(), Map16PsiAxToStatCEReduced()

    def get_z_khax_plus(self, z_ext_ind: Tensor) -> Tensor:
        z_fza = z_ext_ind[:, self.passthr_dim:self.passthr_dim + self.fza_dim]
        return self.rot_transform(z_fza) @ self.rotfza2types()


class SubDecoderPassthrSocAxTo8TypeAx(SubDecoderPassthrSocAxTo12KhAx):
    socax_dim: int = 8  # 7 when only 7 axes are converted to type axes
    khax_dim: int = 8
    khax_trans_dim: int = 8
    socextra_dim: int = 0
    for_cls_dim: int = 8
    khax_real_dim: int = 8

    @staticmethod
    def get_maps() -> Tuple[Map12KhAxTo15RnAx, Map12KhAxToStatCEReduced]:
        return Map8TypeAxTo13RnAx(), Map8TypeAxToStatCEReduced()  # Map8TypeAxTo13RnAx Map8TypeAxTo15RnAx


class SubDecoderPassthrSocAxTo12KhAxPer3(SubDecoderPassthrSocAxTo12KhAx):
    @staticmethod
    def get_maps() -> Tuple[Map12KhAxTo15RnAx, Map12KhAxToStatCEReduced]:
        return Map12KhAxTo15RnAx(), Map12KhAxToStatCEReducedPer3()


class SubDecoderPassthrSocAxTo12KhAxProbStat(SubDecoderPassthrSocAxTo12KhAx):
    khax_trans_dim: int = 12 + 1

    def z_ext_trans_dep(self, z_ext_dep: Tensor) -> Tensor:
        return self.z_ext_trans_dep_(z_ext_dep, self.map_khax_to_oy_ce_reduced.prob_stat)


class SubDecoderPassthrSocAxTo8KhAxProbStat(SubDecoderPassthrSocAxTo8KhAx):
    khax_trans_dim: int = 8 + 1

    def z_ext_trans_dep(self, z_ext_dep: Tensor) -> Tensor:
        return self.z_ext_trans_dep_(z_ext_dep, self.map_khax_to_oy_ce_reduced.prob_stat)


class SubDecoderPassthrSocAxTo12KhAx15Rn(SubDecoderPassthrSocAxTo12KhAx):
    khax_dim: int = 12
    khax_trans_dim: int = 12 + 15
    for_cls_dim: int = 12

    def z_ext_trans_dep(self, z_ext_dep: Tensor) -> Tensor:
        return self.z_ext_trans_dep_(z_ext_dep, self.map_khax2rnax.z_rnax)

    def z_dep_and_trans_dep(self, z_ext_ind: Tensor, z_ext_dep: Tensor) -> Tensor:
        return self.z_ext_trans_dep(z_ext_dep)[:, self.passthr_dim:]


class SubDecoderPassthrSocAxTo12KhAxSemi6(SubDecoderPassthrSocAxTo12KhAx):
    ind_axs_to_cls: Tuple[int, ...] = (
        (0, 1, 2, 2, 3, 3),  # S=>N F=>T IR=>ERr IRr=>ER
        (0, 1, 2, 3, 4, 4, 5, 5, 6),  # S=>N Bg=>Ad Bg=>Ad F=>T IR=>ERr IRr=>ER Bg-12=>Ad+12
        (0, 0, 1, 1, 2, 3, 3, 4, 4),  # SBg=>NAd SAd=>NBg F=>T IR=>ERr IRr=>ER
        (0, 1, 2, 2),  # S=>N F=>T IRr=>ER
        (0, 1, 2, 2, 3, 3, 4, 4, 5, 5),  # S=>N F=>T IR=>ERr IRr=>ER IBg=>EAd IAd=>EBg
    )[SEMI]


class SubDecoderPassthrSocAxTo8KhAxSemi6(SubDecoderPassthrSocAxTo8KhAx):
    ind_axs_to_cls: Tuple[int, ...] = SubDecoderPassthrSocAxTo12KhAxSemi6.ind_axs_to_cls


class SubDecoderPassthrSocAxTo12KhAxSemi6AKhMod(SubDecoderPassthrSocAxTo12KhAxSemi6):
    khax_dim: int = 8
    khax_real_dim: int = 12
    khax_cls_dim: int = 8 + 12
    khax_trans_dim: int = 12
    for_cls_dim: int = 8 + 12
    map_akh: Perceptron

    def __post_init__(self):
        super(SubDecoderPassthrSocAxTo12KhAxSemi6AKhMod, self).__post_init__()
        self.map_akh = self.Map(dims=(self.passthr_dim + self.khax_dim + self.socextra_dim, *self.h_dims,
                                      self.khax_trans_dim + self.socextra_dim),
                                activation_fn=self.map.activation_fn.a,
                                output_activation=(self.map.output_activation.a
                                                   if (self.map.output_activation is not None)
                                                   else None)
                                )

    def get_z_khax_plus(self, z_ext_ind: Tensor) -> Tensor:
        n = self.passthr_dim + self.socax_dim
        s = z_ext_ind[:, :self.passthr_dim]
        z_akhax_plus = self.map.__call__(z_ext_ind[:, :n])
        return self.map_akh.__call__(tr.cat([s, z_akhax_plus], dim=1))

    def get_z_akhax(self, z_ext_ind: Tensor) -> Tensor:
        z_ext_soc_ind = z_ext_ind[:, :self.passthr_dim + self.socax_dim]
        return self.map.__call__(z_ext_soc_ind)

    def z_ext_for_cls(self, z_ext_ind: Tensor) -> Tensor:
        z_ext_dep = self.z_ext_dep(z_ext_ind)
        z_akhax = self.get_z_akhax(z_ext_ind)
        s = z_ext_dep[:, :self.passthr_dim]
        z_dep = z_ext_dep[:, self.passthr_dim:]

        z_ext_for_cls = tr.cat([s, z_akhax, z_dep], dim=1)[:, :self.passthr_dim + self.for_cls_dim]
        if self.ind_axs_to_cls__ext:
            return tr.cat([z_ext_for_cls, z_ext_ind[:, self.ind_axs_to_cls__ext]], dim=1)
        return z_ext_for_cls

    def z_dep_and_trans_dep(self, z_ext_ind: Tensor, z_ext_dep: Tensor) -> Tensor:
        return tr.cat([self.get_z_akhax(z_ext_ind), z_ext_dep[:, self.passthr_dim:]], dim=1)


class SubDecoderPassthrSocAxTo8AKhAx(SubDecoderPassthrSocAxTo12KhAx):
    ind_axs_to_cls: Tuple[int, ...] = (0, 1)  # (0, 1, 2, 2, 3, 3)  # S=>N F=>T IR=>ERr IRr=>ER

    khax_dim: int = 8
    khax_cls_dim: int = 12
    khax_trans_dim: int = 12  # 8 + 12
    for_cls_dim: int = 12  # 8

    akhax2khax_12kh_x_8akh: Tensor
    k_akh2kh_1pos: Parameter
    k_akh2kh_6pos: Tensor

    def __post_init__(self):
        super(SubDecoderPassthrSocAxTo8AKhAx, self).__post_init__()
        m = (self.khax_dim - 2) * 2
        self.register_buffer('akhax2khax_12kh_x_8akh',
                             tr.tensor(MAP_12KHAX_X_8AKHAX, dtype=tr.float)[:m, :self.khax_dim])
        self.k_akh2kh_1pos = Parameter(tr.tensor([0.7], dtype=tr.float), requires_grad=True)
        self.register_buffer('k_akh2kh_6pos', tr.tensor([1. for _ in range(self.khax_dim - 2)]))

    @staticmethod
    def get_maps() -> Tuple[Map12KhAxTo15RnAx, Map12KhAxToStatCEReduced]:
        return Map8AKhAxTo9RnAx(), Map8KhZgAxToStatCEReduced()

    def z_ext_trans_dep(self, z_ext_dep: Tensor) -> Tensor:
        return self.z_ext_trans_dep_(z_ext_dep, self.akhax2khax_, replace=True)

    def z_dep_and_trans_dep(self, z_ext_ind: Tensor, z_ext_dep: Tensor) -> Tensor:
        return self.z_ext_trans_dep_(z_ext_dep, self.akhax2khax_)[:, self.passthr_dim:]

    def akhax2khax(self) -> Tensor:
        """ return shape is (8, 12) """
        k = self.k_akh2kh_1pos**2 + (0.51 + 0.25)
        return (self.akhax2khax_12kh_x_8akh * tr.cat([k, k, self.k_akh2kh_6pos])).transpose(0, 1)

    def akhax2khax_(self, z_akhax: Tensor) -> Tensor:
        return z_akhax @ self.akhax2khax()

    # noinspection PyMethodMayBeStatic
    def repr_learnables(self) -> str:  # API
        return (f'abskhax2complrnkh:\n{self.map_khax2rnax.abskhax2complrnkh()}\n'
                + f'khax2rnkh:\n{self.map_khax2rnax.khax2rnkh()}\n'
                + f'akh2kh:\n{self.akhax2khax()}')

    def z_ext_for_cls(self, z_ext_ind: Tensor) -> Tensor:
        z_ext_for_cls = self.z_ext_trans_dep(self.z_ext_dep(z_ext_ind))[:, :self.passthr_dim + self.for_cls_dim]
        if self.ind_axs_to_cls__ext:
            return tr.cat([z_ext_for_cls, z_ext_ind[:, self.ind_axs_to_cls__ext]], dim=1)
        return z_ext_for_cls


class SubDecoderPassthrSocAxTo6AKhAx(SubDecoderPassthrSocAxTo8AKhAx):
    khax_dim: int = 6
    khax_cls_dim: int = 8
    khax_trans_dim: int = 8  # 6 + 8
    for_cls_dim: int = 8  # 6

    @staticmethod
    def get_maps() -> Tuple[Map12KhAxTo15RnAx, Map12KhAxToStatCEReduced]:
        return Map6AKhAxTo7RnAx(), Map6KhZgAxToStatCEReduced()


class SubDecoderPassthrSocAxTo12KhAx12Dummy(SubDecoderPassthrSocAxTo12KhAx):
    def forward_(self, x: Tensor) -> Tensor:
        return x

    def z_ext_for_cls(self, z_ext_ind: Tensor) -> Tensor:
        return z_ext_ind

    def regul_loss_reduced(self, z_ext_ind: Tensor, z_ext_dep: Tensor) -> Tensor:  # API
        return self.zero


class SubDecoderPassthrSocAxTo12KhAx3Dim(SubDecoderPassthrSocAxTo12KhAx):
    ind_axs_to_cls: Tuple[int, ...] = tuple(range(3))


class SubDecoderPassthrSocAxTo12KhAx4Dim(SubDecoderPassthrSocAxTo12KhAx):
    ind_axs_to_cls: Tuple[int, ...] = tuple(range(4))


SubDecoder: Type[SubDecoderPassthrSocAxTo12KhAx] = SubDecoderPassthrSocAxTo12KhAx2  # TODO
SubDecoderDecSwap: Type[SubDecoderPassthrSocAxTo12KhAx] = SubDecoderPassthrSocAxTo12KhAx6
# SubDecoderPassthrSocAxTo12KhAx  SubDecoderPassthrSocAxTo8KhAx  SubDecoderPassthrSocAxTo12KhAx2
# SubDecoderPassthrSocAxTo12KhAx3  SubDecoderPassthrSocAxTo12KhAx4  --  test
# SubDecoderPassthrSocAxTo12KhAx12Dummy
# SubDecoderPassthrSocAxTo8KhAx4ExtraPairs  SubDecoderPassthrSocAxTo8KhAx4ExtraPairs2
# SubDecoderPassthrSocAxTo8KhZgAx
# SubDecoderPassthrSocAxTo12KhAxSemi6
# SubDecoderPassthrSocAxTo8TypeAx
# SubDecoderPassthrSocAxTo16PsiAx
# SubDecoderPassthrSocAxLinTo16PsiAx
# SubDecoderPassthrSocAxLinTo8KhAx


class ClassifierJATSAxesAlignNTDummy(ClassifierJATSAxesAlign5RnKhAx):
    _subdecoder: Tuple[SubDecoderPassthrSocAxTo12KhAx, ...] = ()

    axs_ce_dim: int = 2
    thrs: Tuple[Tuple[float, ...], ...] = (
        (0.5, 0.5),
        (1.0, 1.0),
    )
    # noinspection PyTypeChecker
    q_types: Tuple[Tuple[int, ...], ...] = (
        N[1], T[1],  # MAP POS
        N[0], T[0],  # MAP NEG
    )

    def set_subdecoder(self, subdecoder: SubDecoderPassthrSocAxTo12KhAx):
        self._subdecoder = (subdecoder,)

    @property
    def subdecoder(self) -> SubDecoderPassthrSocAxTo12KhAx:
        assert self._subdecoder
        return self._subdecoder[0]


_o = missing_types
KHAX1, KHAX0 = tuple(tpl(
        inv(SIR)[i], NIR[i], inv(SER)[i], NER[i], inv(FIR)[i], TIR[i], inv(FER)[i], TER[i],
        inv(DIR)[i], QIR[i], inv(DER)[i], QER[i]
    ) for i in (1, 0))
KHAX_OTHER = (
    _o(inv(SIR)), _o(NIR), _o(inv(SER)), _o(NER), _o(inv(FIR)), _o(TIR), _o(inv(FER)), _o(TER),
    _o(inv(DIR)), _o(QIR), _o(inv(DER)), _o(QER)
    )
RNKHAX1, RNKHAX0 = tuple(tpl(
        E[i], inv(R)[i], N[i], AD[i], T[i], AB[i], Q[i], AG[i]
    ) for i in (1, 0))
COMRNKHAX1, COMRNKHAX0 = tuple(tpl(
        STATIC[i], NESI[i], NRRSR[i], TEFI[i], TRRFR[i], QEDI[i], QRRDR[i]
    ) for i in (1, 0))
AKHAX1, AKHAX0 = tuple(tpl(
        inter(E, inv(R))[i], inter(E, R)[i],
        cat(inv(SIR), NIR)[i], cat(inv(SER), NER)[i], cat(inv(FIR), TIR)[i], cat(inv(FER), TER)[i],
        cat(inv(DIR), QIR)[i], cat(inv(DER), QER)[i]
    ) for i in (1, 0))
AKHAX_OTHER = (
    _o(inter(E, inv(R))), _o(inter(E, R)), _o(cat(inv(SIR), NIR)), _o(cat(inv(SER), NER)),
    _o(cat(inv(FIR), TIR)), _o(cat(inv(FER), TER)), _o(cat(inv(DIR), QIR)), _o(cat(inv(DER), QER))
    )


assert SEMI in (0, 1, 2, 3, 4)
# 0: S=>N F=>T IR=>ERr IRr=>ER
# 1: S=>N Bg=>Ad Bg=>Ad F=>T IR=>ERr IRr=>ER Bg-12=>Ad+12
# 2: SBg=>NAd SAd=>NBg F=>T IR=>ERr IRr=>ER
# 3: S=>N F=>T IRr=>ER
# 4: S=>N F=>T IR=>ERr IRr=>ER IBg=>EAd IAd=>EBg
SEMI1, SEMI0 = tuple(tpl(
        *cond(2, inter(N, AD)[i], _o(inter(N, AD)), inter(N, inv(AD))[i], _o(inter(N, inv(AD)))),
        *cond((0, 1, 3, 4), N[i]),
        *cond(1, AD[i], AD[i]),
        T[i],
        *cond((0, 1, 2, 4), inter(E, inv(R))[i], _o(inter(E, inv(R)))),
        inter(E, R)[i], _o(inter(E, R)),
        *cond(1, AD12[i]),
        *cond(4, inter(E, AD)[i], _o(inter(E, AD)), inter(E, inv(AD))[i], _o(inter(E, inv(AD)))),
    ) for i in (1, 0))


class ClassifierJATSAxesAlignKhAx(ClassifierJATSAxesAlignNTDummy):
    sigmoid_scl: float = 1

    # this combination gives reference reconstruction error:
    thrs_: Tuple[Tuple[float, ...], ...] = ((0., 1., 1.), (0.5, 1., 1.), (1.5, 1.5, 1.5))
    thrs_extra: Tuple[Tuple[float, ...], ...] = ((1.2,), (1.5,), (1.8,))
    thrs: Tuple[Tuple[float, ...], ...] = ()

    axs_ce_dim: int
    kh_ax_dim: int
    extra_dim: int
    socextra_dim: int

    q_types: Tuple[Tuple[int, ...], ...] = ()
    SubDecoder_: Type[SubDecoder] = SubDecoder  # should be non-dynamic property

    def __init__(self, dims: Tuple[int, Tuple[int, ...], int], activation_fn: Callable[[Tensor], Tensor]=tr.selu):
        _, hidden_dims, y_dim = dims
        sd = self.SubDecoder_
        self.extra_dim = len(sd.ind_axs_to_cls)
        self.kh_ax_dim = sd.khax_cls_dim if (sd.khax_cls_dim > 0) else sd.khax_dim
        self.socextra_dim = sd.socextra_dim
        self.axs_ce_dim = sd.for_cls_dim + self.extra_dim

        assert len(self.thrs_) == len(self.thrs_extra)
        self.thrs = self.get_thrs()

        super(ClassifierJATSAxesAlignKhAx, self).__init__(
            (sd.passthr_dim + sd.for_cls_dim + self.extra_dim + sd.axs_ce_dim_modify,
             hidden_dims,
             y_dim),
            activation_fn=activation_fn
        )

    def get_thrs(self) -> Tuple[Tuple[float, ...], ...]:
        return tuple(
            tuple(thrs[0] for _ in range(self.kh_ax_dim))
            + tuple(thrs[1] for _ in range(self.kh_ax_dim))
            # can be added: + tuple(thrs[2] for _ in range(self.socextra_dim))
            + tuple(thrs_extra[min(i, len(thrs_extra) - 1)] for i in range(self.extra_dim))
            for i, (thrs, thrs_extra) in enumerate(zip(self.thrs_, self.thrs_extra))
        )

    def get_axes(self) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]:
        axes = tuple(tpl(
            N[i],
            AD[i], T[i],  # rot (TAd, TBg)
            inv(R)[i], E[i],  # rot (ERr, ER)
            AD[i], E[i]  # rot (EAd, EBg)
        )[:self.extra_dim] for i in (0, 1))
        return axes[0], axes[1]

    def get_khaxes(self) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]:
        return (KHAX0[:self.kh_ax_dim] + KHAX_OTHER[:self.kh_ax_dim],
                KHAX1[:self.kh_ax_dim] + KHAX_OTHER[:self.kh_ax_dim])

    def get_q_types(self) -> Tuple[Tuple[int, ...], ...]:
        khaxes, axes = self.get_khaxes(), self.get_axes()
        return (khaxes[1] + axes[1]  # MAP POS
                + khaxes[0] + axes[0])  # MAP NEG

    def transform2(self, z_trans1: Tensor) -> Tensor:
        z_khax = z_trans1[:, :self.kh_ax_dim]
        return tr.cat([z_khax, z_trans1], dim=1)

    def forward_(self, x: Tensor, y: Tensor=None) -> Tuple[Tensor, Tensor]:
        return super(ClassifierJATSAxesAlignKhAx, self).forward_(self.subdecoder.z_ext_for_cls(x), y)


class ClassifierJATSAxesAlignKhAxRot(ClassifierJATSAxesAlignKhAx):
    # this combination gives reference basis axes:
    thrs_extra: Tuple[Tuple[float, ...], ...] = (
        (1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2),
        (1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5),
        (1.8, 1.8, 1.8, 1.8, 1.8, 1.5, 1.5),
    )

    @staticmethod
    def get_transform() -> Tensor:
        """
        >>> '''
        >>> [IRr] {Rr} [ERr]
        >>>       I=>{E}
        >>>  IR    R    ER
        >>>
        >>> [ER] {E} [ERr]
        >>>     R=>{Rr}
        >>>  IR   I   IRr
        >>>
        >>> [IAd] {Ad} [EAd]
        >>>       I=>{E}
        >>>  IBg   Bg   EBg
        >>>
        >>> [EBg] {E} [EAd]
        >>>      Bg=>{Ad}
        >>>  IBg   I   IAd
        >>>
        >>> [TBg] {T} [TAd]
        >>>      Bg=>{Ad}
        >>>  FBg   F   FAd
        >>>
        >>> [ST] {T} [NT]
        >>>      S=>{N}
        >>>  SF   F   NF
        >>> '''

        Original axes:
        a. IR=>[ERr]
        b. ER=>[IRr]

        a. IR=>[ERr]
        b. IRr=>[ER] !!!

        a. IBg=>[EAd]
        b. EBg=>[IAd]

        a. IBg=>[EAd]
        b. IAd=>[EBg] !!!

        a. FBg=>[TAd]
        b. FAd=>[TBg] !!!

        a. SF=>[NT] !!!
        b. NF=>[ST]

        New axes after rot(-45):
        a. I=>{E}
        b. R=>{Rr}

        a. R=>{Rr}
        b. I=>{E} !!!

        a. I=>{E}
        b. Bg=>{Ad}

        a. Bg=>{Ad}
        b. I=>{E} !!!

        a. Bg=>{Ad}
        b. F=>{T} !!!

        a. S=>N !!!
        b. F=>T
        """
        cos, sin, π = math.cos, math.sin, math.pi
        rot45 = tr.tensor([[cos(π / 4), -sin(π / 4)],
                           [sin(π / 4), +cos(π / 4)]])
        rot_inv_45 = rot45.transpose(0, 1)
        return rot_inv_45

    def rot_transform(self, z_extra: Tensor) -> Tensor:
        return tr.cat([
            z_extra[:, (0,)],
            z_extra[:, (1, 2)] @ self.transform,
            z_extra[:, (3, 4)] @ self.transform,
            z_extra[:, (5, 6)] @ self.transform,
            # z_extra[:, :5],
        ], dim=1)

    def transform2(self, z_trans1: Tensor) -> Tensor:
        z_khax = z_trans1[:, :self.kh_ax_dim]
        z_extra = z_trans1[:, self.kh_ax_dim:]
        return tr.cat([z_khax, z_khax, self.rot_transform(z_extra)], dim=1)


class ClassifierJATSAxesAlignKhAxRot3(ClassifierJATSAxesAlignKhAxRot):
    # part 0 and 1
    thrs_: Tuple[Tuple[float, ...], ...] = ((0., 1.), (0.5, 1.), (1.8, 1.8))
    # ρ =                                   0         1          2
    thrs_extra: Tuple[Tuple[float, ...], ...] = (
        (1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2),
        (1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5),
        (1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8),
    )


class ClassifierJATSAxesAlign(ClassifierJATSAxesAlignKhAxRot3):  # TODO
    pass
    # ClassifierJATSAxesAlignKhAxRot  ClassifierJATSAxesAlignKhAxRot2  !ClassifierJATSAxesAlignKhAx5
    # ClassifierJATSAxesAlignNTDummy  ClassifierJATSAxesAlignKhAx  ClassifierJATSAxesAlignKhAx3
    # ClassifierJATSAxesAlignKhAxSemi6  ClassifierJATSAxesAlignKhAxRot  ClassifierJATSAxesAlignKhAx4ExtraPairs2
    # ClassifierJATSAxesAlign8TypeAxRot
    # ClassifierJATSAxesAlign16PsiAx
    # ClassifierJATSAxesAlign16TypeAx
    # !ClassifierJATSAxesAlignKhAxPerTypeNegLogP  !ClassifierJATSAxesAlignKhAxRot2 !ClassifierJATSAxesAlignKhAxRot3
    # experimental                                free mode                         relearn fixed mode


class ClassifierJATSAxesAlignDecSwap(ClassifierJATSAxesAlign):
    # from:                                 35,       36,37,43; 43,43...  44,44,... 45,45,...  42,42,...   41,41,...
    thrs_: Tuple[Tuple[float, ...], ...] = ((0., 1.), (0., 0.5),          (0., 1.), (0.5, 1.), (0.7, 1.2), (1., 1.2))
    # ρ =                                   0         1                   2         3          4           5
    thrs_extra: Tuple[Tuple[float, ...], ...] = (
        (1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2),
        (1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5),
        (1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8),
        (1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8),
        (1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8),
        (1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8),
    )


class ClassifierJATSAxesAlignKhAxRotPerTypeNegLogP(ClassifierJATSAxesAlignKhAxRot):
    # this combination gives reference reconstruction error:
    thrs_: Tuple[Tuple[float, ...], ...] = ((0.5,), (1.,), (1.,))  # was 0., 0.5, 0.5
    # this combination gives reference basis axes:
    thrs_extra: Tuple[Tuple[float, ...], ...] = (
        (1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2),
        (1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5),
        (1.8, 1.8, 1.8, 1.8, 1.8, 1.5, 1.5),
    )

    def get_khaxes(self) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]:
        return KHAX0[:self.kh_ax_dim], KHAX1[:self.kh_ax_dim]

    def transform2(self, z_trans1: Tensor) -> Tensor:
        z_khax = z_trans1[:, :self.kh_ax_dim]
        z_extra = z_trans1[:, self.kh_ax_dim:]
        return tr.cat([z_khax, self.rot_transform(z_extra)], dim=1)

    def get_thrs(self) -> Tuple[Tuple[float, ...], ...]:
        return tuple(
            tuple(thrs[0] for _ in range(self.kh_ax_dim))
            + tuple(thrs_extra[min(i, len(thrs_extra) - 1)] for i in range(self.extra_dim))
            for i, (thrs, thrs_extra) in enumerate(zip(self.thrs_, self.thrs_extra))
        )


# thrs_ =
#   was ((0., 1., 1.), (0.5, 1., 1.), (1., 1., 1.))
#   was ..., (1., 1., 1.)); was ..., (1.5, 1.5, 1.5))
#   changed to ((0., 1., 1.), (0.5, 1., 1.), (1.2, 1.2, 1.2))
#     was ((0., 1., 1.), (0.5, 1., 1.), (0.5, 1., 1.))
#     was ((0., 1., 1.), (0.5, 1., 1.), (1., 1., 1.))
#     was ((0., 1., 1.), (0.5, 1., 1.), (3.5, 3.5, 3.5))
#   ??? ((0., 0.7, 0.7), (0.5, 0.7, 0.7), (0.5, 0.7, 0.7))
# thrs_extra = (
#   was ..., (3.5, 3.5, 3.5, 1.8, 1.8, 1.8, 1.8),)
#   was ..., (3.5, 3.5, 3.5, 1.8, 1.8, 3.5, 3.5),)


class ClassifierJATSAxesAlignKhAx20ExtraPairs(ClassifierJATSAxesAlignKhAxRot):
    def get_thrs(self) -> Tuple[Tuple[float, ...], ...]:
        return tuple(
            tuple(thrs[0] for _ in range(self.kh_ax_dim))
            + tuple(thrs[1] for _ in range(self.kh_ax_dim))
            + tuple(thrs[0] for _ in range(self.socextra_dim))
            + tuple(thrs[1] for _ in range(self.socextra_dim))
            + tuple(thrs_extra[min(i, len(thrs_extra) - 1)] for i in range(self.extra_dim))
            for i, (thrs, thrs_extra) in enumerate(zip(self.thrs_, self.thrs_extra))
        )

    def get_q_types(self) -> Tuple[Tuple[int, ...], ...]:
        assert self.kh_ax_dim == 8
        assert self.socextra_dim == 20
        o = missing_types
        socextra1, socextra0, socextra_o = tuple(
            tpl(ENT[i], INT[i], ENF[i], INF[i],
                inv(SFR)[i], NTR[i], inv(STR)[i], NFR[i],
                inv(SFBG)[i], NTBG[i], inv(STBG)[i], NFBG[i],
                inv(IFBG)[i], inv(EFBG)[i], ETBG[i], ITBG[i],
                inv(FRBG)[i], TRAD[i], inv(FRAD)[i], TRBG[i])
            for i in (1, 0)) + ((
                                    o(ENT), o(INT), o(ENF), o(INF),
                                    o(inv(SFR)), o(NTR), o(inv(STR)), o(NFR),
                                    o(inv(SFBG)), o(NTBG), o(inv(STBG)), o(NFBG),
                                    o(inv(IFBG)), o(inv(EFBG)), o(ETBG), o(ITBG),
                                    o(inv(FRBG)), o(TRAD), o(inv(FRAD)), o(TRBG)
            ),)
        khaxes, axes = self.get_khaxes(), self.get_axes()
        return (khaxes[1] + socextra1 + socextra_o + axes[1]  # MAP POS
                + khaxes[0] + socextra0 + socextra_o + axes[0])  # MAP NEG

    def transform2(self, z_trans1: Tensor) -> Tensor:
        z_khax = z_trans1[:, :self.kh_ax_dim]
        z_socextra = z_trans1[:, self.kh_ax_dim:self.kh_ax_dim + self.socextra_dim]
        z_extra = z_trans1[:, self.kh_ax_dim + self.socextra_dim:]
        return tr.cat([z_khax, z_khax, z_socextra, z_socextra, self.rot_transform(z_extra)], dim=1)


class ClassifierJATSAxesAlignKhAx4ExtraPairs(ClassifierJATSAxesAlignKhAx20ExtraPairs):
    def get_q_types(self) -> Tuple[Tuple[int, ...], ...]:
        assert self.kh_ax_dim == 8
        assert self.socextra_dim == 4
        o = missing_types
        socextra1, socextra0, socextra_o = tuple(
            tpl(inv(SFBG)[i], NTBG[i], inv(STBG)[i], NFBG[i])
            for i in (1, 0)) + ((
                                    o(inv(SFBG)), o(NTBG), o(inv(STBG)), o(NFBG)
            ),)
        khaxes, axes = self.get_khaxes(), self.get_axes()
        return (khaxes[1] + socextra1 + socextra_o + axes[1]  # MAP POS
                + khaxes[0] + socextra0 + socextra_o + axes[0])  # MAP NEG


class ClassifierJATSAxesAlignKhAx4ExtraPairs2(ClassifierJATSAxesAlignKhAx20ExtraPairs):
    def get_q_types(self) -> Tuple[Tuple[int, ...], ...]:
        assert self.kh_ax_dim == 8
        assert self.socextra_dim == 4
        o = missing_types
        socextra1, socextra0, socextra_o = tuple(
            tpl(ETAD[i], ITAD[i], EFAD[i], IFAD[i])
            for i in (1, 0)) + ((
                                    o(ETAD), o(ITAD), o(EFAD), o(IFAD)
            ),)
        khaxes, axes = self.get_khaxes(), self.get_axes()
        return (khaxes[1] + socextra1 + socextra_o + axes[1]  # MAP POS
                + khaxes[0] + socextra0 + socextra_o + axes[0])  # MAP NEG


class ClassifierJATSAxesAlign16PsiAx(ClassifierJATSAxesAlignKhAx):
    def get_q_types(self) -> Tuple[Tuple[int, ...], ...]:
        assert self.kh_ax_dim == 16
        psi1, psi0, psi_missing = (tuple(psi[1] for psi in PSI),
                                   tuple(psi[0] for psi in PSI),
                                   tuple(missing_types(psi) for psi in PSI),)
        axes = self.get_axes()
        return (psi1 + psi_missing + axes[1]  # MAP POS
                + psi0 + psi_missing + axes[0])  # MAP NEG


class ClassifierJATSAxesAlign16TypeAx(ClassifierJATSAxesAlignKhAx):
    def get_q_types(self) -> Tuple[Tuple[int, ...], ...]:
        assert self.kh_ax_dim == 16
        tp1, tp0, tp_missing = (tuple(tp[1] for tp in TYPE16),
                                tuple(tp[0] for tp in TYPE16),
                                tuple(missing_types(tp) for tp in TYPE16),)
        axes = self.get_axes()
        return (tp1 + tp_missing + axes[1]  # MAP POS
                + tp0 + tp_missing + axes[0])  # MAP NEG


class ClassifierJATSAxesAlignKhAx3(ClassifierJATSAxesAlignKhAx):
    thrs_extra: Tuple[Tuple[float, ...], ...] = ((0.5, 0.5, 0.5), (1.5, 1.5, 1.),)


T1, T1_, T101, T05, T12, T15, T2, T22, T22_, T3 = 1., 1., 1.01, 0.5, 1.2, 1.7, 2., 2.2, 2.4, 3.
# T1, T1_, T101, T05, T1___, T1____, T2, T2_, T2__, T3 = 1., 1., 1.01, 0.5, 1., 1., 2., 2., 2., 3.
# T1, T15, T13, T13_, T13__, T23, T2, T2_ = 1., 1.5, 1.3, 1.3, 1.3, 2.3, 2., 2.
# T1, T15, T11, T13, T13_, T18_, T2, T2_ = 1., 1.5, 1.1, 1.3, 1.3, 1.8, 2., 2.
# T1, T15, T11, T13, T18, T18_, T2, T23 = 1., 1.5, 1.1, 1.3, 1.8, 1.8, 2., 2.3
# T1, T15, T12, T13, T23, T18 = 1., 1.5, 1.2, 1.3, 2.3, 1.8
# 1., 1.5, 1.19, 1.2, 2.3, 1.8
# good: 1., 1.5, 1.3, 1.6, 2.3, 1.8
# bad: 1., 1.5, 1.1, 1.2, 2.3, 1.8


class ClassifierJATSAxesAlignKhAxSemi6(ClassifierJATSAxesAlignKhAx):
    thrs_extra: Tuple[Tuple[float, ...], ...] = (
        # S=>N F=>T IR=>ERr IRr=>ER:
        ((T1, T1, T05, T2, T05, T2), (T1, T1, T12, T22, T12, T22), (T1_, T1_, T15, T22_, T15, T22_),),
        # S=>N Bg=>Ad Bg=>Ad F=>T IR=>ERr IRr=>ER Bg-12=>Ad+12:
        ((T1, T1, T1, T1, T05, T2, T05, T2, T1), (T1, T101, T101, T1, T12, T22, T12, T22, T3),
         (T1_, T101, T101, T1_, T15, T22_, T15, T22_, T3),),
        # SBg=>NAd SAd=>NBg F=>T IR=>ERr IRr=>ER:
        ((T05, T22_, T05, T2, T1, T05, T2, T05, T2), (T12, T22, T12, T22, T1, T12, T22, T12, T22),
         (T15, T22_, T15, T22_, T1_, T15, T22_, T15, T22_),),
        # S=>N F=>T IRr=>ER:
        ((T1, T1, T05, T2), (T1, T1, T12, T22), (T1_, T1_, T15, T22_),),
        # S=>N F=>T IR=>ERr IRr=>ER IBg=>EAd IAd=>EBg:
        ((T1, T1, T05, T2, T05, T2, T05, T2, T05, T2), (T1, T1, T12, T22, T12, T22, T12, T22, T12, T22),
         (T1_, T1_, T15, T22_, T15, T22_, T15, T22_, T15, T22_),),
    )[SEMI]

    def get_q_types(self) -> Tuple[Tuple[int, ...], ...]:
        return (
                KHAX1[:self.kh_ax_dim] + KHAX_OTHER[:self.kh_ax_dim] + SEMI1[:self.extra_dim]  # MAP POS
                + KHAX0[:self.kh_ax_dim] + KHAX_OTHER[:self.kh_ax_dim] + SEMI0[:self.extra_dim]  # MAP NEG
        )


class ClassifierJATSAxesAlignKhAxSemi6AKhMod(ClassifierJATSAxesAlignKhAxSemi6):
    thrs_: Tuple[Tuple[float, ...], ...] = ((0.5, 0., 1.5, 1., 1.), (0.5, 0.5, 1.5, 1., 1.),)
    # 578: ((0., 0., 1., 1., 1.), (0.5, 0.5, 1.5, 1., 1.),)

    def get_thrs(self) -> Tuple[Tuple[float, ...], ...]:
        akh_ax_dim = {14: 6, 20: 8}[self.kh_ax_dim]  # 6+8=14, 8+12=20
        kh_ax_dim = {14: 8, 20: 12}[self.kh_ax_dim]
        return tuple(
            tuple(thrs[0] for _ in range(akh_ax_dim)) + tuple(thrs[1] for _ in range(kh_ax_dim))
            + tuple(thrs[2] for _ in range(akh_ax_dim)) + tuple(thrs[3] for _ in range(kh_ax_dim))
            # + tuple(thrs[4] for _ in range(self.socextra_dim))
            + tuple(thrs_extra[min(i, len(thrs_extra) - 1)] for i in range(self.extra_dim))
            for i, (thrs, thrs_extra) in enumerate(zip(self.thrs_, self.thrs_extra))
        )

    def get_q_types(self) -> Tuple[Tuple[int, ...], ...]:
        akh_ax_dim = {14: 6, 20: 8}[self.kh_ax_dim]
        kh_ax_dim = {14: 8, 20: 12}[self.kh_ax_dim]
        return (
                AKHAX1[:akh_ax_dim] + KHAX1[:kh_ax_dim]
                + AKHAX_OTHER[:akh_ax_dim] + KHAX_OTHER[:kh_ax_dim] + SEMI1[:self.extra_dim]  # MAP POS
                + AKHAX0[:akh_ax_dim] + KHAX0[:kh_ax_dim]
                + AKHAX_OTHER[:akh_ax_dim] + KHAX_OTHER[:kh_ax_dim] + SEMI0[:self.extra_dim]  # MAP NEG
        )


class PerTypeNegLogP(ModuleXOptYToX):  # TODO
    idxs: Tuple[int, ...] = (-3, -2)  # mind sex or it's absence
    scl: float = 0.2  # 0.2  0.15

    zero: Tensor  # buffer
    dists16: List[Distrib]
    dists: nn.ModuleList

    def __init__(self):
        super(PerTypeNegLogP, self).__init__()
        self.register_buffer('zero', tr.tensor([0.]).mean())
        dists, self.dists16 = self.get_dists_dists16()
        self.dists = nn.ModuleList(dists)

    # noinspection PyMethodMayBeStatic
    def get_dists_dists16(self) -> Tuple[List[Distrib], List[Distrib]]:
        """ ``dists16`` would be used as ``cycle(dists16)`` """
        dist = Normal()
        return [dist], [dist]

    def forward_(self, x: Tensor, y: Tensor=None) -> Tensor:
        if (y is None) or (self.scl <= 0):
            return self.zero
        z = x[:, self.idxs]
        y_int = y.max(dim=1)[1]  # y_one_hot => y_int
        neg_log_p = [
            -dist.log_p(x=zi, p_params=dist.get_params(zi.size())).view(zi.shape[0], -1).sum(dim=1).sum()
            for (j, zi), dist in zip(((i, z[y_int == i]) for i in range(16)), cycle(self.dists16))
            if zi.shape[0] > 0
        ]  # list of tensors of shape (batch_subset_size,)
        return sum(neg_log_p) * (self.scl / z.shape[0])


class PerTypeNegLogPBimodalNT(PerTypeNegLogP):
    idxs: Tuple[int, ...] = (1, 2)
    _dist: ZeroSymmetricBimodalNormal

    def get_dists_dists16(self) -> Tuple[List[Distrib], List[Distrib]]:
        self._dist = ZeroSymmetricBimodalNormal(z_dim=2, μ=(1.1, 1.2), σ=(6, 6),
                                                # μ=(0.5, 0.5), σ=(0.83, 0.83)
                                                # μ=(1., 1.), σ=(0.67, 0.67),
                                                learnable_μ=False, learnable_σ=True, mode=1)
        dists_ = [
            ZeroSymmetricBimodalNormalTwin(z_dim=2, μ=μ, σ=(1, 1),
                                           learnable_μ=False, learnable_σ=False, mode=1)
            for μ in ((-1, -1), (-1, 1), (1, -1), (1, 1))
        ]
        for dist in dists_:
            dist.set_twin_dist(self._dist)
        dists: List[Distrib] = [dist for dist in dists_]

        return dists, [dists[3], dists[3], dists[0], dists[0],
                       dists[1], dists[1], dists[2], dists[2],
                       dists[0], dists[0], dists[3], dists[3],
                       dists[2], dists[2], dists[1], dists[1]]


class PerTypeNegLogPKhAx12(PerTypeNegLogP):
    idxs: Tuple[int, ...] = tuple(range(12))
    type_indep_dims: Tuple[Tuple[int, ...], ...]  # (16, 8)
    scl: float = 5

    def __init__(self):
        super(PerTypeNegLogPKhAx12, self).__init__()
        asymmetric_types = (
                (1, 13, 6, 10),  (2, 14, 5, 9),  (7, 11,  4, 16), (8, 12,  3, 15),
                (1, 5,  10, 14), (2, 6,  9, 13), (11, 15, 4, 8),  (12, 16, 3, 7),
                (1, 9,  14, 6),  (2, 10, 13, 5), (15, 7,  4, 12), (16, 8,  3, 11),
            )
        self.type_indep_dims = tuple(
            tuple(i for i in range(12) if type_ not in asymmetric_types[i])
            for type_ in range(1, 17)
        )

    def forward_(self, x: Tensor, y: Tensor=None) -> Tensor:
        if (y is None) or (self.scl <= 0):
            return self.zero
        z = x[:, self.idxs]
        y_int = y.max(dim=1)[1]  # y_one_hot => y_int
        neg_log_p = [
            -dist.log_p(x=zi, p_params=dist.get_params(zi.size())).view(zi.shape[0], -1).sum(dim=1).sum()
            for (j, zi), dist in zip(
                ((i, z[y_int == i][:, self.type_indep_dims[i]]) for i in range(16)),
                cycle(self.dists16)
            ) if zi.shape[0] > 0
        ]  # list of tensors of shape (batch_subset_size,)
        return sum(neg_log_p) * (self.scl / z.shape[0])


class ClassifierJATSAxesAlignKhAxPerTypeNegLogP(ClassifierJATSAxesAlignKhAxRotPerTypeNegLogP):  # TODO
    # ClassifierJATSAxesAlignKhAx2 ClassifierJATSAxesAlignKhAx
    # ClassifierJATSAxesAlignKhAxRot3 ClassifierJATSAxesAlignKhAxRotPerTypeNegLogP
    per_type_neg_log_p: PerTypeNegLogP

    def __post_init__(self):
        super(ClassifierJATSAxesAlignKhAxPerTypeNegLogP, self).__post_init__()
        self.per_type_neg_log_p = PerTypeNegLogPKhAx12()  # PerTypeNegLogP PerTypeNegLogPBimodalNT PerTypeNegLogPKhAx12

    def forward_(self, x: Tensor, y: Tensor=None) -> Tuple[Tensor, Tensor]:
        z_for_cls = self.subdecoder.z_ext_for_cls(x)
        p, ce = super(ClassifierJATSAxesAlignKhAx, self).forward_(z_for_cls, y)
        return p, ce + self.per_type_neg_log_p.__call__(z_for_cls, y)

    def __forward__(self, x: Tensor, y: Tensor=None) -> Tuple[Tensor, Tensor]:  # OFF
        p, ce = super(ClassifierJATSAxesAlignKhAxPerTypeNegLogP, self).forward_(x, y)
        return p, ce + self.per_type_neg_log_p.__call__(x, y)

    def __set_ρ__(self, ρ: int) -> None:  # OFF
        super(ClassifierJATSAxesAlignKhAxPerTypeNegLogP, self).set_ρ(ρ)
        self.per_type_neg_log_p.scl = self.per_type_neg_log_p.scl if (ρ < 2) else -0.000001


class ClassifierJATSAxesAlignKhAx1(ClassifierJATSAxesAlignKhAx):
    thrs_extra: Tuple[Tuple[float, ...], ...] = (
        (0.5, 0.5, 0.), (1.5, 1.5, 0.5),
    )

    def get_q_types(self) -> Tuple[Tuple[int, ...], ...]:
        return (
                KHAX1[:self.kh_ax_dim] + KHAX_OTHER[:self.kh_ax_dim] + (N[1], T[1], E[1], E[1])  # MAP POS
                + KHAX0[:self.kh_ax_dim] + KHAX_OTHER[:self.kh_ax_dim] + (N[0], T[0], E[0], E[0])  # MAP NEG
        )

    def get_thrs(self) -> Tuple[Tuple[float, ...], ...]:
        return tuple(
            tuple(thrs[0] for _ in range(self.kh_ax_dim))
            + tuple(thrs[1] for _ in range(self.kh_ax_dim))
            # + tuple(thrs[2] for _ in range(self.socextra_dim))
            + tuple(thrs_extra[min(i, len(thrs_extra) - 1)] for i in range(self.extra_dim + 1))
            for i, (thrs, thrs_extra) in enumerate(zip(self.thrs_, self.thrs_extra))
        )

    def transform2(self, z_trans1: Tensor) -> Tensor:
        z_khax = z_trans1[:, :self.kh_ax_dim]
        return tr.cat([z_khax, z_trans1, z_trans1[:, (-1,)]], dim=1)


class ClassifierJATSAxesAlign8TypeAxRot(ClassifierJATSAxesAlignKhAxRot):
    # ClassifierJATSAxesAlignKhAx ClassifierJATSAxesAlignKhAxRot
    def get_q_types(self) -> Tuple[Tuple[int, ...], ...]:
        assert self.kh_ax_dim == 8
        type1, type0, type_missing = (tuple(type_[1] for type_ in TYPE8),
                                      tuple(type_[0] for type_ in TYPE8),
                                      tuple(missing_types(type_) for type_ in TYPE8),)
        axes = self.get_axes()
        return (
            type1 + type_missing + axes[1]  # MAP POS
            + type0 + type_missing + axes[0]  # MAP NEG
        )


class ClassifierJATSAxesAlignAKhAx(ClassifierJATSAxesAlignKhAx):
    # thrs_extra: Tuple[Tuple[float, ...], ...] = ((1., 1., 1.3, 2.3, 1.3, 2.3), (1., 1., 1.6, 2.3, 1.6, 2.3),)

    def get_q_types(self) -> Tuple[Tuple[int, ...], ...]:
        return (
                AKHAX1[:self.kh_ax_dim] + AKHAX_OTHER[:self.kh_ax_dim] + SEMI1[:self.extra_dim]  # MAP POS
                + AKHAX0[:self.kh_ax_dim] + AKHAX_OTHER[:self.kh_ax_dim] + SEMI0[:self.extra_dim]  # MAP NEG
        )


class ClassifierJATSAxesAlignKhAxPlus(ClassifierJATSAxesAlignKhAx):
    thrs_: Tuple[Tuple[float, ...], ...] = ((0., 1., 0., 0.5), (0.5, 1., 0., 1.),)

    def get_q_types(self) -> Tuple[Tuple[int, ...], ...]:
        assert self.kh_ax_dim == 12
        assert self.socextra_dim == 15
        axes = self.get_axes()
        return (KHAX1 + KHAX_OTHER + RNKHAX1 + COMRNKHAX1 + axes[1]  # MAP POS
                + KHAX0 + KHAX_OTHER + RNKHAX0 + COMRNKHAX0 + axes[0])  # MAP NEG


# noinspection PyPep8Naming
class __ClassifierJATSAxesAlign12KhAx12ENTLikeAx__(ClassifierJATSAxesAlignKhAx):
    def get_q_types(self) -> Tuple[Tuple[int, ...], ...]:
        assert self.kh_ax_dim == 12
        o = missing_types
        return (
                KHAX1 + (ENT[1], inv(INT)[1], inv(INF)[1], ENF[1],
                         ETQ[1], inv(ITQ)[1], inv(IFQ)[1], EFQ[1],
                         ENQ[1], inv(INQ)[1], ESQ[1], inv(ISQ)[1])
                + KHAX_OTHER + (o(ENT), o(inv(INT)), o(inv(INF)), o(ENF), o(ETQ), o(inv(ITQ)), o(inv(IFQ)), o(EFQ),
                                o(ENQ), o(inv(INQ)), o(ESQ), o(inv(ISQ))) + (N[1], T[1])  # MAP POS
                + KHAX0 + (ENT[0], inv(INT)[0], inv(INF)[0], ENF[0],
                           ETQ[0], inv(ITQ)[0], inv(IFQ)[0], EFQ[0],
                           ENQ[0], inv(INQ)[0], ESQ[0], inv(ISQ)[0])
                + KHAX_OTHER + (o(ENT), o(inv(INT)), o(inv(INF)), o(ENF), o(ETQ), o(inv(ITQ)), o(inv(IFQ)), o(EFQ),
                                o(ENQ), o(inv(INQ)), o(ESQ), o(inv(ISQ))) + (N[0], T[0])  # MAP NEG
        )


class ClassifierJATSAxesAlign12KhAx1StatAx8RnKhAx(ClassifierJATSAxesAlignKhAx):
    thrs_: Tuple[Tuple[float, ...], ...] = ((0., 1., 0., 0.5), (0.5, 1., 1., 1.),)
    map_ax_dim: int = 9 + 7

    def get_q_types(self) -> Tuple[Tuple[int, ...], ...]:
        assert self.kh_ax_dim == 12
        return (
            KHAX1 + KHAX_OTHER + RNKHAX1 + COMRNKHAX1 + (N[1], T[1])  # MAP POS
            + KHAX0 + KHAX_OTHER + RNKHAX0 + COMRNKHAX0 + (N[0], T[0])  # MAP NEG
        )

    def get_thrs(self) -> Tuple[Tuple[float, ...], ...]:
        return tuple(
            tuple(thrs[0] for _ in range(self.kh_ax_dim))
            + tuple(thrs[1] for _ in range(self.kh_ax_dim))
            + tuple(thrs[2] for _ in range(self.map_ax_dim))
            + tuple(thrs[3] for _ in range(self.extra_dim))
            for thrs in self.thrs_
        )

    def transform2(self, z_trans1: Tensor) -> Tensor:
        z_khax = z_trans1[:, :self.kh_ax_dim]
        z_extra = z_trans1[:, self.kh_ax_dim:]
        z_ornkhax = self.subdecoder.map_khax2rnax.__call__(z_khax)
        return tr.cat([z_khax, z_khax, z_ornkhax, z_extra], dim=1)


class ClassifierJATSAxesAlign12KhAx1StatAx(ClassifierJATSAxesAlign12KhAx1StatAx8RnKhAx):
    thrs_: Tuple[Tuple[float, ...], ...] = ((0., 1., 0., 0.5), (0.5, 1., 0.5, 1.),)
    map_ax_dim: int = 1

    def get_q_types(self) -> Tuple[Tuple[int, ...], ...]:
        assert self.kh_ax_dim == 12
        return (
                KHAX1 + KHAX_OTHER + (STATIC[1], N[1], T[1])  # MAP POS
                + KHAX0 + KHAX_OTHER + (STATIC[0], N[0], T[0])  # MAP NEG
        )


class ClassifierJATSAxesAlignKhAx1StatAxBinary(ClassifierJATSAxesAlignKhAx):
    # ClassifierJATSAxesAlignKhAx2 ClassifierJATSAxesAlignKhAx
    oy_mult: float = 0.25  # was 1; was 0.66; was 2; was 1; was 0.5;

    def set_ρ(self, ρ: int) -> None:
        super(ClassifierJATSAxesAlignKhAx1StatAxBinary, self).set_ρ(ρ)
        if ρ <= 0:
            self.oy_mult = 1
        elif ρ == 1:
            self.oy_mult = 0.6
        else:
            self.oy_mult = 0.3

    def axes_cross_entropy(self, z_trans1: Tensor, z_trans2: Tensor, y: Tensor) -> Tensor:
        axes_cross_entropy = super(ClassifierJATSAxesAlignKhAx1StatAxBinary, self).axes_cross_entropy(
            z_trans1, z_trans2, y)
        if self.oy_mult <= 0:
            return axes_cross_entropy

        z_khax = z_trans1[:, :self.kh_ax_dim]
        oy_cross_entropy = self.subdecoder.map_khax_to_oy_ce_reduced.__call__(z_khax, y) * self.oy_mult
        return axes_cross_entropy + oy_cross_entropy


class DecoderPassthroughTwinSex(DecoderPassthroughTwin):
    SubDecoder_: Type[SubDecoder] = SubDecoder  # should be non-dynamic property

    def neg_log_p(self, x: Tensor) -> Opt[Tensor]:
        return neg_log_standard_bernoulli(x).view(x.shape[0], -1).sum(dim=1)


class DecoderPassthrTwinSexKhAx0(DecoderPassthroughTwinSex):
    subdecoder: SubDecoderPassthrSocAxTo12KhAx
    prefix_hidden_dims: Tuple[int, ...] = (64, 64, 64)

    def __init__(self, dims: Tuple[int, Tuple[int, ...], int],
                 passthrough_dim: int=1,
                 activation_fn: Callable[[Tensor], Tensor]=tr.selu,
                 output_activation: Opt[Callable[[Tensor], Tensor]]=None):
        latent_dim, hidden_dims, input_dim = dims
        with self.disable_final_init():
            super(DecoderPassthrTwinSexKhAx0, self).__init__(
                dims=(latent_dim, self.prefix_hidden_dims + hidden_dims, input_dim), passthrough_dim=passthrough_dim,
                activation_fn=activation_fn, output_activation=output_activation)
        self.subdecoder = self.SubDecoder_(passthrough_dim=self.passthr_dim)
        self.__final_init__()


class DecoderPassthrTwinSexKhAx1(DecoderPassthroughTwinSex):
    subdecoder: SubDecoderPassthrSocAxTo12KhAx

    def __init__(self, dims: Tuple[int, Tuple[int, ...], int],
                 passthrough_dim: int=1,
                 activation_fn: Callable[[Tensor], Tensor]=tr.selu,
                 output_activation: Opt[Callable[[Tensor], Tensor]]=None):
        latent_dim, hidden_dims, input_dim = dims
        sd = self.SubDecoder_
        latent_dim += sd.khax_trans_dim + sd.socextra_dim - sd.socax_dim + sd.socpass_dim
        with self.disable_final_init():
            super(DecoderPassthrTwinSexKhAx1, self).__init__(
                dims=(latent_dim, hidden_dims, input_dim), passthrough_dim=passthrough_dim,
                activation_fn=activation_fn, output_activation=output_activation)
        self.subdecoder = self.SubDecoder_(passthrough_dim=self.passthr_dim)
        self.__final_init__()

    def extend_x(self, x: Tensor) -> Tensor:
        # SubDecoder.khax_trans_dim + SubDecoder.socextra_dim - SubDecoder.socax_dim
        z_ext_ind = super(DecoderPassthrTwinSexKhAx1, self).extend_x(x)
        z_ext_trans_dep = self.subdecoder.__call__(z_ext_ind)
        return z_ext_trans_dep

    def _extend_x__1(self, x: Tensor) -> Tensor:  # OFF
        # SubDecoder.khax_trans_dim * 2 + SubDecoder.socextra_dim - SubDecoder.socax_dim
        z_ext_ind = super(DecoderPassthrTwinSexKhAx1, self).extend_x(x)
        z_ext_trans_dep = self.subdecoder.__call__(z_ext_ind)
        n = self.passthr_dim + self.subdecoder.khax_dim
        s = z_ext_trans_dep[:, :self.passthr_dim]
        z_khax = z_ext_trans_dep[:, self.passthr_dim:n]
        z_extra = z_ext_trans_dep[:, n:]
        return tr.cat([s, z_khax, self.subdecoder.map_khax2rnax.abs(z_khax), z_extra], dim=1)

    def _extend_x__2(self, x: Tensor) -> Tensor:  # OFF
        # SubDecoder.khax_trans_dim + SubDecoder.socextra_dim
        z_ext_ind = super(DecoderPassthrTwinSexKhAx1, self).extend_x(x)
        z_trans_dep = self.subdecoder.__call__(z_ext_ind)[:, self.passthr_dim:]
        return tr.cat([z_ext_ind[:, :self.passthr_dim + self.subdecoder.socax_dim], z_trans_dep], dim=1)


class DecoderPassthrTwinSexKhAx(DecoderPassthrTwinSexKhAx1):  # TODO
    # DecoderPassthrTwinSexKhAx0 DecoderPassthrTwinSexKhAx1
    pass


class DecoderPassthrTwinSexKhAxDecSwap(DecoderPassthrTwinSexKhAx):
    SubDecoder_: Type[SubDecoder] = SubDecoderDecSwap  # should be non-dynamic property


class EncoderSplitMixin:
    hidden2: List[nn.Linear]


class EncoderTwinSplit1(EncoderTwin, EncoderSplitMixin):
    hidden2: List[nn.Linear]
    sample2: GaussianSampleTwin
    split_stage: bool = False
    z_subdims1: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    z_subdims2: Tuple[int, ...] = (7,)
    _twin_encoder2: Tuple[EncoderSplitMixin]

    def __post_init__(self):
        super(EncoderTwinSplit1, self).__post_init__()
        MetaTrimmer.__init__(self)

        x_dim, h_dim, z_dim = self.dims
        hidden2, sample2 = self.get_hidden_and_sample(GaussianSampleTwin, z_dim, x_dim, *h_dim)
        assert isinstance(sample2, GaussianSampleTwin)
        self.hidden2, self.sample2 = hidden2, sample2

        self.set_twin_encoder(self)

    def set_twin_encoder(self, encoder: Encoder):
        super(EncoderTwinSplit1, self).set_twin_encoder(encoder)
        assert isinstance(encoder, EncoderTwinSplit1)
        encoder_: EncoderTwinSplit1 = encoder
        self._twin_encoder2 = (encoder_,)
        self.sample2.set_twin_sampler(encoder_.sample2)

    @property
    def hidden2_twin(self) -> List[nn.Linear]:
        return self._twin_encoder2[0].hidden2

    def copy_twins_to_twins2(self) -> None:
        """ Do not use inside guide's step! """
        hidden: nn.ModuleList
        hidden2: nn.ModuleList
        hidden_, hidden2_ = self.hidden_twin, self.hidden2_twin
        assert isinstance(hidden_, nn.ModuleList) and isinstance(hidden2_, nn.ModuleList)
        hidden, hidden2 = hidden_, hidden2_
        hidden2.load_state_dict(hidden.state_dict())

        μ, μ2 = self.sample.μ_twin, self.sample2.μ_twin
        μ2.load_state_dict(μ.state_dict())

    def copy_σ_to_σ2(self) -> None:
        log_σ, log_σ2 = self.sample.log_σ, self.sample2.log_σ
        log_σ2.load_state_dict(log_σ.state_dict())

    def subcat(self, x1: Tensor, x2: Tensor) -> Tensor:
        return tr.cat((x1[:, self.z_subdims1], x2[:, self.z_subdims2]), dim=1)

    def first(self, x: Tensor) -> Tensor:
        """ select first sub-dimensions """
        return x[:, self.z_subdims1]

    def second(self, x: Tensor) -> Tensor:
        """ select second complementary sub-dimensions """
        return x[:, self.z_subdims2]

    def forward_(self, x: Tensor) -> XTupleYi:
        z, (μ, log_σ) = self.subforward(x, self.hidden_twin, self.sample)
        if self.split_stage:
            z2, (μ2, log_σ2) = self.subforward(x, self.hidden2_twin, self.sample2)
            z, μ, log_σ = self.subcat(z, z2), self.subcat(μ, μ2), self.subcat(log_σ, log_σ2)
        self.set_trim(μ, log_σ)
        return z, (μ, log_σ)


class Encoder700(Encoder):
    sample: GaussianSample

    def __init__(self, dims: Tuple[int, Tuple[int, ...], int]):
        super(Encoder700, self).__init__(dims, GaussianSample, tr.selu)


class EncoderTwinFrozen700Base(EncoderTwin):
    # encoders: List[Encoder700]
    encoders2: Tuple[EncoderTwin, ...] = ()
    _encoders2: Opt[nn.ModuleList] = None
    _encoders2inner: Opt[nn.ModuleList] = None

    def set_encoders2_(self, encoders_interface: List[EncoderTwin], encoders_inner: List[EncoderTwin]):
        self.encoders2 = tuple(encoders_interface)
        self._encoders2 = nn.ModuleList(list(encoders_interface))
        self._encoders2inner = nn.ModuleList(list(encoders_inner))


class EncoderTwinFrozen700(EncoderTwinFrozen700Base):
    n_average: int = 6
    inv8_start: int = 4  # counting from 0
    # _encoders: nn.ModuleList
    _twin_encoder: Tuple[EncoderTwinFrozen700Base]
    first_stage: bool = True
    encoders_here: bool = False

    @property
    def twin_encoder(self) -> EncoderTwinFrozen700Base:
        return self._twin_encoder[0]

    def __init__(self, dims: Tuple[int, Tuple[int, ...], int],
                 sample_layer: Type[GaussianSampleTwin]=GaussianSampleTwin,
                 activation_fn: Callable[[Tensor], Tensor]=tr.selu):
        super(EncoderTwin, self).__init__(dims=dims, sample_layer=sample_layer, activation_fn=activation_fn)

    def __post_init__(self):
        super(EncoderTwin, self).__post_init__()
        MetaTrimmer.__init__(self)
        # self.encoders = [Encoder700(dims=self.dims) for _ in range(self.n_average)]
        # noinspection PyTypeChecker
        # self._encoders = nn.ModuleList(self.encoders)
        self.set_twin_encoder(self)
        if self.encoders_here:
            encoders_interface = [EncoderTwin(dims=self.dims) for _ in range(self.n_average)]
            encoders_inner = [EncoderTwin(dims=self.dims) for _ in range(self.n_average)]
            for enc_int, enc_inn in zip(encoders_interface, encoders_inner):
                enc_int.set_twin_encoder(enc_inn)
            self.twin_encoder.set_encoders2_(encoders_interface, encoders_inner)

    def set_encoders2(self, models: List[nn.Module]):
        # assert len(models) == len(self.encoders)
        encoders_interface: List[EncoderTwin] = []
        encoders_inner: List[EncoderTwin] = []
        for model_ in models:
            assert isinstance(model_, VAEPassthroughClassifyTwinMMD)
            enc = model_.encoder
            assert isinstance(enc, EncoderTwin)
            encoders_interface.append(enc)
            twin_encoder = enc.twin_encoder
            assert isinstance(twin_encoder, EncoderTwin)
            encoders_inner.append(twin_encoder)
        self.twin_encoder.set_encoders2_(encoders_interface, encoders_inner)

    def set_twin_encoder(self, encoder: Encoder):
        assert isinstance(encoder, EncoderTwinFrozen700Base)
        self._twin_encoder = (encoder,)
        self.sample.set_twin_sampler(encoder.sample)

    def subforward_(self, x: Tensor, hidden: List[nn.Linear], lin: nn.Linear) -> Tensor:
        h: Tensor
        for layer in hidden:
            h = self.activation_fn.a(layer.__call__(x))
            x = h
        return lin.__call__(x)

    def subforward_μ(self, x: Tensor, encoder:  EncoderTwin) -> Tensor:
        h: Tensor
        for layer in encoder.hidden_twin:
            h = self.activation_fn.a(layer.__call__(x))
            x = h
        return encoder.sample.μ_twin.__call__(x)

    @staticmethod
    def inv8(x: Tensor) -> Tensor:
        return tr.cat([x[:, :7], -x[:, 7:]], dim=1)

    def forward_(self, x: Tensor) -> XTupleYi:
        log_σ = self.subforward_(x, self.twin_encoder.hidden, self.sample.log_σ)
        if self.first_stage:
            μ = self.subforward_(x, self.twin_encoder.hidden, self.sample.μ_twin) * 0.
            # dummy that is never useful and should run only when 38
        else:
            encoders = self.twin_encoder.encoders2
            encoder = encoders[0]
            μ = self.subforward_μ(x, encoder)
            for encoder in encoders[1:self.inv8_start]:
                μ += self.subforward_μ(x, encoder)
            for encoder in encoders[self.inv8_start:]:
                μ += self.inv8(self.subforward_μ(x, encoder))
            μ = μ / self.n_average

            # μ = self.subforward_(x, self.twin_encoder.hidden, self.sample.μ_twin)  # DEBUG
        self.set_trim(μ, log_σ)
        return self.sample.reparametrize(μ, log_σ), (μ, log_σ)

    # @staticmethod
    # def copy(from_: nn.Linear, to_: nn.Linear):
    #     to_.load_state_dict(from_.state_dict())

    # def copy_encoders(self, *models: nn.Module):
    #     assert len(models) == len(self.encoders)
    #     for model_, encoder_ in zip(models, self.encoders):
    #         assert isinstance(model_, VAEPassthroughClassifyTwinMMD)
    #         model_enc_ = model_.encoder
    #         assert isinstance(model_enc_, EncoderTwin)
    #         model_enc = model_enc_
    #         # encoder_.hidden = model_enc.hidden_twin
    #         # encoder_.sample.μ = model_enc.sample.μ_twin
    #         for lin_from, lin_to in zip(model_enc.hidden_twin, encoder_.hidden):
    #             self.copy(lin_from, lin_to)
    #         self.copy(model_enc.sample.μ_twin, encoder_.sample.μ)


class EncoderTwinFrozen700Part2(EncoderTwinFrozen700):
    first_stage: bool = False
    encoders_here: bool = False


class EncoderTwinFrozen700Part2Inner(EncoderTwinFrozen700):
    first_stage: bool = False
    encoders_here: bool = True


class EncoderTwinJATS(EncoderTwin):
    # EncoderTwin EncoderTwinSplit0 EncoderTwinSplit1
    pass


class ClassifierJATSAxesAlignNTRrAdeAdi(ClassifierJATSAxesAlign5RnKhAx):
    axs_ce_dim: int = 5
    thrs: Tuple[Tuple[float, ...], ...] = (
        tuple(1. for _ in range(axs_ce_dim)),
        tuple(1. for _ in range(axs_ce_dim)),
        tuple(1.5 for _ in range(axs_ce_dim)),
    )
    # noinspection PyTypeChecker
    q_types: Tuple[Tuple[int, ...], ...] = (
        # IBg=>EAd, EBg=>IAd
        #     I=>E, Bg=>Ad
        N[1], T[1], inv(R)[1], E[1], AD[1],  # MAP POS
        N[0], T[0], inv(R)[0], E[0], AD[0],  # MAP NEG
    )

    @staticmethod
    def get_transform() -> Tensor:
        """
        >>> '''
        >>> IAd   Ad   EAd
        >>>      I=>E
        >>> IBg   Bg   EBg
        >>> '''

        Original axes:
        0. IBg=>EAd
        1. EBg=>IAd

        New axes after rot(-45):
        0. I=>E
        1. Bg=>Ad
        """
        cos, sin, π = math.cos, math.sin, math.pi
        rot45 = tr.tensor([[cos(π / 4), -sin(π / 4)],
                           [sin(π / 4), +cos(π / 4)]])
        rot_inv_45 = rot45.transpose(0, 1)
        return rot_inv_45

    def transform2(self, z_trans1: Tensor) -> Tensor:
        return tr.cat((z_trans1[:, :3], z_trans1[:, 3:] @ self.transform), dim=1)

    def get_z_new(self, z_ax: Tensor) -> Tensor:
        return tr.cat((self.transform2(z_ax[:, :self.axs_ce_dim]), z_ax[:, self.axs_ce_dim:]), dim=1)


class ClassifierJATSAxesAlignNTRrAdeAdiAd12(ClassifierJATSAxesAlignNTRrAdeAdi):
    axs_ce_dim: int = 6
    thrs: Tuple[Tuple[float, ...], ...] = (
        (1.0, 1.0, 1.0, 1.0, 1.0, 1.8),
        (1.0, 1.0, 1.0, 1.0, 1.0, 2.2),
        (1.5, 1.5, 1.5, 1.5, 1.5, 6),
    )
    # noinspection PyTypeChecker
    q_types: Tuple[Tuple[int, ...], ...] = (
        # IBg=>EAd, EBg=>IAd
        #     I=>E, Bg=>Ad
        N[1], T[1], inv(R)[1], E[1], AD[1], AD12[1],  # MAP POS
        N[0], T[0], inv(R)[0], E[0], AD[0], AD12[0],  # MAP NEG
    )

    def transform2(self, z_trans1: Tensor) -> Tensor:
        return tr.cat((z_trans1[:, :3], z_trans1[:, (3, 4)] @ self.transform, z_trans1[:, 5:]), dim=1)


THRS = (0., 1.5)  # (0., 2.); (0., 1.5); (-0.5, 1.5); (-0.5, 1.)


class ClassifierPassthrJATS24KhCFBase(ClassifierJATSAxesAlign5RnKhAx):
    extra_append_dims: int = 2
    axs_dim: int = 12
    thrs: Tuple[Tuple[float, ...], ...] = (
        tuple(THRS[0] for _ in range(axs_dim)) + tuple(THRS[1] for _ in range(axs_dim)),
    )
    axes_cross_entropy_mult: float = 2  # was: 1, [2], {2}
    mse_mult: float = 0.5  # was: 0.5
    mmd_mult: float = 500  # was: 500
    trim_mult: float = 300
    trim_thr: float = 3

    axs_ce_dim: int = 0  # it was only used in axs_ce_dims_sel for forward_
    # noinspection PyTypeChecker
    q_types: Tuple[Tuple[int, ...], ...] = ()

    passthr_dim: int = 1
    encoder: Perceptron
    decoder: Perceptron
    dist: Distrib

    PerceptronEncoder: Type[Perceptron] = Perceptron
    PerceptronDecoder: Type[Perceptron] = Perceptron

    encoder_h_dims: Tuple[int, ...] = (64, 128, 128)
    decoder_h_dims: Tuple[int, ...] = (128, 128, 64)
    types16_cross_entropy_mult: int = 0

    def __init__(self, dims: Tuple[int, Tuple[int, ...], int], activation_fn: Callable[[Tensor], Tensor]=tr.selu):
        z_dim_plus_passthr_dim, h_dims, y_dims = dims
        z_dim = z_dim_plus_passthr_dim - self.passthr_dim

        with self.disable_final_init():
            super(ClassifierPassthrJATS24KhCFBase, self).__init__(
                (self.axs_dim + self.passthr_dim, h_dims, y_dims),
                activation_fn=activation_fn)
        self.dims = dims

        self.encoder = self.PerceptronEncoder(
            dims=(z_dim + self.passthr_dim + self.extra_append_dims, *self.encoder_h_dims, self.axs_dim),
            activation_fn=activation_fn, output_activation=None)

        self.decoder = self.PerceptronDecoder(
            dims=(self.axs_dim + self.passthr_dim, *self.decoder_h_dims, z_dim),
            activation_fn=activation_fn, output_activation=None)

        self.dist = Normal()
        self.__final_init__()

    def transform2(self, z_trans1: Tensor) -> Tensor:
        return tr.cat([z_trans1, z_trans1], dim=1)

    def get_q_types(self) -> Tuple[Tuple[int, ...], ...]:
        return (
                KHAX1[:self.axs_dim] + KHAX_OTHER[:self.axs_dim]  # MAP POS
                + KHAX0[:self.axs_dim] + KHAX_OTHER[:self.axs_dim]  # MAP NEG
        )

    def get_z_new(self, z_ax: Tensor) -> Tensor:
        raise RuntimeError('method is deprecated for subclass')

    @staticmethod
    def get_transform() -> Tensor:
        return ClassifierJATSAxesAlignNTRrAdeAdi.get_transform()

    def append_extra_dims(self, z_ext: Tensor) -> Tensor:
        return tr.cat([z_ext, z_ext[:, (4, 5)] @ self.transform], dim=1)

    def get__z_new__z_new_ext(self, z_ext: Tensor) -> Tuple[Tensor, Tensor]:
        z_new = self.encoder.__call__(self.append_extra_dims(z_ext))
        return z_new, tr.cat([z_ext[:, :self.passthr_dim], z_new], dim=1)

    def probs__types_cross_entropy(self, z_new_ext: Tensor, y: Tensor=None) -> Tuple[Tensor, Tensor]:
        probs = tr.softmax(self.get_logits(z_new_ext), dim=-1)
        if self.types_cross_entropy_mult <= 0:
            return probs, self.zero
        return probs, self.cross_entropy(probs, y).mean() * self.types_cross_entropy_mult

    def forward_(self, x: Tensor, y: Tensor=None) -> Tuple[Tensor, Tensor]:
        z_new, z_new_ext = self.get__z_new__z_new_ext(x)
        probs, types_cross_entropy = self.probs__types_cross_entropy(z_new_ext, y)
        if y is None:
            return probs, types_cross_entropy
        return probs, self.axes_cross_entropy(z_new, self.transform2(z_new), y) + types_cross_entropy

    def get__mse__reg_loss(self, z_ext: Tensor, z_ext_mse: Tensor=None) -> Tuple[Tensor, Tensor]:
        """ mse is unredused, reg_loss is reduced """
        z = z_ext[:, self.passthr_dim:] if (z_ext_mse is None) else z_ext_mse[:, self.passthr_dim:]
        z_new, z_new_ext = self.get__z_new__z_new_ext(z_ext)
        z_rec = self.decoder.__call__(z_new_ext)
        trim = (tr.relu(z_new - self.trim_thr)  # not neg. thr. at right
                + tr.relu(-z_new - self.trim_thr)  # not pos. thr. at left
                ).view(z_ext.shape[0], -1).sum(dim=1).mean() * self.trim_mult
        mse = ((z - z_rec)**2).view(z_ext.shape[0], -1).sum(dim=1) * self.mse_mult
        mmd = self.dist.mmd(z_new) * self.mmd_mult if (self.mmd_mult > 0) else self.zero
        return mse, trim + mmd

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def probs_verbose(self, z_ext: Tensor) -> Opt[Tensor]:  # pylint: disable=unused-argument
        return None


class EncoderCustom2(EncoderCustom):
    pass


class VAEPassthroughClassifyMMD(VAEPassthroughClassify):
    mmd: Union[Tensor, int] = 0
    τ: float = 0

    def forward_vae_to_z(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        z, (μ, log_σ) = super(VAEPassthroughClassifyMMD, self).forward_vae_to_z(x=x, y=y)

        if self.τ != 0:
            self.mmd = self.kld.prior_dist.mmd(self.kld.inv_flow_pz(μ))
        else:
            self.mmd = 0
        return z, (μ, log_σ)


class VAEPassthroughClassifyTwinMMD(VAEPassthroughClassifyTwin):
    mmd: Union[Tensor, int] = 0
    τ: float = 0

    def forward_vae_to_z(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        z, (μ, log_σ) = super(VAEPassthroughClassifyTwinMMD, self).forward_vae_to_z(x=x, y=y)

        if self.τ != 0:
            self.mmd = self.kld.prior_dist.mmd(self.kld.inv_flow_pz(μ))
        else:
            self.mmd = 0
        return z, (μ, log_σ)


class DecoderPassthroughSex(DecoderPassthrough):
    def neg_log_p(self, x: Tensor) -> Opt[Tensor]:
        return neg_log_standard_bernoulli(x).view(x.shape[0], -1).sum(dim=1)


# -------------------------------------------------------------------------
# TRASH:  # TODO
# -------------------------------------------------------------------------
# noinspection PyPep8Naming
class __SubDecoderPassthrSocAxTo8KhAx2__(SubDecoderPassthrSocAxTo8KhAx):
    socax_dim: int = 5
    Map: Type[Perceptron] = Perceptron3


# noinspection PyPep8Naming
class __ClassifierJATSAxesAlignKhAx5__(ClassifierJATSAxesAlignKhAx):
    thrs_: Tuple[Tuple[float, ...], ...] = ((0., 1., 1.), (0.5, 1., 1.), (1.5, 1.5, 1.5))
    thrs_extra: Tuple[Tuple[float, ...], ...] = (
        (1.2, 1.2, 1.2, 1.2, 1.2, 1.2),
        (1.5, 1.5, 1.5, 1.5, 1.5, 1.5),
        (1.8, 1.8, 1.8, 1.8, 1.8, 1.8),
    )
    # axs_ce_dim_modify: int = -2

    def get_axes(self) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]:
        axes = tuple(tpl(
            N[i],
            inv(R)[i],
            E[i],

            T[i],
            inv(AD)[i],
            E[i],
        )[:self.extra_dim] for i in (0, 1))
        return axes[0], axes[1]


# noinspection PyPep8Naming
class __ClassifierJATSAxesAlignProjectionNTRrEAd__(ClassifierJATSAxesAlignProjectionNTRrEAd):
    k: Parameter
    _k0: float = 1
    _k_fixed_dim: float = 0.7
    target_std: float = 1
    trim_mult: float = 300
    mse_mult: float = 20

    thrs: Tuple[Tuple[float, ...], ...] = (
        tuple(0. for _ in range(8)) + tuple(1.5 for _ in range(8)),
    )

    def __post_init__(self):
        super(__ClassifierJATSAxesAlignProjectionNTRrEAd__, self).__post_init__()
        self.k = Parameter(tr.tensor([1.], dtype=tr.float), requires_grad=True)

    def transform2(self, z_trans1: Tensor) -> Tensor:
        return z_trans1 @ self.trans_khax_dim_x_socax_dim().transpose(0, 1)

    def trans_khax_dim_x_socax_dim(self) -> Tensor:
        return self.const_khax_dim_x_socax_dim * (self.k_socax_dim ** 2 + self.k0) * self.k ** 2

    def get_regul_loss_reduced(self, zμ: Tensor) -> Tensor:
        zμ_new = self.get_z_new(zμ)
        std_mse = ((zμ_new.std(dim=0) - self.target_std)**2).sum() * self.mse_mult
        return std_mse

    def get_regul_loss_reduced__0(self, zμ: Tensor) -> Tensor:
        zμ_new = self.get_z_new(zμ)
        trim = (tr.relu(zμ_new - self.target_std * 3)  # not neg. thr. at right
                + tr.relu(-zμ_new - self.target_std * 3)  # not pos. thr. at left
                ).view(zμ.shape[0], -1).sum(dim=1).mean() * self.trim_mult
        return trim


# noinspection PyPep8Naming
class __SubDecoderPassthr11KhZgAxTo12KhAx__(ModuleXToX, PostInit):
    soc_ax_dim: int = 11  # independent axes
    trans_ax_dim: int = 12  # linear map axes
    kh_ax_dim: int = 12
    q_indx: int = 4  # without passthrough axes
    q_map_n: int = 4  # axes to q_map (without passthrough axes)
    fixed_dim: int = 5

    mmd_mult: float = 750
    trim_mult: float = 200
    trim_thr: float = 3
    dist: Distrib
    zero: Tensor  # buffer
    passthr_dim: int = 1
    q_map: Perceptron

    const_khax_dim_x_transax_dim: Tensor
    k0: Tensor
    k_fixed_dim: Tensor
    k_transax_dim: Parameter
    _k0: float = 0.5
    _k_fixed_dim: float = math.sqrt(0.5)

    def __init__(self, h_dims: Tuple[int, ...]=(64, 64), passthrough_dim: int=1,
                 activation_fn: Callable[[Tensor], Tensor]=tr.selu,
                 output_activation: Opt[Callable[[Tensor], Tensor]]=None):
        """
        In case of 12 Khizhnyak axes:

        z_ext_ind: (sex, Rr, E, N, T, Q', ?Ad, ..., extra1, ...)
        z_ext_dep: (sex, Rr, E, N, T, Qᵈᵉᵖ, Q', ?Ad, ..., extra1, ...)
        z_ext_khax: (sex, NERr, ..., QER,  extra1, extra2, ...)

        Supplemental Q-map: (sex, Rr, E, N, T, ...) => (Qᵈᵉᵖ,)
        Map: z_ext_ind => z_ext_dep => z_ext_khax

        In case of 12 Khizhnyak axes (Alt1):

        z_ext_ind: (sex, Rr, E, N, T, Q, ?Ad, ..., extra1, ...)
        z_ext_dep:= z_ext_ind
        z_ext_khax: (sex, NERr, ..., QER,  extra1, ...)

        Dummy identity supplemental map.
        Map: z_ext_ind => z_ext_dep => z_ext_khax

        In case of 12 Khizhnyak axes (Alt2):

        z_ext_ind: (sex, Rr, E, N, T, Q', ?Ad, ..., extra1, ...)
        z_ext_dep:= (sex, Rr, E, N, T, Qᵈᵉᵖ, ?Ad, ..., extra1, ...)
        z_ext_khax: (sex, NERr, ..., QER,  extra1, ...)

        Supplemental Q-map: (sex, Rr, E, N, T, ...) => (Qᵈᵉᵖ,)
        Map: z_ext_ind => z_ext_dep => z_ext_khax

        In case of 8 Khizhnyak axes:

        z_ext_ind: (sex, Rr, E, N, T, ?Ad, ..., extra1, ...)
        z_ext_dep:= z_ext_ind
        z_ext_khax: (sex, NERr, ..., TER,  extra1, ...)

        Dummy identity supplemental map.
        Map: z_ext_ind => z_ext_dep => z_ext_khax
        """
        super(__SubDecoderPassthr11KhZgAxTo12KhAx__, self).__init__()
        assert self.passthr_dim == passthrough_dim
        self.q_map = Perceptron(dims=(self.passthr_dim + self.q_map_n, *h_dims, 1),
                                activation_fn=activation_fn,
                                output_activation=output_activation)

        self.register_buffer('const_khax_dim_x_transax_dim', self.get_const())
        self.k_transax_dim = Parameter(
            tr.tensor([self._k_fixed_dim for _ in range(self.trans_ax_dim - self.fixed_dim)], dtype=tr.float),
            requires_grad=True)
        self.register_buffer('k_fixed_dim',
                             tr.tensor([self._k_fixed_dim for _ in range(self.trans_ax_dim)], dtype=tr.float))
        self.register_buffer('k0', tr.tensor([self._k0], dtype=tr.float))

        self.register_buffer('zero', tr.tensor([0.]).mean())
        self.dist = Normal()
        self.__final_init__()

    @staticmethod
    def get_const() -> Tensor:
        #     Q = Qᵈᵉᵖ(Rr,E,N,T)
        #     Rr  E  N  T  Q  Q'  EAd IAd EAb IAb EAg IAg
        return tr.tensor([
            [+1,  1, 1, 0, 0, 0,  1,  0,  0,  0,  0,  0],  # 6+10=>1+13; ISxR=>ENxRr
            [-1, -1, 1, 0, 0, 0,  0,  1,  0,  0,  0,  0],  # 5+9=>2+14;  ESxRr=>INxR
            [+1, -1, 1, 0, 0, 0, -1,  0,  0,  0,  0,  0],  # 4+16=>7+11; ESxR=>INxRr
            [-1,  1, 1, 0, 0, 0,  0, -1,  0,  0,  0,  0],  # 3+15=>8+12; ISxRr=>ENxR
            [+1,  1, 0, 1, 0, 0,  0,  0,  1,  0,  0,  0],  # 10+14=>1+5; IxFR=>ExTRr
            [-1, -1, 0, 1, 0, 0,  0,  0,  0,  1,  0,  0],  # 9+13=>2+6;  ExFRr=>IxTR
            [+1, -1, 0, 1, 0, 0,  0,  0, -1,  0,  0,  0],  # 4+8=>11+15; ExFR=>IxTRr
            [-1,  1, 0, 1, 0, 0,  0,  0,  0, -1,  0,  0],  # 3+7=>12+16; IxFRr=>ExTR
            [+1,  1, 0, 0, 1, 1,  0,  0,  0,  0,  1,  0],  # 6+14=>1+9;  IxxRD=>ExxRrQ
            [-1, -1, 0, 0, 1, 1,  0,  0,  0,  0,  0,  1],  # 5+13=>2+10; ExxRrD=>IxxRQ
            [+1, -1, 0, 0, 1, 1,  0,  0,  0,  0, -1,  0],  # 4+12=>7+15; ExxRD=>IxxRrQ
            [-1,  1, 0, 0, 1, 1,  0,  0,  0,  0,  0, -1],  # 3+11=>8+16; IxxRrD=>ExxRQ
        ], dtype=tr.float)

    # noinspection PyMethodMayBeStatic
    def repr_learnables(self) -> str:
        return (f'\ntrans_{self.kh_ax_dim}khax_dim_x_{self.trans_ax_dim}transax_dim:\n'
                + f'{self.trans_khax_dim_x_transax_dim()}\n')

    def regul_loss_reduced(self, z_ext_dep: Tensor) -> Tensor:
        q = z_ext_dep[:, (self.passthr_dim + self.q_indx,)]

        trim = (tr.relu(q - self.trim_thr)  # not neg. thr. at right
                + tr.relu(-q - self.trim_thr)  # not pos. thr. at left
                ).view(q.shape[0], -1).sum(dim=1).mean() * self.trim_mult
        mmd = self.dist.mmd(q) * self.mmd_mult if (self.mmd_mult > 0) else self.zero
        return trim + mmd

    def z_ext_khax(self, z_ext_dep: Tensor) -> Tensor:
        n = self.passthr_dim + self.trans_ax_dim
        z = z_ext_dep[:, self.passthr_dim:n]
        mat = self.trans_khax_dim_x_transax_dim()
        z_khax = z @ (mat.transpose(0, 1) / mat.norm(dim=1))
        if z_ext_dep.shape[1] > n:
            return tr.cat([z_ext_dep[:, :self.passthr_dim], z_khax, z_ext_dep[:, n:]], dim=1)
        return tr.cat([z_ext_dep[:, :self.passthr_dim], z_khax], dim=1)

    def trans_khax_dim_x_transax_dim(self) -> Tensor:
        k_ax_dim = tr.cat([self.k_fixed_dim[:, :self.fixed_dim], self.k_transax_dim], dim=0)
        return self.const_khax_dim_x_transax_dim * (k_ax_dim ** 2 + self.k0)

    def forward_(self, x: Tensor) -> Tensor:
        z_ext_ind = x
        return self.z_ext_khax(self.z_ext_dep(z_ext_ind))

    def z_ext_dep(self, z_ext_ind: Tensor) -> Tensor:
        n = self.passthr_dim + self.q_map_n
        z_pref = z_ext_ind[:, :n]
        q = self.q_map.__call__(z_pref)
        return tr.cat([z_pref, q, z_ext_ind[:, n:]], dim=1)

    def z_ext_for_cls(self, z_ext_ind: Tensor) -> Tensor:
        z_ext_dep = self.z_ext_dep(z_ext_ind)
        # qi = self.passthr_dim + self.q_indx
        return tr.cat([self.z_ext_khax(z_ext_dep)[:, :self.passthr_dim + self.kh_ax_dim],
                       # z_ext_dep[:, (qi - 2, qi - 1, qi)]
                       ], dim=1)

    # @staticmethod
    # def extra_for_cls_dim(kh_ax_dim: int) -> int:
    #     return kh_ax_dim // 4


# noinspection PyPep8Naming
class __SubDecoderPassthr8RnKhAxTo12KhAx__(__SubDecoderPassthr11KhZgAxTo12KhAx__):
    soc_ax_dim: int = 8
    trans_ax_dim: int = 9
    kh_ax_dim: int = 12
    q_indx: int = 4
    q_map_n: int = 4
    fixed_dim: int = 5

    @staticmethod
    def get_const() -> Tensor:
        #     Q = Qᵈᵉᵖ(Rr,E,N,T)
        #     Rr  E  N  T  Q  Q'  Ad  Ab  Ag
        return tr.tensor([
            [+1,  1, 1, 0, 0, 0,  1,  0,  0],  # 6+10=>1+13; ISxR=>ENxRr
            [-1, -1, 1, 0, 0, 0,  1,  0,  0],  # 5+9=>2+14;  ESxRr=>INxR
            [+1, -1, 1, 0, 0, 0, -1,  0,  0],  # 4+16=>7+11; ESxR=>INxRr
            [-1,  1, 1, 0, 0, 0, -1,  0,  0],  # 3+15=>8+12; ISxRr=>ENxR
            [+1,  1, 0, 1, 0, 0,  0,  1,  0],  # 10+14=>1+5; IxFR=>ExTRr
            [-1, -1, 0, 1, 0, 0,  0,  1,  0],  # 9+13=>2+6;  ExFRr=>IxTR
            [+1, -1, 0, 1, 0, 0,  0, -1,  0],  # 4+8=>11+15; ExFR=>IxTRr
            [-1,  1, 0, 1, 0, 0,  0, -1,  0],  # 3+7=>12+16; IxFRr=>ExTR
            [+1,  1, 0, 0, 1, 1,  0,  0,  1],  # 6+14=>1+9;  IxxRD=>ExxRrQ
            [-1, -1, 0, 0, 1, 1,  0,  0,  1],  # 5+13=>2+10; ExxRrD=>IxxRQ
            [+1, -1, 0, 0, 1, 1,  0,  0, -1],  # 4+12=>7+15; ExxRD=>IxxRrQ
            [-1,  1, 0, 0, 1, 1,  0,  0, -1],  # 3+11=>8+16; IxxRrD=>ExxRQ
        ], dtype=tr.float)


# noinspection PyPep8Naming
class __SubDecoderPassthr8RnKhAxTo12KhAxAlt2__(__SubDecoderPassthr11KhZgAxTo12KhAx__):
    soc_ax_dim: int = 8
    trans_ax_dim: int = 8
    kh_ax_dim: int = 12
    q_indx: int = 4
    q_map_n: int = 5
    fixed_dim: int = 5

    @staticmethod
    def get_const() -> Tensor:
        #     Q = Qᵈᵉᵖ(Rr,E,N,T,Q')
        #     Rr  E  N  T  Q   Ad  Ab  Ag
        return tr.tensor([
            [+1,  1, 1, 0, 0,  1,  0,  0],  # 6+10=>1+13; ISxR=>ENxRr
            [-1, -1, 1, 0, 0,  1,  0,  0],  # 5+9=>2+14;  ESxRr=>INxR
            [+1, -1, 1, 0, 0, -1,  0,  0],  # 4+16=>7+11; ESxR=>INxRr
            [-1,  1, 1, 0, 0, -1,  0,  0],  # 3+15=>8+12; ISxRr=>ENxR
            [+1,  1, 0, 1, 0,  0,  1,  0],  # 10+14=>1+5; IxFR=>ExTRr
            [-1, -1, 0, 1, 0,  0,  1,  0],  # 9+13=>2+6;  ExFRr=>IxTR
            [+1, -1, 0, 1, 0,  0, -1,  0],  # 4+8=>11+15; ExFR=>IxTRr
            [-1,  1, 0, 1, 0,  0, -1,  0],  # 3+7=>12+16; IxFRr=>ExTR
            [+1,  1, 0, 0, 1,  0,  0,  1],  # 6+14=>1+9;  IxxRD=>ExxRrQ
            [-1, -1, 0, 0, 1,  0,  0,  1],  # 5+13=>2+10; ExxRrD=>IxxRQ
            [+1, -1, 0, 0, 1,  0,  0, -1],  # 4+12=>7+15; ExxRD=>IxxRrQ
            [-1,  1, 0, 0, 1,  0,  0, -1],  # 3+11=>8+16; IxxRrD=>ExxRQ
        ], dtype=tr.float)

    def z_ext_dep(self, z_ext_ind: Tensor) -> Tensor:
        q = self.q_map.__call__(z_ext_ind[:, :self.passthr_dim + self.q_map_n])
        n = self.passthr_dim + self.q_indx
        return tr.cat([z_ext_ind[:, :n], q, z_ext_ind[:, n + 1:]], dim=1)


# noinspection PyPep8Naming
class __SubDecoderPassthr8KhZgAxTo8KhAx__(__SubDecoderPassthr11KhZgAxTo12KhAx__):
    soc_ax_dim: int = 8
    trans_ax_dim: int = 8
    kh_ax_dim: int = 8
    q_indx: int = 4  # the place where it should have been
    fixed_dim: int = 4

    @staticmethod
    def get_const() -> Tensor:
        #     Rr  E  N  T  EAd IAd EAb IAb
        return tr.tensor([
            [+1,  1, 1, 0,  1,  0,  0,  0],  # 6+10=>1+13; ISxR=>ENxRr
            [-1, -1, 1, 0,  0,  1,  0,  0],  # 5+9=>2+14;  ESxRr=>INxR
            [+1, -1, 1, 0, -1,  0,  0,  0],  # 4+16=>7+11; ESxR=>INxRr
            [-1,  1, 1, 0,  0, -1,  0,  0],  # 3+15=>8+12; ISxRr=>ENxR
            [+1,  1, 0, 1,  0,  0,  1,  0],  # 10+14=>1+5; IxFR=>ExTRr
            [-1, -1, 0, 1,  0,  0,  0,  1],  # 9+13=>2+6;  ExFRr=>IxTR
            [+1, -1, 0, 1,  0,  0, -1,  0],  # 4+8=>11+15; ExFR=>IxTRr
            [-1,  1, 0, 1,  0,  0,  0, -1],  # 3+7=>12+16; IxFRr=>ExTR
        ], dtype=tr.float)

    def regul_loss_reduced(self, z_ext_dep: Tensor) -> Tensor:
        return self.zero

    def z_ext_dep(self, z_ext_ind: Tensor) -> Tensor:
        return z_ext_ind

    def z_ext_for_cls(self, z_ext_ind: Tensor) -> Tensor:
        z_ext_dep = self.z_ext_dep(z_ext_ind)
        # qi = self.passthr_dim + self.q_indx
        return tr.cat([self.z_ext_khax(z_ext_dep)[:, :self.passthr_dim + self.kh_ax_dim],
                       # z_ext_dep[:, (qi - 2, qi - 1)]
                       ], dim=1)


# noinspection PyPep8Naming
class __SubDecoderPassthr6RnKhAxTo8KhAx__(__SubDecoderPassthr8KhZgAxTo8KhAx__):
    soc_ax_dim: int = 6
    trans_ax_dim: int = 6
    kh_ax_dim: int = 8
    q_indx: int = 4  # the place where it should have been
    fixed_dim: int = 4

    @staticmethod
    def get_const() -> Tensor:
        #     Rr  E  N  T   Ad  Ab
        return tr.tensor([
            [+1,  1, 1, 0,  1,  0],  # 6+10=>1+13; ISxR=>ENxRr
            [-1, -1, 1, 0,  1,  0],  # 5+9=>2+14;  ESxRr=>INxR
            [+1, -1, 1, 0, -1,  0],  # 4+16=>7+11; ESxR=>INxRr
            [-1,  1, 1, 0, -1,  0],  # 3+15=>8+12; ISxRr=>ENxR
            [+1,  1, 0, 1,  0,  1],  # 10+14=>1+5; IxFR=>ExTRr
            [-1, -1, 0, 1,  0,  1],  # 9+13=>2+6;  ExFRr=>IxTR
            [+1, -1, 0, 1,  0, -1],  # 4+8=>11+15; ExFR=>IxTRr
            [-1,  1, 0, 1,  0, -1],  # 3+7=>12+16; IxFRr=>ExTR
        ], dtype=tr.float)


# noinspection PyPep8Naming
class __SubDecoderPassthr6RnKhAxTo8KhAxAlt3__(__SubDecoderPassthr8KhZgAxTo8KhAx__):
    soc_ax_dim: int = 6
    trans_ax_dim: int = 6
    kh_ax_dim: int = 8
    q_indx: int = 4  # the place where it should have been
    fixed_dim: int = 4

    @staticmethod
    def get_const() -> Tensor:
        #     Rr  E  N  T  EAd IAd
        return tr.tensor([
            [+1,  1, 1, 0,  1,  0],  # 6+10=>1+13; ISxR=>ENxRr
            [-1, -1, 1, 0,  0,  1],  # 5+9=>2+14;  ESxRr=>INxR
            [+1, -1, 1, 0, -1,  0],  # 4+16=>7+11; ESxR=>INxRr
            [-1,  1, 1, 0,  0, -1],  # 3+15=>8+12; ISxRr=>ENxR
            [+1,  1, 0, 1,  0,  0],  # 10+14=>1+5; IxFR=>ExTRr
            [-1, -1, 0, 1,  0,  0],  # 9+13=>2+6;  ExFRr=>IxTR
            [+1, -1, 0, 1,  0,  0],  # 4+8=>11+15; ExFR=>IxTRr
            [-1,  1, 0, 1,  0,  0],  # 3+7=>12+16; IxFRr=>ExTR
        ], dtype=tr.float)


# noinspection PyPep8Naming
class __SubDecoderPassthr8RnKhAxTo12KhAxAlt1__(__SubDecoderPassthr8KhZgAxTo8KhAx__):
    soc_ax_dim: int = 8
    trans_ax_dim: int = 8
    kh_ax_dim: int = 12
    q_indx: int = 4  # the place where it should have been (real Q not bonus one)
    fixed_dim: int = 8

    @staticmethod
    def get_const() -> Tensor:
        #     Rr  E  N  T  Q   Ad  Ab  Ag
        return tr.tensor([
            [+1,  1, 1, 0, 0,  1,  0,  0],  # 6+10=>1+13; ISxR=>ENxRr
            [-1, -1, 1, 0, 0,  1,  0,  0],  # 5+9=>2+14;  ESxRr=>INxR
            [+1, -1, 1, 0, 0, -1,  0,  0],  # 4+16=>7+11; ESxR=>INxRr
            [-1,  1, 1, 0, 0, -1,  0,  0],  # 3+15=>8+12; ISxRr=>ENxR
            [+1,  1, 0, 1, 0,  0,  1,  0],  # 10+14=>1+5; IxFR=>ExTRr
            [-1, -1, 0, 1, 0,  0,  1,  0],  # 9+13=>2+6;  ExFRr=>IxTR
            [+1, -1, 0, 1, 0,  0, -1,  0],  # 4+8=>11+15; ExFR=>IxTRr
            [-1,  1, 0, 1, 0,  0, -1,  0],  # 3+7=>12+16; IxFRr=>ExTR
            [+1,  1, 0, 0, 1,  0,  0,  1],  # 6+14=>1+9;  IxxRD=>ExxRrQ
            [-1, -1, 0, 0, 1,  0,  0,  1],  # 5+13=>2+10; ExxRrD=>IxxRQ
            [+1, -1, 0, 0, 1,  0,  0, -1],  # 4+12=>7+15; ExxRD=>IxxRrQ
            [-1,  1, 0, 0, 1,  0,  0, -1],  # 3+11=>8+16; IxxRrD=>ExxRQ
        ], dtype=tr.float)

    def trans_khax_dim_x_transax_dim(self) -> Tensor:
        return self.const_khax_dim_x_transax_dim * (self.k_fixed_dim**2 + self.k0)

    # @staticmethod
    # def extra_for_cls_dim(kh_ax_dim: int) -> int:
    #     return 2


# noinspection PyPep8Naming
class __SubDecoderPassthrSocAxTo12KhAx3__(SubDecoderPassthrSocAxTo12KhAx):
    mmd_mult: float = 750

    h_dims: Tuple[int, ...] = (32, 32)
    h_dims1: Tuple[int, ...] = (3, 16, 16, 3)
    h_dims2: Tuple[int, ...] = (3, 16, 16, 3)
    subset1: Tuple[int, ...] = (0, 3, 4)
    subset2: Tuple[int, ...] = (1, 2, 6)

    ind_axs_to_cls: Tuple[int, ...] = tuple(range(6))  # only dimension matters!
    for_cls_dim: int = 12 + 0
    socax_dim: int = 8
    semi_type_indep: Tuple[int, ...] = ()  # semi-type-indep. axes
    socpass_dim: int = 0   # M semi-type-indep. axes

    map1: Perceptron
    map2: Perceptron
    semi_type_indep__ext: Tuple[int, ...]  # with passthr
    subset1__ext: Tuple[int, ...]  # with passthr
    subset2__ext: Tuple[int, ...]  # with passthr
    rot_45: Tensor

    def __post_init__(self):
        super(__SubDecoderPassthrSocAxTo12KhAx3__, self).__post_init__()
        output_activation = self.map.output_activation.a if self.map.output_activation else None
        self.map1 = self.Map(dims=(self.passthr_dim + self.h_dims1[0], *self.h_dims1[1:]),
                             activation_fn=self.map.activation_fn.a, output_activation=output_activation)
        self.map2 = self.Map(dims=(self.passthr_dim + self.h_dims2[0], *self.h_dims2[1:]),
                             activation_fn=self.map.activation_fn.a, output_activation=output_activation)
        self.semi_type_indep__ext = tuple(i + self.passthr_dim for i in self.semi_type_indep)
        self.subset1__ext = tuple(range(self.passthr_dim)) + tuple(i + self.passthr_dim for i in self.subset1)
        self.subset2__ext = tuple(range(self.passthr_dim)) + tuple(i + self.passthr_dim for i in self.subset2)
        self.rot_45 = self.get_rot_45()

    @staticmethod
    def get_rot_45() -> Tensor:
        cos, sin, π = math.cos, math.sin, math.pi
        rot45 = tr.tensor([[cos(π / 4), -sin(π / 4)],
                           [sin(π / 4), +cos(π / 4)]])
        return rot45

    def z_ext_dep(self, z_ext_ind: Tensor) -> Tensor:
        z_khax_plus = self.get_z_khax_plus(z_ext_ind)  # == cat(..., z_khax, z_socextra)
        s = z_ext_ind[:, :self.passthr_dim]

        if self.semi_type_indep__ext:
            return tr.cat([s, z_khax_plus, z_ext_ind[:, self.semi_type_indep__ext]], dim=1)

        n = self.passthr_dim + self.socax_dim
        if z_ext_ind.shape[1] > n:
            return tr.cat([s, z_khax_plus, z_ext_ind[:, n:]], dim=1)  # == cat(..., z_khax_plus, z_nonsocextra)

        return tr.cat([s, z_khax_plus], dim=1)

    def get_z31_z32(self, z_ext_ind: Tensor) -> Tuple[Tensor, Tensor]:
        return (self.map1.__call__(z_ext_ind[:, self.subset1__ext]),
                self.map2.__call__(z_ext_ind[:, self.subset2__ext]))

    def get_rots(self, z3: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return z3[:, (0, 1)] @ self.rot_45, z3[:, (0, 2)] @ self.rot_45, z3[:, (1, 2)] @ self.rot_45

    def z_ext_for_cls(self, z_ext_ind: Tensor) -> Tensor:
        z31, z32 = self.get_z31_z32(z_ext_ind)
        z_ext_dep = self.z_ext_dep(z_ext_ind)
        if self.semi_type_indep__ext:
            n = z_ext_dep.shape[1] - len(self.semi_type_indep__ext)
            return tr.cat([z_ext_dep[:, :n], z31, z32, z_ext_dep[:, n:]], dim=1)
        return tr.cat([z_ext_dep, z31, z32], dim=1)

    def z_for_plot(self, z_ext_ind: Tensor) -> Tensor:
        z_ext_dep = self.z_ext_dep(z_ext_ind)  # == cat(..., z_socextra, z_nonsocextra)
        z31, z32 = self.get_z31_z32(z_ext_ind)
        z_khax = z_ext_dep[:, self.passthr_dim:self.passthr_dim + self.khax_real_dim]
        return tr.cat([self.z_dep_and_trans_dep(z_ext_ind, z_ext_dep),
                       z31, z32, *self.get_rots(z31), *self.get_rots(z32),
                       self.map_khax_to_oy_ce_reduced.prob_stat(z_khax),
                       self.map_khax2rnax.z_rnax(z_khax)], dim=1)

    def regul_loss_reduced(self, z_ext_ind: Tensor, z_ext_dep: Tensor) -> Tensor:  # API
        z_khax = z_ext_dep[:, self.passthr_dim:self.passthr_dim + self.khax_real_dim]
        trim = (tr.relu(z_khax - self.trim_thr)  # not neg. thr. at right
                + tr.relu(-z_khax - self.trim_thr)  # not pos. thr. at left
                ).view(z_khax.shape[0], -1).sum(dim=1).mean() * self.trim_mult
        mmd = (self.dist.mmd(tr.cat([*self.get_z31_z32(z_ext_ind)], dim=1)) * self.mmd_mult
               if (self.mmd_mult > 0)
               else self.zero)
        return trim + mmd


# noinspection PyPep8Naming
class __SubDecoderPassthrSocAxTo12KhAx4__(__SubDecoderPassthrSocAxTo12KhAx3__):
    for_cls_dim: int = 12 + 2
    semi_type_indep: Tuple[int, ...] = (5, 7)  # semi-type-indep. axes
    socpass_dim: int = 2   # M semi-type-indep. axes


# noinspection PyPep8Naming
class __SubDecoderPassthrSocAxTo12KhAx5__(SubDecoderPassthrSocAxTo12KhAx):
    # this method doesn't work as triplets are too restrictive fot optimization to find independent axes
    h_dims: Tuple[int, ...] = (32, 32)
    subset1: Tuple[int, ...] = (0, 1, 2)
    subset2: Tuple[int, ...] = (3, 4, 5)

    ind_axs_to_cls: Tuple[int, ...] = tuple(range(6))  # only dimension matters!
    for_cls_dim: int = 12 + 2
    socax_dim: int = 6
    semi_type_indep: Tuple[int, ...] = ()  # semi-type-indep. axes
    socpass_dim: int = 0   # M semi-type-indep. axes
    # WARNING: semi-type-indep. axes CANNOT BE combined with number of socax_dim less than all axes.

    semi_type_indep__ext: Tuple[int, ...]  # with passthr
    subset1__ext: Tuple[int, ...]  # with passthr
    subset2__ext: Tuple[int, ...]  # with passthr
    rot45: Tensor
    invrot45: Tensor
    rot01_div_pi: List[float] = [1/8, 1/4, 1/32]
    rot02_div_pi: List[float] = [1/8, 1/4, 1/32]
    rot1: Parameter
    rot2: Parameter

    def __post_init__(self):
        super(__SubDecoderPassthrSocAxTo12KhAx5__, self).__post_init__()
        self.semi_type_indep__ext = tuple(i + self.passthr_dim for i in self.semi_type_indep)
        self.rot45 = self.get_rot45()
        self.invrot45 = self.get_rot45().transpose(0, 1)
        self.subset1__ext = tuple(i + self.passthr_dim for i in self.subset1)
        self.subset2__ext = tuple(i + self.passthr_dim for i in self.subset2)
        self.rot1 = Parameter(tr.tensor(self.rot01_div_pi, dtype=tr.float) * math.pi, requires_grad=True)
        self.rot2 = Parameter(tr.tensor(self.rot02_div_pi, dtype=tr.float) * math.pi, requires_grad=True)

    @staticmethod
    def get_rot45() -> Tensor:
        cos, sin, π = math.cos, math.sin, math.pi
        rot45 = tr.tensor([[cos(π / 4), -sin(π / 4)],
                           [sin(π / 4), +cos(π / 4)]])
        return rot45

    @staticmethod
    def rot_3d(rot: Tensor) -> Tensor:
        """
        Here I assume that starting axes position is (N,Rr,E1) or (T,Bg,E2) - see ``rot.png``.
        Hence I need inverse transform in order to get maps like:

        * (N~NRr,ERr,ER~?NR) => inv.rot. => (N,Rr,E1) => rot.(π/8,π/4,0) => (basis 3-subset #1)
        * (T~TBg,EBg,EAd~TAd) => inv.rot. => (T,Bg,E2) => rot.(π/8,π/4,0) => (basis 3-subset #2)

        First we hold Z (blue-Z) and rotate α, then hold ex-X (now N) and rotate β, then hold ex-Z (red-Z) and rotate γ.
        (X,Y,Z)=(N,Rr,E1); Z=E1, rot.(+π/8), now ex-X is N~NRr; ex-X=N~NRr, rot.(+π/4), now ex-Y is ~ERr, ex-Z is ~ER.
        (X,Y,Z)=(T,Bg,E2); Z=E2, rot.(+π/8), now ex-X is T~TBg; ex-X=N~NRr, rot.(+π/4), now ex-Y is ~ERr, ex-Z is ~EAd.

        Hence (...).transpose(0, 1)
        """
        cos_rot, sin_rot = tr.cos(rot), tr.sin(rot)
        cosα, cosβ, cosγ = cos_rot[0], cos_rot[1], cos_rot[2]
        sinα, sinβ, sinγ = sin_rot[0], sin_rot[1], sin_rot[2]
        return tr.tensor([
            [cosα * cosγ - cosβ * sinα * sinγ,  -cosγ * sinα - cosα * cosβ * sinγ,   sinβ * sinγ],
            [cosβ * cosγ * sinα + cosα * sinγ,   cosα * cosβ * cosγ - sinα * sinγ,  -cosγ * sinβ],
            [sinα * sinβ,                        cosα * sinβ,                        cosβ]
        ], dtype=tr.float).transpose(0, 1)

    def get_z_rot(self, z_ext_ind: Tensor) -> Tensor:
        z31, z32 = z_ext_ind[:, self.subset1__ext], z_ext_ind[:, self.subset2__ext]
        return tr.cat([z31 @ self.rot_3d(self.rot1), z32 @ self.rot_3d(self.rot2)], dim=1)

    def z_ext_dep(self, z_ext_ind: Tensor) -> Tensor:
        z_khax_plus = self.get_z_khax_plus(z_ext_ind)  # == cat(..., z_khax, z_socextra)
        s = z_ext_ind[:, :self.passthr_dim]

        if self.semi_type_indep__ext:
            return tr.cat([s, z_khax_plus, z_ext_ind[:, self.semi_type_indep__ext]], dim=1)

        n = z_ext_ind.shape[1] - (self.passthr_dim + self.socax_dim)
        if n > 0:
            return tr.cat([s, z_khax_plus, z_ext_ind[:, -n:]], dim=1)  # == cat(..., z_khax_plus, z_nonsocextra)

        return tr.cat([s, z_khax_plus], dim=1)

    def z_ext_for_cls(self, z_ext_ind: Tensor) -> Tensor:
        z_rot = self.get_z_rot(z_ext_ind)
        z_ext_dep = self.z_ext_dep(z_ext_ind)

        if self.semi_type_indep__ext:
            n = z_ext_dep.shape[1] - len(self.semi_type_indep__ext)
            return tr.cat([z_ext_dep[:, :n], z_rot, z_ext_dep[:, n:]], dim=1)

        n = z_ext_ind.shape[1] - (self.passthr_dim + self.socax_dim)
        if n > 0:
            return tr.cat([z_ext_dep[:, :-n], z_rot, z_ext_dep[:, -n:]], dim=1)

        return tr.cat([z_ext_dep, z_rot], dim=1)

    def get_rots(self, z_rot: Tensor) -> Tensor:
        z31, z32 = z_rot[:, :3], z_rot[:, 3:]
        return tr.cat([z31[:, (0, 1)] @ self.rot45, z31[:, (0, 2)] @ self.rot45, z31[:, (1, 2)] @ self.rot45,
                       z32[:, (0, 1)] @ self.rot45, z32[:, (0, 2)] @ self.rot45, z32[:, (1, 2)] @ self.rot45], dim=1)

    def z_for_plot(self, z_ext_ind: Tensor) -> Tensor:
        z_ext_dep = self.z_ext_dep(z_ext_ind)  # == cat(..., z_socextra, z_nonsocextra)
        z_rot = self.get_z_rot(z_ext_ind)
        z_khax = z_ext_dep[:, self.passthr_dim:self.passthr_dim + self.khax_real_dim]
        return tr.cat([self.z_dep_and_trans_dep(z_ext_ind, z_ext_dep),
                       z_rot, self.get_rots(z_rot),
                       self.map_khax_to_oy_ce_reduced.prob_stat(z_khax),
                       self.map_khax2rnax.z_rnax(z_khax)], dim=1)

    # noinspection PyMethodMayBeStatic
    def repr_learnables(self) -> str:  # API
        return (super(__SubDecoderPassthrSocAxTo12KhAx5__, self).repr_learnables()
                + f'\nRotations (α,β,γ)/π:\n{self.rot1 / math.pi};{self.rot2 / math.pi}')


# noinspection PyPep8Naming
class __SubDecoderPassthrSocAxTo12KhAx6__(__SubDecoderPassthrSocAxTo12KhAx5__):
    rot01_div_pi: List[float] = [1/8, 1/4, 1/32]
    rot02_div_pi: List[float] = [1/8, 1/4, 1/32]

    def get_z_rot(self, z_ext_ind: Tensor) -> Tensor:
        """
        >>> '''
        >>> [ER] {E} [ERr]
        >>>     R=>{Rr}
        >>>  IR   I   IRr
        >>>
        >>> [EAd] {E} [EBg]
        >>>      Ad=>{Bg}
        >>>  IAd   I   IBg
        >>> '''

        Original axes:
        a. IR=>[ERr]
        b. IRr=>[ER]

        a. IAd=>[EBg]
        b. IBg=>[EAd]

        New axes after rot(-45):
        a. R=>{Rr}
        b. I=>{E}

        a. Ad=>{Bg}
        b. I=>{E}
        """
        z31, z32 = z_ext_ind[:, self.subset1__ext], z_ext_ind[:, self.subset2__ext]
        return tr.cat([z31[:, (0,)], z31[:, (1, 2)] @ self.invrot45,
                       z32[:, (0,)], z32[:, (1, 2)] @ self.invrot45], dim=1)

    def get_z_unrot(self, z_rot: Tensor) -> Tuple[Tensor, Tensor]:
        z31, z32 = z_rot[:, :3], z_rot[:, 3:]
        return (tr.cat([z31[:, (0,)], z31[:, (1, 2)] @ self.rot45], dim=1),
                tr.cat([z32[:, (0,)], z32[:, (1, 2)] @ self.rot45], dim=1))

    def get_rots(self, z_rot: Tensor) -> Tensor:
        z31, z32 = self.get_z_unrot(z_rot)
        z31, z32 = z31 @ self.rot_3d(self.rot1), z32 @ self.rot_3d(self.rot2)
        return tr.cat([
            z31, z31[:, (0, 1)] @ self.rot45, z31[:, (0, 2)] @ self.rot45, z31[:, (1, 2)] @ self.rot45,
            z32, z32[:, (0, 1)] @ self.rot45, z32[:, (0, 2)] @ self.rot45, z32[:, (1, 2)] @ self.rot45
        ], dim=1)


# noinspection PyPep8Naming
class __ClassifierJATSAxesAlignKhAx__(ClassifierJATSAxesAlign5RnKhAx):
    thrs_: Tuple[Tuple[float, ...], ...] = ((0., 1.5, 0.5), (0., 1.5, 1.5),)
    q_extra_dim: int = 0

    thrs: Tuple[Tuple[float, ...], ...] = ()
    axs_ce_dim: int
    kh_ax_dim: int
    # extra_dim: int
    q_types: Tuple[Tuple[int, ...], ...] = ()
    _subdecoder: Tuple[SubDecoderPassthrSocAxTo12KhAx, ...] = ()
    SubDecoder_: Type[SubDecoder] = SubDecoder  # should be non-dynamic property

    def __init__(self, dims: Tuple[int, Tuple[int, ...], int], activation_fn: Callable[[Tensor], Tensor]=tr.selu):
        self.kh_ax_dim = self.SubDecoder_.khax_dim
        self.axs_ce_dim = self.kh_ax_dim  # + SubDecoder.extra_for_cls_dim(self.kh_ax_dim)
        # self.extra_dim = {2: 2, 3: 3 + self.q_extra_dim}[self.axs_ce_dim - self.kh_ax_dim]

        self.thrs = tuple(
            tuple(thrs[0] for _ in range(self.kh_ax_dim))
            + tuple(thrs[1] for _ in range(self.kh_ax_dim))
            # + tuple(thrs[2] for _ in range(self.extra_dim))
            for thrs in self.thrs_
        )

        _, hidden_dims, y_dim = dims
        super(__ClassifierJATSAxesAlignKhAx__, self).__init__(
            (self.SubDecoder_.passthr_dim + self.axs_ce_dim, hidden_dims, y_dim), activation_fn=activation_fn)

    def set_subdecoder(self, subdecoder: SubDecoderPassthrSocAxTo12KhAx):
        self._subdecoder = (subdecoder,)

    @property
    def subdecoder(self) -> SubDecoderPassthrSocAxTo12KhAx:
        assert self._subdecoder
        return self._subdecoder[0]

    def get_q_types(self) -> Tuple[Tuple[int, ...], ...]:
        return (  # MAP POS:
                KHAX1[:self.kh_ax_dim] + KHAX_OTHER[:self.kh_ax_dim]
                # + (N[1], T[1], Q[1], Q[1], Q[1])[:self.extra_dim]  # MAP NEG:
                + KHAX0[:self.kh_ax_dim] + KHAX_OTHER[:self.kh_ax_dim]
                # + (N[0], T[0], Q[0], Q[0], Q[0])[:self.extra_dim]
        )

    def transform2(self, z_trans1: Tensor) -> Tensor:
        return tr.cat([z_trans1, z_trans1], dim=1)
        # z_khax = z_trans1[:, :self.kh_ax_dim]
        # return tr.cat([z_khax, z_trans1  # , z_trans1[:, (-1, -1)]
        #                ], dim=1)

    def forward_(self, x: Tensor, y: Tensor=None) -> Tuple[Tensor, Tensor]:
        return super(__ClassifierJATSAxesAlignKhAx__, self).forward_(self.subdecoder.z_ext_for_cls(x), y)
