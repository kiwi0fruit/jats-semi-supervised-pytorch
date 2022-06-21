from typing import Tuple, List

import numpy as np
from numpy.typing import NDArray as Array
import torch as tr
from torch import Tensor, nn


Ax = Tuple[Tuple[int, ...], Tuple[int, ...]]

KHAXDIM = 12

E: Ax = ((2, 3, 6, 7, 10, 11, 14, 15), (1, 4, 5, 8, 9, 12, 13, 16))
R: Ax = ((1, 3, 5, 7, 9, 11, 13, 15), (2, 4, 6, 8, 10, 12, 14, 16))
N: Ax = ((3, 4, 5, 6, 9, 10, 15, 16), (1, 2, 7, 8, 11, 12, 13, 14))
T: Ax = ((3, 4, 7, 8, 9, 10, 13, 14), (1, 2, 5, 6, 11, 12, 15, 16))
AD: Ax = ((5, 6, 7, 8, 9, 10, 11, 12), (1, 2, 3, 4, 13, 14, 15, 16))
AB: Ax = ((9, 10, 11, 12, 13, 14, 15, 16), (1, 2, 3, 4, 5, 6, 7, 8))
SAB: Ax = ((11, 12, 13, 14), (3, 4, 5, 6))
SAB_OTHER: Ax = ((1, 2, 7, 8, 9, 10, 15, 16), (1, 2, 7, 8, 9, 10, 15, 16))
TAD: Ax = ((7, 8, 9, 10), (1, 2, 15, 16))
TAD_OTHER: Ax = ((3, 4, 5, 6, 11, 12, 13, 14), (3, 4, 5, 6, 11, 12, 13, 14))

SIR: Ax = ((1, 13), (6, 10))  # (NERr, SIR)
NIR: Ax = ((5, 9), (2, 14))  # (SERr, NIR)
SER: Ax = ((7, 11), (4, 16))  # (NIRr, SER)
NER: Ax = ((3, 15), (8, 12))  # (SIRr, NER)

FIR: Ax = ((1, 5), (10, 14))  # (TERr, FIR)
TIR: Ax = ((9, 13), (2, 6))  # (FERr, TIR)
FER: Ax = ((11, 15), (4, 8))  # (TIRr, FER)
TER: Ax = ((3, 7), (12, 16))  # (FIRr, TER)

DIR: Ax = ((1, 9), (6, 14))  # (QERr, DIR)
QIR: Ax = ((5, 13), (2, 10))  # (DERr, QIR)
DER: Ax = ((7, 15), (4, 12))  # (QIRr, DER)
QER: Ax = ((3, 11), (8, 16))  # (DIRr, QER)


def inv(ax: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """ inversion """
    return ax[1], ax[0]


def tpl(*ax: Tuple[int, ...]) -> Tuple[Tuple[int, ...], ...]:
    return ax


def missing_types(ax: Ax) -> Tuple[int, ...]:
    neg, pos = ax
    used = neg + pos
    return tuple(i for i in range(1, 17) if i not in used)


KHAX1, KHAX0 = tuple(tpl(
        inv(SIR)[i], NIR[i], inv(SER)[i], NER[i],
        inv(FIR)[i], TIR[i], inv(FER)[i], TER[i],
        inv(DIR)[i], QIR[i], inv(DER)[i], QER[i]
    ) for i in (1, 0))
KHAX_OTHER = tuple(missing_types(ax)
                   for ax in (inv(SIR), NIR, inv(SER), NER,
                              inv(FIR), TIR, inv(FER), TER,
                              inv(DIR), QIR, inv(DER), QER))


class JATSRegularizerTested(nn.Module):
    q_subsets: Tuple[Tuple[int, ...], ...]
    thr: Tuple[Tensor]

    thrs_: Tuple[tuple, ...] = (
        (1.8, 1.8, tuple(1.8 for _ in range(7))),
        (0.5, 1., tuple(1.5 for _ in range(7))),
        (0., 1., tuple(1.2 for _ in range(7)))
    )
    disabled: bool = False
    sigmoid_scl: float = 1

    def __init__(self):
        """ Loss reduction is sum. """
        super(JATSRegularizerTested, self).__init__()
        self.zero = tr.tensor([0.]).mean()
        self.inv_rot_45 = self.get_inv_rot(tr.tensor(0.25))
        self.thrs = Tensor(self.expand_thrs(self.thrs_))
        self.thr = (self.thrs[-1],)

        self.q_subsets = self.get_q_subsets()
        if len(self.q_subsets) != 16: raise RuntimeError

    def expand_thrs(self, thrs_: Tuple[tuple, ...]) -> Tuple[Tuple[float, ...], ...]:
        return tuple(
            tuple(thrs[0] for _ in range(KHAXDIM))
            + tuple(thrs[1] for _ in range(KHAXDIM))
            + thrs[2]
            for thrs in thrs_
        )

    @staticmethod
    def get_khaxes() -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]:
        return (KHAX0[:KHAXDIM] + KHAX_OTHER[:KHAXDIM],
                KHAX1[:KHAXDIM] + KHAX_OTHER[:KHAXDIM])

    def get_q_types(self) -> Tuple[Tuple[int, ...], ...]:
        khaxes, axes = self.get_khaxes(), self.get_axes()
        return (khaxes[1] + axes[1] +  # MAP POS
                khaxes[0] + axes[0])  # MAP NEG

    def get_q_subsets(self, verbose: bool = False) -> Tuple[Tuple[int, ...], ...]:
        """
        Classes from  ``self.get_q_types()`` should be from 1..16.
        :returns q_subsets spec for each of 16 types.
        """
        q_subsets_: List[List[int]] = [[] for _ in range(1, 17)]
        for axis_i, axis_types in enumerate(self.get_q_types()):
            for type_ in axis_types:
                q_subsets_[type_ - 1].append(axis_i)
        q_subsets = tuple(tuple(sorted(subset)) for subset in q_subsets_)
        if verbose:
            print('types specs: ', [[i + 1, spec] for i, spec in enumerate(q_subsets)])
        return q_subsets

    # @staticmethod
    # def get_transform() -> Tensor:
    #     cos, sin, pi = math.cos, math.sin, math.pi
    #     rot45 = tr.tensor([[cos(pi / 4), -sin(pi / 4)],
    #                        [sin(pi / 4), +cos(pi / 4)]])
    #     rot_inv_45 = rot45.transpose(0, 1)
    #     return rot_inv_45

    @staticmethod
    def get_inv_rot(angle_pi_mult: Tensor) -> Tensor:
        angle = angle_pi_mult * tr.pi
        return tr.tensor([[tr.cos(angle), -tr.sin(angle)],
                          [tr.sin(angle), +tr.cos(angle)]]).transpose(0, 1)

    def rot_transform(self, z: Tensor) -> Tensor:
        """
        >>> '''
        >>> [TBg] {T} [TAd]   [FAd] {Ad} [TAd]
        >>>     Bg=>{Ad}            F=>{T}
        >>>  FBg   F   FAd     FBg   Bg   TBg
        >>> [-ER] {-R} [-IR]  [+ER] {E} [-IR]
        >>>       I=>{E}          +R=>{-R}
        >>>  +IR   +R  +ER     +IR   I   -ER
        >>> [IAd] {Ad} [EAd]  [EBg] {E} [EAd]
        >>>       I=>{E}          Bg=>{Ad}
        >>>  IBg   Bg   EBg    IBg   I   IAd
        >>> '''
        Original axes => New axes after rot(-45)
        0. S=>[N]     => S=>{N}
        1. FBg=>[TAd] => Bg=>{Ad}
        2. FAd=>[TBg] => F=>{T}
        3. +IR=>[-IR] => +R=>{-R}
        4. -ER=>[+ER] => I=>{E}
        5. IBg=>[EAd] => Bg=>{Ad}
        6. IAd=>[EBg] => I=>{E}
        """
        return tr.cat([
            z[:, (0,)],
            z[:, (1, 2)] @ self.inv_rot_45,
            z[:, (3, 4)] @ self.inv_rot_45,
            z[:, (5, 6)] @ self.inv_rot_45,
        ], dim=1)

    @staticmethod
    def get_axes() -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]:
        axes = tuple(tpl(
            N[i],
            AD[i], T[i],  # rot (TAd, TBg)
            inv(R)[i], E[i],  # rot (-IR, +ER)
            AD[i], E[i]  # rot (EAd, EBg)
        ) for i in (0, 1))
        return axes[0], axes[1]

    def set_threshold(self, value: int) -> None:
        """
        Sets threshold.

        :param value: (value - 1) - index of the thresholds set from self.thrs.
          If value == 0 then set self.disabled=True
        """
        if value == 0:
            self.disabled = True
            return
        if value >= 1:
            self.thr = (self.thrs[value - 1],)
            self.disabled = False
            return
        raise ValueError('value < 0')

    def get_q(self, z_: Tensor) -> Tensor:
        """
        Returns array of quasi-probabilities vectors (batch_size, len(q_types)).
        Elements of the quasi-probability vectors correspond with ``self.q_types`` (same length).
        """
        q_pos = tr.sigmoid((z_ + self.thr[0]) * self.sigmoid_scl)
        q_neg = tr.sigmoid((-z_ + self.thr[0]) * self.sigmoid_scl)
        return tr.cat([q_pos, q_neg], dim=1)

    def axes_cross_entropy(self, z_: Tensor, y: Tensor) -> Tensor:
        if self.disabled:
            return self.zero
        q = self.get_q(z_)
        axes_neg_cross_entropy_ = [
            tr.log(q[mask][:, subset] + 1e-8).sum()
            for subset, mask in zip(self.q_subsets, (y == i for i in range(16)))
            if z_[mask].shape[0] > 0
        ]  # list of tensors of shape (batch_subset_size,)
        return sum(axes_neg_cross_entropy_) * (-1)

    def forward(self, z: Tensor, subdec_z: Tensor, y: Tensor) -> Tensor:
        """ Loss reduction is sum. """
        return self.axes_cross_entropy(tr.cat([subdec_z, subdec_z, self.rot_transform(z)], dim=1), y)

    def extra_repr(self) -> str:
        return (f'thrs_={self.thrs_}, disabled={self.disabled}' +
                f'khaxes={self.get_khaxes()}, axes={self.get_axes()}, sigmoid_scl={self.sigmoid_scl}')


# there was an axis found that has something like 12,13,14 <=> 6,8,(16,10)
class JATSRegularizerUntested(JATSRegularizerTested):
    thrs_: Tuple[tuple, ...] = ((1.8, 1.8, (1.8, 4.0, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8)),
                                (0.5, 1.0, (1.5, 4.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5)),
                                (0.0, 1.0, (1.2, 4.0, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2)),
                                (0.0, 1.0, (1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2)),
                                )
    angle_pi_mults = [0.167, 0.167, 0.25, 0.25]  # [0.0833, 0.167, 0.25]

    def __init__(self):
        """ Loss reduction is sum. """
        super(JATSRegularizerUntested, self).__init__()
        self.inv_rotations = [self.get_inv_rot(tr.tensor(angle_pi_mult)) for angle_pi_mult in self.angle_pi_mults]
        self.thr = (self.thrs[-1],)
        self.lastthr = True

    def rot_transform(self, z: Tensor) -> Tensor:
        """
        >>> '''
        >>> [NGd] {N} [NAb]   [SAb] {Ab} [NAb]
        >>>     Gd=>{Ab}            S=>{N}
        >>>  SGd   S   SAb     SGd   Gd   NGd
        >>>
        >>> [TBg] {T} [TAd]   [FAd] {Ad} [TAd]
        >>>     Bg=>{Ad}            F=>{T}
        >>>  FBg   F   FAd     FBg   Bg   TBg
        >>>
        >>> [-ER] {-R} [-IR]  [+ER] {E} [-IR]
        >>>       I=>{E}          +R=>{-R}
        >>>  +IR   +R  +ER     +IR   I   -ER
        >>>
        >>> [IAd] {Ad} [EAd]  [EBg] {E} [EAd]
        >>>       I=>{E}          Bg=>{Ad}
        >>>  IBg   Bg   EBg    IBg   I   IAd
        >>> '''
        Basis axes => New axes after rot(-alpha)

        SGd=>[NAb] => S=>{N}
        NGd=>[SAb] => Gd=>{Ab}

        FBg=>[TAd] => Bg=>{Ad}
        FAd=>[TBg] => F=>{T}

        +IR=>[-IR] => +R=>{-R}
        -ER=>[+ER] => I=>{E}

        IBg=>[EAd] => Bg=>{Ad}
        IAd=>[EBg] => I=>{E}
        """
        return tr.cat([
            z[:, (0, 7)] @ self.inv_rotations[0] if self.lastthr else z[:, (0, 1)],
            z[:, (1, 2)] @ self.inv_rotations[2 if self.lastthr else 1],
            z[:, (3, 4)] @ self.inv_rotations[2],
            z[:, (5, 6)] @ self.inv_rotations[3],
        ], dim=1)

    def cat_rot_2d(self, z: Array) -> Array:
        return np.concatenate([
            z[:, (0, 7)],  # @ self.inv_rotations[0].numpy(),
            z[:, (1, 2)] @ self.inv_rotations[1].numpy(),
            z[:, (3, 4)] @ self.inv_rotations[2].numpy(),
            z[:, (5, 6)] @ self.inv_rotations[3].numpy(),
        ], axis=1)

    @staticmethod
    def get_axes() -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]:
        axes = tuple(tpl(
            N[i], AB[i],
            AD[i], T[i],
            inv(R)[i], E[i],
            AD[i], E[i],
            # N[i], SAB[i], SAB_OTHER[i],
            # TAD_OTHER[i], TAD[i], T[i],
            # E[i], inv(R)[i],
        ) for i in (0, 1))
        return axes[0], axes[1]

    def expand_thrs(self, thrs_: Tuple[tuple, ...]) -> Tuple[Tuple[float, ...], ...]:
        return tuple(
            tuple(thrs[0] for _ in range(KHAXDIM))
            + tuple(thrs[1] for _ in range(KHAXDIM))
            + thrs[2]
            for thrs in thrs_
        )

    def set_threshold(self, value: int) -> None:
        """
        Sets threshold.

        :param value: (value - 1) - index of the thresholds set from self.thrs.
          If value == 0 then set self.disabled=True
        """
        if value == 0:
            self.disabled = True
            return
        if value >= 1:
            self.thr = (self.thrs[value - 1],)
            self.disabled = False
            if value == self.thrs.shape[0]:
                self.lastthr = True
            else:
                self.lastthr = False
            return
        raise ValueError('value < 0')

    def extra_repr(self) -> str:
        return super(JATSRegularizerUntested, self).extra_repr() + f'angle_pi_mults={self.angle_pi_mults}'
