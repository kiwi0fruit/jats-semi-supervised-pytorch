from typing import Tuple, List, Union
import torch as tr
from torch import Tensor

Funcs = Tuple[Tuple[int, ...], ...]

CF1: Funcs = (
    (7, 11),  # N±ER
    (1, 13),  # N±IR
    (3, 15),  # S±ER
    (5, 9),  # S±IR
    (2, 6),  # T±IR
    (12, 16),  # T±ER
    (10, 14),  # F±IR
    (4, 8),  # F±ER
    (2, 10),  # Q±IR
    (8, 16),  # Q±ER
    (6, 14),  # D±IR
    (4, 12),  # D±ER
)
CF2: Funcs = (
    (8, 12),  # N±ER
    (2, 14),  # N±IR
    (4, 16),  # S±ER
    (6, 10),  # S±IR
    (1, 5),  # T±IR
    (11, 15),  # T±ER
    (9, 13),  # F±IR
    (3, 7),  # F±ER
    (1, 9),  # Q±IR
    (7, 15),  # Q±ER
    (5, 13),  # D±IR
    (3, 11),  # D±ER
)
CF3: Funcs = (
    (3, 15),  # N±ER
    (5, 9),  # N±IR
    (7, 11),  # S±ER
    (1, 13),  # S±IR
    (10, 14),  # T±IR
    (4, 8),  # T±ER
    (2, 6),  # F±IR
    (12, 16),  # F±ER
    (6, 14),  # Q±IR
    (4, 12),  # Q±ER
    (2, 10),  # D±IR
    (8, 16),  # D±ER
)
CF4: Funcs = (
    (4, 16),  # N±ER
    (6, 10),  # N±IR
    (8, 12),  # S±ER
    (2, 14),  # S±IR
    (9, 13),  # T±IR
    (3, 7),  # T±ER
    (1, 5),  # F±IR
    (11, 15),  # F±ER
    (5, 13),  # Q±IR
    (3, 11),  # Q±ER
    (1, 9),  # D±IR
    (7, 15),  # D±ER
)
CF5: Funcs = (
    (5, 9),  # N±ER
    (3, 15),  # N±IR
    (1, 13),  # S±ER
    (7, 11),  # S±IR
    (4, 8),  # T±IR
    (10, 14),  # T±ER
    (12, 16),  # F±IR
    (2, 6),  # F±ER
    (4, 12),  # Q±IR
    (6, 14),  # Q±ER
    (8, 16),  # D±IR
    (2, 10),  # D±ER
)
CF6: Funcs = (
    (6, 10),  # N±ER
    (4, 16),  # N±IR
    (2, 14),  # S±ER
    (8, 12),  # S±IR
    (3, 7),  # T±IR
    (9, 13),  # T±ER
    (11, 15),  # F±IR
    (1, 5),  # F±ER
    (3, 11),  # Q±IR
    (5, 13),  # Q±ER
    (7, 15),  # D±IR
    (1, 9),  # D±ER
)
CF7: Funcs = (
    (1, 13),  # N±ER
    (7, 11),  # N±IR
    (5, 9),  # S±ER
    (3, 15),  # S±IR
    (12, 16),  # T±IR
    (2, 6),  # T±ER
    (4, 8),  # F±IR
    (10, 14),  # F±ER
    (8, 16),  # Q±IR
    (2, 10),  # Q±ER
    (4, 12),  # D±IR
    (6, 14),  # D±ER
)
CF8: Funcs = (
    (2, 14),  # N±ER
    (8, 12),  # N±IR
    (6, 10),  # S±ER
    (4, 16),  # S±IR
    (11, 15),  # T±IR
    (1, 5),  # T±ER
    (3, 7),  # F±IR
    (9, 13),  # F±ER
    (7, 15),  # Q±IR
    (1, 9),  # Q±ER
    (3, 11),  # D±IR
    (5, 13),  # D±ER
)
V15: Funcs = (
    (7, 11, 5, 9),  # N±ER S±IR
    (1, 13, 3, 15),  # N±IR S±ER
    (2, 6, 4, 8),  # T±IR F±ER
    (12, 16, 10, 14),  # T±ER F±IR
    (2, 10, 4, 12),  # Q±IR D±ER
    (8, 16, 6, 14),  # Q±ER D±IR
)
V26: Funcs = (
    (8, 12, 6, 10),  # N±ER S±IR
    (2, 14, 4, 16),  # N±IR S±ER
    (1, 5, 3, 7),  # T±IR F±ER
    (11, 15, 9, 13),  # T±ER F±IR
    (1, 9, 3, 11),  # Q±IR D±ER
    (7, 15, 5, 13),  # Q±ER D±IR
)
V37: Funcs = (
    (1, 13, 3, 15),  # N±ER S±IR
    (7, 11, 5, 9),  # N±IR S±ER
    (12, 16, 10, 14),  # T±IR F±ER
    (2, 6, 4, 8),  # T±ER F±IR
    (8, 16, 6, 14),  # Q±IR D±ER
    (2, 10, 4, 12),  # Q±ER D±IR
)
V48: Funcs = (
    (2, 14, 4, 16),  # N±ER S±IR
    (8, 12, 6, 10),  # N±IR S±ER
    (11, 15, 9, 13),  # T±IR F±ER
    (1, 5, 3, 7),  # T±ER F±IR
    (7, 15, 5, 13),  # Q±IR D±ER
    (1, 9, 3, 11),  # Q±ER D±IR
)

Ax = Tuple[Tuple[int, ...], Tuple[int, ...]]


def missing_types(ax: Ax) -> Tuple[int, ...]:
    neg, pos = ax
    used = neg + pos
    return tuple(i for i in range(1, 17) if i not in used)


def inter(ax1: Tuple[Tuple[int, ...], Tuple[int, ...]],
          ax2: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """ intersection """
    neg1, pos1 = ax1
    neg2, pos2 = ax2
    return tuple(i for i in neg1 if i in neg2), tuple(i for i in pos1 if i in pos2)


def cat(ax1: Tuple[Tuple[int, ...], Tuple[int, ...]],
        ax2: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """ cat """
    neg1, pos1 = ax1
    neg2, pos2 = ax2
    return neg1 + neg2, pos1 + pos2


def inv(ax: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """ inversion """
    return ax[1], ax[0]


def tpl(*ax: Tuple[int, ...]) -> Tuple[Tuple[int, ...], ...]:
    return ax


def cond(condition: Union[Tuple[int, ...], int], *ax: Tuple[int, ...], const: int) -> Tuple[Tuple[int, ...], ...]:
    if isinstance(condition, int):
        if const != condition:
            return ()
    elif isinstance(condition, tuple):
        if const not in condition:
            return ()
    return ax


def swap(target: Ax, swap_subset: Ax) -> Ax:
    neg, pos = target
    neg2, pos2 = swap_subset
    return tuple(i for i in neg if i not in neg2) + pos2, tuple(i for i in pos if i in pos2) + neg2


E: Ax = ((2, 3, 6, 7, 10, 11, 14, 15), (1, 4, 5, 8, 9, 12, 13, 16))
R: Ax = ((1, 3, 5, 7, 9, 11, 13, 15), (2, 4, 6, 8, 10, 12, 14, 16))
STATIC: Ax = ((3, 4, 7, 8, 11, 12, 15, 16), (1, 2, 5, 6, 9, 10, 13, 14))

N: Ax = ((3, 4, 5, 6, 9, 10, 15, 16), (1, 2, 7, 8, 11, 12, 13, 14))
NESI: Ax = ((2, 4, 5, 7, 9, 11, 14, 16), (1, 3, 6, 8, 10, 12, 13, 15))
NRRSR: Ax = ((2, 3, 5, 8, 9, 12, 14, 15), (1, 4, 6, 7, 10, 11, 13, 16))
AD: Ax = ((5, 6, 7, 8, 9, 10, 11, 12), (1, 2, 3, 4, 13, 14, 15, 16))

T: Ax = ((3, 4, 7, 8, 9, 10, 13, 14), (1, 2, 5, 6, 11, 12, 15, 16))
TEFI: Ax = ((2, 4, 6, 8, 9, 11, 13, 15), (1, 3, 5, 7, 10, 12, 14, 16))
TRRFR: Ax = ((2, 3, 6, 7, 9, 12, 13, 16), (1, 4, 5, 8, 10, 11, 14, 15))
AB: Ax = ((9, 10, 11, 12, 13, 14, 15, 16), (1, 2, 3, 4, 5, 6, 7, 8))

Q: Ax = ((3, 4, 5, 6, 11, 12, 13, 14), (1, 2, 7, 8, 9, 10, 15, 16))
QEDI: Ax = ((2, 4, 5, 7, 10, 12, 13, 15), (1, 3, 6, 8, 9, 11, 14, 16))
QRRDR: Ax = ((2, 3, 5, 8, 10, 11, 13, 16), (1, 4, 6, 7, 9, 12, 14, 15))
AG: Ax = ((5, 6, 7, 8, 13, 14, 15, 16), (1, 2, 3, 4, 9, 10, 11, 12))

AD12: Ax = ((5, 6, 7, 8, 9, 10, 11), (1, 2, 3, 4, 13, 14, 15, 16, 12))  # (Bg12, Ad12)


# SIR, NIR, SER, NER,  FIR, TIR, FER, TER, DIR, QIR, DER, QER
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

# ENT, INT, ENF, INF,  ETQ, ITQ, EFQ, IFQ,  ENQ, INQ, ESQ, ISQ
ENT: Ax = ((3, 10), (1, 12))
INT: Ax = ((4, 9), (2, 11))
ENF: Ax = ((6, 15), (8, 13))
INF: Ax = ((5, 16), (7, 14))

ETQ: Ax = ((3, 14), (1, 16))
ITQ: Ax = ((4, 13), (2, 15))
EFQ: Ax = ((6, 11), (8, 9))
IFQ: Ax = ((5, 12), (7, 10))

ENQ: Ax = ((3, 6), (1, 8))
INQ: Ax = ((4, 5), (2, 7))
ESQ: Ax = ((11, 14), (9, 16))
ISQ: Ax = ((12, 13), (10, 15))

# NTR, SFR, STR, NFR,  TQR, FDR, TDR, FQR,  QNR, DSR, QSR, DNR
NTR: Ax = ((3, 9), (2, 12))
SFR: Ax = ((1, 11), (4, 10))
STR: Ax = ((7, 13), (6, 16))
NFR: Ax = ((5, 15), (8, 14))

TQR: Ax = ((3, 13), (2, 16))
FDR: Ax = ((1, 15), (4, 14))
TDR: Ax = ((7, 9), (6, 12))
FQR: Ax = ((5, 11), (8, 10))

QNR: Ax = ((3, 5), (2, 8))
DSR: Ax = ((1, 7), (4, 6))
QSR: Ax = ((11, 13), (10, 16))
DNR: Ax = ((9, 15), (12, 14))

# NTBg, SFBg, STBg, NFBg
NTBG: Ax = ((3, 4), (11, 12))
SFBG: Ax = ((1, 2), (9, 10))
STBG: Ax = ((13, 14), (5, 6))
NFBG: Ax = ((15, 16), (7, 8))

# IFBg, EFBg, ETBg, ITBg
IFBG: Ax = ((1, 16), (7, 10))
EFBG: Ax = ((2, 15), (8, 9))
ETBG: Ax = ((3, 14), (5, 12))
ITBG: Ax = ((4, 13), (6, 11))

# FRBg, TRAd, FRAd, TRBg
FRBG: Ax = ((1, 15), (8, 10))
TRAD: Ax = ((7, 9), (2, 16))
FRAD: Ax = ((5, 11), (4, 14))
TRBG: Ax = ((3, 13), (6, 12))

# NTQ, SFQ, STD, NFD
NTQ: Ax = ((3, 4), (1, 2))      # SF±ER=>NT±IR
SFQ: Ax = ((11, 12), (9, 10))   # NT±ER=>SF±IR
STD: Ax = ((7, 8), (5, 6))      # NF±ER=>ST±IR
NFD: Ax = ((15, 16), (13, 14))  # ST±ER=>NF±IR

# ETAd, ITAd, EFAd, IFAd
ETAD: Ax = ((7, 10), (1, 16))
ITAD: Ax = ((8, 9), (2, 15))
EFAD: Ax = ((6, 11), (4, 13))
IFAD: Ax = ((5, 12), (3, 14))


TYPE8: Tuple[Ax, Ax, Ax, Ax, Ax, Ax, Ax, Ax] = (
    ((10,), (1,)),
    ((9,), (2,)),
    ((12,), (3,)),
    ((11,), (4,)),
    ((14,), (5,)),
    ((13,), (6,)),
    ((16,), (7,)),
    ((15,), (8,)),
)

TYPE16: Tuple[Ax, Ax, Ax, Ax, Ax, Ax, Ax, Ax, Ax, Ax, Ax, Ax, Ax, Ax, Ax, Ax] = (
    ((10,), (1,)),
    ((9,), (2,)),
    ((12,), (3,)),
    ((11,), (4,)),
    ((14,), (5,)),
    ((13,), (6,)),
    ((16,), (7,)),
    ((15,), (8,)),
    ((2,), (9,)),
    ((1,), (10,)),
    ((4,), (11,)),
    ((3,), (12,)),
    ((6,), (13,)),
    ((5,), (14,)),
    ((8,), (15,)),
    ((7,), (16,)),
)

PSI: Tuple[Ax, Ax, Ax, Ax, Ax, Ax, Ax, Ax, Ax, Ax, Ax, Ax, Ax, Ax, Ax, Ax] = (
    ((6, 10, 14), (1,)),
    ((5, 9, 13), (2,)),
    ((8, 12, 16), (3,)),
    ((7, 11, 15), (4,)),
    ((2, 10, 14), (5,)),
    ((1, 9, 13), (6,)),
    ((4, 12, 16), (7,)),
    ((3, 11, 15), (8,)),
    ((2, 6, 14), (9,)),
    ((1, 5, 13), (10,)),
    ((4, 8, 16), (11,)),
    ((3, 7, 15), (12,)),
    ((2, 6, 10), (13,)),
    ((1, 5, 9), (14,)),
    ((4, 8, 12), (15,)),
    ((3, 7, 11), (16,)),
)


MAP_12KHAX_X_8RNKHAX = [
    # +E +Rr +N  +Ad +T  +Ab +Q  +Ag
    # -I -R  -S  -Bg -F  -Gd -D  -Bd
    [+1,  1,  3,  3,  0,  0,  0,  0],  # SIR=>NERr
    [-1, -1,  3,  3,  0,  0,  0,  0],  # SERr=>NIR
    [-1,  1,  3, -3,  0,  0,  0,  0],  # SER=>NIRr
    [+1, -1,  3, -3,  0,  0,  0,  0],  # SIRr=>NER
    [+1,  1,  0,  0,  3,  3,  0,  0],  # FIR=>TERr
    [-1, -1,  0,  0,  3,  3,  0,  0],  # FERr=>TIR
    [-1,  1,  0,  0,  3, -3,  0,  0],  # FER=>TIRr
    [+1, -1,  0,  0,  3, -3,  0,  0],  # FIRr=>TER
    [+1,  1,  0,  0,  0,  0,  3,  3],  # DIR=>QERr
    [-1, -1,  0,  0,  0,  0,  3,  3],  # DERr=>QIR
    [-1,  1,  0,  0,  0,  0,  3, -3],  # DER=>QIRr
    [+1, -1,  0,  0,  0,  0,  3, -3],  # DIRr=>QER
]

MAP_ABS12KHAX_X_7COMPLRNKHAX = [
    # +static
    #        +NRrSR  +TRrFR  +QRrDR
    #    +NESI   +TEFI   +QEDI
    [+1,  3,  3,  0,  0,  0,  0],  # |SIR=>NERr|
    [+1, -3, -3,  0,  0,  0,  0],  # |SERr=>NIR|
    [-1, -3,  3,  0,  0,  0,  0],  # |SER=>NIRr|
    [-1,  3, -3,  0,  0,  0,  0],  # |SIRr=>NER|
    [+1,  0,  0,  3,  3,  0,  0],  # |FIR=>TERr|
    [+1,  0,  0, -3, -3,  0,  0],  # |FERr=>TIR|
    [-1,  0,  0, -3,  3,  0,  0],  # |FER=>TIRr|
    [-1,  0,  0,  3, -3,  0,  0],  # |FIRr=>TER|
    [+1,  0,  0,  0,  0,  3,  3],  # |DIR=>QERr|
    [+1,  0,  0,  0,  0, -3, -3],  # |DERr=>QIR|
    [-1,  0,  0,  0,  0, -3,  3],  # |DER=>QIRr|
    [-1,  0,  0,  0,  0,  3, -3],  # |DIRr=>QER|
]

MAP_8AKHAX_X_8RNKHAX = [
    # +E +Rr +N +Ad +T +Ab +Q +Ag
    # -I -R  -S -Bg -F -Gd -D -Bd
    [1,  1,  0,  0,  0,  0,  0,  0],  # IR=>ERr
    [1, -1,  0,  0,  0,  0,  0,  0],  # IRr=>ER
    [0,  0,  1,  1,  0,  0,  0,  0],  # S±IR=>N±IR
    [0,  0,  1, -1,  0,  0,  0,  0],  # S±ER=>N±ER
    [0,  0,  0,  0,  1,  1,  0,  0],  # F±IR=>T±IR
    [0,  0,  0,  0,  1, -1,  0,  0],  # F±ER=>T±ER
    [0,  0,  0,  0,  0,  0,  1,  1],  # D±IR=>Q±IR
    [0,  0,  0,  0,  0,  0,  1, -1],  # D±ER=>Q±ER
]

MAP_12KHAX_X_8AKHAX = [
    # +ERr+ER +N±IR +N±ER +T±IR +T±ER +Q±IR +Q±ER
    # -IR -IRr-S±IR -S±ER -F±IR -F±ER -D±IR -D±ER
    [+1,  0,  3,  0,  0,  0,  0,  0],  # SIR=>NERr
    [-1,  0,  3,  0,  0,  0,  0,  0],  # SERr=>NIR
    [0., -1,  0,  3,  0,  0,  0,  0],  # SER=>NIRr
    [0.,  1,  0,  3,  0,  0,  0,  0],  # SIRr=>NER
    [+1,  0,  0,  0,  3,  0,  0,  0],  # FIR=>TERr
    [-1,  0,  0,  0,  3,  0,  0,  0],  # FERr=>TIR
    [0., -1,  0,  0,  0,  3,  0,  0],  # FER=>TIRr
    [0.,  1,  0,  0,  0,  3,  0,  0],  # FIRr=>TER
    [+1,  0,  0,  0,  0,  0,  3,  0],  # DIR=>QERr
    [-1,  0,  0,  0,  0,  0,  3,  0],  # DERr=>QIR
    [0., -1,  0,  0,  0,  0,  0,  3],  # DER=>QIRr
    [0.,  1,  0,  0,  0,  0,  0,  3],  # DIRr=>QER
]

MAP_ABS8AKHAX_X_STATAX = [
    # DYNAMIC=>STATIC
    [+1],  # |IR=>ERr|
    [-1],  # |IRr=>ER|
    [+1],  # |S±IR=>N±IR|
    [-1],  # |S±ER=>N±ER|
    [+1],  # |F±IR=>T±IR|
    [-1],  # |F±ER=>T±ER|
    [+1],  # |D±IR=>Q±IR|
    [-1],  # |D±ER=>Q±ER|
]


# Select 3 axes from 5 ENTRC axes:
# ENT.., EN.R., EN..Bg,    E.TR., E.T.Bg, E..RBg,    .NTR., .NT.Bg, .N.RBg,    ..TRBg
a = 1
a_ = a + 3
b, c, g = a_ / 6, a_ / 2, a_ / (a + 2)
MAP_8KHAX20EXTRAPAIRS_X_6RNKHAX = [
    # +E +Rr +N  +Ad +T  +Ab
    # -I -R  -S  -Bg -F  -Gd
    # EN.R. = EN..Bg = E..RBg = .N.RBg:
    [+a,  a,  a,  a,  0,  0],  # ISxRBg=>ENxRrAd  6,10=>1,13
    [-a, -a,  a,  a,  0,  0],  # ESxRrBg=>INxRAd   5,9=>2,14
    [-a,  a,  a, -a,  0,  0],  # ESxRAd=>INxRrBg  4,16=>7,11
    [+a, -a,  a, -a,  0,  0],  # ISxRrAd=>ENxRBg  3,15=>8,12
    # E.TR.:
    [+1,  1,  0,  0,  b,  c],  # IxFR=>ExTRr   10,14=>1,5
    [-1, -1,  0,  0,  b,  c],  # ExFRr=>IxTR    9,13=>2,6
    [-1,  1,  0,  0,  b, -c],  # ExFR=>IxTRr     4,8=>11,15
    [+1, -1,  0,  0,  b, -c],  # IxFRr=>ExTR     3,7=>12,16
    # ENT..:
    [+1,  0,  1,  0,  b,  0],  # ISFx=>ENTx    3,10=>1,12
    [-1,  0,  1,  0,  b,  0],  # ESFx=>INTx     4,9=>2,11
    [+1,  0,  1,  0, -b,  0],  # ISTx=>ENFx    6,15=>8,13
    [-1,  0,  1,  0, -b,  0],  # ESTx=>INFx    5,16=>7,14
    # .NTR.:
    [0.,  1,  1,  0,  b,  0],  # xSFR=>xNTRr    4,10=>1,11
    [0., -1,  1,  0,  b,  0],  # xSFRr=>xNTR     3,9=>2,12
    [0.,  1,  1,  0, -b,  0],  # xSTR=>xNFRr    6,16=>7,13
    [0., -1,  1,  0, -b,  0],  # xSTRr=>xNFR    5,15=>8,14
    # .NT.Bg:
    [0.,  0,  1,  1,  b,  c],  # xSFxBg=>xNTxAd  9,10=>1,2
    [0.,  0,  1, -1,  b, -c],  # xSFxAd=>xNTxBg   3,4=>11,12
    [0.,  0,  1,  1, -b, -c],  # xSTxBg=>xNFxAd   6,5=>13,14
    [0.,  0,  1, -1, -b,  c],  # xSTxAd=>xNFxBg 15,16=>7,8
    # E.T.Bg:
    [+1,  0,  0,  1,  b,  0],  # IxFxBg=>ExTxAd  7,10=>1,16
    [-1,  0,  0,  1,  b,  0],  # ExFxBg=>IxTxAd   8,9=>2,15
    [+1,  0,  0, -1,  b,  0],  # IxFxAd=>ExTxBg  3,14=>5,12
    [-1,  0,  0, -1,  b,  0],  # ExFxAd=>IxTxBg  4,13=>6,11
    # ..TRBg:
    [0.,  1,  0,  1,  b,  0],  # xxFRBg=>xxTRrAd  8,10=>1,15
    [0., -1,  0,  1,  b,  0],  # xxFRrBg=>xxTRAd   7,9=>2,16
    [0.,  1,  0, -1,  b,  0],  # xxFRAd=>xxTRrBg  4,14=>5,11
    [0., -1,  0, -1,  b,  0],  # xxFRrAd=>xxTRBg  3,13=>6,12
    # +E -R  +N  +Ad +T  +Ab
    # -I +R  -S  -Bg -F  -Gd
]

MAP_ABS8KHAX20EXTRAPAIRS_X_STATAX = [
    # +static
    [+a * g],  # ISxRBg=>ENxRrAd 6,10=>1,13
    [+a * g],  # ESxRrBg=>INxRAd  5,9=>2,14
    [-a * g],  # ESxRAd=>INxRrBg 4,16=>7,11
    [-a * g],  # ISxRrAd=>ENxRBg  3,15=>8,12

    [+g],  # IxFR=>ExTRr   10,14=>1,5
    [+g],  # ExFRr=>IxTR    9,13=>2,6
    [-g],  # ExFR=>IxTRr     4,8=>11,15
    [-g],  # IxFRr=>ExTR     3,7=>12,16

    [0.],  # ISFx=>ENTx    3,10=>1,12
    [0.],  # ESFx=>INTx     4,9=>2,11
    [0.],  # ISTx=>ENFx    6,15=>8,13
    [0.],  # ESTx=>INFx    5,16=>7,14

    [0.],  # xSFR=>xNTRr    4,10=>1,11
    [0.],  # xSFRr=>xNTR     3,9=>2,12
    [0.],  # xSTR=>xNFRr    6,16=>7,13
    [0.],  # xSTRr=>xNFR    5,15=>8,14

    [+g],  # xSFxBg=>xNTxAd  9,10=>1,2
    [-g],  # xSFxAd=>xNTxBg   3,4=>11,12
    [+g],  # xSTxBg=>xNFxAd   6,5=>13,14
    [-g],  # xSTxAd=>xNFxBg 15,16=>7,8

    [0.],  # IxFxBg=>ExTxAd  7,10=>1,16
    [0.],  # ExFxBg=>IxTxAd   8,9=>2,15
    [0.],  # IxFxAd=>ExTxBg  3,14=>5,12
    [0.],  # ExFxAd=>IxTxBg  4,13=>6,11

    [0.],  # xxFRBg=>xxTRrAd  8,10=>1,15
    [0.],  # xxFRrBg=>xxTRAd   7,9=>2,16
    [0.],  # xxFRAd=>xxTRrBg  4,14=>5,11
    [0.],  # xxFRrAd=>xxTRBg  3,13=>6,12
]


MAP_8KHAX4EXTRAPAIRS_X_6RNKHAX = [
    # +E +Rr +N  +Ad +T  +Ab
    # -I -R  -S  -Bg -F  -Gd
    [+1,  1,  1,  1,  0,  0],  # ISxR=>ENxRr    6,10=>1,13
    [-1, -1,  1,  1,  0,  0],  # ESxRr=>INxR     5,9=>2,14
    [-1,  1,  1, -1,  0,  0],  # ESxR=>INxRr    4,16=>7,11
    [+1, -1,  1, -1,  0,  0],  # ISxRr=>ENxR    3,15=>8,12
    [+1,  1,  0,  0,  1,  1],  # IxFR=>ExTRr   10,14=>1,5
    [-1, -1,  0,  0,  1,  1],  # ExFRr=>IxTR    9,13=>2,6
    [-1,  1,  0,  0,  1, -1],  # ExFR=>IxTRr     4,8=>11,15
    [+1, -1,  0,  0,  1, -1],  # IxFRr=>ExTR     3,7=>12,16
    [0.,  0,  1,  1,  1,  1],  # xSFxBg=>xNTxAd  9,10=>1,2
    [0.,  0,  1, -1,  1, -1],  # xSFxAd=>xNTxBg   3,4=>11,12
    [0.,  0,  1,  1, -1, -1],  # xSTxBg=>xNFxAd   6,5=>13,14
    [0.,  0,  1, -1, -1,  1],  # xSTxAd=>xNFxBg 15,16=>7,8
]
# +QRrDR?
# +QEDI?
MAP_ABS8KHAX4EXTRAPAIRS_X_5COMPLRNKHAX2QAG = [
    # +static
    #        +NRrSR  +TRrFR
    #    +NESI   +TEFI   +Q  +Ag
    [+1,  3,  3,  0,  0,  0,  0],  # |ISxR=>ENxRr|    6,10=>1,13
    [+1, -3, -3,  0,  0,  0,  0],  # |ESxRr=>INxR|     5,9=>2,14
    [-1, -3,  3,  0,  0,  0,  0],  # |ESxR=>INxRr|    4,16=>7,11
    [-1,  3, -3,  0,  0,  0,  0],  # |ISxRr=>ENxR|    3,15=>8,12
    [+1,  0,  0,  3,  3,  0,  0],  # |IxFR=>ExTRr|   10,14=>1,5
    [+1,  0,  0, -3, -3,  0,  0],  # |ExFRr=>IxTR|    9,13=>2,6
    [-1,  0,  0, -3,  3,  0,  0],  # |ExFR=>IxTRr|     4,8=>11,15
    [-1,  0,  0,  3, -3,  0,  0],  # |IxFRr=>ExTR|     3,7=>12,16
    [+1,  0,  0,  0,  0,  3,  3],  # |xSFxBg=>xNTxAd|  9,10=>1,2
    [-1,  0,  0,  0,  0, -3,  3],  # |xSFxAd=>xNTxBg|   3,4=>11,12
    [+1,  0,  0,  0,  0, -3, -3],  # |xSTxBg=>xNFxAd|   6,5=>13,14
    [-1,  0,  0,  0,  0,  3, -3],  # |xSTxAd=>xNFxBg| 15,16=>7,8
]


MAP_8KHAX4EXTRAPAIRS2_X_6RNKHAX = [
    # +E +Rr +N  +Ad +T  +Ab
    # -I -R  -S  -Bg -F  -Gd
    [+2,  3,  6,  3,  0,  0],  # ISxR=>ENxRr    6,10=>1,13
    [-2, -3,  6,  3,  0,  0],  # ESxRr=>INxR     5,9=>2,14
    [-2,  3,  6, -3,  0,  0],  # ESxR=>INxRr    4,16=>7,11
    [+2, -3,  6, -3,  0,  0],  # ISxRr=>ENxR    3,15=>8,12
    [+2,  3,  0,  0,  3,  6],  # IxFR=>ExTRr   10,14=>1,5
    [-2, -3,  0,  0,  3,  6],  # ExFRr=>IxTR    9,13=>2,6
    [-2,  3,  0,  0,  3, -6],  # ExFR=>IxTRr     4,8=>11,15
    [+2, -3,  0,  0,  3, -6],  # IxFRr=>ExTR     3,7=>12,16
    [+2,  0,  0,  3,  3,  0],  # IxFxBg=>ExTxAd  7,10=>1,16
    [-2,  0,  0,  3,  3,  0],  # ExFxBg=>IxTxAd   8,9=>2,15
    [+2,  0,  0,  3, -3,  0],  # IxTxBg=>ExFxAd  6,11=>4,13
    [-2,  0,  0,  3, -3,  0],  # ExTxBg=>IxFxAd  5,12=>3,14
]
# +QRrDR?
# +QEDI?
# +Ag?
MAP_ABS8KHAX4EXTRAPAIRS2_X_5COMPLRNKHAX2QAG = [
    # +static
    #        +NRrSR  +TRrFR
    #    +NESI   +TEFI   +Q
    [+1,  2,  2,  0,  0,  0],  # |ISxR=>ENxRr|    6,10=>1,13
    [+1, -2, -2,  0,  0,  0],  # |ESxRr=>INxR|     5,9=>2,14
    [-1, -2,  2,  0,  0,  0],  # |ESxR=>INxRr|    4,16=>7,11
    [-1,  2, -2,  0,  0,  0],  # |ISxRr=>ENxR|    3,15=>8,12
    [+1,  0,  0,  1,  2,  0],  # |IxFR=>ExTRr|   10,14=>1,5
    [+1,  0,  0, -1, -2,  0],  # |ExFRr=>IxTR|    9,13=>2,6
    [-1,  0,  0, -1,  2,  0],  # |ExFR=>IxTRr|     4,8=>11,15
    [-1,  0,  0,  1, -2,  0],  # |IxFRr=>ExTR|     3,7=>12,16
    [+0,  0,  0,  1,  0,  2],  # |IxFxBg=>ExTxAd|  7,10=>1,16
    [+0,  0,  0, -1,  0,  2],  # |ExFxBg=>IxTxAd|   8,9=>2,15
    [+0,  0,  0, -1,  0, -2],  # |IxTxBg=>ExFxAd|  6,11=>4,13
    [+0,  0,  0,  1,  0, -2],  # |ExTxBg=>IxFxAd|  5,12=>3,14
]


MAP_8TYPEAX_X_6RNKHAX = [
    # +E -R  +N  +Ad +T  +Ab
    # -I -R  -S  -Bg -F  -Gd
    [+1,  1,  1,  1,  1,  1],  # ISFR=>ENTRr 10=>1
    [-1, -1,  1,  1,  1,  1],  # ESFRr=>INTR  9=>2
    [-1,  1, -1,  1, -1,  1],  # ENTR=>ISFRr 12=>3
    [+1, -1, -1,  1, -1,  1],  # INTRr=>ESFR 11=>4
    [+1,  1, -1, -1,  1,  1],  # INFR=>ESTRr 14=>5
    [-1, -1, -1, -1,  1,  1],  # ENFRr=>ISTR 13=>6
    [-1,  1,  1, -1, -1,  1],  # ESTR=>INFRr 16=>7
    [+1, -1,  1, -1, -1,  1],  # ISTRr=>ENFR 15=>8
]
# +QRrDR?
# +QEDI?
MAP_ABS8TYPEAX_X_5COMPLRNKHAX2QAG = [
    # +static
    #        +NRrSR  +TRrFR
    #    +NESI   +TEFI   +Q  +Ag
    [+1,  1,  1,  1,  1,  1,  1],  # |ISFR=>ENTRr| 10=>1
    [+1, -1, -1, -1, -1,  1,  1],  # |ESFRr=>INTR|  9=>2
    [-1,  1, -1,  1, -1, -1,  1],  # |ENTR=>ISFRr| 12=>3
    [-1, -1,  1, -1,  1, -1,  1],  # |INTRr=>ESFR| 11=>4
    [+1, -1, -1,  1,  1, -1, -1],  # |INFR=>ESTRr| 14=>5
    [+1,  1,  1, -1, -1, -1, -1],  # |ENFRr=>ISTR| 13=>6
    [-1, -1,  1,  1, -1,  1, -1],  # |ESTR=>INFRr| 16=>7
    [-1,  1, -1, -1,  1,  1, -1],  # |ISTRr=>ENFR| 15=>8
]

MAP_RELU8TYPEAXTO16TYPEAX_X_2QAG7COMPLRNKHAX = [
    # +Q  +Ag +static +NRrSR  +TRrFR  +QRrDR
    # -D  -Bd     +NESI   +TEFI   +QEDI
    [+1,  1,  1,  1,  1,  1,  1,  1,  1],  # RELU(ISFR=>ENTRr)  10=>1
    [+1,  1,  1, -1, -1, -1, -1, -1, -1],  # RELU(ESFRr=>INTR)   9=>2
    [-1,  1, -1,  1, -1,  1, -1,  1, -1],  # RELU(ENTR=>ISFRr)  12=>3
    [-1,  1, -1, -1,  1, -1,  1, -1,  1],  # RELU(INTRr=>ESFR)  11=>4
    [-1, -1,  1, -1, -1,  1,  1, -1, -1],  # RELU(INFR=>ESTRr)  14=>5
    [-1, -1,  1,  1,  1, -1, -1,  1,  1],  # RELU(ENFRr=>ISTR)  13=>6
    [+1, -1, -1, -1,  1,  1, -1, -1,  1],  # RELU(ESTR=>INFRr)  16=>7
    [+1, -1, -1,  1, -1, -1,  1,  1, -1],  # RELU(ISTRr=>ENFR)  15=>8
    [+1,  1,  1, -1, -1, -1, -1,  1,  1],  # RELU(-ESFRr=>-INTR) 2=>9
    [+1,  1,  1,  1,  1,  1,  1, -1, -1],  # RELU(-ISFR=>-ENTRr) 1=>10
    [-1,  1, -1, -1,  1, -1,  1,  1, -1],  # RELU(-INTRr=>-ESFR) 4=>11
    [-1,  1, -1,  1, -1,  1, -1, -1,  1],  # RELU(-ENTR=>-ISFRr) 3=>12
    [-1, -1,  1,  1,  1, -1, -1, -1, -1],  # RELU(-ENFRr=>-ISTR) 6=>13
    [-1, -1,  1, -1, -1,  1,  1,  1,  1],  # RELU(-INFR=>-ESTRr) 5=>14
    [+1, -1, -1,  1, -1, -1,  1, -1,  1],  # RELU(-ISTRr=>-ENFR) 8=>15
    [+1, -1, -1, -1,  1,  1, -1,  1, -1],  # RELU(-ESTR=>-INFRr) 7=>16
]


MAP_16PSIAX_X_15RNAX = [
    # +E +Rr +N  +Ad +T  +Ab +Q  +Ag +static +NRrSR  +TRrFR  +QRrDR
    # -I -R  -S  -Bg -F  -Gd -D  -Bd     +NESI   +TEFI   +QEDI
    [+1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # IxxR(-2)=>ENTRr 6,10,14=>1
    [-1, -1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1],  # ExxRr(-1)=>INTR  5,9,13=>2
    [-1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1],  # ExxR(-4)=>ISFRr 8,12,16=>3
    [+1, -1, -1,  1, -1,  1, -1,  1, -1, -1,  1, -1,  1, -1,  1],  # IxxRr(-3)=>ESFR 7,11,15=>4
    [+1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1],  # IxxR(-6)=>ESTRr 2,10,14=>5
    [-1, -1, -1, -1,  1,  1, -1, -1,  1,  1,  1, -1, -1,  1,  1],  # ExxRr(-5)=>ISTR  1,9,13=>6
    [-1,  1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1, -1, -1,  1],  # ExxR(-8)=>INFRr 4,12,16=>7
    [+1, -1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1],  # IxxRr(-7)=>ENFR 3,11,15=>8
    [+1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1,  1,  1],  # IxxR(-10)=>ESFRr 2,6,14=>9
    [-1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1, -1, -1],  # ExxRr(-9)=>ISFR  1,5,13=>10
    [-1,  1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1, -1],  # ExxR(-12)=>INTRr 4,8,16=>11
    [+1, -1,  1, -1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1,  1],  # IxxRr(-11)=>ENTR 3,7,15=>12
    [+1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1],  # IxxR(-14)=>ENFRr 2,6,10=>13
    [-1, -1,  1,  1, -1, -1, -1, -1,  1, -1, -1,  1,  1,  1,  1],  # ExxRr(-13)=>INFR  1,5,9=>14
    [-1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1],  # ExxR(-16)=>ISTRr 4,8,12=>15
    [+1, -1, -1,  1,  1, -1,  1, -1, -1, -1,  1,  1, -1,  1, -1],  # IxxRr(-15)=>ESTR 3,7,11=>16
]


# The matrix would be split in three: a**2 * A_abs1 + b**2 * A_abs2 + c**2 * A_abs3
# a,b,c are learnable vectors of size 8
_MAP_8FZA4_X_16TYPES = [
    # 1   2   3   4    5,  6,  7,  8    9  10  11  12   13  14  15  16
    [+1,  2, -2, -1,  -2, -1,  1,  2,  -2, -1,  1,  2,   1,  2, -2, -1],  # S=>N
    [+1,  1,  0,  0,   0,  0, -2, -3,  -2, -2,  0,  0,   0,  0,  1,  1],  # FBg=>TAd
    [+2,  2, -1, -1,   1,  1, -3, -2,  -2, -2,  1,  1,  -1, -1,  2,  2],  # FAd=>TBg
    [+1, -1,  2,  2,   1, -1,  2,  2,   1, -1, -2, -2,   1, -1, -2, -3],  # IR=>ERr (Ij=>Ep)
    [+0,  0, -1,  1,   0,  0, -1,  1,   0,  0, -1,  1,   0,  0, -1,  1],  # IRr=>ER (Ip=>Ej)
    [+2, -3,  3,  1,   3, -1, -2, -1,   3, -2, -1,  3,   1, -3, -3,  2],  # IBg=>EAd
    [+2, -1, -1,  2,   1, -2, -2,  1,   1, -3, -2,  1,   3, -1, -1,  2],  # IAd=>EBg
    [-1, -2, -2, -3,  -1, -1, -2, -1,   1,  2,  3,  1,   1,  2,  2,  1],  # -AV=>+AV (Ab=>Gd maybe)
]
# Wrong abs would be replaced by zeros:
MAP_3_X_8FZA4_X_16TYPES = [
    [[(i / 1 if (abs(i) == 1) else 0) for i in row] for row in _MAP_8FZA4_X_16TYPES],
    [[(i / 2 if (abs(i) == 2) else 0) for i in row] for row in _MAP_8FZA4_X_16TYPES],
    [[(i / 3 if (abs(i) == 3) else 0) for i in row] for row in _MAP_8FZA4_X_16TYPES]
]


_MAP_ROT8FZA4_X_16TYPES = [
    # 1   2   3   4    5,  6,  7,  8    9  10  11  12   13  14  15  16
    [+1,  1, -1, -1,  -1, -1,  1,  1,  -1, -1,  1,  1,   1,  1, -1, -1],  # S=>N
    [+2,  2,  1,  1,  -1, -1,  3, -2,  -2, -2, -1, -1,   1,  1,  2,  2],  # Bg=>Ad
    [+1,  1, -1, -1,   1,  1, -1, -1,  -1, -1,  1,  1,  -1, -1,  1,  1],  # F=>T
    [+1, -1,  1, -1,   1, -2,  1, -3,   1, -1,  2, -1,   1, -1,  1, -1],  # R=>Rr (j=>p)
    [+1, -1, -1,  1,   1, -1, -2,  1,   1, -1, -1,  3,   1, -1, -1,  2],  # I=>E
    [+1,  1,  1,  1,  -1, -1, -1, -3,  -1, -2, -1, -1,   1,  1,  1,  1],  # Bg=>Ad
    [+2, -1, -2,  2,   1, -2, -2,  2,   1, -1, -1,  1,   1, -1, -1,  2],  # I=>E
    [-1, -1, -1, -1,  -1, -1, -1, -1,   1,  1,  1,  1,   1,  1,  1,  1],  # -AV=>+AV (Ab=>Gd maybe)
]
MAP_3_X_ROT8FZA4_X_16TYPES = [
    [[(i / 1 if (abs(i) == 1) else 0) for i in row] for row in _MAP_ROT8FZA4_X_16TYPES],
    [[(i / 2 if (abs(i) == 2) else 0) for i in row] for row in _MAP_ROT8FZA4_X_16TYPES],
    [[(i / 3 if (abs(i) == 3) else 0) for i in row] for row in _MAP_ROT8FZA4_X_16TYPES]
]
MAP_16TYPES_X_8KHAX = [
    # NERrNIR NIRrNER TERrTIR TIRrTER  (+)
    # SIR SERrSER SIRrFIR FERrFER FIRr (-)
    [+1,  0,  0,  0,  1,  0,  0,  0],  # 10=>1
    [+0,  1,  0,  0,  0,  1,  0,  0],  # _9=>2
    [+0,  0,  0, -1,  0,  0,  0, -1],  # 12=>3
    [+0,  0, -1,  0,  0,  0, -1,  0],  # 11=>4
    [+0, -1,  0,  0,  1,  0,  0,  0],  # 14=>5
    [-1,  0,  0,  0,  0,  1,  0,  0],  # 13=>6
    [+0,  0,  1,  0,  0,  0,  0, -1],  # 16=>7
    [+0,  0,  0,  1,  0,  0, -1,  0],  # 15=>8
    [+0, -1,  0,  0,  0, -1,  0,  0],  # _2=>9
    [-1,  0,  0,  0, -1,  0,  0,  0],  # _1=>10
    [+0,  0,  1,  0,  0,  0,  1,  0],  # _4=>11
    [+0,  0,  0,  1,  0,  0,  0,  1],  # _3=>12
    [+1,  0,  0,  0,  0, -1,  0,  0],  # _6=>13
    [+0,  1,  0,  0, -1,  0,  0,  0],  # _5=>14
    [+0,  0,  0, -1,  0,  0,  1,  0],  # _8=>15
    [+0,  0, -1,  1,  0,  0,  0,  1],  # _7=>16
]


def select_type(k: int, cf_n: int, cf_subset: int=None,
                to_temper: bool=False, join_valuable: bool=False, join_strong: bool=False) -> Tuple[int, ...]:
    """
    Input:

    >>> '''               0                     1                     2
    >>>  0     1    2     3    4     5    6     7    8     9    10    11
    >>> (NERr, NIR, NIRr, NER, TERr, TIR, TIRr, TER, QERr, QIR, QIRr, QER,
    >>>  SERr, SIR, SIRr, SER, FERr, FIR, FIRr, FER, DERr, DIR, DIRr, DER)
    >>>  12    13   14    15   16    17   18    19   20    21   22    23
    >>>                   3                     4                     5 '''

    Output: In case of 1st type we would have 8 triplets in the order:

    >>> '''
    >>> 1: NERr, TERr, QERr
    >>> 2: NIR, TIR, QIR
    >>> 3: SERr, FERr, DERr
    >>> 4: SIR, FIR, DIR
    >>> 5: SIRr, FIRr, DIRr
    >>> 6: SER, FER, DER
    >>> 7: NIRr, TIRr, QIRr
    >>> 8: NER, TER, QER '''

    :param k: type n
    :param cf_n:
    :param cf_subset: select only 0, 1 or 2 column from output
    :param to_temper: convert output to temper format
    :param join_valuable: convert output to valuable together format
    :param join_strong: convert output to strong together format
    :return:
    """
    i, e = E
    r, rr = R
    if (k in e) and (k in rr):
        tempers = [0, 1, 0, 1, 2, 3, 2, 3]
    elif (k in i) and (k in r):
        tempers = [1, 0, 1, 0, 3, 2, 3, 2]
    elif (k in i) and (k in rr):
        tempers = [2, 3, 2, 3, 0, 1, 0, 1]
    elif (k in e) and (k in r):
        tempers = [3, 2, 3, 2, 1, 0, 1, 0]
    else:
        raise ValueError(f'k={k} or axes constants bug')

    s, n = N
    f, t = T
    d, q = Q
    if (k in n) and (k in t) and (k in q):
        strong24, weak24 = [0, 1, 2], [3, 4, 5]
        strong16, weak16 = [0, 1], [2, 3]
    elif (k in s) and (k in f) and (k in d):
        strong24, weak24 = [3, 4, 5], [0, 1, 2]
        strong16, weak16 = [2, 3], [0, 1]

    elif (k in s) and (k in t) and (k in d):
        strong24, weak24 = [3, 1, 5], [0, 4, 2]
        strong16, weak16 = [2, 1], [0, 3]
    elif (k in n) and (k in f) and (k in q):
        strong24, weak24 = [0, 4, 2], [3, 1, 5]
        strong16, weak16 = [0, 3], [2, 1]

    elif (k in s) and (k in f) and (k in q):
        strong24, weak24 = [3, 4, 2], [0, 1, 5]
        strong16, weak16 = [2, 3], [0, 1]
    elif (k in n) and (k in t) and (k in d):
        strong24, weak24 = [0, 1, 5], [3, 4, 2]
        strong16, weak16 = [0, 1], [2, 3]

    elif (k in n) and (k in f) and (k in d):
        strong24, weak24 = [0, 4, 5], [3, 1, 2]
        strong16, weak16 = [0, 3], [2, 1]
    elif (k in s) and (k in t) and (k in q):
        strong24, weak24 = [3, 1, 2], [0, 4, 5]
        strong16, weak16 = [2, 1], [0, 3]
    else:
        raise ValueError(f'k={k} or axes constants bug')

    #         N 0           T 1           Q 2             S 3               F 4                D 5
    cfs24 = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19],  [20, 21, 22, 23]]
    #         N 0           T 1           S 2             F 3
    cfs16 = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    if cf_n == 24:
        cfs, strong_cf, weak_cf, subtuple_l = cfs24, strong24, weak24, 3
    elif cf_n == 16:
        cfs, strong_cf, weak_cf, subtuple_l = cfs16, strong16, weak16, 2
    else:
        raise ValueError(f'cf_n={cf_n}')
    blocks_cf_subtuples_idxs = [strong_cf, strong_cf, weak_cf, weak_cf, weak_cf, weak_cf, strong_cf, strong_cf]

    def transform(cf_subtuples_idxs: List[int]) -> List[int]:
        """ By default Q and D functions is always the last in the triplet. Pronal CF is the first. """
        if cf_subset is not None:
            return [cf_subtuples_idxs[cf_subset]]
        if (k in r) and not to_temper:
            return [cf_subtuples_idxs[1], cf_subtuples_idxs[0]] + cf_subtuples_idxs[2:]
        return cf_subtuples_idxs

    ret = tuple(cfs[cf_subtuple_i][temper_i]
                for temper_i, cf_subtuples_idxs, block_i in zip(tempers, blocks_cf_subtuples_idxs, range(8))
                for cf_subtuple_i in transform(cf_subtuples_idxs))

    if to_temper:
        assert len(ret) == 8 * subtuple_l
        split = [ret[i * subtuple_l:(i + 1) * subtuple_l] for i in range(8)]
        return tuple(sorted(split[1 - 1] + split[3 - 1]) + sorted(split[2 - 1] + split[4 - 1]) +
                     sorted(split[7 - 1] + split[5 - 1]) + sorted(split[8 - 1] + split[6 - 1]))
    if join_valuable or join_strong:
        assert cf_subset is not None
        assert len(ret) == 8
        if join_valuable:
            return ret[0:2] + ret[4:6] + ret[2:4] + ret[6:8]
        if join_strong:
            return ret[0:2] + ret[6:8] + ret[2:4] + ret[4:6]
        raise RuntimeError
    return ret


SEL_16TYPE_X_24CF = tuple(select_type(k, cf_n=24) for k in range(1, 17))
SEL_16TYPE_X_16CF = tuple(select_type(k, cf_n=16) for k in range(1, 17))
SEL_4TEMP_X_24CF = tuple(select_type(k, cf_n=24, to_temper=True) for k in (1, 2, 3, 4))
SEL_4TEMP_X_16CF = tuple(select_type(k, cf_n=16, to_temper=True) for k in (1, 2, 3, 4))

SEL_8NS_X_8CF__24CF = tuple(select_type(k, cf_n=24, cf_subset=0) for k in (1, 2, 7,  8,  5, 6,  3, 4))
SEL_8NS_X_8CF__16CF = tuple(select_type(k, cf_n=16, cf_subset=0) for k in (1, 2, 7,  8,  5, 6,  3, 4))
# #                                                                  alt:        11, 12, 9, 10
SEL_8TF_X_8CF__24CF = tuple(select_type(k, cf_n=24, cf_subset=1) for k in (1, 2, 11, 12, 9, 10, 3, 4))
SEL_8TF_X_8CF__16CF = tuple(select_type(k, cf_n=16, cf_subset=1) for k in (1, 2, 11, 12, 9, 10, 3, 4))
# #                                                                  alt:        15, 16, 13,14
SEL_8QD_X_8CF__24CF = tuple(select_type(k, cf_n=24, cf_subset=2) for k in (1, 2, 7, 8, 5, 6, 3, 4))
# #                                                                  alt:        15, 16, 13,14
SEL_2NS_X_8CF__24CF = tuple(select_type(k, cf_n=24, cf_subset=0, join_strong=True) for k in (1, 5))
SEL_2NS_X_8CF__16CF = tuple(select_type(k, cf_n=16, cf_subset=0, join_strong=True) for k in (1, 5))
SEL_2TF_X_8CF__24CF = tuple(select_type(k, cf_n=24, cf_subset=1, join_strong=True) for k in (1, 9))
SEL_2TF_X_8CF__16CF = tuple(select_type(k, cf_n=16, cf_subset=1, join_strong=True) for k in (1, 9))
SEL_2QD_X_8CF__24CF = tuple(select_type(k, cf_n=24, cf_subset=2, join_strong=True) for k in (1, 5))

SEL_2ADBG_X_8CF__24CF = tuple(select_type(k, cf_n=24, cf_subset=0, join_valuable=True) for k in (1, 5))
SEL_2ADBG_X_8CF__16CF = tuple(select_type(k, cf_n=16, cf_subset=0, join_valuable=True) for k in (1, 5))
SEL_2ABGD_X_8CF__24CF = tuple(select_type(k, cf_n=24, cf_subset=1, join_valuable=True) for k in (1, 9))
SEL_2ABGD_X_8CF__16CF = tuple(select_type(k, cf_n=16, cf_subset=1, join_valuable=True) for k in (1, 9))
SEL_2AGBD_X_8CF__24CF = tuple(select_type(k, cf_n=24, cf_subset=2, join_valuable=True) for k in (1, 5))


def zip_cat(*tuples_of_tuples: Tuple[Tuple[int, ...], ...]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(elem for tuple_ in tuples_slice for elem in tuple_) for tuples_slice in zip(*tuples_of_tuples))


class JatsProbs:
    @staticmethod
    def p_stat_dyn(probs: Tensor) -> Tensor:
        """ (static, dynamic) """
        p_temper = JatsProbs.p_temper(probs)
        return tr.stack([
            p_temper[:, [0, 1]].sum(dim=1),
            p_temper[:, [2, 3]].sum(dim=1),
        ], dim=1)

    @staticmethod
    def p_ei(probs: Tensor) -> Tensor:
        """ (E, I) """
        p_temper = JatsProbs.p_temper(probs)
        return tr.stack([
            p_temper[:, [0, 3]].sum(dim=1),
            p_temper[:, [1, 2]].sum(dim=1),
        ], dim=1)

    @staticmethod
    def p_rr_r(probs: Tensor) -> Tensor:
        """ (Rr, R) """
        p_temper = JatsProbs.p_temper(probs)
        return tr.stack([
            p_temper[:, [0, 2]].sum(dim=1),
            p_temper[:, [1, 3]].sum(dim=1),
        ], dim=1)

    @staticmethod
    def p_ns(probs: Tensor) -> Tensor:
        """ (N, S) """
        batch_size = probs.shape[0]
        probs = probs.view(batch_size, 8, 2)
        return tr.stack([
            probs[:, quatro, :].view(batch_size, -1).sum(dim=1)
            for quatro in ([1 - 1, 4 - 1, 6 - 1, 7 - 1], [2 - 1, 3 - 1, 5 - 1, 8 - 1])
        ], dim=1)

    @staticmethod
    def p_tf(probs: Tensor) -> Tensor:
        """ (T, F) """
        batch_size = probs.shape[0]
        probs = probs.view(batch_size, 8, 2)
        return tr.stack([
            probs[:, quatro, :].view(batch_size, -1).sum(dim=1)
            for quatro in ([1 - 1, 3 - 1, 6 - 1, 8 - 1], [2 - 1, 4 - 1, 5 - 1, 7 - 1])
        ], dim=1)

    @staticmethod
    def p_qd(probs: Tensor) -> Tensor:
        """ (Q, D) """
        batch_size = probs.shape[0]
        probs = probs.view(batch_size, 8, 2)
        return tr.stack([
            probs[:, quatro, :].view(batch_size, -1).sum(dim=1)
            for quatro in ([1 - 1, 4 - 1, 5 - 1, 8 - 1], [2 - 1, 3 - 1, 6 - 1, 7 - 1])
        ], dim=1)

    @staticmethod
    def p_ad_bg(probs: Tensor) -> Tensor:
        """ (Ad, Bg) """
        p_quadra = JatsProbs.p_quadra(probs)  # (batch_size, 4)
        return tr.stack([
            p_quadra[:, 0] + p_quadra[:, 3],
            p_quadra[:, 1] + p_quadra[:, 2],
        ], dim=1)

    @staticmethod
    def p_ab_gd(probs: Tensor) -> Tensor:
        """ (Ab, Gd) """
        p_quadra = JatsProbs.p_quadra(probs)  # (batch_size, 4)
        return tr.stack([
            p_quadra[:, 0] + p_quadra[:, 1],
            p_quadra[:, 2] + p_quadra[:, 3],
        ], dim=1)

    @staticmethod
    def p_ag_bd(probs: Tensor) -> Tensor:
        """ (Ag, Bd) """
        p_quadra = JatsProbs.p_quadra(probs)  # (batch_size, 4)
        return tr.stack([
            p_quadra[:, 0] + p_quadra[:, 2],
            p_quadra[:, 1] + p_quadra[:, 3],
        ], dim=1)

    @staticmethod
    def p_type(probs: Tensor) -> Tensor:
        return probs

    @staticmethod
    def p_temper(probs: Tensor) -> Tensor:
        """ (ERr/-IR, IR/+IR, IRr/-ER, ER/+ER) """
        return probs.view(probs.shape[0], 4, 4).sum(dim=1)

    @staticmethod
    def p_8ns(probs: Tensor) -> Tensor:
        """ (NERr, NIR, NIRr, NER, SERr, SIR, SIRr, SER) """
        return tr.stack([
            probs[:, 1 - 1] + probs[:, 13 - 1],
            probs[:, 2 - 1] + probs[:, 14 - 1],
            probs[:, 7 - 1] + probs[:, 11 - 1],
            probs[:, 8 - 1] + probs[:, 12 - 1],
            probs[:, 5 - 1] + probs[:, 9 - 1],
            probs[:, 6 - 1] + probs[:, 10 - 1],
            probs[:, 3 - 1] + probs[:, 15 - 1],
            probs[:, 4 - 1] + probs[:, 16 - 1],
        ], dim=1)

    @staticmethod
    def p_8tf(probs: Tensor) -> Tensor:
        """ (TERr, TIR, TIRr, TER, FERr, FIR, FIRr, FER) """
        return tr.stack([
            probs[:, 1 - 1] + probs[:, 5 - 1],
            probs[:, 2 - 1] + probs[:, 6 - 1],
            probs[:, 11 - 1] + probs[:, 15 - 1],
            probs[:, 12 - 1] + probs[:, 16 - 1],
            probs[:, 9 - 1] + probs[:, 13 - 1],
            probs[:, 10 - 1] + probs[:, 14 - 1],
            probs[:, 3 - 1] + probs[:, 7 - 1],
            probs[:, 4 - 1] + probs[:, 8 - 1],
        ], dim=1)

    @staticmethod
    def p_8qd(probs: Tensor) -> Tensor:
        """ (QERr, QIR, QIRr, QER, DERr, DIR, DIRr, DER) """
        return tr.stack([
            probs[:, 1 - 1] + probs[:, 9 - 1],
            probs[:, 2 - 1] + probs[:, 10 - 1],
            probs[:, 7 - 1] + probs[:, 15 - 1],
            probs[:, 8 - 1] + probs[:, 16 - 1],
            probs[:, 5 - 1] + probs[:, 13 - 1],
            probs[:, 6 - 1] + probs[:, 14 - 1],
            probs[:, 3 - 1] + probs[:, 11 - 1],
            probs[:, 4 - 1] + probs[:, 12 - 1],
        ], dim=1)

    @staticmethod
    def p_quadra(probs: Tensor) -> Tensor:
        """ (a, b, g, d) """
        return probs.view(probs.shape[0], 4, 4).sum(dim=2)

    @staticmethod
    def p_club(probs: Tensor) -> Tensor:
        """ (NT, SF, ST, NF) """
        batch_size = probs.shape[0]
        probs = probs.view(batch_size, 8, 2)
        return tr.stack([
            probs[:, pair, :].view(batch_size, -1).sum(dim=1)
            for pair in ([1 - 1, 6 - 1], [2 - 1, 5 - 1], [3 - 1, 8 - 1], [4 - 1, 7 - 1])
        ], dim=1)

    @staticmethod
    def p_quadraclub(probs: Tensor) -> Tensor:
        """ (NTa, SFa, STb, NFb, SFg, NTg, NFd, STd) """
        return probs.view(probs.shape[0], 8, 2).sum(dim=2)

    @staticmethod
    def p_dom(probs: Tensor) -> Tensor:
        """ (N±ER, N±IR, S±ER, S±IR, T±IR, T±ER, F±IR, F±ER) """
        return tr.stack([
            probs[:, 7 - 1] + probs[:, 11 - 1],
            probs[:, 1 - 1] + probs[:, 13 - 1],
            probs[:, 3 - 1] + probs[:, 15 - 1],
            probs[:, 5 - 1] + probs[:, 9 - 1],
            probs[:, 2 - 1] + probs[:, 6 - 1],
            probs[:, 12 - 1] + probs[:, 16 - 1],
            probs[:, 10 - 1] + probs[:, 14 - 1],
            probs[:, 4 - 1] + probs[:, 8 - 1],
        ], dim=1)

    @staticmethod
    def p_4th(probs: Tensor) -> Tensor:
        """ (N±ER, N±IR, S±ER, S±IR, T±IR, T±ER, F±IR, F±ER) """
        return tr.stack([
            probs[:, 4 - 1] + probs[:, 16 - 1],
            probs[:, 6 - 1] + probs[:, 10 - 1],
            probs[:, 8 - 1] + probs[:, 12 - 1],
            probs[:, 2 - 1] + probs[:, 14 - 1],
            probs[:, 9 - 1] + probs[:, 13 - 1],
            probs[:, 3 - 1] + probs[:, 7 - 1],
            probs[:, 1 - 1] + probs[:, 5 - 1],
            probs[:, 11 - 1] + probs[:, 15 - 1],
        ], dim=1)

    @staticmethod
    def p_ie_sn_ft(probs: Tensor) -> Tensor:
        """ (ENT, INT, ISF, ESF, EST, IST, INF, ENF) """
        return tr.stack([
            probs[:, 1 - 1] + probs[:, 12 - 1],
            probs[:, 2 - 1] + probs[:, 11 - 1],
            probs[:, 3 - 1] + probs[:, 10 - 1],
            probs[:, 4 - 1] + probs[:, 9 - 1],
            probs[:, 5 - 1] + probs[:, 16 - 1],
            probs[:, 6 - 1] + probs[:, 15 - 1],
            probs[:, 7 - 1] + probs[:, 14 - 1],
            probs[:, 8 - 1] + probs[:, 13 - 1],
        ], dim=1)
