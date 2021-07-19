"""
---
pandoctools:
  profile: Default
  out: "*.ipynb"
input: True
eval: False
echo: True
...
"""

# %%
# from importlib import reload
from typing import Tuple, Optional as Opt, Any, Iterable, Type, List
import warnings
from dataclasses import dataclass
from itertools import chain
import math
# noinspection PyPep8Naming
from numpy import ndarray as Array
import numpy as np
import torch as tr
from torch import nn
from torch.nn import Module, functional as func
from torch.optim.optimizer import Optimizer
import matplotlibhelper as mh  # pylint: disable=wrong-import-order
from kiwi_bugfix_typechecker import test_assert
from vae import EncoderCustom, EncoderPassthrCustom, BernoulliLoss, ClassifierCustom, KLDLossTwin, BetaTCKLDLossTwin
from vae.loop_tools import Do, StrArg, IntArg, FloatArg, TupleIntArg
from vae.linear_component_analyzer import LinearAnalyzer
from beta_tcvae_typed import BetaTC, BetaTCKLDLoss, Normal, Laplace, ZeroSymmetricBimodalNormal, KLDLoss
from semi_supervised_typed import (VariationalAutoencoder, Decoder, SVI, AuxiliaryDeepGenerativeModel,
                                   DeepGenerativeModel, ADGMPassthrough, Classifier, DecoderPassthrough, Encoder,
                                   EncoderPassthrough, VAEPassthrough, DGMPassthrough, VAEClassify,
                                   VAEPassthroughClassify, PassthroughMeta, DGMPassthroughSeparateClassifier)
from semi_supervised_typed.inference import ImportanceWeightedSampler
from semi_supervised_typed.layers import GaussianSample
from factor_vae import Discriminator
from jats_vae.semi_supervised import (DecoderPassthroughSex, EncoderCustom2, ClassifierJATSCustom,
                                      DecoderPassthroughTwinSex, VAEPassthroughClassifyMMD,
                                      VAEPassthroughClassifyTwinMMD, EncoderTwinJATS, EncoderTwinFrozen700Part2Inner,
                                      ClassifierJATSAxesAlign, ClassifierJATSAxesAlign5RnKhAx,
                                      ClassifierJATSAxesAlignNTRrAdeAdiAd12, ClassifierJATSAxesAlignNTRrAdeAdi,
                                      ClassifierPassthrJATS24KhCFBase, EncoderTwinFrozen700, EncoderTwinFrozen700Part2,
                                      DecoderPassthrTwinSexKhAx, EncoderTwinSplit0, EncoderTwinSplit1,
                                      ClassifierJATSAxesAlignDecSwap, DecoderPassthrTwinSexKhAxDecSwap)
from jats_vae.latents_explorer import LatentsExplorerPassthr
from jats_display import check_cov
from socionics_db import debug_loader, debug_lbl_loader
from normalizing_flows_typed import flows, NormalizingFlows, PlanarNormalizingFlow, BNAFs
from ready import Global
from guides import GuideTwinVAE, GuideVAE, Universal, GuideDGM, GuideADGM
warnings.filterwarnings('ignore')

_: Any = (
    BetaTC, BetaTCKLDLoss, BNAFs, NormalizingFlows, Normal, flows, EncoderPassthrCustom, ClassifierJATSAxesAlign5RnKhAx,
    DeepGenerativeModel, PlanarNormalizingFlow, Classifier, DecoderPassthrough, Encoder, EncoderPassthrough,
    EncoderCustom, ClassifierJATSAxesAlign, Decoder, ADGMPassthrough, ImportanceWeightedSampler, VAEPassthrough,
    DecoderPassthroughSex, GuideDGM, GuideADGM, EncoderCustom2, DGMPassthrough, VAEClassify, VAEPassthroughClassify,
    ClassifierJATSCustom, PassthroughMeta, DecoderPassthroughTwinSex, VAEPassthroughClassifyMMD, ClassifierCustom,
    Laplace, DGMPassthroughSeparateClassifier, VAEPassthroughClassifyTwinMMD, EncoderTwinJATS, BetaTCKLDLossTwin,
    LatentsExplorerPassthr, AuxiliaryDeepGenerativeModel, ZeroSymmetricBimodalNormal, KLDLoss, KLDLossTwin, math.inf,
    ClassifierJATSAxesAlignNTRrAdeAdiAd12, ClassifierJATSAxesAlignNTRrAdeAdi, ClassifierPassthrJATS24KhCFBase,
    DecoderPassthrTwinSexKhAx, EncoderTwinSplit0,
)
del _

DBNAME = ('solti', 'bolti')[0]
H_DIMS: Tuple[int, ...] = dict(solti=(64, 32), bolti=(128,))[DBNAME]  # type: ignore
# was (128, 128, 128)  (128, 128, 128, 128, 128)  (128, 96, 64)

DEBUG = False
δ = 1e-8
mh.ready(font_size=12, ext='svg', hide=True, magic='agg')
test_assert()
Z_DIM = 8  # TODO [was 8]

gl = Global(labels='type',  # xir_xer temper type dom lc mr
            h_dims=H_DIMS,
            # check_pca=(9, 8, 7, 6, 5, Z_DIM),
            check_pca=(Z_DIM + 1, Z_DIM - 1, Z_DIM,),
            db_name=DBNAME,
            )
if DEBUG:
    debug_loader(gl.data_loader)
    debug_lbl_loader(gl.data_loader)


# %% ----------------------------------------------------
# PCA and FA
# -------------------------------------------------------
pca: LinearAnalyzer
fa: LinearAnalyzer
try:
    # noinspection PyUnresolvedReferences,PyUnboundLocalVariable
    pca, fa = pca, fa
    EVAL_PCA = False  # False True
except NameError:
    EVAL_PCA = True
if EVAL_PCA:
    pca, fa = gl.get_pca_fa(plot_pca=False, print_pca=True)


# %% ----------------------------------------------------
# Consts
# -------------------------------------------------------
BATCH_SIZE = 128
A_DIM = 9  # 12
HD_DIMS = tuple(1024 for _ in range(5))
HY_DIMS = (32,)  # (32,) (128,) (64,)
gl.dims.set_z(Z_DIM)
gl.dims.set_a(A_DIM)
gl.dims.set_hy(*HY_DIMS)

PRINT_EVERY_EPOCH = dict(solti=10, bolti=29)[DBNAME]

SET_FLOW = False
USE_TC_KLD = True
IS_DGM = False
IS_FVAE = False

ADDITIONAL_EXPLORE = False
MERGE_FZA = (False, True)[0]  # TODO
TRAIN = (False, True)[1]
_P = int(not TRAIN)

EXPLORE = (True, False)[_P]
# EXPLORE = True
PLOT = (True, False)[_P]
# PLOT = True

TO_DO_LIST: Tuple[int, ...]
MODELS_IDXS: Tuple[int, ...]
USE_REPLACE_ENC_AND_DEC = True
if not IS_DGM:
    TO_DO_LIST = ((
                      # 8,
                      # 34,
                      # # 40,
                      # 20,
                      # 28,
                      # 29,  # never was needed 30, 30,  # the last with rho=1
                      # ----------------------
                      # 21,
                      # 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                      # 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                      # 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                      # 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                      # 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                      # 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                      # ----------------------
                      # # 46,
                      35,
                      36, 37,  # next use only for models that have compact center, with potence to merge that center:
                      # 43, 43, 43, 43, 43, 43, 43, 43,
                      # after 43,... should be ~ 0.1411; 0.1424
                      # after this should be clear if model is good (pairs of types are in sync on NIM,QIM)
                      # 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
                      # 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
                      # 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
                      # 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
                      # # 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
                      # after 43,... should be ~ 0.1373; 0.1398
                      # after this should be clear if model is good (pairs of types are in sync on NIM,QIM - on coin. types)
                      # ----------------------
                      # 44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
                      # after 44,... should be ~ 0.1358; 0.1386
                      # ----------------------
                      # 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
                      # 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
                      # 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
                      # 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
                      # 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
                      # after 45,... should be ~ 0.1341; 0.1372
                      # ----------------------
                      # 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                      # after 42,... should be ~ 0.1336; 0.1368
                      # ----------------------
                      # 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
                      # 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
                      # 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
                      # 41, 41, 41, 41, 41, 41, 41, 41,
                      # after 41,... should be ~ 0.1332; 0.1365
                      # ----------------------
                      # ----------------------
                      # # 39,
                      # 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
                      # 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
                      # 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
                      # 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
                      # 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
                      # 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
                      # ----------------------
                      # 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
                      # 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
                      # 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
                      # 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
                      # -- CF begin --
                      # 6, 7,
                      # 13, 14,
                      # 14, 14, 14,
                      # 31, 31, 31, 31,
                      # # 31, 31, 31, 31, 31, 31, 31, 31, 31,  # 6040
                      # # 31, 31, 31, 31, 31, 31, 31, 31, 31,
                      # 32, 33, 33, 33,
                      # 33, 33, 33, 33, 33,
                      # 33, 33, 33, 33, 33, 33, 33, 33, 33,
                      # -- CF end --
                      # -- Flow begin --
                      # 10,
                      # 11, 11, 11, 11, 11, 11, 11, 11,
                      # 11, 11, 11, 11, 11, 11, 11, 11,
                      # -- Flow end --
                      # -- Flow log begin --
                      # 10,  # thrs 1.5
                      # 11, 11, 11, 11, 11, 11, 11, 11,  # thrs 1.5
                      # 11,  # thrs 0.5
                      # -- Flow log end --
                      # 15, 16,
                      # 16, 16,
                  ), (38,))[_P]
    MODELS_IDXS = ((7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009,), (700,))[_P]
    # ((700,), (700,))[_P] -- for averaging base axes @ 38,39 and adjusting log_sigma @ 46
    # ((7000, 7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009,), (700,))[_P] -- for tests series @ 35,36,37
    # 7004, -- for tests further
    # 7001, 7002, 7003, 7005, 7006, 7007, 7008, 7009, -- for tests series @ 35,36,37

    # 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815
    # 704, 714, 717, 720, 725, 728
    # 700, 701, 702, 703, 705, 706, 707, 708, 709, 710, 711, 712, 713, 
    # .                715, 716, 718, 719, 721, 722, 723, 724, 726, 727, 729, 730, 731, 732, 733,
    # 8, 20, 28, 29, 30,
    # 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
    # 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
    # 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
    # 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
    # (518, 519, 520)
    # 8, 20, 28, 29,
    # 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
    # 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
    # 8, 20, 28, 29, 21, 27, 27, 27, 27,
    # 8, 20, 28, 28, 29, 21, 27, 27, 27, 27,
    # 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    # 6, 13, 14,
    # 15, 16,
    # 29, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
    ID_PREF = 'vae'
else:
    TO_DO_LIST = (17, 18, 18, 18, 18, 18,)
    MODELS_IDXS = (2,)
    ID_PREF = ('dgm', 'adgm')[0]
SUCCESSFUL_MODELS_N = len(MODELS_IDXS)
# SUCCESSFUL_MODELS_N = (1, 5)[_P]
ID_PREF += ('', 'nf')[0]


ITERS_PER_EPOCH = gl.iters_per_epoch(BATCH_SIZE, IS_FVAE)
# FORMULA = 'BCE(x)+β*(γTC+λDW)(.)+ηCE(y)-log_p_sex+Trim(.)'
FORMULA = 'BCE(x)+(βKLD+(γ-1)TC)(.)+ηCE(y)-log_p_sex+Trim(.)'
# FORMULA = 'BCE+β*KLD+Trim()'


def s(x_: float): return x_ + 1


A0_5 = 0.5
A1 = 1
A1_25 = 1.25
A2 = 2
B0_005 = 0.005
B1 = 1
B2_5 = 2.5
H0 = 0
H0_5 = 0.5
H30 = 30  # was30;was20;             [was30 @550]
H10 = 10  # was10.5;was5.5;          [was10 @550]
H5 = 5  # was20;was10;was5;was2;     [was5 @550]
H1 = 5  # was5;was20;was10;was5;was1;[was5 @550]
S0 = s(0)
S2 = s(2)
S7 = s(7)
E0_1 = 0.1  # 0.33; was 0.25;was 0.33;[was 0.25]
E0_75 = 0.75  # was 0.67; was 0.75
E0 = 0
T00 = 1000  # [was 0 @550]
T0 = 1000  # [was 0 @550]
T500 = 1000  # was 500; [was 1000] [was 750 @550]
M0_2 = 0.2  # [was 0.4 @550]
M0_4 = 0.4
M0_5 = 0.5
M0_9 = 0.9
W2 = 2
W2_0 = 4  # [was 2]
W8_0 = 4  # 8;[was 2]
EPOCHS = dict(solti=40, bolti=290)[DBNAME]  # was solti=80


@dataclass
class Do1(Do):
    α: FloatArg = A1
    β: FloatArg = B1
    γ: FloatArg = 1
    δ: FloatArg = 0
    ε: FloatArg = 0
    η: FloatArg = 0
    λ: FloatArg = 0
    μ: FloatArg = M0_2  # deterministic part of the cross-entropy for classifier
    ρ: FloatArg = 0
    τ: FloatArg = T500
    ω: FloatArg = W2_0
    epochs: IntArg = EPOCHS * 2
    iters_per_epoch: IntArg = ITERS_PER_EPOCH
    batch_size: IntArg = BATCH_SIZE
    formula: StrArg = FORMULA
    max_failure: IntArg = 9


BASIS_STRIP = ()
# MS = (0.75, 0.75, 0.75, 0.75, 0.75, 0, 0, 0, 0, 0, 0, 0, 0)[:Z_DIM]
# ΣS = (0.75, 0.75, 0.75, 0.75, 0.75, 1, 1, 1, 1, 1, 1, 1, 1)[:Z_DIM]
MS = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)[:Z_DIM]
ΣS = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)[:Z_DIM]
MS2 = (0.75, 0.75, 0.75, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20)[:Z_DIM]
ΣS2 = (0.75, 0.75, 0.75, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93)[:Z_DIM]


@dataclass
class DoTwinVAE(Do1):
    γ: FloatArg = S7
    ε: FloatArg = E0_1  # E0_5@427
    basis_strip: TupleIntArg = BASIS_STRIP

# .                                                      39, 40 were here
FIRSTS, AXES_ALIGN, NELBO_FIT, CLASS_FIT = (8, 34, 40), (20, 28, 29, 30, 21, 27), (6, 7), (13, 14, 31)
# REPLACE_ENC_DEC = (35, 36, 37, 38, 39)
REPLACE_ENCODER = (38, 39, 46)
REPLACE_DECODERS = (35, 36, 37)
REPLACE_DECODERS2 = (43, 44, 45, 42, 41)
AXES_ALIGN2 = (48, 47)
CLASS_FIT_FROZEN, FLOW_FIT, FLOW_MAP, PRIORS_FIT = (32, 33), (10, 11), (12, 19), (15, 16)
REPLACE_DECODERS_DEL = (
    'decoder.hidden.0.weight',
    'decoder.hidden.0.bias',
    'decoder.hidden.1.weight',
    'decoder.hidden.1.bias',
    '_twin_vae_registered.decoder.hidden.0.weight',
    '_twin_vae_registered.decoder.hidden.0.bias',
    '_twin_vae_registered.decoder.hidden.1.weight',
    '_twin_vae_registered.decoder.hidden.1.bias',

    'decoder.reconstruction.weight',
    'decoder.reconstruction.bias',
    '_twin_vae_registered.decoder.reconstruction.weight',
    '_twin_vae_registered.decoder.reconstruction.bias',
    'classifier.transform',
    '_twin_vae_registered.classifier.transform',

    'decoder.subdecoder.map.layers.0.weight',
    'decoder.subdecoder.map.layers.0.bias',
    'decoder.subdecoder.map.layers.1.weight',
    'decoder.subdecoder.map.layers.1.bias',
    'decoder.subdecoder.map.layers.2.weight',
    'decoder.subdecoder.map.layers.2.bias',
    '_twin_vae_registered.decoder.subdecoder.map.layers.0.weight',
    '_twin_vae_registered.decoder.subdecoder.map.layers.0.bias',
    '_twin_vae_registered.decoder.subdecoder.map.layers.1.weight',
    '_twin_vae_registered.decoder.subdecoder.map.layers.1.bias',
    '_twin_vae_registered.decoder.subdecoder.map.layers.2.weight',
    '_twin_vae_registered.decoder.subdecoder.map.layers.2.bias',

    'decoder.subdecoder.map_khax2rnax.k_kh2crnkh_7pos',
    'decoder.subdecoder.map_khax2rnax.k_kh2rnkh_8pos',
    'decoder.subdecoder.map_khax2rnax.abskhax2complrnkh_12kh_x_7crnkh',
    'decoder.subdecoder.map_khax2rnax.khax2rnkhax_12kh_x_8rnkh',
    '_twin_vae_registered.decoder.subdecoder.map_khax2rnax.k_kh2crnkh_7pos',
    '_twin_vae_registered.decoder.subdecoder.map_khax2rnax.k_kh2rnkh_8pos',
    '_twin_vae_registered.decoder.subdecoder.map_khax2rnax.abskhax2complrnkh_12kh_x_7crnkh',
    '_twin_vae_registered.decoder.subdecoder.map_khax2rnax.khax2rnkhax_12kh_x_8rnkh',

    'classifier.perceptron.layers.0.weight',
    '_twin_vae_registered.classifier.perceptron.layers.0.weight',

    'decoder.subdecoder.dist.log_σ',
    'decoder.subdecoder.dist.μ',
    'decoder.subdecoder.dist.normalization',
    '_twin_vae_registered.decoder.subdecoder.dist.log_σ',
    '_twin_vae_registered.decoder.subdecoder.dist.μ',
    '_twin_vae_registered.decoder.subdecoder.dist.normalization',
)
REPLACE_DECODERS_DEL2 = (
    'decoder.subdecoder.dist.log_σ',
    'decoder.subdecoder.dist.μ',
    'decoder.subdecoder.dist.normalization',
    '_twin_vae_registered.decoder.subdecoder.dist.log_σ',
    '_twin_vae_registered.decoder.subdecoder.dist.μ',
    '_twin_vae_registered.decoder.subdecoder.dist.normalization',
)
REPLACE_DECODERS_DEL__ = (
                  # 'decoder.hidden.0.weight',
                  # 'decoder.hidden.0.bias',
                  # 'decoder.hidden.1.weight',
                  # 'decoder.hidden.1.bias',
                  # '_twin_vae_registered.decoder.hidden.0.weight',
                  # '_twin_vae_registered.decoder.hidden.0.bias',
                  # '_twin_vae_registered.decoder.hidden.1.weight',
                  # '_twin_vae_registered.decoder.hidden.1.bias',
                  #
                  # 'decoder.reconstruction.weight',
                  # 'decoder.reconstruction.bias',
                  # '_twin_vae_registered.decoder.reconstruction.weight',
                  # '_twin_vae_registered.decoder.reconstruction.bias',
                  #
                  # 'decoder.subdecoder.map.layers.0.weight',
                  # 'decoder.subdecoder.map.layers.0.bias',
                  # 'decoder.subdecoder.map.layers.1.weight',
                  # 'decoder.subdecoder.map.layers.1.bias',
                  # 'decoder.subdecoder.map.layers.2.weight',
                  # 'decoder.subdecoder.map.layers.2.bias',
                  # '_twin_vae_registered.decoder.subdecoder.map.layers.0.weight',
                  # '_twin_vae_registered.decoder.subdecoder.map.layers.0.bias',
                  # '_twin_vae_registered.decoder.subdecoder.map.layers.1.weight',
                  # '_twin_vae_registered.decoder.subdecoder.map.layers.1.bias',
                  # '_twin_vae_registered.decoder.subdecoder.map.layers.2.weight',
                  # '_twin_vae_registered.decoder.subdecoder.map.layers.2.bias',
                  #
                  # 'decoder.subdecoder.map_khax2rnax.k_kh2crnkh_7pos',
                  # 'decoder.subdecoder.map_khax2rnax.k_kh2rnkh_8pos',
                  # 'decoder.subdecoder.map_khax2rnax.abskhax2complrnkh_12kh_x_7crnkh',
                  # 'decoder.subdecoder.map_khax2rnax.khax2rnkhax_12kh_x_8rnkh',
                  # '_twin_vae_registered.decoder.subdecoder.map_khax2rnax.k_kh2crnkh_7pos',
                  # '_twin_vae_registered.decoder.subdecoder.map_khax2rnax.k_kh2rnkh_8pos',
                  # '_twin_vae_registered.decoder.subdecoder.map_khax2rnax.abskhax2complrnkh_12kh_x_7crnkh',
                  # '_twin_vae_registered.decoder.subdecoder.map_khax2rnax.khax2rnkhax_12kh_x_8rnkh',
                  #
                  'classifier.transform',
                  '_twin_vae_registered.classifier.transform',
                  'classifier.perceptron.layers.0.weight',
                  '_twin_vae_registered.classifier.perceptron.layers.0.weight',
              )
QUICK_DEL = (
    'classifier.transform', '_twin_vae_registered.classifier.transform',
    'classifier.perceptron.layers.0.weight', '_twin_vae_registered.classifier.perceptron.layers.0.weight',
    # "decoder.subdecoder.map_khax2rnax.abskhaxextra2oy_28kh_x_1stat",
    # "decoder.subdecoder.map_khax2rnax.khaxextra2rnkhax_28kh_x_8rnkh",
    # "_twin_vae_registered.decoder.subdecoder.map_khax2rnax.abskhaxextra2oy_28kh_x_1stat",
    # "_twin_vae_registered.decoder.subdecoder.map_khax2rnax.khaxextra2rnkhax_28kh_x_8rnkh",
    # '_twin_vae_registered._kld._prior_dist.μ',
    # '_twin_vae_registered._kld._prior_dist.log_σ',
    # '_kld._prior_dist.μ',
    # '_kld._prior_dist.log_σ',
    # '_twin_vae_registered.classifier.transform',
    # '_twin_vae_registered.classifier.perceptron.layers.0.weight',
    # '_twin_vae_registered.classifier.perceptron.layers.1.weight',
    # '_twin_vae_registered.classifier.signs',
    # 'classifier.perceptron.layers.0.weight',
    # 'classifier.perceptron.layers.1.weight',
    # '_twin_vae_registered.classifier.rot_2d_45',
    # '_twin_vae_registered.classifier.ax2type_transp_temperdim_typedim_x_axdim',
    # '_twin_vae_registered.classifier.fun2type_transp_8pos_x_typedim_x_fundim',
    # '_twin_vae_registered.classifier.adabag2type_transp_2pos_x_typedim_x_3ax',
    # '_twin_vae_registered.classifier.values_scl',
    # '_twin_vae_registered.classifier.softmax_scl',
)

W_MOD = 1

DO_SPECS = (
    # ---- βTC-VAE -> VAE -> βTC-VAE ------------------
    # 0, 22, 1, [800] 3, 3, 3, 3, [1440] 23, [1760] 2, [1920]
    # 0, 22, 1, 3, 3, 3, 3, 23,
    # 2,
    Do1(id=0, γ=s(1.5), γ0=[S0, s(1.5)], η0=[H30, 0], β0=[B0_005, B2_5], λ0=[1, 0], τ=0,
        anneal_epochs=EPOCHS, epochs=EPOCHS * 4),
    Do1(id=22, γ=s(2), γ0=[s(1.5), s(2)], anneal_epochs=EPOCHS, epochs=EPOCHS * 4),
    Do1(id=1, γ=s(0.01), γ0=[s(2), s(0.01)], anneal_epochs=EPOCHS),
    Do1(id=3, γ=s(0.01)),
    Do1(id=23, γ=s(2), γ0=[S0, s(2)], anneal_epochs=EPOCHS, epochs=EPOCHS * 4),
    Do1(id=2, γ=s(2)),
    # ------------------------------
    # 8, 20, 9, 9, 6, [960] 7, 7, 7, 7, 7, 7, 7,
    # 8, 20, 9, 9, 9, 9 [1120], 6 [1280], 13 [1320], 10, 11 [1640], 12 [1800], 19,
    # 8, 20, 9, 9, 9, 9, 6, 13, 10, 11, 12, 19,
    # 24, 25, 26, 26, 26, 9, 9, 9, 9,
    # 8 [240], 20 [400], 21, 21, 21, 21, 21, 21, 21 [1520], 6 [1680], 13 [1733], 10, [1893] 11, 12, 19,
    DoTwinVAE(id=8, ρ=0, basis_strip=(), anneal_epochs=2 * EPOCHS,  # 2 * EPOCHS
              # epochs=EPOCHS * (8 if Z_DIM >= 8 else 5),
              epochs=EPOCHS * (16 - 12),  # TODO [was 16 at 8 dim] [was 5 @ 550]; was 8 at bests
              # was (6 if Z_DIM >= 7 else 3) with EPOCHS = dict(solti=80...
              # α=A0_5, α0=[B0_005, A0_5], β=B1, β0=[B0_005, B1], γ=S2, γ0=[S0, S2],
              α=A0_5, α0=[B0_005, A0_5], β=B1, β0=[B0_005, B1], γ=S7, γ0=[S0, S7],
              # α=A1, α0=[B0_005, A1], β=B1, β0=[B0_005, B1], γ=S7, γ0=[S0, S7],
              η=H10, η0=[H30, H10], λ0=[1, 0], τ=T0, τ0=[T00, T0], ε=E0_75, ω=W2, μ=M0_4,
              ),
    DoTwinVAE(id=34, ρ=0, α=A0_5, β=B1, γ=S7, η=H10, τ=T0, ε=E0_75, ω=W2, μ=M0_4, epochs=EPOCHS * 13, max_failure=1),
    DoTwinVAE(id=40, ρ=0, α=A0_5, β=B1, γ=S7, η=H10, τ=T0, ε=E0_75, ω=W2, μ=M0_4, epochs=EPOCHS * 18, max_failure=1),
    # DoTwinVAE(id=20, ρ=1, basis_strip=(), anneal_epochs=EPOCHS,
    #           α=A1_25, α0=[A0_5, A1_25], β=B1_25, β0=[B1, B1_25], η=H10, η0=[H10, H10], γ0=[S2, S7], γ=S7,
    #           μ=M0_2, μ0=[M0_4, M0_2]),

    DoTwinVAE(id=20, α=A0_5, β=B1, η=H5, η0=[H10, H5], γ=S7, ε0=[E0_75, E0_1],
              τ0=[T0, T500], ω=W2_0, ω0=[W2, W2_0], μ=M0_2, μ0=[M0_4, M0_2],
              ρ=1, anneal_epochs=EPOCHS * 2, epochs=EPOCHS * 5),
    DoTwinVAE(id=390, α=A0_5, β=B1, η=H5, η0=[H10, H5], γ=S7, ε0=[E0_75, E0_1],
              τ0=[T0, T500], ω=W2_0, ω0=[W2, W2_0], μ=M0_2, μ0=[M0_4, M0_2],
              ρ=1, anneal_epochs=EPOCHS * 2, epochs=EPOCHS * 5,
              model_load_skip=QUICK_DEL, optimizer_load_skip=('RESET',)),
    DoTwinVAE(id=28, α=A0_5, β=B1, η=H5, γ=S7, ρ=1),

    # model_load_skip=QUICK_DEL,
    # model_load_skip=('',), optimizer_load_skip=('',),
    # optimizer_load_skip=('',),
    # model_load_skip=('RESET',)
    # optimizer_load_skip=('RESET',)

    DoTwinVAE(
        id=29, α=A0_5, α0=[A0_5, A0_5], β=B1, η=H1, η0=[H5, H1], ω=W8_0, ω0=[W2_0, W8_0], γ=S7,
        ρ=1, anneal_epochs=EPOCHS,
    ),
    DoTwinVAE(id=30, α=A0_5, β=B1, η=H1, ω=W8_0, γ=S7, ρ=1),

    DoTwinVAE(id=3900, ρ=0, basis_strip=(), anneal_epochs=2 * EPOCHS, epochs=EPOCHS * 6,
              α=A0_5, α0=[B0_005, A0_5], β=B1, β0=[B0_005, B1], γ=S7, γ0=[S0, S7],
              η=H10, η0=[H30, H10], λ0=[1, 0], τ=T0, τ0=[T00, T0], ε=E0_75, ω=W2, μ=M0_4,
              model_load_skip=REPLACE_DECODERS_DEL, optimizer_load_skip=('RESET',)),
    DoTwinVAE(id=4000, α=A0_5, β=B1, η=H5, η0=[H10, H5], γ=S7, ε0=[E0_75, E0_1],
              τ0=[T0, T500], ω=W2_0, ω0=[W2, W2_0], μ=M0_2, μ0=[M0_4, M0_2],
              ρ=1, anneal_epochs=EPOCHS * 2, epochs=EPOCHS * 4),
    DoTwinVAE(id=350, α=A0_5, α0=[A0_5, A0_5], β=B1, η=H1, η0=[H5, H1], ω=W8_0, ω0=[W2_0, W8_0], γ=S7,
              ρ=0, anneal_epochs=2 * EPOCHS, epochs=EPOCHS * 4, model_load_skip=REPLACE_DECODERS_DEL),
    DoTwinVAE(id=360, α=A0_5, β=B1, η=H1, ω=W8_0, γ=S7, ρ=0),
    DoTwinVAE(id=3700, α=A0_5, β=B1, η=H1, ω=W8_0, γ=S7, ρ=1),
    DoTwinVAE(id=135, α=A0_5, β=B1, η=H1, ω=W8_0, γ=S7, ρ=0, epochs=EPOCHS * 4, model_load_skip=REPLACE_DECODERS_DEL),
    DoTwinVAE(id=136, α=A0_5, β=B1, η=H1, ω=W8_0, γ=S7, ρ=1),

    DoTwinVAE(id=35, ρ=0, basis_strip=(), anneal_epochs=2 * EPOCHS, epochs=EPOCHS * 6,
              α=A0_5, α0=[B0_005, A0_5], β=B1, β0=[B0_005, B1], γ=S7, γ0=[S0, S7],
              η=H10, η0=[H30, H10], λ0=[1, 0], τ=T0, τ0=[T00, T0], ε=E0_75, ω=W2 * W_MOD, μ=M0_4,
              model_load_skip=REPLACE_DECODERS_DEL, optimizer_load_skip=('RESET',)),
    DoTwinVAE(id=36, α=A0_5, β=B1, η=H5, η0=[H10, H5], γ=S7, ε0=[E0_75, E0_1],
              τ0=[T0, T500], ω=W2_0 * W_MOD, ω0=[W2 * W_MOD, W2_0 * W_MOD], μ=M0_2, μ0=[M0_4, M0_2],
              ρ=1, anneal_epochs=EPOCHS * 2, epochs=EPOCHS * 4),
    DoTwinVAE(id=37, α=A0_5, α0=[A0_5, A0_5], β=B1, η=H1, η0=[H5, H1], ω=W8_0 * W_MOD, ω0=[W2_0 * W_MOD, W8_0 * W_MOD],
              γ=S7, ρ=1, anneal_epochs=EPOCHS),
    # , model_load_skip=REPLACE_DECODERS_DEL2, optimizer_load_skip=('RESET',)
    DoTwinVAE(id=43, α=A0_5, β=B1, η=H1, ω=W8_0 * W_MOD, γ=S7, ρ=1),
    DoTwinVAE(id=44, α=A0_5, β=B1, η=H1, ω=W8_0 * W_MOD, γ=S7, ρ=2),
    DoTwinVAE(id=45, α=A0_5, β=B1, η=H1, ω=W8_0 * W_MOD, γ=S7, ρ=3),
    DoTwinVAE(id=42, α=A0_5, β=B1, η=H1, ω=W8_0 * W_MOD, γ=S7, ρ=4),
    DoTwinVAE(id=41, α=A0_5, β=B1, η=H1, ω=W8_0 * W_MOD, γ=S7, ρ=5),

    DoTwinVAE(id=38, α=A0_5, β=B1, η=H1, ω=W8_0 * W_MOD, γ=S7, ρ=2,
              optimizer_load_skip=('RESET',)),  # ρ like in 21 [was 1 without averaging]
    # , model_load_skip=('',), optimizer_load_skip=('',)),
    DoTwinVAE(id=39, α=A0_5, β=B1, η=H1, ω=W8_0 * W_MOD, γ=S7, ρ=2,
              optimizer_load_skip=('RESET',)),  # ρ like in 21 [was 3 without averaging]
    DoTwinVAE(id=46, α=A0_5, β=B1, η=H1, ω=W8_0 * W_MOD, γ=S7, ρ=2),

    DoTwinVAE(id=21, α=A0_5, β=B1, η=H1, ω=W8_0, γ=S7, ρ=2,
              model_load_skip=('',), optimizer_load_skip=('RESET',)  # TODO this line is needed when averaging
              ),
    DoTwinVAE(id=48, α=A0_5, β=B1, η=H1, ω=W8_0, γ=S7, ρ=1),

    # DoTwinVAE(id=20, α=A1, β=B1, η=H10, γ=S5, ρ=1),
    # DoTwinVAE(id=21, α=A1_25, β=B2_5, η=H1, γ=S5, ε=E0_2, ρ=1),

    DoTwinVAE(id=27, α=A0_5, β=B1, η=H1, ω=W8_0, γ=S7, ρ=2),
    DoTwinVAE(id=47, α=A0_5, β=B1, η=H1, ω=W8_0, γ=S7, ρ=2),
    # DoTwinVAE(id=31, α=A0_5, β=B1, η=H1, ω=W8_0, γ=S7, ε=E0, ρ=2),

    # DoTwinVAE(id=27, β=B1, β0=[B1_25, B1], η=H0, γ=S7, ρ=1, anneal_epochs=EPOCHS),
    # DoTwinVAE(id=28, β=B1, η=H0, γ=S7, ρ=1),

    # DoTwinVAE(id=24, γ=S7, η0=[0, H8], η=H8,
    #           ρ=1, basis_strip=(), anneal_epochs=round(EPOCHS * 0.85), epochs=round(EPOCHS * 1.5)),
    # DoTwinVAE(id=25, γ=S7 + 4, γ0=[S7, S7 + 4], η0=[H8, H2], η=H2,
    #           ρ=1, anneal_epochs=round(EPOCHS * 0.85), epochs=round(EPOCHS * 1.5)),
    # DoTwinVAE(id=26, γ=S7 + 4, η=H2, ρ=1),  # τ=T0, α=A0,
    # DoTwinVAE(id=9, γ=S7, ρ=1),

    # # SOLTI ALT.
    # # 8, 20, 21, [800] 9, 9, 6, [1280] 7, 7, 7, 7, 7, 7, 7,
    # # 8, 20, 21, 9, 9, 6, 7, 7, 7, 7, 7, 7, 7,
    # DoTwinVAE(id=8, ε=0, γ=s(1.5), γ0=[S0, s(1.5)], η0=H0, β0=B0, λ0=[1, 0],
    #           anneal_epochs=EPOCHS, epochs=EPOCHS * 4),
    # DoTwinVAE(id=20, ε=0, γ=S2, γ0=[s(1.5), S2], anneal_epochs=EPOCHS, epochs=EPOCHS * 4),
    # DoTwinVAE(id=21, ε0=[0, E0], γ0=[S2, S2], anneal_epochs=EPOCHS),

    DoTwinVAE(id=6, α=A1, β=B1, η=H0, ω=W8_0, γ=S7, ρ=2),
    DoTwinVAE(id=7, α=A1, β=B1, η=H0, ω=W8_0, γ=S7, ρ=2),

    DoTwinVAE(id=13, α=1, β=1, γ=S0, η=1, ω=W8_0, ρ=5,  # μ=M0_8,  # was ρ=3
              model_load_skip=('',),
              # epochs=round(EPOCHS / 2)
              ),
    DoTwinVAE(id=14, α=1, β=1, γ=S0, η=1, ω=W8_0, ρ=5  # μ=M0_8,  # was ρ=3
              # epochs=round(EPOCHS / 2)
              ),
    DoTwinVAE(id=31, α=1, β=1, γ=S0, η=1, ω=W8_0, ρ=5,  # μ=M0_9,  # was ρ=4
              ),
    DoTwinVAE(id=32, α=1, β=1, γ=S0, η=1, ω=W8_0, ρ=5,  # μ=M0_9,
              ),
    DoTwinVAE(id=33, α=1, β=1, γ=S0, η=1, ω=W8_0, ρ=5,  # μ=M0_9,
              ),

    DoTwinVAE(id=10, β=1, α=1, γ=S0, η=H1, ω=W8_0, ρ=2, τ=0,
              model_load_skip=('_twin_vae_registered.classifier.transform',),
              ),
    DoTwinVAE(id=11, β=1, α=1, γ=S0, η=H1, ω=W8_0, ρ=2, τ=0),

    DoTwinVAE(id=12, β=B2_5, α=A2, α0=[A1, A2], γ=S7, η=H0_5, η0=[H0_5, H0_5], ω=W8_0, ρ=2, anneal_epochs=EPOCHS),
    DoTwinVAE(id=19, β=B2_5, α=A2, γ=S7, η=H0_5, ω=W8_0, ρ=2),

    DoTwinVAE(id=15, α=1, β=1, γ=S0, η=H0, ω=W8_0,
              model_load_skip=('_twin_vae_registered._kld._prior_dist.μ',
                               '_twin_vae_registered._kld._prior_dist.log_σ',
                               '_kld._prior_dist.μ',
                               '_kld._prior_dist.log_σ',),
              ρ=2),
    DoTwinVAE(id=16, α=1, β=1, γ=S0, η=H0, ω=W8_0, ρ=2),

    # ---- DGM ---------------------
    # 17, 18, 18, 18, 18, 18,
    Do1(id=17, λ=1, β0=[B0_005, B2_5], anneal_epochs=EPOCHS, epochs=EPOCHS * 2),
    Do1(id=18, λ=1, epochs=EPOCHS),
)


# %% ----------------------------------------------------
# Model getters
# -------------------------------------------------------
nll = BernoulliLoss()
gl.data_loader.upd_nll_state(nll)

# noinspection PyUnusedLocal
def modify_vae_inplace(spec: Do, model: Module) -> None:  # pylint: disable=unused-argument
    if not SET_FLOW:
        return
    assert not isinstance(model, AuxiliaryDeepGenerativeModel)
    assert isinstance(model, VariationalAutoencoder)
    vae: VariationalAutoencoder = model

    flows0 = NormalizingFlows(flows=[flows.NSFAR(gl.dims.z) for _ in range(4)])
    # flows0 = NormalizingFlows(flows=[flows.MAF(gl.dims.z) for _ in range(16)])
    # flows0 = NormalizingFlows(flows=[PlanarNormalizingFlow(gl.dims.z) for _ in range(16)])
    # flows0 = BNAFs(dim=gl.dims.z, flows_n=16)
    # flows0 = NormalizingFlows(dim=gl.dims.z)
    vae.kld.prior_dist.set_inv_pz_flow(flows0)

    flows1 = NormalizingFlows(flows=[flows.NSFAR(gl.dims.z) for _ in range(4)])
    # flows1 = BNAFs(dim=gl.dims.z, flows_n=8)
    # flows1 = NormalizingFlows(flows=[flows.MAF(gl.dims.z) for _ in range(8)])
    # flows1 = NormalizingFlows(flows=[flows.RealNVP(gl.dims.z) for _ in range(16)])
    # flows1 = NormalizingFlows(dim=gl.dims.z)
    vae.kld.set_qz_x_flow(flows1)
    return


# noinspection PyUnusedLocal
def modify_dgm_inplace(spec: Do, model: Module) -> None:  # pylint: disable=unused-argument
    pass


# noinspection PyUnusedLocal
def get_vae(spec: Do, module: Opt[Module]) -> Opt[Module]:  # pylint: disable=unused-argument
    if spec.post_load:
        return None
    model = VAEPassthroughClassifyMMD(  # VAEPassthrough VAEPassthroughClassify VAEPassthroughClassifyJATSMover
        (gl.dims.x, gl.dims.z, gl.dims.h, gl.dims.y, gl.dims.hy),
        # (gl.dims.x, gl.dims.z, gl.dims.h),
        Encode=EncoderCustom2,  # EncoderCustom EncoderCustomJATSMover
        Decode=DecoderPassthroughSex,  # DecoderPassthrCustomSex
        # Classify=ClassifierJATSCustom,
    )
    if USE_TC_KLD:
        model.set_kld(
            BetaTCKLDLoss(
                kl=BetaTC(λ_kld__γmin1_tc=True),
                mss=gl.tc_kld_mss,
            ))
    modify_vae_inplace(spec, model)
    return model


# noinspection PyUnusedLocal
def get_optimizer(spec: Do, model: Module) -> Opt[Optimizer]:
    if spec.post_load:
        return None
    return tr.optim.Adam(model.parameters(), lr=gl.learning_rate)


# noinspection PyUnusedLocal
def get_optimizer_default(spec: Do, model: Module) -> Opt[Optimizer]:
    if spec.post_load:
        return None
    return tr.optim.Adam(model.parameters())


def get_twin_vae(spec: Do, module: Opt[Module]) -> Opt[Module]:
    ClassF: Type[ClassifierJATSAxesAlign]
    if spec.id in (FIRSTS + AXES_ALIGN + REPLACE_ENCODER):
        ClassF = ClassifierJATSAxesAlign
        Dec = DecoderPassthrTwinSexKhAx
    else:
        ClassF = ClassifierJATSAxesAlignDecSwap
        Dec = DecoderPassthrTwinSexKhAxDecSwap
    # this works only if classifiers do not differ in loadable parameters

    def get_inner_vae(interface_: VAEPassthroughClassifyTwinMMD) -> VAEPassthroughClassifyTwinMMD:
        vae_ = interface_.vae_twin
        assert isinstance(vae_, VAEPassthroughClassifyTwinMMD)
        return vae_

    # noinspection PyUnusedLocal
    def axes_align(interface_: VAEPassthroughClassifyTwinMMD) -> None:  # pylint: disable=unused-argument
        pass
        # vae_ = get_inner_vae(interface_)
        # vae_.kld.set_prior_dist(ZeroSymmetricBimodalNormal(z_dim=gl.dims.z, μ=μs, σ=σs))
        # interface_.kld.set_prior_dist(ZeroSymmetricBimodalNormal(z_dim=gl.dims.z, μ=μs, σ=σs))

    # noinspection PyUnusedLocal
    def replace_encoder(interface_: VAEPassthroughClassifyTwinMMD) -> None:  # pylint: disable=unused-argument
        pass

    # noinspection PyUnusedLocal
    def replace_decoders(interface_: VAEPassthroughClassifyTwinMMD) -> None:  # pylint: disable=unused-argument
        pass

    # noinspection PyUnusedLocal
    def nelbo_fit(interface_: VAEPassthroughClassifyTwinMMD) -> None:  # pylint: disable=unused-argument
        pass

    def class_fit(interface_: VAEPassthroughClassifyTwinMMD) -> None:
        # _ = (interface_,)
        dims0 = interface_.classifier.dims  # TODO SWITCH CLASSIFIER
        z_dim, _, y_dim = dims0
        dims1 = (z_dim, (16,), y_dim)  # 16
        interface_.classifier = Classifier(dims1)  # 16  ClassifierJATSCustom  Classifier
        vae_ = get_inner_vae(interface_)
        dims0 = vae_.classifier.dims
        z_dim, _, y_dim = dims0
        dims1 = (z_dim, (16,), y_dim)  # 16
        vae_.classifier = Classifier(dims1)

    # noinspection PyUnusedLocal
    def class_fit_frozen(interface_: VAEPassthroughClassifyTwinMMD) -> None:  # pylint: disable=unused-argument
        pass

    def flow_fit(interface_: VAEPassthroughClassifyTwinMMD) -> None:
        # _ = (interface_,)
        vae_ = get_inner_vae(interface_)
        z_dim, _, y_dim = vae_.classifier.dims
        vae_.classifier = ClassF((z_dim, (16,), y_dim))
        # ClassifierJATSAxesAlignNTPAdeAdi ClassifierJATSAxesAlignNTPEAdTypeSymm  ClassifierJATSAxesAlignNTPEAd

        # vae_.kld.set_prior_dist(ZeroSymmetricBimodalNormal(z_dim=gl.dims.z, μ=MS, σ=ΣS,
        #                                                    # learnable_μ=True, learnable_σ=True
        #                                                    ))
        vae_.kld.prior_dist.set_inv_pz_flow(
            # NormalizingFlows(flows=[flows.NSFAR(gl.dims.z, hidden_dim=gl.dims.z) for _ in range(8)])
            # was: NSFAR, MAF, RealNVP (buggy at dim=5)
            BNAFs(dim=gl.dims.z, hidden_dim=2 * gl.dims.z, flows_n=8)
        )
        assert interface_.kld.pz_inv_flow is None

    def flow_map(interface_: VAEPassthroughClassifyTwinMMD) -> None:
        _ = (interface_,)

    def priors_fit(interface_: VAEPassthroughClassifyTwinMMD) -> None:
        interface_.vae_twin.kld.set_prior_dist(ZeroSymmetricBimodalNormal(
            z_dim=gl.dims.z, μ=MS2, σ=ΣS2, learnable_μ=True, learnable_σ=True,
        ))

    if spec.post_load:
        assert isinstance(module, VAEPassthroughClassifyTwinMMD)
        interface = module
        ret: Opt[Module] = interface
        if spec.id == AXES_ALIGN[0]:
            axes_align(interface)
        elif spec.id == REPLACE_ENCODER[0]:
            replace_encoder(interface)
        elif spec.id == AXES_ALIGN2[0]:
            replace_encoder(interface)
        elif spec.id == NELBO_FIT[0]:
            nelbo_fit(interface)
        elif spec.id == CLASS_FIT[0]:
            class_fit(interface)
        elif spec.id == CLASS_FIT_FROZEN[0]:
            class_fit_frozen(interface)
        elif spec.id == FLOW_FIT[0]:
            flow_fit(interface)
        elif spec.id == FLOW_MAP[0]:
            flow_map(interface)
        elif spec.id == PRIORS_FIT[0]:
            priors_fit(interface)
        else:
            ret = None
        return ret

    # ---- firsts begin ----
    def twin_vae(Classify: Type[Classifier], inner: bool):
        if spec.id == REPLACE_ENCODER[0]:
            Enc = EncoderTwinFrozen700
        elif spec.id in (FIRSTS + AXES_ALIGN):
            Enc = EncoderTwinJATS
        elif inner:
            Enc = EncoderTwinFrozen700Part2Inner
        else:
            Enc = EncoderTwinFrozen700Part2
        return VAEPassthroughClassifyTwinMMD(
            # VAEΣ1Σ2PassthroughClassify VAEPassthroughClassify VAEPassthroughClassifyJATSMover VAEPassthrough
            (gl.dims.x, gl.dims.z, gl.dims.h, gl.dims.y, gl.dims.hy),
            Encode=Enc,  # EncoderΣ1Σ2 EncoderCustom EncoderTrimCustom EncoderTrimCustomJATSMover
            Decode=Dec,
            # DecoderPassthrTwinSexKhAx, DecoderPassthroughTwinSex, DecoderPassthrCustomSex
            Classify=Classify,  # ClassifierCustom ClassifierJATSCustom Classifier
        )

    def set_subdecoder(vae_: VAEPassthroughClassifyTwinMMD) -> None:
        assert isinstance(vae_.classifier, ClassifierJATSAxesAlign)
        _cls: ClassifierJATSAxesAlign = vae_.classifier
        assert isinstance(vae_.decoder, DecoderPassthrTwinSexKhAx)
        _dec: DecoderPassthrTwinSexKhAx = vae_.decoder
        _cls.set_subdecoder(_dec.subdecoder)

    vae = twin_vae(ClassF, inner=True)
    vae.set_kld(BetaTCKLDLoss(
        kl=BetaTC(λ_kld__γmin1_tc=True),
        mss=gl.tc_kld_mss,
        # prior_dist=None,
        prior_dist=ZeroSymmetricBimodalNormal(
            z_dim=gl.dims.z, μ=MS, σ=ΣS,
            # learnable_μ=True, learnable_σ=True,
        ),
    ))
    interface = twin_vae(ClassF, inner=False)  # [Classifier @550] ClassifierJATS0sex1ie2sn3ft
    # interface.set_kld(BetaTCKLDLossTwin(
    #     kl=BetaTC(kld__γmin1_tc=True),
    #     mss=gl.tc_kld_mss,
    # ))
    interface.set_kld(BetaTCKLDLoss(
        kl=BetaTC(λ_kld__γmin1_tc=True),
        mss=gl.tc_kld_mss,
        # prior_dist=None,
        prior_dist=ZeroSymmetricBimodalNormal(
            z_dim=gl.dims.z, μ=MS, σ=ΣS,
            # learnable_μ=True, learnable_σ=True,
        ),
    ))
    interface.set_twin_vae(vae)
    interface.use_svi = False

    set_subdecoder(vae)
    set_subdecoder(interface)

    # assert isinstance(interface.classifier, ClassifierJATSAxesAlignRot)
    # assert isinstance(vae.classifier, ClassifierJATSAxesAlignRot)
    # interface_cls: ClassifierJATSAxesAlignRot = interface.classifier
    # inner_cls: ClassifierJATSAxesAlignRot = vae.classifier
    # interface_cls.β = 0
    # inner_cls.α = 0
    # ---- firsts end ----
    previous = FIRSTS

    if spec.id in AXES_ALIGN:
        if spec.id != AXES_ALIGN[0]:
            axes_align(interface)
    elif spec.id not in previous:
        axes_align(interface)
    previous = previous + AXES_ALIGN

    if spec.id in REPLACE_ENCODER:
        if spec.id != REPLACE_ENCODER[0]:
            replace_encoder(interface)
    elif spec.id not in previous:
        replace_encoder(interface)
    previous = (previous + REPLACE_ENCODER) if USE_REPLACE_ENC_AND_DEC else previous

    if spec.id in REPLACE_DECODERS:
        if spec.id != REPLACE_DECODERS[0]:
            replace_decoders(interface)
    elif spec.id not in previous:
        replace_decoders(interface)
    previous = (previous + REPLACE_DECODERS) if USE_REPLACE_ENC_AND_DEC else previous

    if spec.id in REPLACE_DECODERS2:
        if spec.id != REPLACE_DECODERS2[0]:
            replace_decoders(interface)
    elif spec.id not in previous:
        replace_decoders(interface)
    previous = (previous + REPLACE_DECODERS2) if USE_REPLACE_ENC_AND_DEC else previous

    if spec.id in AXES_ALIGN2:
        if spec.id != AXES_ALIGN2[0]:
            axes_align(interface)
    elif spec.id not in previous:
        axes_align(interface)
    previous = (previous + AXES_ALIGN2) if USE_REPLACE_ENC_AND_DEC else previous

    if spec.id in NELBO_FIT:
        if spec.id != NELBO_FIT[0]:
            nelbo_fit(interface)
    elif spec.id not in previous:
        nelbo_fit(interface)
    previous = previous + NELBO_FIT

    if spec.id in CLASS_FIT:
        if spec.id != CLASS_FIT[0]:
            class_fit(interface)
    elif spec.id not in previous:
        class_fit(interface)
    previous = previous + CLASS_FIT

    if spec.id in CLASS_FIT_FROZEN:
        if spec.id != CLASS_FIT_FROZEN[0]:
            class_fit_frozen(interface)
    elif spec.id not in previous:
        class_fit_frozen(interface)
    previous = previous + CLASS_FIT_FROZEN

    if spec.id in FLOW_FIT:
        if spec.id != FLOW_FIT[0]:
            flow_fit(interface)
    elif spec.id not in previous:
        flow_fit(interface)
    previous = previous + FLOW_FIT

    if spec.id in FLOW_MAP:
        if spec.id != FLOW_MAP[0]:
            flow_map(interface)
    elif spec.id not in previous:
        flow_map(interface)
    previous = previous + FLOW_MAP

    if spec.id in PRIORS_FIT:
        if spec.id != PRIORS_FIT[0]:
            priors_fit(interface)
    elif spec.id not in previous:
        priors_fit(interface)

    return interface


def get_optimizer_twin_vae(spec: Do, model: Module) -> Opt[Optimizer]:
    ret: Opt[Optimizer]
    interface: VAEPassthroughClassifyTwinMMD
    inner_vae: VAEPassthroughClassifyTwinMMD

    assert isinstance(model, VAEPassthroughClassifyTwinMMD)
    interface = model
    _vae_twin = interface.vae_twin
    assert isinstance(_vae_twin, VAEPassthroughClassifyTwinMMD)
    inner_vae = _vae_twin
    assert isinstance(interface.encoder.sample, GaussianSample)
    interface_sample: GaussianSample = interface.encoder.sample
    assert isinstance(inner_vae.encoder.sample, GaussianSample)
    inner_sample: GaussianSample = inner_vae.encoder.sample

    # tmp1 = interface.kld.prior_dist
    # assert isinstance(tmp1, Normal)
    # p_dist_interface: Normal = tmp1
    tmp2 = inner_vae.kld.prior_dist
    assert isinstance(tmp2, Normal)
    p_dist_inner: Normal = tmp2

    def firsts() -> Optimizer:
        # assert isinstance(inner_vae.classifier, ClassifierJATSAxesAlignRot)
        # cls: ClassifierJATSAxesAlignRot = inner_vae.classifier
        # cls.mults.requires_grad = False
        return tr.optim.Adam(model.parameters(), lr=gl.learning_rate)

    def axes_align() -> Optimizer:
        # assert isinstance(inner_vae.classifier, ClassifierJATSAxesAlignRot)
        # cls: ClassifierJATSAxesAlignRot = inner_vae.classifier
        # cls.mults.requires_grad = False

        if isinstance(interface.encoder, EncoderTwinSplit0) or isinstance(inner_vae.encoder, EncoderTwinSplit0):
            assert isinstance(interface.encoder, EncoderTwinSplit0)
            assert isinstance(inner_vae.encoder, EncoderTwinSplit0)
            intfc_enc0: EncoderTwinSplit0 = interface.encoder
            inner_enc0: EncoderTwinSplit0 = inner_vae.encoder
            for lin in chain(intfc_enc0.encoder2.hidden, inner_enc0.encoder2.hidden,
                             [intfc_enc0.encoder2.sample.μ, inner_enc0.encoder2.sample.μ]):
                lin.weight.requires_grad = False
                if lin.bias is not None:
                    lin.bias.requires_grad = False

        elif isinstance(interface.encoder, EncoderTwinSplit1) or isinstance(inner_vae.encoder, EncoderTwinSplit1):
            assert isinstance(interface.encoder, EncoderTwinSplit1)
            assert isinstance(inner_vae.encoder, EncoderTwinSplit1)
            intfc_enc1: EncoderTwinSplit1 = interface.encoder
            inner_enc1: EncoderTwinSplit1 = inner_vae.encoder
            for lin in chain(intfc_enc1.hidden2, inner_enc1.hidden2,
                             [intfc_enc1.sample2.μ, inner_enc1.sample2.μ]):
                lin.weight.requires_grad = False
                if lin.bias is not None:
                    lin.bias.requires_grad = False

        return tr.optim.Adam(model.parameters(), lr=gl.learning_rate)

    def replace_encoder() -> Optimizer:
        if interface.svi is not None:
            raise NotImplementedError
        interface_enc_hidden, inner_vae_enc_hidden = interface.encoder.hidden, inner_vae.encoder.hidden
        assert isinstance(interface_enc_hidden, nn.ModuleList)
        assert isinstance(inner_vae_enc_hidden, nn.ModuleList)
        return tr.optim.Adam(chain(
            interface.classifier.parameters(),
            inner_vae.classifier.parameters(),
            interface.decoder.parameters(),
            inner_vae.decoder.parameters(),
            interface.kld.parameters(),
            inner_vae.kld.parameters(),
            inner_sample.log_σ.parameters(),
            interface_sample.log_σ.parameters(),
            interface_enc_hidden.parameters(),
            inner_vae_enc_hidden.parameters(),
        ), lr=gl.learning_rate)

    def replace_decoders() -> Optimizer:
        if interface.svi is not None:
            raise NotImplementedError
        interface_enc_hidden, inner_vae_enc_hidden = interface.encoder.hidden, inner_vae.encoder.hidden
        assert isinstance(interface_enc_hidden, nn.ModuleList)
        assert isinstance(inner_vae_enc_hidden, nn.ModuleList)
        return tr.optim.Adam(chain(
            interface.classifier.parameters(),
            inner_vae.classifier.parameters(),
            interface.decoder.parameters(),
            inner_vae.decoder.parameters(),
            interface.kld.parameters(),
            inner_vae.kld.parameters(),
            inner_sample.log_σ.parameters(),
            interface_sample.log_σ.parameters(),
        ), lr=gl.learning_rate)

    def nelbo_fit() -> Optimizer:
        return tr.optim.Adam(chain(
            p_dist_inner.parameters(),
            # p_dist_interface.parameters(),
            inner_sample.log_σ.parameters(),
            # interface_sample.log_σ.parameters(),
        ), lr=gl.learning_rate)

    def class_fit() -> Optimizer:
        if interface.svi is not None:
            return tr.optim.Adam(interface.svi.parameters())
        return tr.optim.Adam(chain(
            interface.classifier.parameters(),  # TODO SWITCH CLASSIFIER
            inner_vae.classifier.parameters()
        ), lr=gl.cls_learning_rate)

    def class_fit_frozen() -> Optimizer:
        if interface.svi is not None:
            raise NotImplementedError
        raise NotImplementedError
        # noinspection PyUnreachableCode
        return class_fit()

    def flow_fit() -> Optimizer:
        assert p_dist_inner.inv_pz_flow is not None
        return tr.optim.Adam(chain(
            p_dist_inner.inv_pz_flow.parameters(),
            inner_sample.log_σ.parameters(),
        ), lr=gl.learning_rate)
        # return class_fit()

    def flow_map() -> Optimizer:
        # priors: ZeroSymmetricBimodalNormal = inner_vae.kld.prior_dist  # type: ignore
        # assert isinstance(priors, ZeroSymmetricBimodalNormal)
        # priors.μ.requires_grad = False
        # priors.log_σ.requires_grad = False
        #
        # return firsts()
        # return tr.optim.Adam(chain(
        #     p_dist_inner.inv_pz_flow.parameters(),
        #     inner_sample.log_σ.parameters(),
        #     inner_vae.classifier.parameters(),
        # ), lr=gl.learning_rate)
        return class_fit()

    def priors_fit() -> Optimizer:
        return tr.optim.Adam(chain(
            inner_sample.log_σ.parameters(),
            (p_dist_inner.μ,
             p_dist_inner.log_σ),
        ), lr=gl.learning_rate)

    if spec.post_load:
        if spec.id == AXES_ALIGN[0]:
            ret = axes_align()
        elif spec.id == REPLACE_ENCODER[0]:
            ret = replace_encoder()
        elif spec.id == REPLACE_DECODERS[0]:
            ret = replace_decoders()
        elif spec.id == REPLACE_DECODERS2[0]:
            ret = replace_encoder()
        elif spec.id == AXES_ALIGN2[0]:
            ret = axes_align()
        elif spec.id == NELBO_FIT[0]:
            ret = nelbo_fit()
        elif spec.id == CLASS_FIT[0]:
            ret = class_fit()
        elif spec.id == CLASS_FIT_FROZEN[0]:
            ret = class_fit_frozen()
        elif spec.id == FLOW_FIT[0]:
            ret = flow_fit()
        elif spec.id == FLOW_MAP[0]:
            ret = flow_map()
        elif spec.id == PRIORS_FIT[0]:
            ret = priors_fit()
        else:
            ret = None
    elif USE_REPLACE_ENC_AND_DEC and spec.id in (AXES_ALIGN[1:] + REPLACE_ENCODER[:1]):
        ret = axes_align()
    elif USE_REPLACE_ENC_AND_DEC and spec.id in (REPLACE_ENCODER[1:] + REPLACE_DECODERS[:1]):
        ret = replace_encoder()
    elif USE_REPLACE_ENC_AND_DEC and spec.id in (REPLACE_DECODERS[1:] + REPLACE_DECODERS2[:1]):
        ret = replace_decoders()
    elif USE_REPLACE_ENC_AND_DEC and spec.id in (REPLACE_DECODERS2[1:] + AXES_ALIGN2[:1]):
        ret = replace_encoder()
    elif USE_REPLACE_ENC_AND_DEC and spec.id in (AXES_ALIGN2[1:] + NELBO_FIT[:1]):
        ret = axes_align()
    elif spec.id in (FIRSTS + AXES_ALIGN[:1]):
        ret = firsts()
    elif spec.id in (AXES_ALIGN[1:] + NELBO_FIT[:1]):
        ret = axes_align()
    elif spec.id in (NELBO_FIT[1:] + CLASS_FIT[:1]):
        ret = nelbo_fit()
    elif spec.id in (CLASS_FIT[1:] + CLASS_FIT_FROZEN[:1]):
        ret = class_fit()
    elif spec.id in (CLASS_FIT_FROZEN[1:] + FLOW_FIT[:1]):
        ret = class_fit_frozen()
    elif spec.id in (FLOW_FIT[1:] + FLOW_MAP[:1]):
        ret = flow_fit()
    elif spec.id in (FLOW_MAP[1:] + PRIORS_FIT[:1]):
        ret = flow_map()
    elif spec.id in PRIORS_FIT[1:]:
        ret = priors_fit()
    else:
        raise NotImplementedError
    return ret


def get_dgm(spec: Do, module: Opt[Module]) -> Opt[Module]:
    _ = (module,)
    if spec.post_load:
        return None

    dgm = DGMPassthrough(
            (gl.dims.x, gl.dims.y, gl.dims.z, gl.dims.h),  # 4<->gl.dims.y
            Decode=DecoderPassthroughSex,  # DecoderPassthrCustomSex DecoderPassthrough
            Classify=Classifier,  # ClassifierJATSCustom Classifier ClassifierJATSCustomTemper
            Encode=EncoderCustom,  # EncoderTrimCustom Encoder
            # Encode=EncoderSELUTrim, AuxEncoder=EncoderSELUTrim, AuxDecoder=EncoderPassthrSELUTrim,
        )
    assert gl.data_loader.labelled is not None
    N_lbl: int = len(gl.data_loader.labelled)
    model = SVI(model=dgm, N_l=N_lbl, N_u=len(gl.data_loader.unlabelled), α0=gl.svi_α0,
                nll=nll, β=1, sampler=ImportanceWeightedSampler(mc=1, iw=1))
    modify_dgm_inplace(spec, model.model)
    return model


def get_adgm(spec: Do, module: Opt[Module]) -> Opt[Module]:
    _ = (module,)
    if spec.post_load:
        return None

    adgm = ADGMPassthrough(  # ADGMPassthrough ADGMPassthroughJATSMover
            (gl.dims.x, gl.dims.y, gl.dims.z, gl.dims.a, gl.dims.h),
            Decode=DecoderPassthroughSex,  # DecoderPassthrCustomSex DecoderPassthrough
            Classify=Classifier,  # ClassifierJATSCustom Classifier
            Encode=EncoderCustom,  # EncoderTrimCustom EncoderTrim
            AuxEncoder=EncoderCustom,  # EncoderTrimCustom EncoderSELUJATSMover EncoderSELU EncoderSELUTrimJATSMover
            AuxDecoder=EncoderCustom,  # EncoderSELU EncoderPassthrough
            # AuxEncoder=Encoder, AuxDecoder=EncoderPassthrough,
            # Encode=EncoderSELUTrim, AuxEncoder=EncoderSELUTrim, AuxDecoder=EncoderPassthrSELUTrim,
        )
    assert gl.data_loader.labelled is not None
    N_lbl: int = len(gl.data_loader.labelled)
    model = SVI(model=adgm, N_l=N_lbl, N_u=len(gl.data_loader.unlabelled), α0=gl.svi_α0,
                nll=nll, β=1, sampler=ImportanceWeightedSampler(mc=1, iw=1))
    modify_dgm_inplace(spec, model.model)
    return model


# noinspection PyUnusedLocal
def get_optimizer_fvae(spec: Do, model: Module) -> Opt[Optimizer]:
    if spec.post_load:
        return None
    return tr.optim.Adam(model.parameters(), lr=gl.learning_rate)


def get_optimizer_fvae_discr(discr: Discriminator) -> Optimizer:
    return tr.optim.Adam(discr.parameters(), lr=gl.discr_learning_rate, betas=gl.discr_betas)


def get_disc() -> Discriminator:
    return Discriminator(
        (gl.dims.z, HD_DIMS),
        activation_fn=func.leaky_relu,
        # activation_fn=tr.selu,
    )


# noinspection PyUnusedLocal
def get_loader(spec: Do) -> Iterable:  # pylint: disable=unused-argument
    return gl.data_loader.get_lbl_loader()


# noinspection PyUnusedLocal
def get_loader_fvae(spec: Do) -> Iterable:  # pylint: disable=unused-argument
    return gl.data_loader.get_double_lbl_loader()


# %% ----------------------------------------------------
# VAEΣ1Σ2
# -------------------------------------------------------
def get_twin_vae_guide(models_idxs: Tuple[int, ...], to_do_list: Tuple[int, ...]) -> Universal:
    return GuideTwinVAE(
        get_model=get_twin_vae, get_optimizer=get_optimizer_twin_vae, get_loader=get_loader,
        id_pref=ID_PREF, models_idxs=models_idxs, successful_models_n=SUCCESSFUL_MODELS_N, to_do=to_do_list,
        do_specs=DO_SPECS,
        logger=gl.logger, dtype=gl.dtype, device=gl.device, glb_spec=gl.spec,
        train=TRAIN, print_every_epoch=PRINT_EVERY_EPOCH,
        context=GuideTwinVAE.ctx(
            loader=gl.data_loader, nll=nll, db=gl.db, pca=pca, fa=fa, data=gl.data, trans=gl.trans,
            dims=gl.dims))


# %% ----------------------------------------------------
# VAE
# -------------------------------------------------------
def get_vae_guide() -> Universal:
    return GuideVAE(
        get_model=get_vae, get_optimizer=get_optimizer, get_loader=get_loader,
        id_pref=ID_PREF, models_idxs=MODELS_IDXS, successful_models_n=SUCCESSFUL_MODELS_N, to_do=TO_DO_LIST,
        do_specs=DO_SPECS,
        logger=gl.logger, dtype=gl.dtype, device=gl.device, glb_spec=gl.spec,
        train=TRAIN, print_every_epoch=PRINT_EVERY_EPOCH,
        context=GuideVAE.ctx(
            loader=gl.data_loader, nll=nll, db=gl.db, pca=pca, fa=fa, data=gl.data, trans=gl.trans,
            dims=gl.dims))


# %% ----------------------------------------------------
# DGM
# -------------------------------------------------------
def get_dgm_guide() -> Universal:
    return GuideDGM(
        get_model=get_dgm, get_optimizer=get_optimizer_default, get_loader=get_loader,
        id_pref=ID_PREF, models_idxs=MODELS_IDXS, successful_models_n=SUCCESSFUL_MODELS_N, to_do=TO_DO_LIST,
        do_specs=DO_SPECS,
        logger=gl.logger, dtype=gl.dtype, device=gl.device, glb_spec=gl.spec,
        train=TRAIN, print_every_epoch=PRINT_EVERY_EPOCH,
        context=GuideDGM.ctx(
            loader=gl.data_loader, nll=nll, db=gl.db, pca=pca, fa=fa, data=gl.data, trans=gl.trans,
            dims=gl.dims))


# %% ----------------------------------------------------
# ADGM
# -------------------------------------------------------
def get_adgm_guide() -> Universal:
    return GuideADGM(
        get_model=get_adgm, get_optimizer=get_optimizer_default, get_loader=get_loader,
        id_pref=ID_PREF, models_idxs=MODELS_IDXS, successful_models_n=SUCCESSFUL_MODELS_N, to_do=TO_DO_LIST,
        do_specs=DO_SPECS,
        logger=gl.logger, dtype=gl.dtype, device=gl.device, glb_spec=gl.spec,
        train=TRAIN, print_every_epoch=PRINT_EVERY_EPOCH,
        context=GuideADGM.ctx(
            loader=gl.data_loader, nll=nll, db=gl.db, pca=pca, fa=fa, data=gl.data, trans=gl.trans,
            dims=gl.dims))


# %% ----------------------------------------------------
# PREPARE
# -------------------------------------------------------
guide = get_twin_vae_guide(MODELS_IDXS, TO_DO_LIST)
# guide = get_vae_guide()
# guide = get_dgm_guide()
# guide = get_adgm_guide()


# %% ----------------------------------------------------
# RUN
# -------------------------------------------------------
guide.explore_model_output = EXPLORE
guide.plot_model_output = PLOT
guide.run_guide()
print(guide.successful_models)
if TRAIN:
    raise RuntimeError()


# %% ----------------------------------------------------
# Merge FZA
# -------------------------------------------------------
if not MERGE_FZA:
    raise RuntimeError()
guides700 = [get_twin_vae_guide((idx,), (21,)) for idx in (704, 714, 717, 725, 720, 728)]
models700: List[nn.Module] = []
for guide_ in guides700:
    guide_.explore_model_output = True
    guide_.plot_model_output = False
    guide_.run_guide()
    models700.append(guide_.model)

model_ = guide.model
assert isinstance(model_, VAEPassthroughClassifyTwinMMD)
model700 = model_
encoder_ = model700.encoder
assert isinstance(encoder_, EncoderTwinFrozen700)
encoder700 = encoder_
encoder700.set_encoders2(models700)
guide.save()

guide = get_twin_vae_guide(MODELS_IDXS, (39,))
guide.explore_model_output = True
guide.plot_model_output = True
guide.run_guide()

if not TRAIN:
    raise RuntimeError()
# model_ = guide.model
# assert isinstance(model_, VAEPassthroughClassifyTwinMMD)
# model700 = model_
# encoder_ = model700.encoder
# assert isinstance(encoder_, EncoderTwinFrozen700)
# encoder700 = encoder_
# print(len(encoder700.encoders2), len(encoder700.twin_encoder.encoders2))


# %% ----------------------------------------------------
# Check covariance of several models
# -------------------------------------------------------
if not ADDITIONAL_EXPLORE:
    raise RuntimeError()
# from importlib import reload
# import jats_display.covariance as cv
# reload(cv)
# from jats_display.covariance import check_cov
# noinspection PyTypeChecker
check_cov(
    models=guide.models, weight_vec=gl.data.weight_vec, path_prefix=gl.logger.prefix_db(),
    n_models_to_find=SUCCESSFUL_MODELS_N, corr_threshold=0.51, score_to_reach=5,
    # n_models_to_find=3, corr_threshold=0.51, score_to_reach=5,
    # n_models_to_find=4, corr_threshold=0.51, score_to_reach=5,
    # n_models_to_find=5, corr_threshold=0.51, score_to_reach=5,
    # n_models_to_find=6, corr_threshold=0.51, score_to_reach=4.5,
)
_ = """
"""


# %% ----------------------------------------------------
# Regenerate test database subset
# -------------------------------------------------------
# from socionics_db.generate_control_groups import write_test_samples, test_test_samples
# write_test_samples()
# test_test_samples(debug=True)
# ---- Export spec file, rename and remove "__" prefix


# %% ----------------------------------------------------
if not ADDITIONAL_EXPLORE:
    raise RuntimeError()
# from importlib import reload
# import vae.latents_explorer as _latents_explorer1
# import jats_vae.latents_explorer as _latents_explorer2
# reload(_latents_explorer1)
# reload(_latents_explorer2)
# from jats_vae.latents_explorer import LatentsExplorerPassthr

LANG = 'ru'
assert len(list(guide.models.values())) == 1
assert isinstance(guide.model, VariationalAutoencoder)


def z_trans__sn__ft__bgi_ade__bge_adi__jp(z: Array) -> Array:
    cos, sin, π = math.cos, math.sin, math.pi
    rot45 = np.array([[cos(π / 4), -sin(π / 4)],
                      [sin(π / 4), cos(π / 4)]])
    return np.concatenate((z[:, :4], z[:, (4, 5)] @ rot45.T, z[:, 6:]), axis=1)

le = LatentsExplorerPassthr(
    vae=guide.model, nll=nll, questions=gl.db.questions,
    prefix_path_db_nn=guide.logger.prefix_nn(), x=gl.data.input, idxs_lbl=gl.data.indxs_labelled,
    w=gl.data.weight_vec, w_lbl=gl.data.weight_vec_labelled,
    y_lbl=(gl.data.target[gl.data.indxs_labelled]
           if (gl.data.target is not None) and (gl.data.indxs_labelled is not None)
           else None),
    dtype=gl.dtype, device=gl.device, lang=LANG,
    override_w=False,
    show_n_questions=60,
    # z_transform=z_trans__sn__ft__bgi_ade__bge_adi__jp
)


# %% first axis is sex so z0 corresponds to {1: ...}
if not ADDITIONAL_EXPLORE:
    raise RuntimeError()
le.inspect_z_spec('z0-[1,inf]', z_spec={1: (1, math.inf)})
le.inspect_z_spec('z0-[-inf,-1]', z_spec={1: (-math.inf, -1)})
le.inspect_z_spec('z0-[0,inf]', z_spec={1: (0, math.inf)})
le.inspect_z_spec('z0-[-inf,0]', z_spec={1: (-math.inf, 0)})
le.inspect_z_spec('z0-[-1,1]', z_spec={1: (-1, 1)})

le.inspect_z_spec('z1-[1,inf]', z_spec={2: (1, math.inf)})
le.inspect_z_spec('z1-[-inf,-1]', z_spec={2: (-math.inf, -1)})
le.inspect_z_spec('z1-[0,inf]', z_spec={2: (0, math.inf)})
le.inspect_z_spec('z1-[-inf,0]', z_spec={2: (-math.inf, 0)})
le.inspect_z_spec('z1-[-1,1]', z_spec={2: (-1, 1)})

le.inspect_z_spec('z2-[1,inf]', z_spec={3: (1, math.inf)})
le.inspect_z_spec('z2-[-inf,-1]', z_spec={3: (-math.inf, -1)})
le.inspect_z_spec('z2-[0,inf]', z_spec={3: (0, math.inf)})
le.inspect_z_spec('z2-[-inf,0]', z_spec={3: (-math.inf, 0)})
le.inspect_z_spec('z2-[-1,1]', z_spec={3: (-1, 1)})

le.inspect_z_spec('z3-[1,inf]', z_spec={4: (1, math.inf)})
le.inspect_z_spec('z3-[-inf,-1]', z_spec={4: (-math.inf, -1)})
le.inspect_z_spec('z3-[0,inf]', z_spec={4: (0, math.inf)})
le.inspect_z_spec('z3-[-inf,0]', z_spec={4: (-math.inf, 0)})
le.inspect_z_spec('z3-[-1,1]', z_spec={4: (-1, 1)})

le.inspect_z_spec('z4-[1,inf]', z_spec={5: (1, math.inf)})
le.inspect_z_spec('z4-[-inf,-1]', z_spec={5: (-math.inf, -1)})
le.inspect_z_spec('z4-[0,inf]', z_spec={5: (0, math.inf)})
le.inspect_z_spec('z4-[-inf,0]', z_spec={5: (-math.inf, 0)})
le.inspect_z_spec('z4-[-1,1]', z_spec={5: (-1, 1)})

le.inspect_z_spec('z5-[1,inf]', z_spec={6: (1, math.inf)})
le.inspect_z_spec('z5-[-inf,-1]', z_spec={6: (-math.inf, -1)})
le.inspect_z_spec('z5-[0,inf]', z_spec={6: (0, math.inf)})
le.inspect_z_spec('z5-[-inf,0]', z_spec={6: (-math.inf, 0)})
le.inspect_z_spec('z5-[-1,1]', z_spec={6: (-1, 1)})

le.inspect_z_spec('z6-[1,inf]', z_spec={7: (1, math.inf)})
le.inspect_z_spec('z6-[-inf,-1]', z_spec={7: (-math.inf, -1)})
le.inspect_z_spec('z6-[0,inf]', z_spec={7: (0, math.inf)})
le.inspect_z_spec('z6-[-inf,0]', z_spec={7: (-math.inf, 0)})
le.inspect_z_spec('z6-[-1,1]', z_spec={7: (-1, 1)})

le.inspect_z_spec('z7-[1,inf]', z_spec={8: (1, math.inf)})
le.inspect_z_spec('z7-[-inf,-1]', z_spec={8: (-math.inf, -1)})
le.inspect_z_spec('z7-[0,inf]', z_spec={8: (0, math.inf)})
le.inspect_z_spec('z7-[-inf,0]', z_spec={8: (-math.inf, 0)})
le.inspect_z_spec('z7-[-1,1]', z_spec={8: (-1, 1)})

le.inspect_z_spec('s-[0.5,inf]', z_spec={0: (0.5, math.inf)})
le.inspect_z_spec('s-[-inf,0.5]', z_spec={0: (-math.inf, 0.5)})
_ = le.inspect_z_spec('s-[-inf,inf]', z_spec={0: (-math.inf, math.inf)})


# # %%
# # THR_, INF_ = 0.5, math.inf
# # le.inspect_z_spec('INP6', z_spec={1: (-INF_, -THR_ / 2), 2: (THR_, INF_), 4: (THR_, INF_)})
# # le.inspect_z_spec('INJ8', z_spec={1: (-INF_, -THR_ / 2), 2: (THR_, INF_), 8: (THR_, INF_)})
# # le.inspect_z_spec('INP6J8', z_spec={1: (-INF_, -THR_ / 2), 2: (THR_, INF_), 4: (THR_, INF_), 8: (THR_, INF_)})
# # le.inspect_z_spec('INJ6P8', z_spec={1: (-INF_, -THR_ / 2), 2: (THR_, INF_), 4: (-INF_, -THR_), 8: (-INF_, -THR_)})
# # le.inspect_z_spec('INFP6', z_spec={1: (-INF_, -THR_ / 2), 2: (THR_, INF_), 3: (-INF_, -THR_), 4: (THR_, INF_)})
# # le.inspect_z_spec('J6P8', z_spec={4: (-INF_, -THR_), 8: (-INF_, -THR_)})
# # le.inspect_z_spec('P6J8', z_spec={4: (THR_, INF_), 8: (THR_, INF_)})
# # le.inspect_z_spec('P6+Ad', z_spec={4: (THR_, INF_), 5: (THR_, INF_)})
# # le.inspect_z_spec('EN', z_spec={1: (1.33, INF_), 2: (1.33, INF_)})
#
#
# # %%
# # le.inspect_z_spec('y1-y14', y_spec=(1, 14,))
# # le.inspect_z_spec('y2-y13', y_spec=(2, 13,))

# noinspection PyProtectedMember
# guide.model._twin_vae_registered.classifier
