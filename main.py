# %%
# from importlib import reload
from typing import Tuple, List, Dict, Optional as Opt, Union, Any
from dataclasses import dataclass
# noinspection PyPep8Naming
from numpy import ndarray as Array
import torch as tr
from torch import Tensor
import matplotlibhelper as mh  # pylint: disable=wrong-import-order

from kiwi_bugfix_typechecker.nn import Module
from vae import EncoderSELU, EncoderSELUTrim, DecoderSELU, BernoulliLoss, TrimLoss, MMDLoss, ClassifierSELU
from vae.loop_tools import Do, Guide, Consts
from vae.data import VAELoaderReturnType, DGMLoaderReturnType
from vae.linear_component_analyzer import LinearAnalyzer
from beta_tcvae_typed import BetaTC, BetaTCKLDLoss, Normal
from semi_supervised_typed import (VariationalAutoencoder, Decoder, SVI, AuxiliaryDeepGenerativeModel,
                                   DeepGenerativeModel)
from semi_supervised_typed.inference import ImportanceWeightedSampler
from jats_vae.utils import OutputChecker1, TestArraysToTensors, get_basis_kld_vec
from jats_vae.data import model_output
from jats_display import get__x__z__plot_batch, plot_jats, check_cov, explore_jats, log_iter
from socionics_db import JATSModelOutput, debug_vae_loader, debug_dgm_loader
from normalizing_flows_typed import flows, NormalizingFlows, PlanarNormalizingFlow, BNAFs
from ready import Global

_ = (BetaTC, BetaTCKLDLoss, get_basis_kld_vec, BNAFs, NormalizingFlows, Normal, flows,
     DeepGenerativeModel, PlanarNormalizingFlow, EncoderSELU, EncoderSELUTrim, ClassifierSELU)
del _
DEBUG = False
δ = 1e-8
mh.ready(font_size=12, ext='svg', hide=True, magic='agg')


gl = Global()
loader = gl.data_loader
if DEBUG:
    debug_vae_loader(loader)
    debug_dgm_loader(loader)


def print_(*objs: Any) -> None: return gl.logger.print(*objs)
def print_i(*objs: Any) -> None: return gl.logger.print_i(*objs)


TupleInt = Tuple[int, ...]
TupleOptFloat = Tuple[Opt[float], ...]
DictIntStr = Dict[int, str]


@dataclass
class Do1(Do):
    formula: Union[str, List[str]] = 'L'
    kld_trim: Union[TupleOptFloat, List[TupleOptFloat]] = (None,)  # (start_val, end_val) or (start_val, None)
    plot: Union[bool, List[bool]] = True
    mmd_dists: Union[Opt[DictIntStr], List[Opt[DictIntStr]]] = None
    max_abs_Σ: Union[float, List[float]] = 1
    allowed_basis_dims: Union[Opt[TupleInt], List[Opt[TupleInt]]] = None


# %% ----------------------------------------------------
# PCA and FA:
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
    pca, fa = gl.get_pca_fa(plot_pca=False, print_pca=False)


# %% ----------------------------------------------------
# NN
# -------------------------------------------------------
FORMULA = 'BCE+β*(KLD+(γ-1)TC)+Trim()'
# FORMULA = 'BCE+β*KLD+Trim()'
model: VariationalAutoencoder
base_model: Module
svi: SVI

BATCH_SIZE = 128
Z_DIM = 8
A_DIM = 8
HA_DIMS = tuple(12 for _ in gl.dims.h)  # 8+8(y+z) to 8(a) -> 12ha; 4+8(y+z) to 8(a) -> 10?
EXPLORE = True
PLOT_QUESTIONS = False
TRAIN = True  # True False

PRINT_EVERY_EPOCH = 10
B = 1
Γ = 4
Λ = 10
KLD_TRIM = 0.05
MAX_ABS_Σ = 1  # 0.167, 0.33
# ALLOWED_BASIS_DIMS = (7,)
ALLOWED_BASIS_DIMS = tuple(range(1, Z_DIM + 1))

gl.dims.set_z(Z_DIM)
mmd = MMDLoss(batch_size=BATCH_SIZE)
nll = BernoulliLoss()
loader.upd_nll_state(nll)
trim: Opt[TrimLoss]
trim = TrimLoss()
# trim = None

do3 = Do1(
    id=3, len=1, formula=FORMULA,
    batch_size=BATCH_SIZE, iters_per_epoch=gl.iters_per_epoch(BATCH_SIZE),
    anneal_epochs=80, epochs=160,
    β0=[0.1, B], β=B, γ=Γ, λ0=[1, Λ], λ=Λ,
    max_failure=15, max_abs_Σ=MAX_ABS_Σ,
    allowed_basis_dims=ALLOWED_BASIS_DIMS,
    plot=True,
)
do4 = Do1(
    id=4, len=1, formula=FORMULA,
    batch_size=BATCH_SIZE, iters_per_epoch=gl.iters_per_epoch(BATCH_SIZE),
    epochs=40,
    β=B, γ=Γ, λ=Λ,
    max_failure=15, max_abs_Σ=MAX_ABS_Σ,
    allowed_basis_dims=ALLOWED_BASIS_DIMS,
    plot=True,
)
do0 = Do1(
    id=0, len=2, formula=FORMULA,
    batch_size=BATCH_SIZE, iters_per_epoch=gl.iters_per_epoch(BATCH_SIZE),
    β0=[0.1, B], β=B, γ=Γ, λ=Λ,
    η0=[0, 0], η=[0, 3],
    anneal_epochs=[50, 12], epochs=[100, 25],
    kld_trim=[(None, KLD_TRIM), (KLD_TRIM,)],
    max_failure=15, max_abs_Σ=MAX_ABS_Σ,
    allowed_basis_dims=ALLOWED_BASIS_DIMS,
    plot=True,
)
do2 = Do1(
    id=2, len=1, formula=FORMULA,
    batch_size=BATCH_SIZE, iters_per_epoch=gl.iters_per_epoch(BATCH_SIZE),
    epochs=160,
    η=3, β=B, γ=Γ, λ=Λ,
    kld_trim=(KLD_TRIM,),
    max_failure=15, max_abs_Σ=MAX_ABS_Σ,
    allowed_basis_dims=ALLOWED_BASIS_DIMS,
    plot=True,
)
do1 = Do1(
    id=1, len=1, formula='BCE+β*MMD(dists)+Trim()',
    batch_size=BATCH_SIZE, iters_per_epoch=gl.iters_per_epoch(BATCH_SIZE),
    epochs=80,
    α=30, β=162,
    kld_trim=[(KLD_TRIM, 0.6)],
    max_failure=6, max_abs_Σ=MAX_ABS_Σ,
    allowed_basis_dims=ALLOWED_BASIS_DIMS,
    plot=True,
)
# TO_DO_LIST = (0,)  # 0, 2, 2
TO_DO_LIST = (3, 4, 4, 4, 4)
MODELS_IDXS = (18,)
ID_PREF = ('vae', 'dgm', 'adgm')[2] + ('', 'nf')[0]


class DecoderReLU(Decoder):
    def __init__(self, dims: Tuple[int, Tuple[int, ...], int]):
        super(DecoderReLU, self).__init__(dims=dims, output_activation=None)


"""
base_model = VariationalAutoencoder(
    (gl.dims.x, gl.dims.z, gl.dims.h),
    Encode=EncoderSELU, Decode=DecoderSELU,
    # Decode=DecoderReLU,
)
# noinspection PyTypeChecker
model = base_model
def get_loader(): return loader.get_vae_loader()
"""
class ADGModel(AuxiliaryDeepGenerativeModel):
    def _kld(self, z: Tensor, qz_params: Tuple[Tensor, ...],
             pz_params: Tuple[Tensor, ...]=None, unmod_kld: bool=False) -> Tuple[Tensor, Tensor]:
        if (self.tc_kld is not None) and not unmod_kld:
            return self.tc_kld.__call__(z=z, qz_params=qz_params, pz_params=pz_params)
        return self._kld_normal(z=z, qz_params=qz_params, pz_params=pz_params)


model = ADGModel(
    (gl.dims.x, gl.dims.y, gl.dims.z, A_DIM, gl.dims.h, HA_DIMS),
    Encode=EncoderSELUTrim, Decode=DecoderSELU, Classify=ClassifierSELU,  # No.12 was without ClassifierSELU
)
"""
model = DeepGenerativeModel(
    (gl.dims.x, gl.dims.y, gl.dims.z, gl.dims.h),
    Encode=EncoderSELU, Decode=DecoderSELU,
)
"""
if loader.labelled is not None:
    N_l = len(loader.labelled)
else:
    raise RuntimeError
base_model = SVI(model=model, N_l=N_l, N_u=len(loader.unlabelled), α0=gl.svi_α0,
                 nll=nll, β=B, sampler=ImportanceWeightedSampler(mc=1, iw=1))
def get_loader(): return loader.get_dgm_loader()

# prior_dist = Normal()
# prior_dist.set_prior_params(z_dim=Z_DIM)
# prior_dist.set_inv_pz_flow(NormalizingFlows(dim=Z_DIM, flows=[flows.MAF(Z_DIM) for _ in range(16)]))
# prior_dist.set_inv_pz_flow(NormalizingFlows(dim=Z_DIM, flows=[PlanarNormalizingFlow(Z_DIM) for _ in range(16)]))
# prior_dist.set_inv_pz_flow(BNAFs(dim=Z_DIM, flows_n=16))
# prior_dist.set_inv_pz_flow(NormalizingFlows(dim=Z_DIM))

# model.set_tc_kld(BetaTCKLDLoss(z_dims=(Z_DIM, A_DIM), kl=BetaTC(kld__γmin1_tc=True), mss=gl.tc_kld_mss))
# model.set_qz_x_flow(NormalizingFlows(dim=Z_DIM))
# model.set_qz_x_flow(BNAFs(dim=Z_DIM, flows_n=16))
# model.tc_kld.set_qz_x_flow(BNAFs(dim=Z_DIM, flows_n=16))

tc_kld = model.tc_kld
# noinspection PyTypeChecker
svi = base_model

base_model = base_model.to(device=gl.device, dtype=gl.dtype)
optimizer = tr.optim.Adam(model.parameters(), lr=gl.learning_rate)
loader.regenerate_loaders(batch_size=BATCH_SIZE)

tdata: TestArraysToTensors = TestArraysToTensors(data=gl.data, nll=nll)
models: Dict[str, JATSModelOutput] = dict()


def do_to_do1(spec: Do) -> Do1:  # noinspection PyTypeChecker
    return spec  # type: ignore


class GuideβTC(Guide):
    # noinspection PyUnusedLocal
    def _step(self, spec: Do, item: VAELoaderReturnType, consts: Consts) -> Tuple[Tensor, Dict[str, float]]:
        """
        NELBO = -ELBO = -(E[log p_θ(x|z)] - β*KL[q_ϕ(z|x)||p(z)]) = NLL + β*KLD
        """
        sp, c = do_to_do1(spec), consts
        vae: VariationalAutoencoder = model
        x, x_nll, _, weight_mat_ = item

        x = self.tensor(x)
        weight_mat = self.tensor(weight_mat_) if loader.weight_mat else None
        x_nll = self.tensor(x_nll) if loader.separate_nll else x

        if trim is not None:
            trim.set_consts(strip_scl=c.η)
        if tc_kld is not None:
            tc_kld.set_γ(c.γ)
            tc_kld.set_dataset_size(loader.unlabelled_dataset_size)
        vae.set_save_qz_params(True)

        if sp.id in (0, 2):
            x_rec, kld = vae.__call__(x, x)
            z, (μ, log_σ) = vae.take_away_qz_params()
            nll_ = nll.__call__(x_params=x_rec, target=x_nll, weight=weight_mat)

            nelbo = nll_ + kld * c.β
            if tr.isnan(nelbo).any():
                raise ValueError('NaN spotted in objective.')
            L = nelbo.mean()

            loss = L
            losses = dict(L=L.item())
        elif sp.id == 1:
            if isinstance(sp.mmd_dists, list):
                raise RuntimeError
            mmd.set_z_dists(sp.mmd_dists)
            z, (μ, log_σ) = vae.encoder.__call__(x)
            x_rec = vae.decoder.__call__(z)
            nll_ = nll.__call__(x_params=x_rec, target=x_nll, weight=weight_mat)
            if tr.isnan(nll_).any():
                raise ValueError('NaN spotted in objective.')
            nll_ = nll_.mean()
            mmd_ = mmd.__call__(z)

            loss = nll_ + mmd_ * c.β
            losses = dict(NLL=nll_.item(), MMD=mmd_.item())
        else:
            raise ValueError
        if trim is not None:
            loss += trim.__call__(μ=μ, log_σ=log_σ)
        return loss, dict(loss=loss.item(), **losses)

    def step(self, spec: Do, item: DGMLoaderReturnType, consts: Consts) -> Tuple[Tensor, Dict[str, float]]:
        c = consts
        adgm: AuxiliaryDeepGenerativeModel = model

        (x, x_nll_, _, weight_mat_, y), (u, u_nll_, _, u_weight_mat_) = item

        weight_mat, u_weight_mat = ((self.tensor(weight_mat_), self.tensor(u_weight_mat_))
                                    if loader.weight_mat else (None, None))
        x, u = self.tensor(x), self.tensor(u)
        x_nll, u_nll = (self.tensor(x_nll_), self.tensor(u_nll_)) if loader.separate_nll else (None, None)

        svi.set_consts(β=c.β)
        adgm.set_λ(c.λ)
        if tc_kld is not None:
            tc_kld.set_γ(c.γ)
            tc_kld.set_dataset_size(loader.labelled_dataset_size)
        L, cross_entropy, probs = svi.__call__(x=x, y=y, weight=weight_mat, x_nll=x_nll)
        if tc_kld is not None:
            tc_kld.set_dataset_size(loader.unlabelled_dataset_size)
        U, _, _ = svi.__call__(x=u, weight=u_weight_mat, x_nll=u_nll)

        J_α = L.mean() + cross_entropy.mean() * svi.α + U.mean()
        return J_α, dict(J_α=J_α.item(), Acc=svi.accuracy(probs, y).mean().item())

    def print(self, epoch: int, iteration: int, spec: Do, item: Any, consts: Consts, losses: Dict[str, float]):
        sp = do_to_do1(spec)
        if not isinstance(self.model, (VariationalAutoencoder, SVI)):
            raise NotImplementedError
        c = consts
        # string, dic = '', dict()
        string, dic = log_iter(
            model=self.model, nll=nll, trans=gl.trans, test_data=tdata, trim=trim, loader=loader)

        id_ = self.logger.run_id
        string = (f"ID {id_}, epoch #{epoch}, iter. #{iteration}, L={sp.formula}; "
                  + ', '.join([f"{k}={v:.2f}" for k, v in losses.items()])
                  + f'; α={c.α:.2f}, β={c.β:.2f}, γ={c.γ:.2f}, λ={c.λ:.2f}; '
                  + string)
        dic = dict(id=id_, epoch=epoch, i=iteration, formula=sp.formula, losses=losses, α=c.α, β=c.β, γ=c.γ, λ=c.λ,
                   basis=trim.basis if (trim is not None) else None, **dic)
        self.logger.print(string)
        self.logger.print_i(string)
        self.log_stats.append(dic)

    def load_model(self, spec: Do):
        sp = do_to_do1(spec)
        if isinstance(sp.max_abs_Σ, list) or isinstance(sp.allowed_basis_dims, list) or isinstance(sp.kld_trim, list):
            raise RuntimeError
        self.set_output_checker(OutputChecker1(
            x=gl.data.input, weights=gl.data.weight_vec,
            max_abs_Σ=sp.max_abs_Σ,
            allowed_basis_dims=sp.allowed_basis_dims,
            line_printers=(print_, print_i),
            dtype=gl.dtype, kld_trim=sp.kld_trim
        ))
        if trim is not None:
            basis, _ = get_basis_kld_vec(self.log_stats, sp.kld_trim[0])
            trim.set_basis(basis)
        super(GuideβTC, self).load_model(sp)

    def save_model_output(self, spec: Do, epoch: int, run_id: str) -> None:
        sp = do_to_do1(spec)
        if isinstance(sp.kld_trim, list):
            raise RuntimeError
        profiles = model_output(x=self.tensor(gl.data.input), model=model,
                                rec_loss=nll, dat=gl.data, trans=gl.trans, log_stats=self.log_stats,
                                kld_drop=sp.kld_trim[-1] if (trim is not None) else None)
        profiles.set_z_norm_refs(gl.get_refs)

        if EXPLORE:
            if isinstance(sp.mmd_dists, list):
                raise RuntimeError
            explore_jats(profiles=profiles, dat=gl.data, dims=gl.dims, trans=gl.trans, types=gl.db.types_tal,
                         types_sex=gl.db.types_tal_sex, types_self=gl.db.types_self, logger=gl.logger,
                         fa=fa, pca=pca, mmd_dists=sp.mmd_dists)
        if sp.plot:
            x_batch, z_batch = get__x__z__plot_batch(model=model, data_loader=gl.data_loader)
            plot_jats(
                x_batch=x_batch, z_batch=z_batch, profiles=profiles, pca=pca, fa=fa,
                weight=gl.data.weight_vec, prefix_path_db_nn=gl.logger.checkpoint(epoch),
                prefix_path_db=gl.logger.prefix_db(), types_tal=gl.db.types_tal, types_self=gl.db.types_self,
                plot_questions=PLOT_QUESTIONS)

        models[run_id] = profiles


guide = GuideβTC(
    model=base_model, optimizer=optimizer, get_loader=get_loader,
    id_pref=ID_PREF, models_idxs=MODELS_IDXS, successful_models_n=1, to_do=TO_DO_LIST,
    do_specs=(do0, do1, do2, do3, do4),
    logger=gl.logger, dtype=gl.dtype, device=gl.device, glb_spec=gl.spec,
    train=TRAIN, print_every_epoch=PRINT_EVERY_EPOCH)

def float_(x: Array) -> Tensor: return guide.tensor(x)
def int_(x: Array) -> Tensor: return guide.tensor(x, dtype=tr.long)
tdata.set_test_tensors(float_, int_)


# %% ----------------------------------------------------
guide.run_guide()
print(guide.successful_models)


# %% ----------------------------------------------------
# noinspection PyTypeChecker
check_cov(models=models, weight_vec=gl.data.weight_vec, path_prefix=gl.logger.prefix_db(),
          allowed_basis_dims=ALLOWED_BASIS_DIMS)
