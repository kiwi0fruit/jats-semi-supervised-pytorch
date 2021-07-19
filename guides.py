# %%
from typing import Tuple, Dict, Optional as Opt, Union, Any, Callable
from abc import abstractmethod
from dataclasses import asdict
import torch as tr
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
import numpy as np
from kiwi_bugfix_typechecker import test_assert
from beta_tcvae_typed import BetaTCKLDLoss, Normal
from vae import Trimmer, BaseWeightedLoss
from vae.linear_component_analyzer import LinearAnalyzer
from vae.loop_tools import Do, Guide, Consts, OutputChecker
from vae.data import LoaderRetType, LblLoaderRetType, DblLblLoaderRetType, TrainLoader, Dim
from vae.utils import ndarr
from vae.tools import get_z_ymax
from semi_supervised_typed import (VariationalAutoencoder, SVI, AuxiliaryDeepGenerativeModel, VAEPassthroughClassify,
                                   Passer)
from factor_vae import Discriminator, FactorVAEContainer
from jats_vae.utils import TestArraysToTensors
from jats_vae.data import model_output
from jats_vae.semi_supervised import (VAEPassthroughClassifyTwin, ClassifierJATSAxesAlignENTRr,
                                      VAEPassthroughClassifyTwinMMD, VAEPassthroughClassifyMMD,
                                      ClassifierPassthrJATS24KhCFBase, DecoderPassthrTwinSexKhAx,
                                      ClassifierJATSAxesAlignProjectionNTRrEAd, SubDecoderPassthrSocAxTo12KhAx,
                                      EncoderTwinSplit1)
from jats_display import get__x__z__plot_batch, plot_jats, explore_jats, log_iter, Batch
from socionics_db import JATSModelOutput, DB, Transforms, Data


δ = 1e-8
test_assert()
DEBUG = False


class OutputCheckerDimensionWiseKLD(OutputChecker):  # TODO
    max_kl_thr: float = 4
    min_kl_thr: float = 0.09
    min_kl_thr_subset: float = 0.42
    subset: Tuple[int, ...] = ()
    pre_last_kl: float = 0
    allowed_min_kl_failures: int = 0
    allowed_min_kl_failures_subset: int = 0

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def check(self, guide: Any) -> bool:  # pylint: disable=unused-argument,no-self-use
        assert isinstance(guide, Universal)
        self_ = guide

        model = self_.model
        vae: VariationalAutoencoder
        vae2: Opt[VariationalAutoencoder] = None
        if isinstance(model, VAEPassthroughClassifyTwin):
            _interface: VAEPassthroughClassifyTwin = model
            vae = model
            vae2 = _interface.vae_twin
        elif isinstance(model, VariationalAutoencoder):
            vae = model
        elif isinstance(model, SVI):
            svi: SVI = model
            vae = svi.model
        elif isinstance(model, FactorVAEContainer):
            fvaecont: FactorVAEContainer = model
            vae = fvaecont.vae
        else:
            raise NotImplementedError

        x, _, _, _ = self_.test_data.get_test_tensors()
        ret = self.get_check(x, vae)
        if vae2 is not None:
            ret = ret and self.get_check(x, vae2)
        return ret

    def get_check(self, x: Tensor, vae: VariationalAutoencoder) -> bool:
        _, _, params, _ = get_z_ymax(x=x, y=None, model=vae, random=False)
        μ, log_σ = params[0], params[1]

        kld_mat = ndarr(Normal.kl((μ, log_σ)))
        kld_vec = [float(s) for s in np.mean(kld_mat, axis=0)]

        ret = True
        if self.min_kl_thr > δ:
            ret = ret and sum(kl < self.min_kl_thr for kl in kld_vec) <= self.allowed_min_kl_failures
        if self.subset and (self.min_kl_thr_subset > δ):
            kld_vec_subset = [s for i, s in enumerate(kld_vec) if i in self.subset]
            ret = ret and sum(kl < self.min_kl_thr_subset for kl in kld_vec_subset) <= self.allowed_min_kl_failures_subset
        if self.max_kl_thr > δ:
            ret = ret and sum(kl > self.max_kl_thr for kl in kld_vec) == 0
        if (self.pre_last_kl > δ) and len(kld_vec) == 8:
            ret = ret and (kld_vec[-2] > self.pre_last_kl)
        return ret


# %% ----------------------------------------------------
class Universal(Guide):
    """ context kwarg should be provides as context=Universal.ctx(...) """

    loader: TrainLoader
    nll: BaseWeightedLoss
    db: DB
    pca: LinearAnalyzer
    fa: LinearAnalyzer
    data: Data
    trans: Transforms

    test_data: TestArraysToTensors
    trimmer: Trimmer
    pass_sex: Passer
    models: Dict[str, JATSModelOutput]
    explore_model_output: bool = False
    plot_model_output: bool = False

    # noinspection PyMethodOverriding
    @staticmethod
    def ctx(loader: TrainLoader, nll: BaseWeightedLoss, db: DB,  # type: ignore # pylint: disable=arguments-differ
            pca: LinearAnalyzer, fa: LinearAnalyzer, data: Data, trans: Transforms, dims: Dim) -> Dict[str, Any]:
        """ ctx method doesn't hold Liskov substitution principle """
        return dict(loader=loader, nll=nll, db=db, pca=pca, fa=fa, data=data, trans=trans, dims=dims)

    def __post_init__(self) -> None:
        super(Universal, self).__post_init__()
        self.models = dict()

        self.loader = self.context['loader']
        assert isinstance(self.loader, TrainLoader)
        self.nll = self.context['nll']
        assert isinstance(self.nll, BaseWeightedLoss)
        self.db = self.context['db']
        assert isinstance(self.db, DB)
        self.pca = self.context['pca']
        assert isinstance(self.pca, LinearAnalyzer)
        self.fa = self.context['fa']
        assert isinstance(self.fa, LinearAnalyzer)
        self.data = self.context['data']
        assert isinstance(self.data, Data)
        self.trans = self.context['trans']
        assert isinstance(self.trans, Transforms)
        self.dims = self.context['dims']
        assert isinstance(self.dims, Dim)

        self.test_data = TestArraysToTensors(data=self.data, nll=self.nll, dtype=self.dtype, device=self.device)
        self.test_data.set_test_tensors()

    @abstractmethod
    def step(self, spec: Do, item: Any, consts: Consts) -> Tuple[Tensor, Dict[str, float], bool]:
        raise NotImplementedError

    def log_iter(self, epoch: int, iteration: int, spec: Do, consts: Consts,
                 losses: Dict[str, float], string: str, dic: Dict[str, Any]):
        sp, id_ = spec, self.logger.run_id
        vars_dict: Dict[str, float] = asdict(consts)
        string = (
            f"ID {id_}, epoch #{epoch}, iter. #{iteration}, L={sp.formula}; "
            + ', '.join([f"{k}={v:.2f}" for k, v in losses.items()]) + '; '
            + ', '.join(f'{letter}={val:.2f}' for letter, val in vars_dict.items()) + '; ' + string
        )
        dic = dict(id=id_, epoch=epoch, i=iteration, formula=sp.formula, losses=losses, **vars_dict, **dic)
        self.logger.print(string)
        self.logger.print_i(string)
        self.log_stats.append(dic)

    def unpack(self, item: Any) -> Tuple[Tensor, Opt[Tensor], Opt[Tensor]]:
        item_: LoaderRetType = item
        x, x_nll_, _, weight_mat_ = item_
        x = self.tensor(x)
        weight_mat = self.tensor(weight_mat_) if self.loader.weight_mat else None
        x_nll = self.tensor(x_nll_) if self.loader.separate_nll else None
        return x, weight_mat, x_nll

    def unpack_lbl(self, item: Any) -> Tuple[
        Tuple[Tensor, Opt[Tensor], Opt[Tensor], Tensor],
        Tuple[Tensor, Opt[Tensor], Opt[Tensor]]
    ]:
        item_: LblLoaderRetType = item
        (x, x_nll_, _, weight_mat_, y), (u, u_nll_, _, u_weight_mat_) = item_

        weight_mat = self.tensor(weight_mat_) if self.loader.weight_mat else None
        u_weight_mat = self.tensor(u_weight_mat_) if self.loader.weight_mat else None
        x, u = self.tensor(x), self.tensor(u)
        x_nll = self.tensor(x_nll_) if self.loader.separate_nll else None
        u_nll = self.tensor(u_nll_) if self.loader.separate_nll else None
        return (x, x_nll, weight_mat, y), (u, u_nll, u_weight_mat)

    def unpack_dbl_lbl(self, item: Any) -> Tuple[
        Tuple[Tensor, Opt[Tensor], Opt[Tensor], Tensor],
        Tuple[Tensor, Opt[Tensor], Opt[Tensor]],
        Tuple[Tensor, Opt[Tensor], Opt[Tensor]]
    ]:
        item_: DblLblLoaderRetType = item
        (x, x_nll_, _, weight_mat_, y), (u1, u1_nll_, _, u1_weight_mat_), (u2, u2_nll_, _, u2_weight_mat_) = item_

        weight_mat = self.tensor(weight_mat_) if self.loader.weight_mat else None
        u1_weight_mat = self.tensor(u1_weight_mat_) if self.loader.weight_mat else None
        u2_weight_mat = self.tensor(u2_weight_mat_) if self.loader.weight_mat else None
        x, u1, u2 = self.tensor(x), self.tensor(u1), self.tensor(u2)
        x_nll = self.tensor(x_nll_) if self.loader.separate_nll else None
        u1_nll = self.tensor(u1_nll_) if self.loader.separate_nll else None
        u2_nll = self.tensor(u2_nll_) if self.loader.separate_nll else None
        return (x, x_nll, weight_mat, y), (u1, u1_nll, u1_weight_mat), (u2, u2_nll, u2_weight_mat)

    def save_model_output(self, spec: Do, epoch: int, run_id: str) -> None:
        vae: VariationalAutoencoder
        interface: VariationalAutoencoder
        if isinstance(self.model, VAEPassthroughClassifyTwin):
            vae, interface = self.model.vae_twin, self.model  # 1 is tc-kld, 2 is class and rec
        elif isinstance(self.model, VariationalAutoencoder):
            vae = interface = self.model
        elif isinstance(self.model, SVI):
            vae = interface = self.model.model
        else:
            raise NotImplementedError
        profiles = model_output(x=self.tensor(self.data.input), model=interface, rec_loss=self.nll,
                                learn_indxs=self.data.learn_indxs, learn_weight_vec=self.data.learn_weight_vec,
                                trans=self.trans)

        self.logger.print('Interface: ', interface.kld.prior_dist)
        self.logger.print('Inner VAE: ', vae.kld.prior_dist)
        self.logger.print_i('Interface: ', interface.kld.prior_dist)
        self.logger.print_i('Inner VAE: ', vae.kld.prior_dist)
        if isinstance(vae.decoder, DecoderPassthrTwinSexKhAx):
            self.logger.print(vae.decoder.subdecoder.dist)
            self.logger.print_i(vae.decoder.subdecoder.dist)

        if self.explore_model_output:
            explore_jats(profiles=profiles, dat=self.data, trans=self.trans, types=self.db.types_tal,
                         types_sex=self.db.types_tal_sex, types_self=self.db.types_self, logger=self.logger,
                         fa=self.fa, pca=self.pca, ids=self.db.df['id'].values)

        if self.plot_model_output:
            x_batch, z_batch = get__x__z__plot_batch(model=vae, data_loader=self.loader)
            assert not isinstance(spec.basis_strip, list)
            plot_jats(
                # lrn_idxs=self.data.learn_indxs, tst_idxs=self.data.test_indxs,
                x_batch=x_batch, z_batch=z_batch, profiles=profiles, pca=self.pca, fa=self.fa,
                weight=self.data.weight_vec, prefix_path_db_nn=self.logger.checkpoint(epoch),
                prefix_path_db=self.logger.prefix_db(), types_tal=self.db.types_tal, types_self=self.db.types_self,
                basis_strip=spec.basis_strip)

        self.models[run_id] = profiles

    def print(self, epoch: int, iteration: int, spec: Do, item: Any, consts: Consts,
              losses: Dict[str, float]):
        sp = spec
        models_: Tuple[Union[VariationalAutoencoder, SVI], ...]
        funcs_: Tuple[Callable[[], None], ...] = (lambda: None,)
        is_fvae = is_dgm = False
        if isinstance(self.model, FactorVAEContainer):
            models_ = (self.model.vae,)
            is_fvae = True
        elif isinstance(self.model, VAEPassthroughClassifyTwin):
            models_ = (self.model.vae_twin, self.model)  # 1st is vae, 2nd is tckld interface
            tckld: Opt[BetaTCKLDLoss] = (
                self.model.vae_twin.kld if isinstance(self.model.vae_twin.kld, BetaTCKLDLoss) else None)
            tckld_interface: Opt[BetaTCKLDLoss] = (
                self.model.kld if isinstance(self.model.kld, BetaTCKLDLoss) else None)
            def set_tckld_γ(tckld_: Opt[BetaTCKLDLoss], γ_: float) -> None:
                return tckld_.set_γ(γ_) if (tckld_ is not None) else None
            assert isinstance(sp.γ, (int, float))
            γ: float = sp.γ
            funcs_ = (lambda: set_tckld_γ(tckld, 1), lambda: set_tckld_γ(tckld_interface, γ))
        elif isinstance(self.model, VariationalAutoencoder):
            models_ = (self.model,)
        elif isinstance(self.model, SVI):
            models_ = (self.model,)
            is_dgm = True
        else:
            raise NotImplementedError
        x: Opt[Tensor]
        y: Opt[Tensor]
        if is_fvae:
            (x, x_nll, w_mat, y), (u, u_nll, w_mat_u), _ = self.unpack_dbl_lbl(item)
        else:
            (x, x_nll, w_mat, y), (u, u_nll, w_mat_u) = self.unpack_lbl(item)
        if not is_dgm:
            x = x_nll = w_mat = y = None

        for (model_, func_) in zip(models_, funcs_):
            func_()
            string, dic = log_iter(
                model=model_, nll=self.nll, trans=self.trans, test_data=self.test_data, loader=self.loader,
                batch=Batch(
                    x=x, x_nll=x_nll, w_mat=w_mat, y=y, u=u, u_nll=u_nll, w_mat_u=w_mat_u
                ))
            self.log_iter(epoch=epoch, iteration=iteration, spec=spec, consts=consts, losses=losses, string=string,
                          dic=dic)

    def load_model(self, spec: Do) -> None:
        sp = spec
        self.set_output_checker(OutputCheckerDimensionWiseKLD())

        # model_enc = model.encoder
        # if isinstance(model_enc, MetaJATSMover):
        #     model_enc.scl = sp.τ

        # if sp.id == 10:
        #     assert vae_cls is not None
        #     vae_cls.classifier = ClassifierCustom((gl.dims.z + 1, gl.dims.hy, 4))
        # if sp.id == 12:
        #     assert vae_cls is not None
        #     # model.set_inv_pz_flow(flows1)
        #     # model.set_qz_x_flow(flows1)
        #     vae_cls.classifier = ClassifierJATSTemperClub((gl.dims.z + 1, gl.dims.hy, gl.dims.y))
        super(Universal, self).load_model(sp)
        # model_: VAEΣ1Σ2Interface = self.model
        # self.optimizer = tr.optim.Adam(model_.σ1σ2vae.classifier.parameters(), lr=gl.learning_rate)

        # if sp.id == 10:
        #     assert vae_cls is not None
        #     # model.set_inv_pz_flow(flows1)
        #     # model.set_qz_x_flow(flows1)
        #     vae_cls.classifier = ClassifierJATSTemperClub((gl.dims.z + 1, gl.dims.hy, gl.dims.y))

        # dic = optimizer_discr.state_dict()
        # dic['lr'] = gl.discr_learning_rate
        # optimizer_discr.load_state_dict(dic)

        assert not isinstance(sp.basis_strip, list)
        for mod in self.trimmer.trimmers:
            mod.set_anti_basis(sp.basis_strip)

        self.loader.regenerate_loaders(batch_size=self.batch_size)

    def set_trimmer_mover_passer(self, model: nn.Module) -> None:
        assert isinstance(model, VariationalAutoencoder)
        self.trimmer = Trimmer(model.encoder)
        self.pass_sex = Passer(model.decoder)

    def modify_model_pre_load(self) -> None:
        self.set_trimmer_mover_passer(self.model)


class GuideFVAE(Universal):
    """ context kwarg should be provides as context=GuideFVAE.ctx(...) """
    discr: Discriminator
    optimizer_discr: Optimizer
    _get_discr: Tuple[Callable[[], Discriminator]]
    _get_optimizer_discr: Tuple[Callable[[Discriminator], Optimizer]]

    hd_dims: Tuple[int, ...] = tuple(1024 for _ in range(5))

    def modify_model_pre_load(self) -> None:
        assert isinstance(self.model, VariationalAutoencoder)
        vae: VariationalAutoencoder = self.model
        assert not isinstance(vae.kld, BetaTCKLDLoss)
        get_discr = self._get_discr[0]
        get_optimizer_discr = self._get_optimizer_discr[0]

        self.discr = get_discr()
        self.optimizer_discr = get_optimizer_discr(self.discr)
        self.model = FactorVAEContainer(vae, self.discr)
        self.set_trimmer_mover_passer(vae)

    # noinspection PyMethodOverriding
    @staticmethod
    def ctx(loader: TrainLoader, nll: BaseWeightedLoss, db: DB,  # type: ignore # pylint: disable=arguments-differ
            pca: LinearAnalyzer, fa: LinearAnalyzer, data: Data, trans: Transforms, dims: Dim,
            get_discr: Callable[[], Discriminator],
            get_optimizer_discr: Callable[[Discriminator], Optimizer]) -> Dict[str, Any]:
        """ ctx method doesn't hold Liskov substitution principle """
        return dict(loader=loader, nll=nll, db=db, pca=pca, fa=fa, data=data, trans=trans, dims=dims,
                    get_discr=(get_discr,), get_optimizer_discr=(get_optimizer_discr,))

    def __post_init__(self) -> None:
        super(GuideFVAE, self).__post_init__()
        self._get_discr = self.context['get_discr']
        assert isinstance(self._get_discr, tuple)
        self._get_optimizer_discr = self.context['get_optimizer_discr']
        assert isinstance(self._get_optimizer_discr, tuple)

    def train_step(self, spec: Do, item: Any, consts: Consts) -> Tuple[Dict[str, float], bool]:
        _, losses_i, skip = self.step(spec=spec, item=item, consts=consts)
        return losses_i, skip

    def step(self, spec: Do, item: Any, consts: Consts) -> Tuple[Tensor, Dict[str, float], bool]:
        c = consts
        # noinspection PyTypeChecker
        fvae_cont: FactorVAEContainer = self.model  # type: ignore
        # noinspection PyTypeChecker
        vae: VAEPassthroughClassify = fvae_cont.vae  # type: ignore
        # x, x_nll, weight_mat = self.unpack_vae(item)
        (xl, _, _, y), (x, x_nll, weight_mat), (x2, _, _) = self.unpack_dbl_lbl(item)

        if self.discr.skip_iter(x, x2):
            return xl, dict(), True

        vae.__call__(xl, y)

        probs, cross_entropy = vae.classify(xl, y)
        cross_entropy = cross_entropy.mean()

        z, qz_params = vae.forward_vae_to_z(x, x)
        x_rec, kld, _ = vae.forward_vae_to_x(z, qz_params)

        trim = self.trimmer.trim()
        neg_log_p_sex = self.pass_sex.neg_log_p_passthr_x

        nll = self.nll.__call__(x_params=x_rec, target=x_nll if (x_nll is not None) else x, weight=weight_mat)
        nelbo = kld * c.β + neg_log_p_sex + nll

        if tr.isnan(nelbo).any():
            raise ValueError('NaN spotted in objective.')
        U = nelbo.mean()

        tc_, d_z = self.discr.__call__(z)
        tc = tc_.mean()

        loss = U + tc * c.γ + cross_entropy * c.η + trim
        losses = dict(U=U.item())

        self.optimizer.zero_grad()
        # noinspection PyArgumentList
        loss.backward(retain_graph=True)  # type: ignore
        self.optimizer.step()

        z2, _ = vae.forward_vae_to_z(x2, x2)
        tc_dicr = self.discr.tc_discr(z2, d_z).mean()

        if spec.id != 6:
            self.optimizer_discr.zero_grad()
            tc_dicr.backward()
            self.optimizer_discr.step()

        return xl, dict(loss=loss.item(), CE=cross_entropy.item(),
                        TC=tc.item(), TC_dicr=tc_dicr.item(),
                        Acc=vae.classifier.accuracy(probs, y).mean().item(),
                        Trim=trim.item() if isinstance(trim, Tensor) else trim,
                        **losses), False


class GuideTwinVAE(Universal):
    """ context kwarg should be provides as context=GuideVAEΣ1Σ2.ctx(...) """
    trimmer_interface: Trimmer
    pass_sex_interface: Passer

    def load_model(self, spec: Do) -> None:
        super(GuideTwinVAE, self).load_model(spec)
        assert isinstance(self.output_checker, OutputCheckerDimensionWiseKLD)
        output_checker = self.output_checker
        output_checker.pre_last_kl = 0.  # 0.46
        output_checker.min_kl_thr = 0.19  # 0.10 if (spec.id == 8) else 0.19  # was 0.15 at best 8 dim TODO
        output_checker.subset = (0, 2, 5, 6)
        output_checker.allowed_min_kl_failures = 1 if (spec.id == 8) else 0

    def modify_model_pre_load(self) -> None:
        assert isinstance(self.model, VAEPassthroughClassifyTwin)
        interface: VAEPassthroughClassifyTwin = self.model
        vae_inner = interface.vae_twin

        self.trimmer_interface = Trimmer(interface.encoder)
        self.pass_sex_interface = Passer(interface.decoder)

        self.trimmer = Trimmer(vae_inner.encoder)
        self.pass_sex = Passer(vae_inner.decoder)

    # noinspection PyUnusedLocal
    def step(self, spec: Do,  # pylint: disable=unused-argument
             item: Any, consts: Consts) -> Tuple[Tensor, Dict[str, float], bool]:
        """
        NELBO = -ELBO = -(E[log p_θ(x|z)] - β*KL[q_ϕ(z|x)||p(z)]) = NLL + β*KLD
        """
        # ---- High TC-KLD σ2-encoder: ----
        # noinspection PyTypeChecker
        interface: VAEPassthroughClassifyTwinMMD = self.model  # type: ignore
        # ---- Reconstruction and classifying σ1-encoder: ----
        # noinspection PyTypeChecker
        vae: VAEPassthroughClassifyTwinMMD = interface.vae_twin  # type: ignore
        interface_tckld: Opt[BetaTCKLDLoss] = interface.kld if isinstance(interface.kld, BetaTCKLDLoss) else None
        inner_tckld: Opt[BetaTCKLDLoss] = vae.kld if isinstance(vae.kld, BetaTCKLDLoss) else None
        # interface_cls: Opt[ClassifierJATSAxesAlignRot] = (
        #     interface.classifier if isinstance(interface.classifier, ClassifierJATSAxesAlignRot) else None)
        inner_cls: Opt[ClassifierJATSAxesAlignENTRr] = (
            vae.classifier if isinstance(vae.classifier, ClassifierJATSAxesAlignENTRr) else None)
        cf_cls: Opt[ClassifierPassthrJATS24KhCFBase] = (
            vae.classifier if isinstance(vae.classifier, ClassifierPassthrJATS24KhCFBase) else None)
        projection_cls: Opt[ClassifierJATSAxesAlignProjectionNTRrEAd] = (
            vae.classifier if isinstance(vae.classifier, ClassifierJATSAxesAlignProjectionNTRrEAd) else None)

        split_enc: Opt[EncoderTwinSplit1] = None
        split_enc_inner: Opt[EncoderTwinSplit1] = None
        if isinstance(interface.encoder, EncoderTwinSplit1):
            assert isinstance(vae.encoder, EncoderTwinSplit1)
            split_enc = interface.encoder
            split_enc_inner = vae.encoder

        subdec_kh: Opt[SubDecoderPassthrSocAxTo12KhAx] = None
        if isinstance(vae.decoder, DecoderPassthrTwinSexKhAx):
            _dec_rn_kh: DecoderPassthrTwinSexKhAx = vae.decoder
            subdec_kh = _dec_rn_kh.subdecoder

        svi = vae.svi

        c = consts
        (x_lbl, x_nll_lbl, weight_mat_lbl, y), (x, x_nll, weight_mat) = self.unpack_lbl(item)

        if x.shape[0] != spec.batch_size:
            return x_lbl, dict(), True

        ρ, is_λ_anneal = int(c.ρ), int(c.λ > δ)
        # is_trimmer = is_λ_anneal | int(ρ == 1) | int(ρ == 0)

        self.trimmer.set_μ_scl(200)  # 200 * is_trimmer
        self.trimmer_interface.set_μ_scl(200)  # 200 * is_trimmer
        interface.α = vae.α = c.μ
        if inner_tckld is not None:
            inner_tckld.set_γ(1)
            inner_tckld.set_λ(1)
            assert inner_tckld.unmod_kld_allowed
            inner_tckld.set_dataset_size(self.loader.unlabelled_dataset_size)
        if interface_tckld is not None:
            interface_tckld.set_γ(c.γ)
            interface_tckld.set_λ(c.β)
            interface_tckld.set_dataset_size(self.loader.unlabelled_dataset_size)
        if inner_cls is not None:
            inner_cls.set_ρ(ρ)
            inner_cls.sigmoid_scl = c.ω
            inner_cls.set_thr(max(ρ, 0))
            # if cf_cls is not None:
            #     cf_cls.mmd_mult = 500 if ρ == 3 else 0
            #     cf_cls.types16_cross_entropy_mult = int(ρ >= 5)
            #     raise RuntimeError('Deprecated')
            inner_cls.types_cross_entropy_mult = int(ρ > 5)  # was: {0}, int(ρ == 4)
            # ρ from (0, 1, 2) is used in ClassifierJATSAxesAlignNTPAdeAdiAd12
        # if subdec_kh is not None:
        #     subdec_kh.mmd_mult = c.τ
        if (split_enc is not None) and (split_enc_inner is not None):
            split_enc.split_stage = split_enc_inner.split_stage = ρ >= 4
        vae.τ = c.τ

        z_lbl, (μ_lbl_ext, log_σ_lbl) = vae.forward_vae_to_z(x_lbl, y)
        vae.forward_vae_to_x(z_lbl, (μ_lbl_ext, log_σ_lbl))
        neg_log_p_sex_lbl = self.pass_sex.neg_log_p_passthr_x

        # x1_rec, kld1, _ = vae.__call__(x, x)
        z1, (μ, log_σ1) = vae.forward_vae_to_z(x, x)  # μ1==μ2==μ
        x1_rec, kld1_mod, _ = vae.forward_vae_to_x(z1, (μ, log_σ1))

        neg_log_p_sex = self.pass_sex.neg_log_p_passthr_x
        trim_u1u2 = self.trimmer.trim()
        nll1 = self.nll.__call__(x_params=x1_rec, target=x_nll if (x_nll is not None) else x, weight=weight_mat)
        nelbo_u1_mod = kld1_mod * c.α + neg_log_p_sex + nll1
        mmd_u1 = vae.mmd

        if tr.isnan(nelbo_u1_mod).any():
            raise ValueError('NaN spotted in objective.')
        U1 = nelbo_u1_mod.mean()

        # x2_rec, kld2_mod, verb_interface = interface.__call__(x, x)
        z2, (μ, log_σ2) = interface.forward_vae_to_z(x, x)  # μ1==μ2==μ
        x2_rec, kld2_mod, verb_interface = interface.forward_vae_to_x(z2, (μ, log_σ2))
        tc = verb_interface.get('tc', 0.)

        neg_log_p_sex = self.pass_sex_interface.neg_log_p_passthr_x
        trim_u1u2 = trim_u1u2 + self.trimmer_interface.trim()
        nll2 = self.nll.__call__(x_params=x2_rec, target=x_nll if (x_nll is not None) else x, weight=weight_mat)
        nelbo_u2_mod = kld2_mod + neg_log_p_sex + nll2

        if tr.isnan(nelbo_u2_mod).any():
            raise ValueError('NaN spotted in objective.')
        U2 = nelbo_u2_mod.mean()

        # --------------------------------
        kld2_split: Union[Tensor, float] = 0.
        tc_split = kld2_split
        if (split_enc is not None) and (split_enc_inner is not None) and (ρ <= 0):
            kld2_split, _, verb_intfc_split = interface.kld.__call__(split_enc.second(z2),
                                                                     (split_enc.second(μ), split_enc.second(log_σ2)))
            if tr.isnan(kld2_split).any():
                raise ValueError('NaN spotted in objective.')
            kld2_split = kld2_split.mean()
            tc_split = verb_intfc_split.get('tc', 0.)

        # --------------------------------
        probs: Union[Tensor, float] = 0.
        J_α = U = L = cross_entropy = cog_funcs_mse_det = cog_funcs_mse_locality = probs

        if c.η < δ:
            pass
        elif (svi is not None) and vae.use_svi:
            svi.set_consts(β=c.β)
            s_lbl, s = x_lbl[:, :1], x[:, :1]
            μ_lbl_cls = tr.cat((s_lbl, μ_lbl_ext), dim=1)
            z_lbl_cls = tr.cat((s_lbl, z_lbl), dim=1)

            L, _cross_entropy_types_det_lbl, probs, _ = svi.__call__(
                x=x_lbl, y=y, weight=weight_mat_lbl, x_nll=x_nll_lbl, x_cls=μ_lbl_cls)
            U, _, _, _ = svi.__call__(
                x=x, weight=weight_mat, x_nll=x_nll, x_cls=tr.cat((s, μ), dim=1))
            L, U = (L + neg_log_p_sex_lbl).mean(), (U + neg_log_p_sex).mean()
            _, _cross_entropy_types_rnd_lbl = svi.model.classify(z_lbl_cls, y)
            cross_entropy = _cross_entropy_types_det_lbl.mean() * c.μ + _cross_entropy_types_rnd_lbl.mean() * (1 - c.μ)
            J_α = L + cross_entropy * svi.α + U
        else:
            if ρ >= -1:  # was: ρ in (3, 4, 5)
                _z1_lbl_ext, _μ_lbl_ext = vae.get_z_μ(x_lbl)
                _z2_lbl_ext, _ = interface.get_z_μ(x_lbl)

                if subdec_kh is not None:
                    s = x[:, :1]
                    μ_ext_ind = tr.cat((s, μ), dim=1)
                    μ_ext_dep = subdec_kh.z_ext_dep(μ_ext_ind)
                    cross_entropy += subdec_kh.regul_loss_reduced(z_ext_ind=μ_ext_ind, z_ext_dep=μ_ext_dep)

                _, _cross_entropy_axes_rnd_lbl1 = vae.classifier.__call__(_z1_lbl_ext, y)
                _, _cross_entropy_axes_rnd_lbl2 = vae.classifier.__call__(_z2_lbl_ext, y)
                probs, _cross_entropy_axes_det_lbl = vae.classifier.__call__(_μ_lbl_ext, y)
                cross_entropy += (
                    _cross_entropy_axes_det_lbl * c.μ + (
                        _cross_entropy_axes_rnd_lbl1 * c.ε + _cross_entropy_axes_rnd_lbl2 * (1 - c.ε)
                    ) * ((1 - c.μ) / 2)  # presumably ... / 2) is a mistake
                ).mean()

                if ρ == -1:  # OFF
                    _, _cross_entropy_types_rnd_lbl2 = interface.classifier.__call__(_z2_lbl_ext, y)
                    probs, _cross_entropy_types_det_lbl = interface.classifier.__call__(_μ_lbl_ext, y)
                    cross_entropy += (
                        _cross_entropy_types_det_lbl * c.μ + _cross_entropy_types_rnd_lbl2 * (1 - c.μ)
                    ).mean()
            else:
                _z2_lbl_ext, _μ_lbl_ext = interface.get_z_μ(x_lbl)
                _, _cross_entropy_axes_rnd_lbl2 = vae.classifier.__call__(_z2_lbl_ext, y)
                _, _cross_entropy_axes_det_lbl = vae.classifier.__call__(_μ_lbl_ext, y)
                cross_entropy = (
                    _cross_entropy_axes_det_lbl * c.μ + _cross_entropy_axes_rnd_lbl2 * (1 - c.μ)
                ).mean()

            J_α = cross_entropy

        if cf_cls is not None:
            _z1_ext, _μ_ext = vae.get_z_μ(x)
            _mse_rnd, _reg_loss_rnd = cf_cls.get__mse__reg_loss(_z1_ext)
            _mse_det, _reg_loss_det = cf_cls.get__mse__reg_loss(_μ_ext)
            _mse_detrnd, _ = cf_cls.get__mse__reg_loss(_z1_ext, _μ_ext)
            cog_funcs_mse_det = _mse_det.mean()
            cog_funcs_mse_locality = _mse_detrnd.mean()
            J_α += (
                    (cog_funcs_mse_det + _reg_loss_det) * c.μ
                    + (_mse_rnd.mean() + _reg_loss_rnd) * (1 - c.μ)
                    + cog_funcs_mse_locality
            )
        elif projection_cls is not None:
            J_α += projection_cls.get_regul_loss_reduced(μ)
            # _reg_loss_det = projection_cls.get_regul_loss_reduced(μ)
            # _reg_loss_rnd = projection_cls.get_regul_loss_reduced(z1)
            # J_α += _reg_loss_det * c.μ + _reg_loss_rnd * (1 - c.μ)

        loss = U1 * c.ε + U2 * (1 - c.ε) + J_α * c.η + mmd_u1 * c.τ + trim_u1u2 + kld2_split

        tc_ = tc.mean().item() if isinstance(tc, Tensor) else tc
        tc_split_ = tc_split.mean().item() if isinstance(tc_split, Tensor) else tc_split
        return loss, dict(
            loss=loss.item(), TC=tc_, TC_max=tc_, TC_splt=tc_split_, CE=get_item(cross_entropy),
            CF_MSE=get_item(cog_funcs_mse_det), CF_MSE_loc=get_item(cog_funcs_mse_locality),
            Acc=interface.classifier.accuracy(probs, y).mean().item() if isinstance(probs, Tensor) else probs,
            U1=U1.item(), U2=U2.item(), MMD=get_item(mmd_u1 * c.τ), L=get_item(L), U=get_item(U), J_α=get_item(J_α),
            Trim=get_item(trim_u1u2),
        ), False

    def do_post_train(self, spec: Do) -> None:
        if self.last_consts.ρ <= 1:
            # noinspection PyTypeChecker
            interface: VAEPassthroughClassifyTwinMMD = self.model  # type: ignore
            # noinspection PyTypeChecker
            vae: VAEPassthroughClassifyTwinMMD = interface.vae_twin  # type: ignore

            if isinstance(interface.encoder, EncoderTwinSplit1):
                infc_encoder: EncoderTwinSplit1 = interface.encoder
                infc_encoder.copy_twins_to_twins2()
                infc_encoder.copy_σ_to_σ2()
                assert isinstance(vae.encoder, EncoderTwinSplit1)
                innr_encoder: EncoderTwinSplit1 = vae.encoder
                innr_encoder.copy_σ_to_σ2()


def get_item(tensor: Union[Tensor, float]) -> float:
    return tensor.item() if isinstance(tensor, Tensor) else tensor


class GuideVAE(Universal):
    """ context kwarg should be provides as context=GuideVAE.ctx(...) """

    def load_model(self, spec: Do) -> None:
        super(GuideVAE, self).load_model(spec)
        assert isinstance(self.output_checker, OutputCheckerDimensionWiseKLD)
        output_checker = self.output_checker
        output_checker.pre_last_kl = 0.
        output_checker.min_kl_thr = 0.2

    def step(self, spec: Do, item: Any, consts: Consts) -> Tuple[Tensor, Dict[str, float], bool]:
        """
        NELBO = -ELBO = -(E[log p_θ(x|z)] - β*KL[q_ϕ(z|x)||p(z)]) = NLL + β*KLD
        """
        c = consts
        # noinspection PyTypeChecker
        vae: VAEPassthroughClassifyMMD = self.model  # type: ignore
        tckld: Opt[BetaTCKLDLoss] = vae.kld if isinstance(vae.kld, BetaTCKLDLoss) else None
        # x, x_nll, weight_mat = self.unpack_vae(item)
        (xl, _, _, y), (x, x_nll, weight_mat) = self.unpack_lbl(item)

        if x.shape[0] != spec.batch_size:
            return xl, dict(), True

        if tckld is not None:
            tckld.set_γ(c.γ)
            tckld.set_λ(1)
            tckld.set_dataset_size(self.loader.labelled_dataset_size)

        vae.__call__(xl, y)

        acc: float
        cross_entropy: Union[float, Tensor]
        if isinstance(vae, VAEPassthroughClassify):
            vae_cls = vae
            probs, cross_entropy = vae_cls.classify(xl, y)
            cross_entropy = cross_entropy.mean()
            acc = vae.classifier.accuracy(probs, y).mean().item()
        else:
            acc = cross_entropy = 0.

        if tckld is not None:
            tckld.set_dataset_size(self.loader.unlabelled_dataset_size)
        vae.τ = c.τ

        x_rec, kld, verb = vae.__call__(x, x)
        # z, qz_params = vae.forward_vae_to_z(x, x)
        # x_rec, kld, _ = vae.forward_vae_to_x(z, qz_params)
        tc = verb.get('tc', 0.)

        neg_log_p_sex = self.pass_sex.neg_log_p_passthr_x
        trim = self.trimmer.trim()
        nll = self.nll.__call__(x_params=x_rec, target=x_nll if (x_nll is not None) else x, weight=weight_mat)
        nelbo = kld * (c.β * c.α) + neg_log_p_sex + nll
        mmd = vae.mmd

        if tr.isnan(nelbo).any():
            raise ValueError('NaN spotted in objective.')
        U = nelbo.mean()

        loss = U + cross_entropy * c.η + mmd * c.τ + trim

        tc_ = tc.mean().item() if isinstance(tc, Tensor) else tc
        return loss, dict(
            loss=loss.item(), TC=tc_, TC_max=tc_, CE=get_item(cross_entropy),  Acc=acc, U=U.item(), MMD=get_item(mmd),
            Trim=get_item(trim),
        ), False


class GuideDGM(Universal):
    """ context kwarg should be provides as context=GuideDGM.ctx(...) """

    def load_model(self, spec: Do) -> None:
        super(GuideDGM, self).load_model(spec)
        assert isinstance(self.output_checker, OutputCheckerDimensionWiseKLD)
        output_checker = self.output_checker
        output_checker.pre_last_kl = 0.
        output_checker.min_kl_thr = 0.

    def set_trimmer_mover_passer_dgm(self, model: SVI) -> None:
        dgm = model.model
        self.trimmer = Trimmer(dgm.encoder, dgm.classifier)
        self.pass_sex = Passer(dgm.decoder)

    def modify_model_pre_load(self) -> None:
        assert isinstance(self.model, SVI)
        self.set_trimmer_mover_passer_dgm(self.model)

    # noinspection PyUnusedLocal
    def step(self, spec: Do, item: Any,  # pylint: disable=unused-argument
             consts: Consts) -> Tuple[Tensor, Dict[str, float], bool]:
        c = consts
        # noinspection PyTypeChecker
        svi: SVI = self.model  # type: ignore
        dgm = svi.model
        tckld: Opt[BetaTCKLDLoss] = dgm.kld if isinstance(dgm.kld, BetaTCKLDLoss) else None
        (x, x_nll, weight_mat, y), (u, u_nll, u_weight_mat) = self.unpack_lbl(item)

        if u.shape[0] != spec.batch_size:
            return x, dict(), True

        svi.set_consts(β=c.β)
        if tckld is not None:
            tckld.set_γ(c.γ)
            tckld.set_λ(c.λ)
            tckld.set_dataset_size(self.loader.labelled_dataset_size)

        L, cross_entropy, probs, _ = svi.__call__(x=x, y=y, weight=weight_mat, x_nll=x_nll)

        trim = self.trimmer.trim()
        if tckld is not None:
            tckld.set_dataset_size(self.loader.unlabelled_dataset_size)

        U, _, _, _ = svi.__call__(x=u, weight=u_weight_mat, x_nll=u_nll)
        L, U = L.mean(), U.mean()
        trim = trim + self.trimmer.trim()
        J_α = L + cross_entropy.mean() * svi.α + U + trim
        return J_α, dict(J_α=J_α.item(), U=U.item(), L=L.item(), Acc=svi.accuracy(probs, y).mean().item(),
                         Trim=trim.item() if isinstance(trim, Tensor) else trim), False


class GuideADGM(GuideDGM):
    """ context kwarg should be provides as context=GuideADGM.ctx(...) """

    def set_trimmer_mover_passer_dgm(self, model: SVI) -> None:
        # noinspection PyTypeChecker
        adgm: AuxiliaryDeepGenerativeModel = model.model  # type: ignore
        self.trimmer = Trimmer(adgm.encoder, adgm.aux_encoder, adgm.aux_decoder)
        self.pass_sex = Passer(adgm.decoder)

    def load_model(self, spec: Do) -> None:
        super(GuideADGM, self).load_model(spec)
        assert isinstance(self.output_checker, OutputCheckerDimensionWiseKLD)
        output_checker = self.output_checker
        output_checker.pre_last_kl = 0.
        output_checker.min_kl_thr = 0.

    # noinspection PyUnusedLocal
    def step(self, spec: Do, item: Any,  # pylint: disable=unused-argument
             consts: Consts) -> Tuple[Tensor, Dict[str, float], bool]:
        c = consts
        # noinspection PyTypeChecker
        svi: SVI = self.model  # type: ignore
        # noinspection PyTypeChecker
        adgm: AuxiliaryDeepGenerativeModel = svi.model  # type: ignore
        tckld: Opt[BetaTCKLDLoss] = adgm.kld if isinstance(adgm.kld, BetaTCKLDLoss) else None
        (x, x_nll, weight_mat, y), (u, u_nll, u_weight_mat) = self.unpack_lbl(item)

        if u.shape[0] != spec.batch_size:
            return x, dict(), True

        svi.set_consts(β=c.β)
        if tckld is not None:
            tckld.set_γ(c.γ)
            tckld.set_λ(c.λ)
            tckld.set_dataset_size(self.loader.labelled_dataset_size)

        L, cross_entropy, probs, _ = svi.__call__(x=x, y=y, weight=weight_mat, x_nll=x_nll)

        trim = self.trimmer.trim()
        if tckld is not None:
            tckld.set_dataset_size(self.loader.unlabelled_dataset_size)

        U, _, _, _ = svi.__call__(x=u, weight=u_weight_mat, x_nll=u_nll)
        L, U = L.mean(), U.mean()
        trim = trim + self.trimmer.trim()
        J_α = L + cross_entropy.mean() * svi.α + U + trim
        return J_α, dict(J_α=J_α.item(), U=U.item(), L=L.item(), Acc=svi.accuracy(probs, y).mean().item(),
                         Trim=trim.item() if isinstance(trim, Tensor) else trim), False
