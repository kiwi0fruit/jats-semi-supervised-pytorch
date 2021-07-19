# %%
from typing import Tuple, Dict, Optional as Opt, Any, Iterable
import torch as tr
from torch import Tensor, nn
from torch.nn import Module
from torch.optim.optimizer import Optimizer
import matplotlibhelper as mh  # pylint: disable=wrong-import-order

from kiwi_bugfix_typechecker import test_assert
from vae import BernoulliLoss, Trimmer
from vae.loop_tools import Do, Consts, OutputChecker
from vae.utils import ndarr
from vae.linear_component_analyzer import LinearAnalyzer
from semi_supervised_typed import Passer, Classifier
from ready import Global
from guides import Universal


# %%
δ = 1e-8
mh.ready(font_size=12, ext='svg', hide=True, magic='agg')
test_assert()
Z_DIM = 8
H_DIM = 16
MODELS_IDXS = (3,)

gl = Global(labels='type', h_dims=(128, 128, 128), check_pca=(Z_DIM,))
nll = BernoulliLoss()
gl.data_loader.upd_nll_state(nll)


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
    pca, fa = gl.get_pca_fa(plot_pca=True, print_pca=False)

la = fa

do20 = Do(
    id=20, len=1, formula='PCA_NN_classify',
    batch_size=128, iters_per_epoch=gl.iters_per_epoch(128, False),
    epochs=40
)


class GuidePCAClassify(Universal):
    def step(self, spec: Do, item: Any, consts: Consts) -> Tuple[Tensor, Dict[str, float], bool]:
        # noinspection PyTypeChecker
        classifier: Classifier = self.model  # type: ignore
        (x_lbl, _, _, y), _ = self.unpack_lbl(item)
        # (x_lbl, _, _, y), (x, _, _) = self.unpack_lbl(item)

        z_lbl = tr.cat((
            x_lbl[:, :1],
            self.tensor(la.transform(ndarr(x_lbl)))
        ), dim=1)
        probs, cross_entropy_ = classifier.__call__(z_lbl, y)
        cross_entropy = cross_entropy_.mean()

        # z = tr.cat((
        #     x[:, :1],
        #     self.tensor(pca.transform(ndarr(x)))
        # ), dim=1)
        # _, entropy_ = classifier.__call__(z)
        # entropy = entropy_.mean()

        loss = cross_entropy  # + entropy

        return loss, dict(
            loss=loss.item(), CE=cross_entropy.item(),  # E=entropy.item(),
            Acc=classifier.accuracy(probs, y).mean().item(),
        ), False

    def save_model_output(self, spec: Do, epoch: int, run_id: str) -> None:
        pass

    def set_trimmer_mover_passer(self, model: nn.Module) -> None:
        assert isinstance(model, Classifier)
        self.trimmer = Trimmer(model)
        self.pass_sex = Passer(model)

    def load_model(self, spec: Do) -> None:
        super(GuidePCAClassify, self).load_model(spec)
        self.set_output_checker(OutputChecker())

    def print(self, epoch: int, iteration: int, spec: Do, item: Any, consts: Consts,
              losses: Dict[str, float]):
        # noinspection PyTypeChecker
        classifier: Classifier = self.model  # type: ignore

        x, _, w_vec, _, y, type_ = self.test_data.get_test_labelled_tensors()

        def wμ(x_: Tensor, w_vec_: Opt[Tensor] = None) -> float:
            if w_vec_ is None:
                return x_.mean().item()
            return (x_ * w_vec_[:, 0]).mean().item()

        z = tr.cat((
            x[:, :1],
            self.tensor(la.transform(ndarr(x)))
        ), dim=1)

        probs, _ = classifier.__call__(z)
        acc = classifier.accuracy(probs, y)
        acc_per_type = {f'{s}{j}': classifier.accuracy(probs[maskj & sex], y[maskj & sex]).mean().item()
                        for j, maskj in [(i, type_ == i) for i in range(1, 17)]
                        for s, sex in (('f', x[:, 0] < 0.5), ('m', x[:, 0] > 0.5))}
        acc_per_type['Av'] = sum(acc_per_type.values()) / len(acc_per_type)

        losses = dict(
            minWAcc_t=min(acc_per_type.values()), wAcc_t=wμ(acc, w_vec), Acc_t=wμ(acc),
            **acc_per_type, **losses)

        string = ', '.join(f'{k}={v:.2f}' for k, v in losses.items())
        dic = dict(test_losses=losses)

        self.log_iter(epoch=epoch, iteration=iteration, spec=spec, consts=consts, losses=losses, string=string,
                      dic=dic)


# noinspection PyUnusedLocal
def get_model(spec: Do, module: Opt[Module]) -> Opt[Module]:  # pylint: disable=unused-argument
    if spec.post_load:
        return None
    return Classifier((Z_DIM + 1, (H_DIM,), 16))


# noinspection PyUnusedLocal
def get_loader(spec: Do) -> Iterable: return gl.data_loader.get_lbl_loader()  # pylint: disable=unused-argument


# noinspection PyUnusedLocal
def get_optimizer(spec: Do, model: Module) -> Opt[Optimizer]:
    if spec.post_load:
        return None
    return tr.optim.Adam(model.parameters(),
                         # lr=gl.learning_rate
                         )


guide = GuidePCAClassify(
    get_model=get_model, get_optimizer=get_optimizer, get_loader=get_loader,
    id_pref='pca_class', models_idxs=MODELS_IDXS, successful_models_n=1, to_do=(20, 20, 20, 20),
    do_specs=(do20,),
    logger=gl.logger, dtype=gl.dtype, device=gl.device, glb_spec=gl.spec,
    train=True, print_every_epoch=5,
    context=GuidePCAClassify.ctx(
        loader=gl.data_loader, nll=nll, db=gl.db, pca=pca, fa=fa, data=gl.data, trans=gl.trans, dims=gl.dims))

guide.run_guide()
