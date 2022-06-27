from typing import Optional as Opt, Tuple
from os import path
from os.path import join
import argparse
# import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch as tr
from torch import nn, Tensor
# from torch.nn import functional as F
# from torch.optim.lr_scheduler import MultiStepLR
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Accuracy, MeanMetric, MaxMetric, SumMetric, CatMetric

from lightningfix import (get_last_checkpoint, GitDirsSHACallback, DeterministicWarmup, tensor_to_dict,
                          LightningModule2)
from mmdloss import MMDNormalLoss
from betatcvae.kld import Distrib, StdNormal, KLDTCLoss
from jatsregularizer import JATSRegularizer
from kldcheckdims import CheckKLDDims
from jats.load import get_target_stratify, get_loader, MAIN_QUEST_N, EXTRA_QUEST
from jats.callbacks import PlotCallback
from jats.utils import probs_temper, probs_quadraclub, expand_quadraclub, expand_temper, expand_temper_to_stat_dyn

# README:
# run ./main.py script and read it's README

DEFAULTNAME = 'train_log'
NUM_WORKERS = 4
BATCH_SIZE = 128
TEST_SIZE = 0.25
RANDOM_STATE = 1
# DIMS:
PSS_D = 1
X_D = 160
X_EXT_D = 4
LAT_D = 8
SUB_D = 12

MAXEPOCHS = 3000  # 350  # 1500  # 3000  # 3001
OFFSET_EP = 0
REAL_MAX_EP = True

if X_EXT_D != len(EXTRA_QUEST) or X_D != MAIN_QUEST_N: raise RuntimeError('Inconsistent constants.')

version: Opt[str]
save: Opt[str]
name, version, save = DEFAULTNAME, None, None
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=str(name))
    parser.add_argument("--ver", default=version)
    parser.add_argument("--save", default=save)
    args = parser.parse_args()
    name, version, save = args.name, args.ver, args.save


parent_dir = path.dirname(path.dirname(path.abspath(__file__)))
preprocess_db = join(parent_dir, 'preprocess_db')
logger = TensorBoardLogger(save_dir=parent_dir, name=name, version=version)
checkpoint_path, epochs, _ = get_last_checkpoint(logger.log_dir)
maxepochs = (MAXEPOCHS + OFFSET_EP + epochs + 1) if not REAL_MAX_EP else MAXEPOCHS + OFFSET_EP
jats_df = pd.read_csv(join(preprocess_db, 'db_final.csv'))


train_set, test_set = train_test_split(
    jats_df.values,
    test_size=TEST_SIZE,
    shuffle=True,
    stratify=get_target_stratify(jats_df),
    random_state=RANDOM_STATE
)
df_train = pd.DataFrame(train_set, columns=jats_df.columns)
df_test = pd.DataFrame(test_set, columns=jats_df.columns)

train_loaders = [get_loader(df_train, 'unlbl', BATCH_SIZE, num_workers=NUM_WORKERS),
                 get_loader(df_train, 'lbl', BATCH_SIZE, num_workers=NUM_WORKERS)]
test_loader = get_loader(df_test, 'both', BATCH_SIZE, num_workers=NUM_WORKERS)


class PlotCallback2(PlotCallback):
    def plot(self, trainer, pl_module):
        if isinstance(pl_module, LightningModule2):
            if pl_module.dummy_validation:
                return
        super(PlotCallback2, self).plot(trainer, pl_module)


plot_callback = PlotCallback2(jats_df, join(preprocess_db, 'ids_interesting.ast'),
                              plot_every_n_epoch=20)
git_dir_sha = GitDirsSHACallback(preprocess_db)


class DoubleVAESampler(nn.Module):
    def __init__(self, dist: Distrib):
        super(DoubleVAESampler, self).__init__()
        self.dist = dist

    def forward(self, x: Tensor):
        mu, log_sigma, log_sigma_beta = tuple(tr.tensor_split(x, 3, dim=1))
        z = self.dist.rsample(mu, log_sigma)
        z_beta = self.dist.rsample(mu, log_sigma_beta)
        return mu, log_sigma, log_sigma_beta, z, z_beta


# noinspection PyAbstractClass
class LightVAE(LightningModule2):
    def __init__(self, learning_rate=1e-4, offset_step: int = 0):
        super().__init__()
        self.learning_rate = learning_rate

        self.encoder = nn.Sequential(nn.Linear(X_D + X_EXT_D, 64), nn.SELU(),
                                     nn.Linear(64, 32), nn.SELU(),
                                     nn.Linear(32, LAT_D * 3),
                                     DoubleVAESampler(StdNormal()))
        self.sub_decoder = nn.Sequential(nn.Linear(LAT_D, 32), nn.SELU(),
                                         nn.Linear(32, 32), nn.SELU(),
                                         nn.Linear(32, SUB_D))
        self.decoder = nn.Sequential(nn.Linear(SUB_D, 32), nn.SELU(),
                                     nn.Linear(32, 64), nn.SELU(),
                                     nn.Linear(64, X_D))
        self.decoder_beta = nn.Sequential(nn.Linear(SUB_D, 32), nn.SELU(),
                                          nn.Linear(32, 64), nn.SELU(),
                                          nn.Linear(64, X_D))

        clsdim = LAT_D * 2 + SUB_D
        self.cls_shifts = nn.Parameter(tr.zeros(clsdim * 2).view(2, clsdim) - 0.01)
        self.cls_logits = nn.Linear(PSS_D + clsdim * 3, 16)
        # nn.Sequential(nn.Linear(, 16), nn.LeakyReLU(), nn.Linear(16, 16), nn.LogSoftmax(dim=1))
        self.cls_linear = nn.Linear(PSS_D + clsdim * 3, PSS_D + clsdim * 3)

        self.jatsregularizer = JATSRegularizer()
        self.nll_logprobs = nn.NLLLoss(reduction='none')
        self.bce_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_probs = nn.MSELoss(reduction='none')
        self.mmd = MMDNormalLoss(BATCH_SIZE)
        self.kld_tc = KLDTCLoss(dataset_size=0, prior_dist=StdNormal(), q_dist=StdNormal())

        # learn and validate metrics:
        self.metr_loss = MeanMetric()
        self.metr_acc_l = Accuracy()
        self.metr_acc_v = SumMetric()
        self.metr_bce_l = MeanMetric()
        self.metr_bce_mu_v = SumMetric()
        self.metr_bce_z_v = SumMetric()
        self.metr_mse_l = MeanMetric()
        self.metr_mse_v = SumMetric()
        self.metr_tc = MeanMetric()
        self.metr_tc_max = MaxMetric()
        self.metr_nelbo = SumMetric()
        self.metr_nelbo_beta = SumMetric()
        self.metr_kld = CatMetric()
        self.metr_kld_beta = CatMetric()

        # 183 steps in 1 epoch with new DB (128 batch)
        # 58 steps in 1 epoch with old DB (128 batch)
        # the first part in 714 ended at 640ep
        # the second part ended at 840ep
        # after old-160-ep ~ 0.77  0.36  0.78  0.56  0.22  0.45  0.58  0.08 / mse 0.153       / bce 91.1
        # after old-640-ep ~ 0.69  0.38  0.74  0.59  0.40  0.55  0.63  0.24 / mse 0.143-0.144 / bce 87.1-87.5
        # after old-840-ep ~ mse 0.142 / bce 86.7
        # TC = 0.48 @ 640ep; 0.33 @ 840ep
        # @640ep NLL_t=88.07 raises max d0.5 then goes down.
        # MSE_test=0.1417 was right before the last rho switch.

        def scl(x): return round(x * 183) + offset_step
        def inv_scl(scl_x): return scl_x // 183
        k25, k50 = scl(25), scl(50)
        k200, k300 = scl(200), scl(300)
        k500, k4000 = scl(500), scl(4000)
        k100, k1000 = scl(100), scl(1000)
        # k3000 = scl(3000)

        self.warmup = DeterministicWarmup(
            1, offset_step,
            dict(alpha=[[0, 0.005], [k50, 0.25], [k100, 0.5], [k4000, 0.5]]),
            dict(beta=[[0, 0.005], [k50, 0.25], [k100, 1], [k4000, 1]]),
            dict(gamma=[[0, 0], [k50, 7], [k4000, 7]]),
            dict(delta=[[0, 0], [k4000, 0]]),  # [k3000, 0], [k3000 + 1, 1], [k4000, 1]
            dict(epsilon=[[0, 0.85], [k50, 0.85], [k100, 0.75], [k200, 0.75], [k300, 0.1], [k4000, 0.1]]),
            dict(eta=[[0, 21], [k50, 7], [k200, 7], [k300, 3], [k4000, 3]]),
            dict(mu=[[0, 0.57], [k200, 0.57], [k300, 0.33], [k4000, 0.33]]),
            dict(rho=[[0, 5], [k25, 5], [k25 + 1, 4], [k200, 4], [k200 + 1, 3], [k500, 3], [k500 + 1, 2], [k4000, 2]]),
            dict(omega=[[0, 2], [k200, 2], [k300, 4], [k4000, 4]]),
        )

        # Untested version:
        # -----------------
        check_kld_dims_delta_min = nn.ModuleList([
            CheckKLDDims(thr=0.001, subset=(0, 1, 4, 5), check_interv=(5 - 1, 1000)),
            CheckKLDDims(thr=0.001, subset=(2, 3), check_interv=(50 - 1, 1000)),
        ])
        check_kld_dims_max = nn.ModuleList([
            CheckKLDDims(True, thr=1.30, subset=(1, 4), check_interv=(5 - 1, 5)),
            CheckKLDDims(True, thr=1.10, subset=(1, 4), check_interv=(10 - 1, 10)),
            CheckKLDDims(True, thr=0.55, subset=(1, 4), check_interv=(25 - 1, 1000)),
        ])
        check_kld_dims_min = nn.ModuleList([
            CheckKLDDims(thr=0.42, subset=(0, 2, 3, 5, 6), check_interv=(80 - 1, inv_scl(k200))),
            CheckKLDDims(thr=0.20, subset=(0, 1, 2, 3, 4, 5, 6), check_interv=(inv_scl(k300) - 1, 1000)),
            CheckKLDDims(thr=0.14, subset=(7,), check_interv=(inv_scl(k300) + 100 - 1, 1000)),
        ])
        self.check_kld_dims = nn.ModuleList([check_kld_dims_delta_min, check_kld_dims_max, check_kld_dims_min])

    @staticmethod
    def get_kld_dims_args(kldvec: Tensor, kldvec_beta: Tensor):
        kld_greater = kldvec_beta.view(1, -1)[:, (5, 6)].view(-1)
        dkld = tr.cat((kld_greater - kldvec_beta[1], kld_greater - kldvec_beta[4], kld_greater - kldvec_beta[7]))
        return (dkld,), (kldvec_beta,), (kldvec, kldvec_beta)

    def load_state_dict(self, state_dict, strict=False):
        return super(LightVAE, self).load_state_dict(state_dict, strict=strict)

    def classify_logprobs(self, pss, z, subdec_z) -> Tuple[Tensor, Tensor]:
        z_z_rot_subdec_z = tr.cat([z, self.jatsregularizer.cat_rot(z), subdec_z], dim=1)
        # expects 1 + (8 + 12) * 3; we have 1 + (8 + 6) * 3
        logits = self.cls_logits(tr.cat([
            pss,
            z_z_rot_subdec_z,
            tr.selu(z_z_rot_subdec_z - self.cls_shifts[0]),
            tr.selu(-z_z_rot_subdec_z - self.cls_shifts[1])
        ], dim=1))[:, :12]
        return tr.log_softmax((logits[:, :8]), dim=1), tr.log_softmax((logits[:, 8:]), dim=1)

    def nll_classify_logprobs(self, pss, z, subdec_z, y) -> Tensor:
        logprobs8, logprobs4 = self.classify_logprobs(pss, z, subdec_z)
        return self.nll_logprobs(logprobs8, probs_quadraclub(y)) + self.nll_logprobs(logprobs4, probs_temper(y))

    def classify_probs(self, pss, z, subdec_z) -> Tensor:
        logprobs8, logprobs4 = self.classify_logprobs(pss, z, subdec_z)
        probs8, probs4 = tr.exp(logprobs8), tr.exp(logprobs4)
        return expand_quadraclub(probs8) * expand_temper(probs4) / expand_temper_to_stat_dyn(probs4)

    def sub_decode(self, *z: Tensor) -> Tensor:
        return self.sub_decoder(z[0])

    def decode_sub_decode(self, *z: Tensor) -> Tensor:
        return self.decoder(self.sub_decoder(z[0]))

    def decode_beta_sub_decode(self, *z: Tensor) -> Tensor:
        return self.decoder_beta(self.sub_decoder(z[0]))

    # def sub_decode(self, *z: Tensor) -> Tensor:
    #     return self.sub_decoder(tr.cat(z, dim=1))
    #
    # def decode_sub_decode(self, *z: Tensor) -> Tensor:
    #     subdec_z = self.sub_decode(*z)
    #     return self.decoder(tr.cat((subdec_z,) + z[1:], dim=1))
    #
    # def decode_beta_sub_decode(self, *z: Tensor) -> Tensor:
    #     subdec_z = self.sub_decode(*z)
    #     return self.decoder_beta(tr.cat((subdec_z,) + z[1:], dim=1))

    def forward(self, x: Tensor, x_ext: Tensor, passth: Tensor):
        """
        :return: (mu, z_beta, subdec_mu, classes_probs_mu)
        """
        mu, _, _, _, z_beta = self.encoder(tr.cat([x, x_ext], dim=1))
        subdec_mu = self.sub_decode(mu, passth)
        classes_probs_mu = self.classify_probs(passth, mu, subdec_mu)
        return mu, z_beta, subdec_mu, classes_probs_mu

    def training_step(self, batch, batch_idx):
        (x1, x1ext, pss1), (x2, x2ext, y2, pss2) = batch

        n1, n2 = x1.shape[0], y2.shape[0]
        # self.tc.dataset_size = self.trainer.train_dataloader.dataset.max_len
        self.kld_tc.dataset_size = 110 * 32

        # Setting warmup coefficients
        # =================
        (alpha, beta, gamma, delta, epsilon,
         eta, mu, rho, omega), warmup_log = self.warmup(self.global_step)
        self.logger.log_metrics(warmup_log, step=self.global_step)

        self.jatsregularizer.set_threshold(round(rho))
        self.jatsregularizer.sigmoid_scl = omega

        # Unlabelled data:
        # =================
        # x.sum(dim=1).mean() == x.mean(dim=0).sum() == x.sum() / n
        mu1, log_sigma, log_sigma_beta, z1, z1_beta = self.encoder(tr.cat([x1, x1ext], dim=1))
        subdec_mu1 = self.sub_decode(mu1, pss1)
        bce = self.bce_logits(self.decode_sub_decode(z1, pss1), x1).sum() / n1
        bce_b = self.bce_logits(self.decode_beta_sub_decode(z1_beta, pss1), x1).sum() / n1

        kld = self.kld_tc.q_dist.kld((mu1, log_sigma)).sum() / n1
        # kld = self.kld_tc.kld(z1, mu1, log_sigma).mean()

        kld_b = self.kld_tc.q_dist.kld((mu1, log_sigma_beta)).sum() / n1
        # kld_b = self.kld_tc.kld(z1_beta, mu1, log_sigma_beta).mean()
        tc_b_unrdcd, _ = self.kld_tc(z1_beta, mu1, log_sigma_beta)
        tc_b = tc_b_unrdcd.mean()

        self.metr_tc(tc_b)
        self.metr_tc_max(tc_b_unrdcd.max() if (n1 == self.mmd.batch_size) else tc_b)

        mmd_mu1 = self.mmd(mu1)
        trim_loss = (tr.relu(mu1 - 2.5) + tr.relu(-mu1 - 2.5)).sum() / n1
        trim_loss_subdec = (tr.relu(subdec_mu1 - 3) +  # #          not neg. thr. at right
                            tr.relu(-subdec_mu1 - 3)).sum() / n1  # not pos. thr. at left

        # Labelled data:
        # =================
        mu2, _, _, z2, z2_beta = self.encoder(tr.cat([x2, x2ext], dim=1))
        subdec_z2 = self.sub_decode(z2, pss2)

        jats_z = self.jatsregularizer(z2, subdec_z2, y2) / n2
        jats_z_b = self.jatsregularizer(z2_beta, self.sub_decode(z2_beta, pss2), y2) / n2
        jats_mu_b = self.jatsregularizer(mu2, self.sub_decode(mu2, pss2), y2) / n2

        if delta < 0.5:
            nll = self.nll_classify_logprobs(pss2, z2.detach(), subdec_z2.detach(), y2).sum() / n2
        else:
            nll = self.nll_classify_logprobs(pss2, z2, subdec_z2, y2).sum() / n2

        # =================
        loss = (
            (bce + kld * alpha) * epsilon +
            (bce_b + kld_b * beta + tc_b * gamma) * (1 - epsilon) +
            trim_loss * 400 +  # was 200 + 200
            trim_loss_subdec * 200 +
            mmd_mu1 * 1000 +
            (jats_mu_b * mu + (jats_z * epsilon + jats_z_b * (1 - epsilon)) * (1 - mu)
             ) * eta +
            nll * 0.01
        )
        # trim_loss was *200 for each of twins. But actually they are the same hence 400
        if tr.isnan(loss).any():
            raise NotImplementedError('NaN spotted in the objective.')

        self.log("train_loss", loss)
        self.metr_loss(loss)
        with tr.no_grad():
            xlogits_mu1 = self.decode_sub_decode(mu1, pss1)
            self.metr_bce_l(self.bce_logits(xlogits_mu1, x1).sum() / n1)
            self.metr_mse_l(self.mse_probs(tr.sigmoid(xlogits_mu1), x1).mean())
            self.metr_acc_l(self.classify_probs(pss2, mu2, self.sub_decode(mu2, pss2)), y2)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x1ext, _y, pss1, w1, _w_lbl, mask = batch
        x2, x2ext, y2, pss2, w2 = x1[mask], x1ext[mask], _y[mask], pss1[mask], _w_lbl[mask]
        # self.tc.dataset_size = self.trainer.val_dataloaders[0].dataset.tensors[0].shape[0]
        n = x1.shape[0]

        # Unlabelled data:
        mu1, log_sigma, log_sigma_beta, z1, z1_beta = self.encoder(tr.cat([x1, x1ext], dim=1))

        bce_z1 = (self.bce_logits(self.decode_sub_decode(z1, pss1), x1).sum(dim=1) * w1).sum()
        bce_z1_b = (self.bce_logits(self.decode_beta_sub_decode(z1_beta, pss1), x1).sum(dim=1) * w1).sum()

        xlogits_mu1 = self.decode_sub_decode(mu1, pss1)
        self.metr_bce_mu_v((self.bce_logits(xlogits_mu1, x1).sum(dim=1) * w1).sum())
        self.metr_mse_v((self.mse_probs(tr.sigmoid(xlogits_mu1), x1).mean(dim=1) * w1).sum())

        kld = (self.kld_tc.q_dist.kld((mu1, log_sigma)) * w1.reshape(n, 1)).sum(dim=0)
        kld_b = (self.kld_tc.q_dist.kld((mu1, log_sigma_beta)) * w1.reshape(n, 1)).sum(dim=0)
        self.metr_bce_z_v(bce_z1)
        self.metr_nelbo(kld.sum() + bce_z1)
        self.metr_nelbo_beta(kld_b.sum() + bce_z1_b)
        self.metr_kld(kld.reshape(1, -1))
        self.metr_kld_beta(kld_b.reshape(1, -1))

        # Labelled data:
        acc_mu = 0.
        if mask.sum() > 0:
            mu2 = self.encoder(tr.cat([x2, x2ext], dim=1))[0]
            probs_mu = self.classify_probs(pss2, mu2, self.sub_decode(mu2, pss2))
            acc_mu = ((y2 == tr.argmax(probs_mu, dim=1)) * w2).sum()
        self.metr_acc_v(acc_mu)

        return {'dummy', 0}

    def training_epoch_end(self, outputs):
        self.set_dummy_validation(outputs)
        self.log("learning_rate", self.learning_rate)

        self.log("loss_train", self.metr_loss)
        self.log("tc_train", self.metr_tc)
        self.log("tc_max_train", self.metr_tc_max)
        self.logg(("bce", "mu_train"), self.metr_bce_l)
        self.logg(("mse", "train"), self.metr_mse_l)
        self.logg(("acc", "train"), self.metr_acc_l)

    def validation_epoch_end(self, outputs):
        self.set_dummy_validation(outputs)

        self.logg(("nelbo", "val"), self.metr_nelbo)
        self.logg(("nelbo", "beta_val"), self.metr_nelbo_beta)
        self.logg(("bce", "mu_val"), self.metr_bce_mu_v)
        self.logg(("bce", "z_val"), self.metr_bce_z_v)
        self.logg(("mse", "val"), self.metr_mse_v)
        self.logg(("acc", "val"), self.metr_acc_v)

        # self.log("cls_shifts", {**tensor_to_dict(self.cls_pos_shifts, 'p'),
        #                         **tensor_to_dict(self.cls_neg_shifts, 'n')})
        klvec = self.logg("kld_val", self.metr_kld, lambda m: tensor_to_dict(m.sum(dim=0))).sum(dim=0)
        klvec_b = self.logg("kld_beta_val", self.metr_kld_beta, lambda m: tensor_to_dict(m.sum(dim=0))).sum(dim=0)
        i = -1.
        for checks, args_ in zip(self.check_kld_dims, self.get_kld_dims_args(klvec, klvec_b)):
            for check in checks:
                i += 1.
                if not check(self.current_epoch, *args_):
                    if not self.dummy_validation:
                        self.log("kld_check_failed", i)
                        self.successful_run = False
                        self.trainer.should_stop = True
                        return

    def configure_optimizers(self):
        return tr.optim.Adam(self.parameters(), lr=self.learning_rate)


autoencoder = LightVAE(offset_step=183 * OFFSET_EP)
trainer_ = pl.Trainer(max_epochs=maxepochs, logger=logger, check_val_every_n_epoch=5 if save is None else 1,
                      callbacks=[plot_callback, git_dir_sha,
                                 ModelCheckpoint(every_n_epochs=10 if save is None else 1, save_top_k=-1)
                                 ])


if __name__ == '__main__':
    trainer_.fit(autoencoder, train_loaders, test_loader, ckpt_path=checkpoint_path if save is None else save)
    if not autoencoder.successful_run:
        raise RuntimeError('Unsuccessful. Skipping this train run.')
    raise RuntimeError('Force skip all train runs.')
