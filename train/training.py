from typing import Optional as Opt, Tuple
from os import path
from os.path import join
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import torch as tr
from torch import nn, Tensor
# from torch.nn import functional as F
# from torch.optim.lr_scheduler import MultiStepLR
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
# from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Accuracy, MeanMetric, MaxMetric, SumMetric, CatMetric

from lightningfix import get_last_checkpoint, GitDirsSHACallback, DeterministicWarmup, tensor_to_dict, LightningModule2
from mmdloss import MMDNormalLoss
from betatcvae.kld import Distrib, StdNormal, KLDTCLoss
from jatsregularizertested import JATSRegularizerUntested  # , JATSRegularizerTested
from kldcheckdims import CheckKLDDims
from jats.load import get_target_stratify, get_loader, MAIN_QUEST_N, EXTRA_QUEST
from jats.callbacks import PlotCallback
from jats.utils import probs_temper, probs_quadraclub, expand_quadraclub, expand_temper, expand_temper_to_stat_dyn

# README: run ./main.py script in order to do first part of the training.

# Tensorboard info: Don't start from CWD or parent, use for example ~
# > conda activate nn
# > tensorboard --logdir <abs-path-to-log-dir>

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

# K = 27  # 6; 11; 27
MAXEPOCHS = 100 + 3000  # the first part: 100; + 3000
# the first part: 220 + 20 * K; + 3000
OFFSET_EP = 0  # 0
REAL_MAX_EP = True

if X_EXT_D != len(EXTRA_QUEST) or X_D != MAIN_QUEST_N: raise RuntimeError('Inconsistent constants.')

version: Opt[str]
name, version = DEFAULTNAME, None
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=str(name))
    parser.add_argument("--ver", default=version)
    args = parser.parse_args()
    name, version = args.name, args.ver


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
            if (not pl_module.successful_run) and (not self.verbose):
                return
            if pl_module.dummy_validation:
                return
        super(PlotCallback2, self).plot(trainer, pl_module)


plot_callback = PlotCallback2(jats_df, join(preprocess_db, 'ids_interesting.ast'),
                              plot_every_n_epoch=20, verbose=True)
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
    def __init__(self, learning_rate=1e-3, offset_step: int = 0):  # 1e-4  max_epochs: int,
        super().__init__()
        self.learning_rate = learning_rate
        # self.lr_warmup = False

        # originally all 48 were 32; all 96 were 64
        self.encoder = nn.Sequential(nn.Linear(X_D + X_EXT_D, 96), nn.SELU(),
                                     nn.Linear(96, 48), nn.SELU(),
                                     nn.Linear(48, LAT_D * 3),
                                     DoubleVAESampler(StdNormal()))
        self.sub_decoder = nn.Sequential(nn.Linear(LAT_D, 48), nn.SELU(),
                                         nn.Linear(48, 48), nn.SELU(),
                                         nn.Linear(48, SUB_D))
        self.decoder = nn.Sequential(nn.Linear(SUB_D, 48), nn.SELU(),
                                     nn.Linear(48, 96), nn.SELU(),
                                     nn.Linear(96, X_D))

        self.cls_pos_shifts = nn.Parameter(tr.zeros(LAT_D + SUB_D) - 0.01)
        self.cls_neg_shifts = nn.Parameter(tr.zeros(LAT_D + SUB_D) + 0.01)
        self.selu = nn.SELU()
        self.classifier_logprobs = nn.Sequential(nn.Linear(PSS_D + (LAT_D + SUB_D) * 3, 16), nn.LogSoftmax(dim=1))
        # nn.Sequential(nn.Linear(, 16), nn.LeakyReLU(), nn.Linear(16, 16), nn.LogSoftmax(dim=1))

        self.jatsregularizer = JATSRegularizerUntested()
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

        # k = 57.1875
        # def s(x): return round(x * 57.1875)
        # def a(x): return x + 80  # 80=>25  160=>50
        # x + 96 + 32
        # def b(x): return a(x) - 320  # 640=>200  d64=>d20
        # a(x) - 96 + 32 + 64 * K
        # def c(x): return b(x) + 120  # 760=>237.5  840=>262.5
        # b(x) + 120 * 4 + scl(200) / k
        # def d(x): return c(x)  # 1040=>325  1120=>350
        # c(x) + scl(200) / k
        def scl(x): return round(x * 183)
        k25, k50 = scl(25), scl(50)
        k100, k175 = scl(100), scl(175)
        k200 = scl(200)

        self.warmup = DeterministicWarmup(
            1, offset_step,
            # dict(alpha=[[0, 0.005], [s(a(80)), 0.5], [s(d(1120)), 0.5]]),  # orig
            # dict(alpha=[[0, 0.005], [s(a(80)), 0.25], [s(d(1120)), 0.25]]),
            # dict(beta=[[0, 0.005], [s(a(80)), 1], [s(d(1120)), 1]]),  # orig
            dict(beta=[[0, 0.005], [k50, 0.5], [k200, 0.5]]),
            dict(gamma=[[0, 0], [k50, 7], [k200, 7]]),
            dict(epsilon=[[0, 0.75], [k100, 0.75], [k175, 0.33], [k200, 0.33]]),  # orig
            # dict(epsilon=[[0, 0.75], [s(b(640)), 0.75], [s(c(760)), 0.2], [s(d(1120)), 0.2]]),
            dict(eta=[[0, 21], [k50, 7], [k100, 7], [k175, 3], [k200, 3]]),  # should be @ (1-mu)
            # dict(eta=[[0, 30], [s(a(80)), 10], [s(b(640)), 10], [s(c(760)), 5], [s(d(1120)), 5]]),  # was @ (1-mu)/2
            dict(mu=[[0, 0.57], [k100, 0.57], [scl(175), 0.33], [k200, 0.33]]),  # should be @ (1-mu)
            # dict(mu=[[0, 0.4], [s(b(640)), 0.4], [s(c(760)), 0.2], [s(d(1120)), 0.2]]),  # was @ (1-mu)/2 bug
            # dict(rho=[[0, 3], [s(b(640)), 3], [s(b(641)), 2], [s(d(1040)), 2], [s(d(1041)), 1], [s(d(1120)), 1]]),
            dict(rho=[[0, 4], [k25, 4], [k25 + 1, 3], [k100, 3], [k100 + 1, 2], [k175, 2],
                      [k175 + 1, 1], [k200, 1]]),
            dict(omega=[[0, 2], [k100, 2], [k175, 4], [k200, 4]]),
        )
        # self.max1 = 540

        # untested version:
        # -----------------
        # sh = offset_step
        self.check_kld_dims_max = nn.ModuleList([
            CheckKLDDims(True, thr=1.30, subset=(1,), check_interv=(4, 5)),
            CheckKLDDims(True, thr=0.55, subset=(1,), check_interv=(35, 100)),
            CheckKLDDims(True, thr=0.55, subset=(4,), check_interv=(65, 100)),
        ])
        self.check_kld_dims_min = nn.ModuleList([
            CheckKLDDims(thr=1.20, subset=(5,), check_interv=(4, 5)),
            CheckKLDDims(thr=0.50, subset=(0, 2, 3, 5, 6), check_interv=(65, 100)),
            # CheckKLDDims(thr=0.32, subset=(0, 2, 6), check_interv=(100, 140),
            # CheckKLDDims(thr=0.42, subset=(0, 2, 6), check_interv=(140, max_epochs)),
            # CheckKLDDims(thr=0.42, subset=(3,), check_interv=(140, max_epochs)),
            # CheckKLDDims(thr=0.23, subset=(0, 1, 2, 3, 4, 5, 6), check_interv=(140, max_epochs)),
            # CheckKLDDims(thr=0.20, subset=(0, 1, 2, 3, 4, 5, 6), check_interv=(220 + 20 * K + sh, max_epochs)),
            # CheckKLDDims(thr=0.14, subset=(7,), check_interv=(220 + 20 * K + sh, max_epochs)),  # 220
        ])

        # tested version:
        # ---------------
        # self.check_kld_dims_0 = CheckKLDDims(thr=0.005, subset=(7,), check_interv=(o(a(160)), max_epochs))
        # self.check_kld_dims_0_i = 0

        # self.check_kld_dims_delta_max = nn.ModuleList([
        #     CheckKLDDims(True, thr=0.07, subset=(7,), check_interv=(o(a(100)), max_epochs))
        # ])
        # self.check_kld_dims_max = nn.ModuleList([
        #     CheckKLDDims(True, thr=0.5, subset=(1,), check_interv=(o(a(200)), max_epochs)),
        #     CheckKLDDims(True, thr=0.55, subset=(4,), check_interv=(o(a(200)), max_epochs)),
        # ])
        # self.check_kld_dims_min = nn.ModuleList([
        #     CheckKLDDims(thr=0.32, subset=(0, 2, 5, 6), check_interv=(o(a(200)), o(a(320)))),
        #     CheckKLDDims(thr=0.42, subset=(0, 2, 5, 6), check_interv=(o(a(320)), max_epochs)),
        #     CheckKLDDims(thr=0.42, subset=(3,), check_interv=(o(a(320)), max_epochs)),
        #     CheckKLDDims(thr=0.23, subset=(0, 1, 2, 3, 4, 5, 6), check_interv=(o(a(320)), o(b(640)))),
        #     CheckKLDDims(thr=0.25, subset=(0, 1, 2, 3, 4, 5, 6), check_interv=(o(b(640)), max_epochs)),
        #     CheckKLDDims(thr=0.15, subset=(7,), check_interv=(o(b(640)), max_epochs)),
        # ])

    # def classify_logprobs__(self, pss, z, subdec_z):
    #     z_subdec_z = tr.cat([z, subdec_z], dim=1)
    #     return self.classifier_logprobs(tr.cat([
    #         pss,
    #         z_subdec_z,
    #         self.selu(z_subdec_z - self.cls_pos_shifts),
    #         self.selu(-z_subdec_z - self.cls_neg_shifts)
    #     ], dim=1))
    #
    # def nll_classify_logprobs__(self, pss, z, subdec_z, y) -> Tensor:
    #     return self.nll_logprobs(self.classify_logprobs__(pss, z, subdec_z), y)
    #
    # def classify_probs__(self, pss, z, subdec_z) -> Tensor:
    #     return tr.exp(self.classify_logprobs__(pss, z, subdec_z))

    def classify_logprobs(self, pss, z, subdec_z) -> Tuple[Tensor, Tensor]:
        linear = self.classifier_logprobs[0]
        logsoftmax = self.classifier_logprobs[1]

        z_subdec_z = tr.cat([z, subdec_z], dim=1)
        logits = linear(tr.cat([
            pss,
            z_subdec_z,
            self.selu(z_subdec_z - self.cls_pos_shifts),
            self.selu(-z_subdec_z - self.cls_neg_shifts)
        ], dim=1))[:, :12]
        return logsoftmax(logits[:, :8]), logsoftmax(logits[:, 8:])

    def nll_classify_logprobs(self, pss, z, subdec_z, y) -> Tensor:
        logprobs8, logprobs4 = self.classify_logprobs(pss, z, subdec_z)
        return self.nll_logprobs(logprobs8, probs_quadraclub(y)) + self.nll_logprobs(logprobs4, probs_temper(y))

    def classify_probs(self, pss, z, subdec_z) -> Tensor:
        logprobs8, logprobs4 = self.classify_logprobs(pss, z, subdec_z)
        probs8, probs4 = tr.exp(logprobs8), tr.exp(logprobs4)
        return expand_quadraclub(probs8) * expand_temper(probs4) / expand_temper_to_stat_dyn(probs4)

    def sub_decode(self, *z: Tensor) -> Tensor:
        return self.sub_decoder(z[0])

    def decode__sub_decode(self, *z: Tensor) -> Tensor:
        return self.decoder(self.sub_decoder(z[0]))

    # def sub_decode(self, *z: Tensor) -> Tensor:
    #     return self.sub_decoder(tr.cat(z, dim=1))
    #
    # def decode__sub_decode(self, *z: Tensor) -> Tensor:
    #     subdec_z = self.sub_decode(*z)
    #     return self.decoder(tr.cat((subdec_z,) + z[1:], dim=1))

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
        (beta, gamma, epsilon,  # (alpha,
         eta, mu, rho, omega), warmup_log = self.warmup(self.global_step)
        self.logger.log_metrics(warmup_log, step=self.global_step)

        self.jatsregularizer.set_threshold(round(rho))
        self.jatsregularizer.sigmoid_scl = omega

        # Unlabelled data:
        # =================
        # x.sum(dim=1).mean() == x.mean(dim=0).sum() == x.sum() / n
        mu1, log_sigma, log_sigma_beta, z1, z1_beta = self.encoder(tr.cat([x1, x1ext], dim=1))
        subdec_mu1 = self.sub_decode(mu1, pss1)
        bce = self.bce_logits(self.decode__sub_decode(z1, pss1), x1).sum() / n1
        bce_b = self.bce_logits(self.decode__sub_decode(z1_beta, pss1), x1).sum() / n1

        kld = self.kld_tc.q_dist.kld((mu1, log_sigma)).sum() / n1
        # kld = self.kld_tc.kld(z1, mu1, log_sigma).mean()

        kld_b = self.kld_tc.q_dist.kld((mu1, log_sigma_beta)).sum() / n1
        # kld_b = self.kld_tc.kld(z1_beta, mu1, log_sigma_beta).mean()
        tc_b_unrdcd, _ = self.kld_tc(z1_beta, mu1, log_sigma_beta)
        tc_b = tc_b_unrdcd.mean()

        self.metr_tc(tc_b)
        self.metr_tc_max(tc_b_unrdcd.max() if (n1 == self.mmd.batch_size) else tc_b)

        mmd = self.mmd(mu1)
        trim_loss = (tr.relu(mu1 - 2.5) + tr.relu(-mu1 - 2.5)).sum() / n1
        # trim_loss was *200 for each of twins. But actually they are the same hence 400
        trim_loss_subdec = (tr.relu(subdec_mu1 - 3) +  # #          not neg. thr. at right
                            tr.relu(-subdec_mu1 - 3)).sum() / n1  # not pos. thr. at left

        # Labelled data:
        # =================
        mu2, _, _, z2, z2_beta = self.encoder(tr.cat([x2, x2ext], dim=1))
        subdec_z2 = self.sub_decode(z2, pss2)

        jats_z = self.jatsregularizer(z2, subdec_z2, y2) / n2
        jats_z_b = self.jatsregularizer(z2_beta, self.sub_decode(z2_beta, pss2), y2) / n2
        jats_mu_b = self.jatsregularizer(mu2, self.sub_decode(mu2, pss2), y2) / n2

        nll = self.nll_classify_logprobs(pss2, z2.detach(), subdec_z2.detach(), y2).sum() / n2

        # Only trim_loss coefficients are OK:
        # =================
        loss = (
            (bce + kld * beta) * epsilon +
            (bce_b + kld_b * beta + tc_b * gamma) * (1 - epsilon) +
            trim_loss * 400 +  # was 200 + 200
            trim_loss_subdec * 200 +
            mmd * 1000 +
            (
                jats_mu_b * mu + jats_z * (epsilon*(1 - mu)) + jats_z_b * ((1 - epsilon)*(1 - mu))
            ) * eta +
            nll
        )
        if tr.isnan(loss).any():
            raise NotImplementedError('NaN spotted in the objective.')

        self.log("train_loss", loss)
        self.metr_loss(loss)
        with tr.no_grad():
            xlogits_mu1 = self.decode__sub_decode(mu1, pss1)
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

        bce_z1 = (self.bce_logits(self.decode__sub_decode(z1, pss1), x1).sum(dim=1) * w1).sum()
        bce_z1_b = (self.bce_logits(self.decode__sub_decode(z1_beta, pss1), x1).sum(dim=1) * w1).sum()

        xlogits_mu1 = self.decode__sub_decode(mu1, pss1)
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

        # self.logg("kld_val", self.metr_kld, lambda m: tensor_to_dict(m.sum(dim=0)))
        # self.logg("kld_beta_val", self.metr_kld_beta, lambda m: tensor_to_dict(m.sum(dim=0)))
        klvec = self.logg("kld_val", self.metr_kld, lambda m: tensor_to_dict(m.sum(dim=0))).sum(dim=0)
        klvec_b = self.logg("kld_beta_val", self.metr_kld_beta, lambda m: tensor_to_dict(m.sum(dim=0))).sum(dim=0)

        # j = 0.
        # if not self.check_kld_dims_0(self.current_epoch, klvec, klvec_b):
        #     if not self.dummy_validation:
        #         self.check_kld_dims_0_i += 1
        #         if self.check_kld_dims_0_i >= 8:
        #             self.log("kld_check_failed", j)
        #             self.successful_run = False
        #             self.trainer.should_stop = True
        #             return
        # elif not self.dummy_validation:
        #     self.check_kld_dims_0_i = 0

        # for checks, args_ in zip([self.check_kld_dims_delta_max, self.check_kld_dims_max, self.check_kld_dims_min],
        #                          [(klvec_b - klvec,), (klvec_b,), (klvec, klvec_b)]):
        i = -1.
        for checks, args_ in zip([self.check_kld_dims_max, self.check_kld_dims_min],
                                 [(klvec_b,), (klvec, klvec_b)]):
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
        # if not self.lr_warmup:
        #     return tr.optim.Adam(self.parameters(), lr=self.learning_rate * 0.1)
        # optimizer = tr.optim.Adam(self.parameters(), lr=self.learning_rate)
        # scheduler = MultiStepLR(optimizer, milestones=[self.max1], gamma=0.1)
        # return [optimizer], [dict(
        #     scheduler=scheduler,
        #     interval='epoch',
        #     frequency=1,
        # )]


autoencoder = LightVAE(offset_step=183 * OFFSET_EP)  # max_epochs=maxepochs,
trainer_ = pl.Trainer(max_epochs=maxepochs, logger=logger, check_val_every_n_epoch=5, callbacks=[
    plot_callback, git_dir_sha,
    # ModelCheckpoint(every_n_epochs=10, save_top_k=-1),
])


if __name__ == '__main__':
    trainer_.fit(autoencoder, train_loaders, test_loader, ckpt_path=checkpoint_path)
    if not autoencoder.successful_run or (MAXEPOCHS < 3000):
        raise RuntimeError('Unsuccessful. Skipping this train run.')
