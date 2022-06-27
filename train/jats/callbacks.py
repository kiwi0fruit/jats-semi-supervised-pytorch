from typing import Tuple
from os import path
import ast
from pandas import DataFrame
import pytorch_lightning as pl
from .plot import (chunk_forward, plot_z_correl, plot_dist, plot_interesting,
                   get_plot_vae_args_weighted_batch, get_plot_vae_args__y__w)


class PlotCallback(pl.Callback):
    def __init__(self, df: DataFrame, interesting_ids_ast_file_path: str, plot_every_n_epoch: int = 0):
        super(PlotCallback, self).__init__()
        self.plot_vae_args_weighted_batch = get_plot_vae_args_weighted_batch(df)
        self.plot_vae_args__y__w = get_plot_vae_args__y__w(df)
        with open(interesting_ids_ast_file_path, 'r', encoding='utf-8') as f:
            self.interesting_ids: Tuple[int, ...] = tuple(ast.literal_eval(f.read()))
        self.df = df
        self.plot_every_n_epoch = plot_every_n_epoch

    def plot(self, trainer, pl_module):
        if trainer.current_epoch == 0:
            return
        if self.plot_every_n_epoch > 0:
            if (trainer.current_epoch != 10) and (trainer.current_epoch % self.plot_every_n_epoch != 0):
                return
        prefix = path.join(pl_module.logger.log_dir, f'epoch-{pl_module.current_epoch}-')

        plot_vae_args, y, w = self.plot_vae_args__y__w
        mu, _, subdec_mu, probs = chunk_forward(pl_module, *plot_vae_args)
        pl_module.logger.log_metrics(dict(max_mu_correl=plot_z_correl(mu, w, prefix + 'mu-corr')),
                                     step=pl_module.global_step)

        mu_wb, z_beta_wb, subdec_mu_wb, _ = chunk_forward(pl_module, *self.plot_vae_args_weighted_batch)
        plot_dist(mu_wb, z_beta_wb, mu, y, prefix + 'mu-dist', axis_name='μ')

        jats = pl_module.jatsregularizer
        mu_rot = jats.cat_rot_np(mu)
        plot_dist(jats.cat_rot_np(mu_wb), jats.cat_rot_np(z_beta_wb), mu_rot, y,
                  prefix + 'mu-rot-dist', axis_name='μ_{rot}')
        plot_dist(subdec_mu_wb, None, subdec_mu, y, prefix + 'subdec-mu-dist', axis_name='s(μ)')

        plot_interesting(mu, mu_rot, subdec_mu, probs, self.df, prefix, self.interesting_ids)

    def on_train_end(self, trainer, pl_module):
        self.plot(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self.plot(trainer, pl_module)
