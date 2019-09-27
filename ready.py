from typing import Tuple, Dict, Any
from dataclasses import asdict
from os import path as p
# noinspection PyPep8Naming
from numpy import ndarray as Array
import torch as tr
from vae.data import Dim, TrainLoader
from vae.display import Log
from vae.linear_component_analyzer import get_pca, get_fa, LinearAnalyzer
from jats_vae.utils import get_refs as _get_refs, print_err_tuple
from jats_display import get_x_plot_batch, plot_jats
from socionics_db import DBs, load, DBSpec, Transforms, DB, Data


class Global:
    data_loader: TrainLoader
    dims: Dim
    data: Data
    db: DB
    spec: Dict[str, Any]
    device: tr.device = tr.device('cuda') if tr.cuda.is_available() else tr.device('cpu')
    learning_rate: float = 0.001
    dir_: str = p.expanduser(p.join('~', 'jats_vae'))
    dtype: tr.dtype = tr.float
    trans: Transforms = Transforms()
    db_spec: DBSpec = DBs['solti']
    check_pca: Tuple[int, ...] = ((18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 9), (9,))[1]
    logger: Log = Log(temp_dir=dir_, run_id='', db_name=db_spec.name)
    use_weight: bool = True  # True False
    use_weight_mat: bool = False
    labels: str = 'type'  # 'type', 'dominant', 'temperament'
    svi_α0: float = 0.1
    tc_kld_mss: bool = True

    def __init__(self):
        self.data_loader, self.dims, self.data, self.db = load(
            trans=self.trans, db_spec=self.db_spec, dtype=self.dtype, use_weight=self.use_weight,
            use_weight_mat=self.use_weight_mat, labels=self.labels)
        if self.labels == 'type':
            self.dims.set_y(16)
        elif self.labels == 'dominant':
            self.dims.set_y(8)
        elif self.labels == 'temperament':
            self.dims.set_y(4)
        else:
            raise ValueError
        self.spec = {
            'trans.α': self.trans.α,
            'trans.round_threshold': self.trans.round_threshold,
            'trans.rep24to15': self.trans.rep24to15,
            'db_spec': self.db_spec.dump(),
            'dims': dict(asdict(self.dims)),
            'use_weight': self.use_weight,
            'use_weight_mat': self.use_weight_mat,
            'labels': self.labels,
            'svi_α0': self.svi_α0,
            'tc_kld_mss': self.tc_kld_mss,
        }

    def get_refs(self, z_all: Array) -> Array:
        return _get_refs(z_all, self.db.types_tal, self.db.types_self)

    @staticmethod
    def iters_per_epoch(batch_size: int) -> int:
        return {128: 51, 256: 26, 512: 13}[batch_size]

    def get_pca_fa(self, plot_pca: bool=True, print_pca: bool=True) -> Tuple[LinearAnalyzer, LinearAnalyzer]:
        self.logger.set_run_id('')

        def _err_printer(x: Array, x_rec: Array, pref: str) -> None:
            def line_printer_(s: str, pr: str=pref) -> None: return self.logger.print(pr + s)
            print_err_tuple(x_rec, self.trans.round_x_rec_arr(x_rec), x=x, data=self.data,
                            line_printer=line_printer_)

        if print_pca:
            pca = get_pca(input_=self.data.input, learn_input=self.data.learn_input,
                          learn_weight_vec=self.data.learn_weight_vec, n_comp_list=self.check_pca,
                          err_printer=_err_printer)
            fa = get_fa(input_=self.data.input, learn_input=self.data.learn_input,
                        learn_weight_vec=self.data.learn_weight_vec, n_comp_list=self.check_pca,
                        err_printer=_err_printer)
        else:
            pca = get_pca(input_=self.data.input, learn_input=self.data.learn_input,
                          learn_weight_vec=self.data.learn_weight_vec, n_comp_list=self.check_pca)
            fa = get_fa(input_=self.data.input, learn_input=self.data.learn_input,
                        learn_weight_vec=self.data.learn_weight_vec, n_comp_list=self.check_pca)
        pca.set_z_norm_refs(self.get_refs)
        fa.set_z_norm_refs(self.get_refs)

        if plot_pca:
            plot_jats(
                x_batch=get_x_plot_batch(self.data_loader), pca=pca, fa=fa,
                weight=self.data.weight_vec, prefix_path_db_nn=self.logger.prefix_nn(),
                prefix_path_db=self.logger.prefix_db(),
                types_tal=self.db.types_tal, types_self=self.db.types_self,
                plot_pca=True
            )
        return pca, fa
