from typing import Tuple, Dict, Any
from dataclasses import asdict
from os import path as p
# noinspection PyPep8Naming
from numpy import ndarray as Array
import torch as tr
from vae.data import Dim, TrainLoader
from vae.display import Log
from vae.linear_component_analyzer import get_pca, get_fa, LinearAnalyzer
from jats_vae.utils import print_err_tuple
from jats_display import get_x_plot_batch, plot_jats
from socionics_db import DBs, load, DBSpec, Transforms, DB, Data
from socionics_db.generate_control_groups import SAMPLES_PER_CLASS_32


class Global:
    data_loader: TrainLoader
    dims: Dim
    data: Data
    db: DB
    spec: Dict[str, Any]
    labels: str
    check_pca: Tuple[int, ...]
    db_spec: DBSpec
    logger: Log
    device: tr.device = tr.device('cuda') if tr.cuda.is_available() else tr.device('cpu')
    learning_rate: float = 1e-4  # 1e-4
    cls_learning_rate: float = 1e-4  # 1e-4
    discr_learning_rate: float = 1e-4  # 1e-4
    discr_betas: Tuple[float, float] = (0.5, 0.9)
    dir_: str = p.expanduser(p.join('~', 'jats_vae'))
    dtype: tr.dtype = tr.float
    trans: Transforms = Transforms()
    use_weight: bool = True  # True False
    use_weight_mat: bool = False
    svi_α0: float = 0.1  # 0.1
    tc_kld_mss: bool = True
    samples_per_class_32 = max(SAMPLES_PER_CLASS_32, 512)

    def __init__(self, labels: str='type', h_dims: Tuple[int, ...]=(), check_pca: Tuple[int, ...]=(9,),
                 db_name: str=('solti', 'bolti')[0]):
        self.labels = labels
        self.check_pca = check_pca
        self.db_spec = DBs[db_name]
        self.logger = Log(temp_dir=self.dir_, run_id='', db_name=self.db_spec.name)

        self.data_loader, self.dims, self.data, self.db = load(
            trans=self.trans, db_spec=self.db_spec, dtype=self.dtype, use_weight=self.use_weight,
            use_weight_mat=self.use_weight_mat, labels=self.labels, samples_per_class_32=self.samples_per_class_32)

        if h_dims:
            self.dims.set_h(*h_dims)

        lbl_size = f'Labelled dataset size: {self.data_loader.labelled_dataset_size}'
        self.logger.print_i(lbl_size)
        self.logger.print(lbl_size)
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
            'samples_per_class_32': self.samples_per_class_32,
        }

    def iters_per_epoch(self, batch_size: int, fvae: bool=False) -> int:
        if self.db_spec.name == 'solti':
            if fvae:
                return {128: 33}[batch_size]
            return {128: 58}[batch_size]  # was {64: 135, 128: 68, 256: 34}
        if self.db_spec.name == 'bolti':
            if fvae:
                raise NotImplementedError
            return {64: 32, 128: 16}[batch_size]
        raise NotImplementedError

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

        if plot_pca:
            plot_jats(
                x_batch=get_x_plot_batch(self.data_loader), pca=pca, fa=fa,
                weight=self.data.weight_vec, prefix_path_db_nn=self.logger.prefix_nn(),
                prefix_path_db=self.logger.prefix_db(),
                types_tal=self.db.types_tal, types_self=self.db.types_self,
                plot_pca=True
            )
        return pca, fa
