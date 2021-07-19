from typing import Optional as Opt, List, Tuple, Callable, Dict, Union
import os
import math
import numpy as np
# noinspection PyPep8Naming
from numpy import ndarray as Array
import pandas as pd
import torch as tr
from torch import Tensor, Size
from beta_tcvae_typed import PostInit
from semi_supervised_typed import VariationalAutoencoder, DeepGenerativeModel, PassthroughMeta, VAEClassifyMeta
from kiwi_bugfix_typechecker import test_assert
from vae import BaseWeightedLoss
from vae.utils import ndarr, get_x_normalized_μ_σ

test_assert()


class LatentsExplorer(PostInit):
    lbl_only: bool
    show_n_questions: int

    y_lbl: Opt[Array]
    w: Opt[Array]
    w_lbl: Opt[Array]
    idxs_lbl: Opt[List[int]]

    y: Opt[Array]
    y_s: Opt[Array]

    x_norm: Array
    x_rec_norm: Array
    z: Array

    x_s_norm: Array
    z_s: Array

    x_lbl_norm: Opt[Array]
    x_lbl_rec_norm: Opt[Array]

    lang: str
    prefix_dir: str
    questions_rus: Array
    questions_eng: Array

    def __init__(self, vae: VariationalAutoencoder, nll: BaseWeightedLoss, questions: List[Tuple[str, str]],
                 prefix_path_db_nn: str, x: Array, idxs_lbl: List[int]=None,
                 w: Array=None, w_lbl: Array=None, y_lbl: Array=None,
                 n_generated: int=20000, show_n_questions: int=40,
                 passthr_sampler: Callable[[Size], Tensor]=lambda size: tr.randint(0, 2, size),
                 z_transform: Callable[[Array], Array]=lambda z: z,
                 override_w: bool=False,
                 lang: str=('ru', 'en')[0],
                 dtype: tr.dtype=tr.float,
                 device: tr.device=tr.device('cuda') if tr.cuda.is_available() else tr.device('cpu'),
                 ) -> None:
        """
        :param vae:
        :param nll:
        :param questions: 1st str is in Russian, 2nd str is in English
        :param prefix_path_db_nn:
        :param x: float of shape (all_samples_N, x_dim)
        :param idxs_lbl: list of len == labelled_samples_M
        :param w: float of shape (all_samples_N,)
        :param w_lbl: float of shape (labelled_samples_M,)
        :param y_lbl int of shape (labelled_samples_M,); positive for labelled categories, negative otherwise
        :param n_generated:
        :param show_n_questions:
        :param passthr_sampler:
        :param override_w:
        :param lang:
        :param dtype:
        :param device:
        """
        assert lang in ('ru', 'en')
        self.lang = lang
        if idxs_lbl is not None:
            assert idxs_lbl and (y_lbl is not None) and (w_lbl is not None)
            assert (len(idxs_lbl),) == tuple(w_lbl.shape)
        else:
            assert (w_lbl is None) and (y_lbl is None)

        self.questions_rus = np.array([rus for rus, eng in questions], dtype=np.str)
        self.questions_eng = np.array([eng for rus, eng in questions], dtype=np.str)

        s = prefix_path_db_nn
        self.prefix_dir = s[:-3] + {'.pt': ''}.get(s[-3:], s[-3:]) + '_dir'
        if not os.path.isdir(self.prefix_dir):
            os.mkdir(self.prefix_dir)

        assert not isinstance(vae, DeepGenerativeModel)
        self.show_n_questions = show_n_questions

        decoder = vae.decoder
        if isinstance(decoder, PassthroughMeta):
            dec: PassthroughMeta = decoder
            passthr_dim = dec.passthr_dim
        else:
            passthr_dim = 0

        x_ = tr.tensor(x, dtype=dtype, device=device)
        _, qz_params = vae.forward_vae_to_z(x_, x_)
        zμ = qz_params[0]
        x_rec, _, _ = vae.forward_vae_to_x(zμ, qz_params)
        x_rec, _ = nll.x_recon(x_rec)
        zμ = vae.kld.inv_flow_pz(vae.kld.flow_qz_x(zμ))
        if passthr_dim > 0:
            zμ = tr.cat((x_[:, :passthr_dim], zμ), dim=-1)
        x_rec_ = ndarr(x_rec)

        params = vae.kld.prior_dist.get_params(Size((n_generated, zμ.shape[1] - passthr_dim)))
        zμ_s = vae.kld.prior_dist.sample(params)
        if isinstance(decoder, PassthroughMeta):
            dec, s = decoder, passthr_sampler(Size((n_generated, passthr_dim))).to(dtype=dtype, device=device)
            dec.set_passthr_x(s)
            x_s_rec = vae.sample(zμ_s, zμ_s)
            zμ_s = tr.cat((s, zμ_s), dim=-1)
        else:
            x_s_rec = vae.sample(zμ_s, zμ_s)

        self.y = self.y_s = None
        if isinstance(vae, VAEClassifyMeta):
            def get_y(vae_cls: VAEClassifyMeta, zμ_: Tensor) -> Tensor:
                if (vae_cls.svi is not None) and vae_cls.use_svi:
                    probs, _ = vae_cls.svi.model.classify(zμ_)
                else:
                    probs, _ = vae_cls.classifier.__call__(zμ_)
                return probs.max(dim=1)[1]

            self.y = ndarr(get_y(vae, zμ))
            self.y_s = ndarr(get_y(vae, zμ_s))

        self.z, self.z_s = ndarr(zμ), ndarr(zμ_s)

        self.y_lbl, self.w, self.w_lbl, self.idxs_lbl = y_lbl, w, w_lbl, idxs_lbl
        if override_w:
            self.override_w()

        self.x_norm, _, _ = get_x_normalized_μ_σ(x=x, weights=self.w)
        self.x_rec_norm, _, _ = get_x_normalized_μ_σ(x=x_rec_, weights=self.w)
        self.x_s_norm, _, _ = get_x_normalized_μ_σ(x=ndarr(x_s_rec))

        if self.idxs_lbl is not None:
            assert self.idxs_lbl
            self.x_lbl_norm, _, _ = get_x_normalized_μ_σ(x=x[self.idxs_lbl], weights=self.w_lbl)
            self.x_lbl_rec_norm, _, _ = get_x_normalized_μ_σ(x=x_rec_[self.idxs_lbl], weights=self.w_lbl)
        else:
            self.x_lbl_norm = self.x_lbl_rec_norm = None

        self.z = z_transform(self.z)
        self.z_s = z_transform(self.z_s)

        self.__final_init__()

    def override_w(self) -> None:  # pylint: disable=no-self-use
        print('override_w is not implemented')

    def transform_spec(self, z_spec: Union[Dict[int, Tuple[float, float]], Tuple[Tuple[float, float], ...]]
                       ) -> Tuple[Tuple[float, float], ...]:
        """
        ``float: -math.inf`` and ``float: math.inf`` are OK.
        """
        z_dim = self.z.shape[1]
        if isinstance(z_spec, dict):
            spec: Dict[int, Tuple[float, float]] = z_spec
            spec = {i if (i >= 0) else (z_dim + i): v for i, v in spec.items()}
            for i in range(z_dim):
                spec.setdefault(i, (-math.inf, math.inf))
            assert len(spec) == z_dim
            return tuple(spec[i] for i in range(z_dim))
        if isinstance(z_spec, tuple):
            assert len(z_spec) == z_dim
            return z_spec
        raise ValueError

    def select_idxs(self, z_spec: Tuple[Tuple[float, float], ...]=None, y_spec: Tuple[int, ...]=None
                    ) -> Tuple[List[int], List[int], Opt[List[int]]]:
        """
        ``float: -math.inf`` and ``float: math.inf`` are OK.
        """
        assert (z_spec is not None) or (y_spec is not None)
        if y_spec is not None:
            assert y_spec

        def ret(z_: Array, y_: Array=None) -> List[int]:
            mask: Array = z_[:, 0] <= math.inf
            if z_spec is not None:
                for j, (zj_0, zj_1) in enumerate(z_spec):
                    mask = mask & (z_[:, j] >= zj_0) & (z_[:, j] <= zj_1)

            mask_y: Opt[Array] = None
            if y_spec is not None:
                assert y_ is not None
                mask_y = y_ == y_spec[0]
                for i in y_spec[1:]:
                    mask_y = mask_y | (y_ == i)

            return [int(s) for s in np.where(mask & mask_y if (mask_y is not None) else mask)[0]]

        return (
            ret(self.z, self.y),
            ret(self.z_s, self.y_s),
            ret(self.z[self.idxs_lbl], self.y_lbl) if (self.idxs_lbl is not None) else None
        )

    def μx_norm(self, idxs: List[int], idxs_s: List[int], idxs_lbl: Opt[List[int]]=None
                ) -> Tuple[Array, Array, Array, Opt[Array], Opt[Array]]:
        """
        :return: (aver_questions, aver_questions_rec, aver_questions_s, aver_questions_lbl, aver_questions_lbl_rec)
        """
        assert idxs and idxs_s
        if idxs_lbl is not None:
            assert idxs_lbl and (self.x_lbl_norm is not None) and (self.x_lbl_rec_norm is not None)
            w_lbl_subset: Opt[Array] = self.w_lbl[idxs_lbl] if (self.w_lbl is not None) else None
            μx_lbl_norm: Opt[Array] = np.average(self.x_lbl_norm[idxs_lbl], weights=w_lbl_subset, axis=0)
            μx_lbl_rec_norm: Opt[Array] = np.average(self.x_lbl_rec_norm[idxs_lbl], weights=w_lbl_subset, axis=0)
        else:
            μx_lbl_norm = μx_lbl_rec_norm = None

        w_subset: Opt[Array] = self.w[idxs] if (self.w is not None) else None
        return (
            np.average(self.x_norm[idxs], weights=w_subset, axis=0),
            np.average(self.x_rec_norm[idxs], weights=w_subset, axis=0),
            np.average(self.x_s_norm[idxs_s], axis=0),
            μx_lbl_norm,
            μx_lbl_rec_norm,
        )

    def top_questions(self, μx_norm: Array, μx_rec_norm: Array, μx_s_norm: Array,
                      μx_lbl_norm: Opt[Array]=None, μx_lbl_rec_norm: Opt[Array]=None) -> str:
        show_n_questions = self.show_n_questions
        if self.lang == 'ru':
            questions = self.questions_rus
        elif self.lang == 'en':
            questions = self.questions_eng
        else:
            raise RuntimeError

        def get_μx_norm_and_question_columns(μΔx_: Array, cap: str='μΔx') -> str:
            sortd = [int(i) for i in np.argsort(np.abs(μΔx_))]
            idxs_ = list(reversed(sortd[-show_n_questions:]))  # + list(reversed(sortd[:show_n_questions]))
            df = pd.DataFrame({cap: np.round(μΔx_[idxs_], 2), 'question': questions[idxs_]})
            return df.to_csv(index=False)

        csv = get_μx_norm_and_question_columns(μx_norm, 'μx_norm') + '\n\n'
        csv += get_μx_norm_and_question_columns(μx_rec_norm, 'μx_rec_norm') + '\n\n'
        csv += get_μx_norm_and_question_columns(μx_s_norm, 'μx_s_norm') + '\n\n'
        if μx_lbl_norm is not None:
            assert μx_lbl_rec_norm is not None
            csv += get_μx_norm_and_question_columns(μx_lbl_norm, 'μx_lbl_norm') + '\n\n'
            csv += get_μx_norm_and_question_columns(μx_lbl_rec_norm, 'μx_lbl_rec_norm')
        return csv

    @staticmethod
    def _get_lbl_stats(y_lbl: Array, idxs_lbl_subset: List[int],
                       mask_lbl: Array=None, mask_postfix: str= '') -> Tuple[str, Dict[int, int]]:
        y_lbl_all = y_lbl
        if mask_lbl is not None:
            y_lbl_all = y_lbl_all[mask_lbl]

        uni_, counts_ = np.unique(y_lbl_all, return_counts=True)
        dic = {int(i): int(n) for i, n in zip(uni_, counts_)}

        y_lbl_subset = y_lbl[idxs_lbl_subset]
        if mask_lbl is not None:
            y_lbl_subset = y_lbl_subset[mask_lbl[idxs_lbl_subset]]

        uni_, counts_ = np.unique(y_lbl_subset, return_counts=True)
        dic_subset = {int(i): round(100 * int(n) / dic[int(i)]) for i, n in zip(uni_, counts_)}

        cls: List[int] = []
        pts: List[int] = []
        for c, pc in dic_subset.items():
            cls.append(c)
            pts.append(pc)

        classes, percents = np.array(cls), np.array(pts)

        sortd = np.argsort(percents)
        df = pd.DataFrame({f'lbl{mask_postfix}': classes[sortd], f'percent{mask_postfix}': percents[sortd]})
        return df.to_csv(index=False), dic

    def get_lbl_stats(self, idxs_lbl_subset: List[int]) -> str:
        assert self.y_lbl is not None
        csv, _ = self._get_lbl_stats(self.y_lbl, idxs_lbl_subset)
        return csv

    def inspect_z_spec(self, id_: str,
                       z_spec: Union[Dict[int, Tuple[float, float]], Tuple[Tuple[float, float], ...]]=None,
                       y_spec: Tuple[int, ...]=None, print_: bool=True) -> str:
        """
        ``-math.inf`` and ``math.inf`` are OK.

        ``y_spec`` contains classes integer labels as in ``self.y``, ``self.y_nn``, ``self.y_c``.
        """
        assert (z_spec is not None) or (y_spec is not None)
        idxs_subset, idxs_s_subset, idxs_lbl_subset = self.select_idxs(
            self.transform_spec(z_spec) if (z_spec is not None) else None, y_spec)
        csv = f'{z_spec}'
        assert '"' not in list(csv)
        csv = f'"{csv}"' + '\n\n' + self.top_questions(*self.μx_norm(idxs_subset, idxs_s_subset, idxs_lbl_subset))
        if idxs_lbl_subset is not None:
            csv += '\n\n' + self.get_lbl_stats(idxs_lbl_subset)
        csv = csv.replace('\r', '')
        if print_:
            print(csv, file=open(os.path.join(self.prefix_dir, f'{id_}.csv'), 'w', encoding='utf-8'))
        return csv
