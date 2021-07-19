from typing import Union, Optional as Opt, List
# import math
# noinspection PyPep8Naming
from numpy import ndarray as Array
# import numpy as np
import torch as tr
from torch import Tensor
from vae.utils import ndarr, get_x_normalized_μ_σ
from vae.losses import BaseWeightedLoss
from vae.tools import get_z_ymax

from kiwi_bugfix_typechecker import test_assert
from semi_supervised_typed import (VariationalAutoencoder, ADGMPassthrough, VAEPassthrough,
                                   AuxiliaryDeepGenerativeModel, DGMPassthrough)
from socionics_db import JATSModelOutput, Transforms
from jats_vae.semi_supervised import (VAEPassthroughClassifyTwin, ClassifierJATSAxesAlignENTRr,
                                      ClassifierPassthrJATS24KhCFBase,
                                      DecoderPassthrTwinSexKhAx
                                      )

test_assert()


def model_output(x: Tensor, model: VariationalAutoencoder, rec_loss: BaseWeightedLoss,
                 learn_indxs: List[int], learn_weight_vec: Opt[Array], trans: Transforms) -> JATSModelOutput:
    model_ = cls_vae = model
    if isinstance(model, VAEPassthroughClassifyTwin):
        _interface: VAEPassthroughClassifyTwin = model
        model_ = _interface.vae_twin
        cls_vae = _interface  # TODO SWITCH CLASSIFIER
    _, _, _, probs = get_z_ymax(x=x, y=None, model=cls_vae, random=False, verbose=True)
    z_, ymax, params, probs2 = get_z_ymax(x=x, y=None, model=model_, random=False)

    μ_, log_σ_ = params[:2]
    μ, σ = ndarr(μ_), ndarr(log_σ_.exp())
    zμ_ = fzμ_ = model_.kld.flow_qz_x(μ_)
    zμ, z = ndarr(zμ_), ndarr(z_)
    fzμ = zμ

    if isinstance(model_, (VAEPassthrough, ADGMPassthrough, DGMPassthrough)):
        passthr: Union[VAEPassthrough, ADGMPassthrough, DGMPassthrough] = model_
        passthr.decoder.set_passthr_x(x)
    x_preparams = model_.sample(z=zμ_, y=ymax, use_pz_flow=False)

    x_rec_cont_, x_rec_sample_ = rec_loss.x_recon(x_preparams)
    x_rec_cont, x_rec_sample = ndarr(x_rec_cont_), ndarr(x_rec_sample_)
    x_rec_disc = trans.round_x_rec_arr(x_rec_cont)

    _, μz, σz = get_x_normalized_μ_σ(z[learn_indxs], learn_weight_vec)

    probs32: Opt[Tensor] = None
    zμ_rec: Opt[Array] = None
    a = b = c = zμ_new = zμ_rec
    log = ''

    if isinstance(model_, AuxiliaryDeepGenerativeModel):
        _, _, a_μ_, _ = params
        a = ndarr(a_μ_)

    elif isinstance(model, VAEPassthroughClassifyTwin):
        interface: VAEPassthroughClassifyTwin = model
        z2, _, _, _ = get_z_ymax(x=x, y=None, model=interface)
        a = ndarr(z2)

        assert isinstance(model_, VAEPassthroughClassifyTwin)
        inner_vae: VAEPassthroughClassifyTwin = model_
        inner_cls = inner_vae.classifier
        inner_decoder = inner_vae.decoder

        pz_inv_flow_interface = interface.kld.pz_inv_flow
        pz_inv_flow_inner = model_.vae_twin.kld.pz_inv_flow

        if pz_inv_flow_interface is not None:
            _fzμ, _ = pz_inv_flow_interface.__call__(zμ_)
            _fz, _ = pz_inv_flow_interface.__call__(z_)
            fzμ = ndarr(_fzμ)
            b = ndarr(_fz)
        elif pz_inv_flow_inner is not None:
            _fzμ, _ = pz_inv_flow_inner.__call__(zμ_)
            _fz, _ = pz_inv_flow_inner.__call__(z_)
            fzμ = ndarr(_fzμ)
            b = ndarr(_fz)

        if (interface.svi is not None) and interface.use_svi:
            _, (_cls_μ, _) = interface.svi.model.encode(x, ymax)
            c = ndarr(_cls_μ)

        elif model_.kld.qz_x_flow is not None:
            c = μ

        elif isinstance(inner_decoder, DecoderPassthrTwinSexKhAx):
            _dec: DecoderPassthrTwinSexKhAx = inner_decoder
            _s = x[:, :_dec.passthr_dim]
            _zμ_ext = tr.cat([_s, zμ_], dim=1)
            zμ_new = c = ndarr(_dec.subdecoder.z_for_plot(_zμ_ext))
            log += _dec.subdecoder.repr_learnables()

        elif isinstance(inner_cls, ClassifierPassthrJATS24KhCFBase):
            _cls: ClassifierPassthrJATS24KhCFBase = inner_cls
            _s = x[:, :_cls.passthr_dim]
            _zμ_ext = tr.cat([_s, zμ_], dim=1)
            _zμ_new, zμ_new_ext = _cls.get__z_new__z_new_ext(_zμ_ext)
            _zμ_rec = _cls.decoder.__call__(zμ_new_ext)
            zμ_new = c = ndarr(_zμ_new)
            zμ_rec = ndarr(_zμ_rec)
            log += _cls.repr_learnables()
            _probs32 = _cls.probs_verbose(_zμ_ext)
            if _probs32 is not None:
                probs32 = _probs32

        elif isinstance(inner_cls, ClassifierJATSAxesAlignENTRr):
            _cls_base: ClassifierJATSAxesAlignENTRr = inner_cls
            zμ_new = c = ndarr(_cls_base.get_z_new(fzμ_))
            log += _cls_base.repr_learnables()

    elif model_.kld.qz_x_flow is not None:
        a = μ

    y_probs = ndarr(probs) if (probs is not None) else None
    y_probs2 = ndarr(probs2) if (probs2 is not None) else None
    y_probs32 = ndarr(probs32) if (probs32 is not None) else None
    return JATSModelOutput(x=ndarr(x), z=z, zμ=zμ, fzμ=fzμ, μ=μ, σ=σ, x_rec_cont_zμ=x_rec_cont,
                           x_rec_disc_zμ=(x_rec_disc,), x_rec_sample_zμ=x_rec_sample, y_probs=y_probs,
                           y_probs2=y_probs2, y_probs32=y_probs32, μz=μz, σz=σz, a=a, b=b, c=c,
                           zμ_new=zμ_new, zμ_rec=zμ_rec, log=log)
