from typing import Tuple, Type, Union, List, Optional as Opt, Callable, Iterator
# import math
import torch as tr
from torch import Tensor, nn
# from torch.nn import functional as func
from kiwi_bugfix_typechecker import test_assert
from beta_tcvae_typed import KLDLoss, Distrib, Flows, BetaTCKLDLoss
from semi_supervised_typed.layers.stochastic import XTupleYi
from semi_supervised_typed.layers import GaussianSample, BaseSample
from semi_supervised_typed import (Encoder, Decoder, LadderDecoder, LadderEncoder, Classifier,
                                   EncoderPassthrough, DecoderPassthrough, VAEPassthroughClassify,
                                   VariationalAutoencoder)
from .loss import TrimLoss

test_assert()


class MetaTrimmer:
    trim: Union[Tensor, int]
    trimmer: TrimLoss

    def __init__(self, max_abs_μ: float=2.5, anti_basis: Tuple[int, ...]=(),
                 inv_min_σ: float=0.1**-1, inv_max_σ: float=1,
                 σ_scl: float=0, μ_scl: float=200,
                 # σ_scl: float=200, μ_scl: float=200,
                 basis_scl: float=200,
                 μ_norm_scl: float=0, μ_norm_std: float=0.833) -> None:
        self.trim = 0
        self.trimmer = TrimLoss(σ_scl=σ_scl, inv_min_σ=inv_min_σ, μ_scl=μ_scl, max_abs_μ=max_abs_μ,
                                basis_scl=basis_scl, anti_basis=anti_basis, inv_max_σ=inv_max_σ,
                                μ_norm_scl=μ_norm_scl, μ_norm_std=μ_norm_std)

    def set_trim(self, μ: Tensor, log_σ: Tensor) -> None:
        self.trim = self.trimmer.__call__(μ, log_σ)


class Trimmer:
    modules: List[MetaTrimmer]

    def __init__(self, *modules: nn.Module):
        self.modules = []
        for mod in modules:
            if isinstance(mod, MetaTrimmer):
                self.modules.append(mod)
        if len(self.modules) != len(modules):
            print(f'WARNING! Trimmer: len(self.modules) != len(modules): {len(self.modules)} != {len(modules)}')

    def trim(self) -> Union[Tensor, int]:
        return sum(mod.trim for mod in self.modules)

    @property
    def trimmers(self) -> Iterator[TrimLoss]:
        return (mod.trimmer for mod in self.modules)

    def set_μ_scl(self, μ_scl: float) -> None:
        for trimmer in self.trimmers:
            trimmer.set_μ_scl(μ_scl)


class GaussianSampleTwin(GaussianSample):
    _twin_sampler: Tuple[GaussianSample]

    def __init__(self, in_features: int, out_features: int):
        super(GaussianSampleTwin, self).__init__(in_features=in_features, out_features=out_features)
        self.set_twin_sampler(self)

    def set_twin_sampler(self, sampler: BaseSample):
        assert isinstance(sampler, GaussianSample)
        self._twin_sampler = (sampler,)

    @property
    def μ_twin(self) -> nn.Linear:
        return self._twin_sampler[0].μ

    def μ_log_σ(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        μ = self.μ_twin.__call__(x)
        log_σ = self.log_σ.__call__(x)
        return μ, log_σ


class EncoderTwin(Encoder, MetaTrimmer):
    sample: GaussianSampleTwin
    _twin_encoder: Tuple[Encoder]

    def __init__(self, dims: Tuple[int, Tuple[int, ...], int],
                 sample_layer: Type[GaussianSampleTwin]=GaussianSampleTwin,
                 activation_fn: Callable[[Tensor], Tensor]=tr.selu):
        super(EncoderTwin, self).__init__(dims=dims, sample_layer=sample_layer, activation_fn=activation_fn)

    def __post_init__(self):
        super(EncoderTwin, self).__post_init__()
        MetaTrimmer.__init__(self)
        self.set_twin_encoder(self)

    def set_twin_encoder(self, encoder: Encoder):
        self._twin_encoder = (encoder,)
        self.sample.set_twin_sampler(encoder.sample)

    @property
    def twin_encoder(self) -> Encoder:
        return self._twin_encoder[0]

    @property
    def hidden_twin(self) -> List[nn.Linear]:
        return self._twin_encoder[0].hidden

    def forward_(self, x: Tensor) -> XTupleYi:
        z, (μ, log_σ) = self.subforward(x, self.hidden_twin, self.sample)
        self.set_trim(μ, log_σ)
        return z, (μ, log_σ)


class EncoderTwinSplit0(EncoderTwin):
    split_idx: int = 5
    encoder2: EncoderTwin

    def __init__(self, dims: Tuple[int, Tuple[int, ...], int],
                 sample_layer: Type[GaussianSampleTwin]=GaussianSampleTwin,
                 activation_fn: Callable[[Tensor], Tensor]=tr.selu):
        x_dim, h_dims, z_dim = dims
        with self.disable_final_init():
            super(EncoderTwinSplit0, self).__init__(dims=(x_dim, h_dims, self.split_idx), sample_layer=sample_layer,
                                                    activation_fn=activation_fn)
        self.encoder2 = EncoderTwin(dims=(x_dim, h_dims, z_dim - self.split_idx), sample_layer=sample_layer,
                                    activation_fn=activation_fn)
        self.__final_init__()

    def set_twin_encoder(self, encoder: Encoder):
        super(EncoderTwinSplit0, self).set_twin_encoder(encoder)
        assert isinstance(encoder, EncoderTwinSplit0)
        encoder_: EncoderTwinSplit0 = encoder
        self.encoder2.set_twin_encoder(encoder_.encoder2)

    @staticmethod
    def cat(x, y):
        return tr.cat([x, y], dim=1)

    def forward_(self, x: Tensor) -> XTupleYi:
        z1, (μ1, log_σ1) = super(EncoderTwinSplit0, self).forward_(x)
        z2, (μ2, log_σ2) = self.encoder2.__call__(x)
        self.trim += self.encoder2.trim
        return self.cat(z1, z2), (self.cat(μ1, μ2), self.cat(log_σ1, log_σ2))


class DecoderPassthroughTwin(DecoderPassthrough):
    _twin_decoder: Tuple[Decoder]

    def __init__(self, dims: Tuple[int, Tuple[int, ...], int],
                 passthrough_dim: int=1,
                 activation_fn: Callable[[Tensor], Tensor]=tr.selu,
                 output_activation: Opt[Callable[[Tensor], Tensor]]=None):
        with self.disable_final_init():
            super(DecoderPassthroughTwin, self).__init__(
                dims=dims, passthrough_dim=passthrough_dim, activation_fn=activation_fn,
                output_activation=output_activation)
        self.set_twin_decoder(self)
        self.__final_init__()

    def set_twin_decoder(self, decoder: Decoder):
        self._twin_decoder = (decoder,)

    @property
    def twin_decoder(self) -> Decoder:
        return self._twin_decoder[0]

    def forward_(self, x: Tensor) -> Tensor:
        x = self.extend_x(x)
        h: Tensor
        for layer in self.twin_decoder.hidden:
            h = self.activation_fn.a(layer.__call__(x))
            x = h
        x_preparams = self.twin_decoder.reconstruction.__call__(x)
        return self.output_activation.a(x_preparams) if (self.output_activation is not None) else x_preparams


class TwinKLDMix:
    _twin_kld: Tuple[KLDLoss]
    _closed_form: bool

    def set_closed_form(self):
        raise NotImplementedError

    def set_twin_kld(self, kld: KLDLoss):
        self._twin_kld = (kld,)
        self.set_closed_form()


# noinspection PyAbstractClass
class KLDLossTwin(KLDLoss, TwinKLDMix):
    @property
    def prior_dist(self) -> Distrib:
        return self._twin_kld[0]._prior_dist  # pylint: disable=protected-access

    def __post_init__(self):
        super(KLDLossTwin, self).__post_init__()
        self.set_twin_kld(self)

    def set_prior_dist(self, prior_dist: Distrib):
        raise NotImplementedError

    def set_pz_inv_flow(self, flow: Flows=None):
        raise NotImplementedError

    def set_closed_form(self):
        self._closed_form = False


# noinspection PyAbstractClass
class BetaTCKLDLossTwin(BetaTCKLDLoss, TwinKLDMix):
    @property
    def prior_dist(self) -> Distrib:
        return self._twin_kld[0]._prior_dist  # pylint: disable=protected-access

    def __post_init__(self):
        super(BetaTCKLDLossTwin, self).__post_init__()
        self.set_twin_kld(self)

    def set_prior_dist(self, prior_dist: Distrib):
        raise NotImplementedError

    def set_pz_inv_flow(self, flow: Flows=None):
        raise NotImplementedError

    def set_closed_form(self):
        self._closed_form = False


class VAEPassthroughClassifyTwin(VAEPassthroughClassify):
    _twin_vae: Tuple[VariationalAutoencoder]
    _twin_vae_registered: Opt[VariationalAutoencoder]

    def __post_init__(self):
        super(VAEPassthroughClassifyTwin, self).__post_init__()
        self.set_twin_vae(self)

    def set_twin_vae(self, vae: VariationalAutoencoder):
        self._twin_vae = (vae,)
        if vae is not self:
            if isinstance(self.encoder, EncoderTwin):
                encoder: EncoderTwin = self.encoder
                encoder.set_twin_encoder(vae.encoder)
            if isinstance(self.decoder, DecoderPassthroughTwin):
                decoder: DecoderPassthroughTwin = self.decoder
                decoder.set_twin_decoder(vae.decoder)
            kld_ = self.kld
            if isinstance(kld_, TwinKLDMix):
                kld: TwinKLDMix = kld_
                kld.set_twin_kld(vae.kld)
            self._twin_vae_registered = vae
        else:
            self._twin_vae_registered = None

    @property
    def vae_twin(self) -> VariationalAutoencoder:
        return self._twin_vae[0]


class EncoderSELU(Encoder):
    def __init__(self, dims: Tuple[int, Tuple[int, ...], int], sample_layer: Type[BaseSample]=GaussianSample):
        super(EncoderSELU, self).__init__(dims=dims, sample_layer=sample_layer, activation_fn=tr.selu)


class EncoderSELUTrim(Encoder, MetaTrimmer):
    def __init__(self, dims: Tuple[int, Tuple[int, ...], int], sample_layer: Type[BaseSample]=GaussianSample):
        with self.disable_final_init():
            super(EncoderSELUTrim, self).__init__(dims=dims, sample_layer=sample_layer, activation_fn=tr.selu)
        MetaTrimmer.__init__(self)
        self.__final_init__()

    def forward_(self, x: Tensor) -> XTupleYi:
        z, (μ, log_σ) = super(EncoderSELUTrim, self).forward_(x)
        self.set_trim(μ, log_σ)
        return z, (μ, log_σ)


class EncoderPassthrSELU(EncoderPassthrough):
    def __init__(self, dims: Tuple[int, Tuple[int, ...], int]):
        super(EncoderPassthrSELU, self).__init__(dims=dims, sample_layer=GaussianSample, activation_fn=tr.selu)


class EncoderPassthrSELUTrim(EncoderPassthrough, MetaTrimmer):
    def __init__(self, dims: Tuple[int, Tuple[int, ...], int], sample_layer: Type[BaseSample]=GaussianSample):
        with self.disable_final_init():
            super(EncoderPassthrSELUTrim, self).__init__(dims=dims, sample_layer=sample_layer, activation_fn=tr.selu)
        MetaTrimmer.__init__(self)
        self.__final_init__()

    def forward_(self, x: Tensor) -> XTupleYi:
        z, (μ, log_σ) = super(EncoderPassthrSELUTrim, self).forward_(x)
        self.set_trim(μ, log_σ)
        return z, (μ, log_σ)


class EncoderCustom(EncoderSELUTrim):
    pass


class EncoderPassthrCustom(EncoderPassthrSELUTrim):
    pass


class DecoderSELU(Decoder):
    def __init__(self, dims: Tuple[int, Tuple[int, ...], int]):
        super(DecoderSELU, self).__init__(dims=dims, activation_fn=tr.selu, output_activation=None)


class DecoderPassthrSELU(DecoderPassthrough):
    def __init__(self, dims: Tuple[int, Tuple[int, ...], int]):
        super(DecoderPassthrSELU, self).__init__(dims=dims, activation_fn=tr.selu, output_activation=None)


class DecoderCustom(DecoderSELU):
    pass


class DecoderPassthrCustom(DecoderPassthrSELU):
    pass


class LadderEncoderSELU(LadderEncoder):
    def __init__(self, dims: Tuple[int, int, int]):
        super(LadderEncoderSELU, self).__init__(dims=dims)

    def act(self, x: Tensor) -> Tensor:
        return tr.selu(x)


class LadderDecoderSELU(LadderDecoder):
    def __init__(self, dims: Tuple[int, int, int]):
        super(LadderDecoderSELU, self).__init__(dims=dims)

    def act1(self, z: Tensor) -> Tensor:
        return tr.selu(z)

    def act2(self, z: Tensor) -> Tensor:
        return tr.selu(z)


class ClassifierSELU(Classifier):
    def __init__(self, dims: Tuple[int, Tuple[int, ...], int]):
        super(ClassifierSELU, self).__init__(dims=dims, activation_fn=tr.selu)


class ClassifierCustom(ClassifierSELU):
    pass
