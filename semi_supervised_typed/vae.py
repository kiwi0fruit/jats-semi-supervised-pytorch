from typing import Callable, Tuple, List, Type, Optional as Opt, Union, Sequence as Seq, NamedTuple, Dict
import torch as tr
from torch import Tensor, nn
from torch.nn import functional as func
from beta_tcvae_typed import XYDictStrZ, KLDLoss, PostInit
from kiwi_bugfix_typechecker import test_assert

from .layers import GaussianSample, GaussianMerge, GumbelSoftmax, BaseSample, ModuleXToXTupleYi, XTupleYi
from .vae_types import (RetLadderEncoder, BaseLadderEncoder, RetLadderDecoder, BaseLadderDecoder, ModuleXToX,
                        ModuleXYToXYDictStrOptZ)

test_assert()
δ = 1e-8


class Act(NamedTuple):
    a: Callable[[Tensor], Tensor]


class Perceptron(ModuleXToX):
    dims: Seq[int]
    activation_fn: Act
    output_activation: Opt[Act]
    layers: List[nn.Linear]

    def __init__(self, dims: Seq[int], activation_fn: Callable[[Tensor], Tensor]=tr.selu,
                 output_activation: Opt[Callable[[Tensor], Tensor]]=None):
        super(Perceptron, self).__init__()
        self.dims = dims
        self.activation_fn = Act(activation_fn)
        self.output_activation = Act(output_activation) if (output_activation is not None) else None
        self.layers = self.get_layers(dims)

    @staticmethod
    def get_layers(dims: Seq[int]) -> List[nn.Linear]:
        # noinspection PyTypeChecker
        return nn.ModuleList(list(map(  # type: ignore
            lambda d: nn.Linear(*d), zip(dims, dims[1:])
        )))

    def forward_i(self, x: Tensor, layers: List[nn.Linear]) -> Tensor:
        h: Tensor
        for i, layer in enumerate(layers):
            h = layer.__call__(x)
            if i < len(layers) - 1:
                h = self.activation_fn.a(h)
            elif self.output_activation is not None:
                h = self.output_activation.a(h)
            x = h
        return x

    def forward_(self, x: Tensor) -> Tensor:
        return self.forward_i(x, self.layers)


class Encoder(ModuleXToXTupleYi, PostInit):
    hidden: List[nn.Linear]
    activation_fn: Act
    sample: BaseSample
    extras: Dict[str, Tensor]
    dims: Tuple[int, Tuple[int, ...], int]

    def __init__(self, dims: Tuple[int, Tuple[int, ...], int],
                 sample_layer: Type[BaseSample]=GaussianSample,
                 activation_fn: Callable[[Tensor], Tensor]=tr.selu):
        """
        Inference network

        Attempts to infer the probability distribution
        p(z|x) from the data by fitting a variational
        distribution q_φ(z|x). Returns the two parameters
        of the distribution (µ, log σ²).

        :param dims: dimensions of the networks
           given by the number of neurons on the form
           [input_dim, [hidden_dims], latent_dim].
        :param sample_layer: subclass of the BaseSample
        """
        super(Encoder, self).__init__()
        self.dims = dims
        x_dim, h_dim, z_dim = dims
        self.activation_fn = Act(activation_fn)
        hidden, sample = self.get_hidden_and_sample(sample_layer, z_dim, x_dim, *h_dim)
        self.hidden, self.sample = hidden, sample
        self.extras = {}
        self.__final_init__()

    @staticmethod
    def get_hidden_and_sample(sample_layer: Type[BaseSample], z_dim: int,
                              *neurons: int) -> Tuple[List[nn.Linear], BaseSample]:
        linear_layers = [nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))]
        # noinspection PyTypeChecker
        hidden: List[nn.Linear] = nn.ModuleList(linear_layers)  # type: ignore
        sample = sample_layer(neurons[-1], z_dim)
        return hidden, sample

    def subforward(self, x: Tensor, hidden: List[nn.Linear], sample: BaseSample) -> XTupleYi:
        h: Tensor
        for layer in hidden:
            h = self.activation_fn.a(layer.__call__(x))
            x = h
        return sample.__call__(x)

    def forward_(self, x: Tensor) -> XTupleYi:
        return self.subforward(x, self.hidden, self.sample)


class Decoder(ModuleXToX, PostInit):
    hidden: List[nn.Linear]
    reconstruction: nn.Linear
    activation_fn: Act
    output_activation: Opt[Act]
    extras: Dict[str, Tensor]

    def __init__(self, dims: Tuple[int, Tuple[int, ...], int],
                 activation_fn: Callable[[Tensor], Tensor]=tr.selu,
                 output_activation: Opt[Callable[[Tensor], Tensor]]=None):
        """
        Generative network

        Generates samples from the original distribution
        p(x) by transforming a latent representation, e.g.
        by finding p_θ(x|z).

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(Decoder, self).__init__()

        z_dim, h_dim, x_dim = dims

        neurons = [z_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))]
        # noinspection PyTypeChecker
        self.hidden = nn.ModuleList(linear_layers)  # type: ignore
        self.reconstruction = nn.Linear(h_dim[-1], x_dim)
        self.output_activation = Act(output_activation) if (output_activation is not None) else None
        self.activation_fn = Act(activation_fn)
        self.extras = {}
        self.__final_init__()

    def forward_(self, x: Tensor) -> Tensor:
        h: Tensor
        for layer in self.hidden:
            h = self.activation_fn.a(layer.__call__(x))
            x = h
        x_preparams = self.reconstruction.__call__(x)
        return self.output_activation.a(x_preparams) if (self.output_activation is not None) else x_preparams


class PassthroughMeta:
    _passthr_x: Opt[Tensor]
    _neg_log_p_passthr_x: Opt[Tensor]

    def __init__(self, passthrough_dim: int=1):
        self.passthr_dim = passthrough_dim
        self._passthr_x = None
        self._neg_log_p_passthr_x = None

    def set_passthr_x(self, x: Tensor) -> None:
        self._passthr_x = x[:, :self.passthr_dim]

    @property
    def passthr_x(self) -> Tensor:
        if self._passthr_x is None:
            raise ValueError
        return self._passthr_x

    @property
    def neg_log_p_passthr_x(self) -> Tensor:
        if self._neg_log_p_passthr_x is None:
            raise ValueError
        neg_log_p_passthr_x = self._neg_log_p_passthr_x
        self._neg_log_p_passthr_x = None
        return neg_log_p_passthr_x

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def neg_log_p(self, x: Tensor) -> Opt[Tensor]:  # pylint: disable=unused-argument,no-self-use
        return None

    def extend_x(self, x: Tensor) -> Tensor:
        passthr_x = self._passthr_x
        self._passthr_x = None
        if passthr_x is None:
            raise ValueError
        self._neg_log_p_passthr_x = self.neg_log_p(passthr_x)  # pylint: disable=assignment-from-none
        return tr.cat((passthr_x, x), dim=-1)


class Passer:
    module: Opt[PassthroughMeta]

    def __init__(self, module: nn.Module=None):
        self.module = module if isinstance(module, PassthroughMeta) else None
        if self.module is None:
            print('WARNING! Passer: not isinstance(module, PassthroughMeta)')

    @property
    def neg_log_p_passthr_x(self) -> Union[int, Tensor]:
        if self.module is None:
            return 0
        return self.module.neg_log_p_passthr_x


class EncoderPassthrough(Encoder, PassthroughMeta):
    def __init__(self, dims: Tuple[int, Tuple[int, ...], int],
                 passthrough_dim: int = 1,
                 sample_layer: Type[BaseSample]=GaussianSample,
                 activation_fn: Callable[[Tensor], Tensor]=tr.selu):
        """
        Extends ``Encoder``. Does passthrough of first ``passthrough_dim`` elements
        provided to ``self.set_passthr_x`` (prepended to ``input_dim``).

        :param dims: dimensions of the networks
           given by the number of neurons on the form
           [input_dim, [hidden_dims], latent_dim].
        :param sample_layer: subclass of the BaseSample
        """
        PassthroughMeta.__init__(self=self, passthrough_dim=passthrough_dim)
        x, h, z = dims
        super(EncoderPassthrough, self).__init__(dims=(x + passthrough_dim, h, z), activation_fn=activation_fn,
                                                 sample_layer=sample_layer)

    def forward_(self, x: Tensor) -> XTupleYi:
        return super(EncoderPassthrough, self).forward_(self.extend_x(x))


class DecoderPassthrough(Decoder, PassthroughMeta):
    def __init__(self, dims: Tuple[int, Tuple[int, ...], int],
                 passthrough_dim: int=1,
                 activation_fn: Callable[[Tensor], Tensor]=tr.selu,
                 output_activation: Opt[Callable[[Tensor], Tensor]]=None):
        """
        Extends ``Decoder``. Does passthrough of first ``passthrough_dim`` of ``x`` elements
        provided to ``self.set_passthr_x`` (prepended to ``latent_dim``).

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        PassthroughMeta.__init__(self=self, passthrough_dim=passthrough_dim)
        z, h, x = dims
        super(DecoderPassthrough, self).__init__(dims=(z + passthrough_dim, h, x), activation_fn=activation_fn,
                                                 output_activation=output_activation)

    def forward_(self, x: Tensor) -> Tensor:
        return super(DecoderPassthrough, self).forward_(self.extend_x(x))


class VariationalAutoencoder(ModuleXYToXYDictStrOptZ, PostInit):
    z_dim: int
    encoder: Encoder
    decoder: Decoder
    _kld: KLDLoss

    def __init__(self, dims: Tuple[int, int, Tuple[int, ...]], Encode: Type[Encoder]=Encoder,
                 Decode: Type[Decoder]=Decoder, kld: KLDLoss=None):
        """
        Variational Autoencoder [Kingma 2013] model
        consisting of an encoder/decoder pair for which
        a variational distribution is fitted to the
        encoder. Also known as the M1 model in [Kingma 2014].

        :param dims: x, z and hidden dimensions of the networks
        """
        super(VariationalAutoencoder, self).__init__()

        x_dim, z_dim, h_dim = dims
        self.z_dim = z_dim
        self.set_kld(kld if (kld is not None) else KLDLoss())

        self.encoder = Encode((x_dim, h_dim, z_dim))
        self.decoder = Decode((z_dim, tuple(reversed(h_dim)), x_dim))
        self.__final_init__()

    @property
    def kld(self) -> KLDLoss:
        return self._kld

    def set_kld(self, kld: KLDLoss):
        """ Do not change kld after setting it! """
        self._kld = kld

    def encode(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:  # pylint: disable=unused-argument
        """
        :param x: input data
        :param y: unused dummy
        :return: (flow_qz_x(z), qz_params)
        """
        z, qz_params = self.encoder.__call__(x)
        return self.kld.flow_qz_x(z), qz_params

    def forward_vae_to_z(self, x: Tensor, y: Tensor  # pylint: disable=unused-argument
                         ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        z, qz_params = self.encoder.__call__(x)
        return z, qz_params

    def forward_vae_to_x(self, z: Tensor, qz_params: Tuple[Tensor, ...]) -> XYDictStrZ:
        """ z is sampled from q(z) """
        kld, z, verb = self.kld.__call__(z, qz_params)

        x_rec = self.decoder.__call__(z)
        return x_rec, kld, verb

    def forward_(self, x: Tensor, y: Tensor) -> XYDictStrZ:  # pylint: disable=unused-argument
        """
        Runs a data point through the model in order
        to provide it's reconstruction.

        :param x: input data
        :param y: unused dummy
        :return: (reconstructed input, kld)
        """
        z, qz_params = self.forward_vae_to_z(x, y)
        return self.forward_vae_to_x(z, qz_params)

    def sample(self, z: Tensor, y: Tensor, use_pz_flow: bool=True) -> Tensor:  # pylint: disable=unused-argument
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: Random normal variable
        :param y: unused dummy
        :param use_pz_flow:
        :return: generated sample
        """
        z = self.kld.flow_pz(z) if use_pz_flow else z
        return self.decoder.__call__(z)


class VAEPassthrough(VariationalAutoencoder):
    decoder: DecoderPassthrough

    def __init__(self, dims: Tuple[int, int, Tuple[int, ...]],
                 Encode: Type[Encoder]=Encoder,
                 Decode: Type[DecoderPassthrough]=DecoderPassthrough,
                 kld: KLDLoss=None) -> None:
        super(VAEPassthrough, self).__init__(dims=dims, Encode=Encode, Decode=Decode, kld=kld)

    def forward_vae_to_z(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        self.decoder.set_passthr_x(x)
        return super(VAEPassthrough, self).forward_vae_to_z(x=x, y=y)


class GumbelAutoencoder(ModuleXYToXYDictStrOptZ):
    z_dim: int
    n_samples: int
    encoder: Perceptron
    sampler: GumbelSoftmax
    decoder: Perceptron
    k: Tensor

    def __init__(self, dims: Tuple[int, int, Tuple[int, ...]], n_samples: int=100,
                 Encode: Type[Perceptron]=Perceptron,
                 Decode: Type[Perceptron]=Perceptron):
        super(GumbelAutoencoder, self).__init__()

        x_dim, z_dim, h_dim = dims
        self.z_dim = z_dim
        self.n_samples = n_samples

        self.encoder = Encode([x_dim, *h_dim])
        self.sampler = GumbelSoftmax(h_dim[-1], z_dim, n_samples)
        self.decoder = Decode([z_dim, *reversed(h_dim), x_dim])

        self.register_buffer('k', tr.tensor([float(self.z_dim)]))

    def _kld(self, qz_params: Tuple[Tensor, ...]) -> Tensor:
        (qz,) = qz_params
        kl = qz * (tr.log(qz + δ) - tr.log(self.k**-1))
        kl = kl.view(-1, self.n_samples, self.z_dim)
        return tr.sum(tr.sum(kl, dim=1), dim=1)

    def set_τ(self, τ: float=1.0):
        self.sampler.set_τ(τ)

    # noinspection PyUnusedLocal
    def encode(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:  # pylint: disable=unused-argument
        """
        :param x: input data
        :param y: unused dummy
        :return: (sample, qz_params)
        """
        x = self.encoder.__call__(x)
        sample, qz_params = self.sampler.__call__(x)
        return sample, qz_params

    def forward_(self, x: Tensor, y: Tensor) -> XYDictStrZ:  # pylint: disable=unused-argument
        """
        Runs a data point through the model in order
        to provide it's reconstruction.

        :param x: input data
        :param y: unused dummy
        :return: (reconstructed input, kld, {})
        """
        x = self.encoder.__call__(x)
        sample, qz_params = self.sampler.__call__(x)

        kld = self._kld(qz_params=qz_params)
        x_rec = self.decoder.__call__(sample)
        return x_rec, kld, {}

    def sample(self, z: Tensor) -> Tensor:
        return self.decoder.__call__(z)


class LadderEncoder(BaseLadderEncoder):
    in_features: int
    out_features: int
    linear: nn.Linear
    batchnorm: nn.BatchNorm1d
    sample: GaussianSample

    def __init__(self, dims: Tuple[int, int, int]):
        """
        The ladder encoder differs from the standard encoder
        by using batch-normalization and LReLU activation.
        Additionally, it also returns the transformation x.

        :param dims: dimensions (input_dim, hidden_dim, latent_dim).
        """
        super(LadderEncoder, self).__init__()

        x_dim, h_dim, self.z_dim = dims
        self.in_features = x_dim
        self.out_features = h_dim

        self.linear = nn.Linear(x_dim, h_dim)
        self.batchnorm = nn.BatchNorm1d(h_dim)
        self.sample = GaussianSample(h_dim, self.z_dim)

    def act(self, x: Tensor) -> Tensor:
        return func.leaky_relu(self.batchnorm.__call__(x), 0.1)

    def forward_(self, x: Tensor) -> RetLadderEncoder:
        x = self.act(self.linear.__call__(x))
        return x, self.sample.__call__(x)


class LadderDecoder(BaseLadderDecoder):
    z_dim: int
    linear1: nn.Linear
    batchnorm1: nn.BatchNorm1d
    merge: GaussianMerge
    linear2: nn.Linear
    batchnorm2: nn.BatchNorm1d
    sample: GaussianSample
    default: bool

    def __init__(self, dims: Tuple[int, int, int]):
        """
        The ladder dencoder differs from the standard encoder
        by using batch-normalization and LReLU activation.
        Additionally, it also returns the transformation x.

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            (latent_dim, hidden_dim, input_dim).
        """
        super(LadderDecoder, self).__init__()

        self.z_dim, h_dim, x_dim = dims

        self.linear1 = nn.Linear(x_dim, h_dim)
        self.batchnorm1 = nn.BatchNorm1d(h_dim)
        self.merge = GaussianMerge(h_dim, self.z_dim)

        self.linear2 = nn.Linear(x_dim, h_dim)
        self.batchnorm2 = nn.BatchNorm1d(h_dim)
        self.sample = GaussianSample(h_dim, self.z_dim)

    def act1(self, z: Tensor) -> Tensor:
        return func.leaky_relu(self.batchnorm1.__call__(z), 0.1)

    def act2(self, z: Tensor) -> Tensor:
        return func.leaky_relu(self.batchnorm2.__call__(z), 0.1)

    def sample2(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Sample from the decoder and send forward
        z = self.act2(self.linear2.__call__(x))
        z, (p_μ, p_log_σ) = self.sample.__call__(z)
        return z, p_μ, p_log_σ

    def forward_(self, x: Tensor, l_μ: Tensor, l_log_σ: Tensor) -> RetLadderDecoder:
        # Sample from this encoder layer and merge
        z = self.act1(self.linear1.__call__(x))
        q_z, (q_μ, q_log_σ) = self.merge.__call__(z, l_μ, l_log_σ)
        # Sample from the decoder and send forward
        z, p_μ, p_log_σ = self.sample2(x)

        return z, (q_z, (q_μ, q_log_σ), (p_μ, p_log_σ))


class LadderKLDLoss(KLDLoss):
    pass


class LadderVariationalAutoencoder(VariationalAutoencoder):
    encoder: List[LadderEncoder]  # type: ignore
    decoder: List[LadderDecoder]  # type: ignore
    reconstruction: Decoder

    def __init__(self, dims: Tuple[int, Tuple[int, ...], Tuple[int, ...]],
                 LadderEncode: Type[LadderEncoder]=LadderEncoder,
                 LadderDecode: Type[LadderDecoder]=LadderDecoder,
                 Decode: Type[Decoder]=Decoder,
                 kld: LadderKLDLoss=None):
        """
        Ladder Variational Autoencoder as described by
        [Sønderby 2016]. Adds several stochastic
        layers to improve the log-likelihood estimate.

        :param dims: x, z and hidden dimensions of the networks
        """
        x_dim, z_dim, h_dim = dims
        super(LadderVariationalAutoencoder, self).__init__((x_dim, z_dim[0], h_dim), kld=kld)

        neurons: List[int] = [x_dim, *h_dim]
        encoder_layers = [LadderEncode((neurons[i - 1], neurons[i], z_dim[i - 1])) for i in range(1, len(neurons))]
        decoder_layers = [LadderDecode((z_dim[i - 1], h_dim[i - 1], z_dim[i])) for i in range(1, len(h_dim))][::-1]

        # noinspection PyTypeChecker
        self.encoder = nn.ModuleList(encoder_layers)  # type: ignore
        # noinspection PyTypeChecker
        self.decoder = nn.ModuleList(decoder_layers)  # type: ignore
        self.reconstruction = Decode((z_dim[0], h_dim, x_dim))

    def set_kld(self, kld: KLDLoss):
        assert isinstance(kld, LadderKLDLoss) and kld.closed_form
        self._kld = kld

    # noinspection PyUnusedLocal
    def forward_lvae_to_z(self, x: Tensor, y: Tensor  # pylint: disable=unused-argument
                          ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        # Gather latent representation
        # from encoders along with final z.
        latents: List[Tuple[Tensor, Tensor]] = []
        h: Tensor
        for encoder in self.encoder:
            h, (z, (μ, log_σ)) = encoder.__call__(x)
            latents.append((μ, log_σ))
            x = h
        # noinspection PyUnboundLocalVariable
        z = z

        latents = list(reversed(latents))
        return z, latents

    def forward_lvae_to_x(self, z: Tensor, latents: List[Tuple[Tensor, Tensor]]) -> XYDictStrZ:
        # If at top, encoder == decoder, use prior for KL:
        l_μ, l_log_σ = latents[0]
        kld, z, _ = self.kld.__call__(z, (l_μ, l_log_σ))
        kl_divergence = kld

        # Perform downward merge of information:
        for i, decoder in enumerate(self.decoder):
            l_μ, l_log_σ = latents[i + 1]
            h, (q_z, q_params, p_params) = decoder.__call__(z, l_μ, l_log_σ)
            kld, _, _ = self.kld.__call__(q_z, q_params, p_params)
            kl_divergence += kld
            z = h

        x_rec = self.reconstruction.__call__(z)
        return x_rec, kl_divergence, {}

    def encode(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        z, latents = self.forward_lvae_to_z(x, y)
        return self.kld.flow_qz_x(z), latents[0]

    def forward_(self, x: Tensor, y: Tensor) -> XYDictStrZ:
        z, latents = self.forward_lvae_to_z(x, y)
        return self.forward_lvae_to_x(z, latents)

    def sample(self, z: Tensor, y: Tensor=None, use_pz_flow: bool=True) -> Tensor:  # pylint: disable=unused-argument
        z = self.kld.flow_pz(z) if use_pz_flow else z
        h: Tensor
        for decoder in self.decoder:
            h, _, _ = decoder.sample2(z)
            z = h
        return self.reconstruction.__call__(z)
