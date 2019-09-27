from typing import Callable, Tuple, List, Type, Optional as Opt, Union, Sequence as Seq, NamedTuple
import torch as tr
from torch import Tensor
from kiwi_bugfix_typechecker import func, nn
from beta_tcvae_typed import BetaTCKLDLoss, Normal
from normalizing_flows_typed.types import SequentialZToXY, ModuleZToXY

from .layers import GaussianSample, GaussianMerge, GumbelSoftmax, BaseSample, ModuleXToXTupleYi, XTupleYi
from .inference import log_gaussian, log_standard_gaussian
from .vae_types import (RetLadderEncoder, BaseLadderEncoder, RetLadderDecoder, BaseLadderDecoder, ModuleXToX,
                        ModuleXYToXY)

δ = 1e-8
ZToXY = Union[SequentialZToXY, ModuleZToXY]


class Act(NamedTuple):
    a: Callable[[Tensor], Tensor]


class Perceptron(ModuleXToX):
    dims: Seq[int]
    activation_fn: Act
    output_activation: Opt[Act]
    layers: List[nn.Linear]

    def __init__(self, dims: Seq[int], activation_fn: Callable[[Tensor], Tensor]=tr.relu,
                 output_activation: Opt[Callable[[Tensor], Tensor]]=None):
        super(Perceptron, self).__init__()
        self.dims = dims
        self.activation_fn = Act(activation_fn)
        self.output_activation = Act(output_activation) if (output_activation is not None) else None
        # noinspection PyTypeChecker
        self.layers = nn.ModuleList(list(map(lambda d: nn.Linear(*d), list(zip(dims, dims[1:])))))

    def forward_(self, x: Tensor) -> Tensor:
        h: Tensor
        for i, layer in enumerate(self.layers):
            h = layer.__call__(x)
            if (i == len(self.layers) - 1) and (self.output_activation is not None):
                h = self.output_activation.a(h)
            else:
                h = self.activation_fn.a(h)
            x = h
        return x


class Encoder(ModuleXToXTupleYi):
    hidden: List[nn.Linear]
    activation_fn: Act
    sample: BaseSample

    def __init__(self, dims: Tuple[int, Tuple[int, ...], int],
                 sample_layer: Type[BaseSample]=GaussianSample,
                 activation_fn: Callable[[Tensor], Tensor]=tr.relu):
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

        x_dim, h_dim, z_dim = dims
        neurons = [x_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))]
        # noinspection PyTypeChecker
        self.hidden = nn.ModuleList(linear_layers)
        self.activation_fn = Act(activation_fn)
        self.sample = sample_layer(h_dim[-1], z_dim)

    def forward_(self, x: Tensor) -> XTupleYi:
        h: Tensor
        for layer in self.hidden:
            h = self.activation_fn.a(layer.__call__(x))
            x = h
        return self.sample.__call__(x)


class Decoder(ModuleXToX):
    hidden: List[nn.Linear]
    reconstruction: nn.Linear
    activation_fn: Act
    output_activation: Opt[Act]

    def __init__(self, dims: Tuple[int, Tuple[int, ...], int],
                 activation_fn: Callable[[Tensor], Tensor]=tr.relu,
                 output_activation: Opt[Callable[[Tensor], Tensor]]=tr.sigmoid):
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
        self.hidden = nn.ModuleList(linear_layers)
        self.reconstruction = nn.Linear(h_dim[-1], x_dim)
        self.output_activation = Act(output_activation) if (output_activation is not None) else None
        self.activation_fn = Act(activation_fn)

    def forward_(self, x: Tensor) -> Tensor:
        h: Tensor
        for layer in self.hidden:
            h = self.activation_fn.a(layer.__call__(x))
            x = h
        x_preparams = self.reconstruction.__call__(x)
        return self.output_activation.a(x_preparams) if (self.output_activation is not None) else x_preparams


class VerboseQZParams:
    _qz_params: Opt[Tuple[Tensor, Tuple[Tensor, ...]]]
    _save_qz_params: bool

    def __init__(self):
        self._qz_params = None
        self._save_qz_params = False

    def set_save_qz_params(self, save_qz_params: bool=False) -> None:
        """
        :param save_qz_params: whether to save z_params during forward pass.
        """
        self._save_qz_params = save_qz_params

    def take_away_qz_params(self) -> Tuple[Opt[Tensor], Tuple[Tensor, ...]]:
        """
        :return: saved self._qz_params: (z, qz_params) or (None, ())
           Also clears self._qz_params.
        """
        _qz_params = self._qz_params
        self._qz_params = None
        if _qz_params is None:
            return None, ()
        z, qz_params = _qz_params
        return z, qz_params


class TCKLDMeta:
    tc_kld: Opt[BetaTCKLDLoss]
    q_params_μ_first: bool

    def __init__(self):
        self.tc_kld = None

    def set_tc_kld(self, beta_tc_kld: BetaTCKLDLoss):
        self.tc_kld = beta_tc_kld
        self.q_params_μ_first = self.tc_kld.q_params_μ_first


class VariationalAutoencoder(ModuleXYToXY, VerboseQZParams, TCKLDMeta):
    z_dim: int
    encoder: Encoder
    decoder: Decoder
    qz_x_flow: Opt[ZToXY]
    q_params_μ_first: bool

    def __init__(self, dims: Tuple[int, int, Tuple[int, ...]], Encode: Type[Encoder]=Encoder,
                 Decode: Type[Decoder]=Decoder):
        """
        Variational Autoencoder [Kingma 2013] model
        consisting of an encoder/decoder pair for which
        a variational distribution is fitted to the
        encoder. Also known as the M1 model in [Kingma 2014].

        :param dims: x, z and hidden dimensions of the networks
        """
        super(VariationalAutoencoder, self).__init__()
        VerboseQZParams.__init__(self=self)
        TCKLDMeta.__init__(self=self)

        x_dim, z_dim, h_dim = dims
        self.z_dim = z_dim

        self.encoder = Encode((x_dim, h_dim, z_dim))
        self.decoder = Decode((z_dim, tuple(reversed(h_dim)), x_dim))

        self.qz_x_flow = None
        self.q_params_μ_first = True

    def _kld_normal(self, z: Tensor, qz_params: Tuple[Tensor, ...],
                    pz_params: Tuple[Tensor, ...]=None, try_closed_form: bool=True) -> Tuple[Tensor, Tensor]:
        """
        Computes the KL-divergence of some element z.

        KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                 = -E[log p(z) - log q(z)]

        :param z: sample from q-distribuion
        :param qz_params: (μ, log_σ) of the q-distribution
        :param pz_params: (μ, log_σ) of the p-distribution
        :param try_closed_form:
        :return: KL(q||p), flow(z)
        """
        if try_closed_form and (self.qz_x_flow is None):
            return Normal.kl(qz_params, pz_params).view(z.shape[0], -1).sum(dim=1), z

        q_μ, q_log_σ = qz_params
        log_qz_x = log_gaussian(z, μ=q_μ, log_σ=q_log_σ)

        sladetj: Union[Tensor, int]
        if self.qz_x_flow is not None:
            fz, sladetj = self.qz_x_flow.__call__(z)
            sladetj = sladetj.view(sladetj.size(0), -1).sum(dim=1)
        else:
            fz, sladetj = z, 0

        if pz_params is not None:
            p_μ, p_log_σ = pz_params
            log_pz = log_gaussian(fz, μ=p_μ, log_σ=p_log_σ)
        else:
            log_pz = log_standard_gaussian(fz)

        kld = log_qz_x - sladetj - log_pz  # sladetj := sum_log_abs_det_jacobian
        return kld, fz

    def _kld(self, z: Tensor, qz_params: Tuple[Tensor, ...], pz_params: Tuple[Tensor, ...]=None,
             unmod_kld: bool=False) -> Tuple[Tensor, Tensor]:  # pylint: disable=unused-argument
        """
        Computes the KL-divergence of
        some element z.

        KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                 = -E[log p(z) - log q(z)]

        :param z: sample from q-distribuion
        :param qz_params: (μ, log_σ) of the q-distribution
        :param pz_params: (μ, log_σ) of the p-distribution
        :param unmod_kld: force unmodified KLD (useful for subclasses)
        :return: KL(q||p), flow(z)
        """
        return self._kld_normal(z=z, qz_params=qz_params, pz_params=pz_params)

    def set_qz_x_flow(self, flow: ZToXY):
        self.qz_x_flow = flow
        self.q_params_μ_first = False

    def forward_(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:  # pylint: disable=unused-argument
        """
        Runs a data point through the model in order
        to provide it's reconstruction.

        :param x: input data
        :param y: unused dummy
        :return: (reconstructed input, kld)
        """
        z, qz_params = self.encoder.__call__(x)

        kld, z = self._kld(z, qz_params)

        if self._save_qz_params:
            self._qz_params = (z, qz_params)

        x_rec = self.decoder.__call__(z)
        return x_rec, kld

    def sample(self, z: Tensor, y: Tensor) -> Tensor:  # pylint: disable=unused-argument
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: Random normal variable
        :param y: unused dummy
        :return: generated sample
        """
        return self.decoder.__call__(z)


class GumbelAutoencoder(ModuleXYToXY, VerboseQZParams):
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
        VerboseQZParams.__init__(self=self)

        x_dim, z_dim, h_dim = dims
        self.z_dim = z_dim
        self.n_samples = n_samples

        self.encoder = Encode([x_dim, *h_dim])
        self.sampler = GumbelSoftmax(h_dim[-1], z_dim, n_samples)
        self.decoder = Decode([z_dim, *reversed(h_dim), x_dim], output_activation=tr.sigmoid)

        self.register_buffer('k', tr.tensor([float(self.z_dim)]))

    def _kld(self, qz_params: Tuple[Tensor, ...]) -> Tensor:
        (qz,) = qz_params
        kl = qz * (tr.log(qz + δ) - tr.log(self.k**-1))
        kl = kl.view(-1, self.n_samples, self.z_dim)
        return tr.sum(tr.sum(kl, dim=1), dim=1)

    def set_τ(self, τ: float=1.0):
        self.sampler.set_τ(τ)

    def forward_(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:  # pylint: disable=unused-argument
        """
        Runs a data point through the model in order
        to provide it's reconstruction.

        :param x: input data
        :param y: unused dummy
        :return: (reconstructed input, kld)
        """
        x = self.encoder.__call__(x)
        sample, qz_params = self.sampler.__call__(x)

        if self._save_qz_params:
            self._qz_params = (sample, qz_params)

        kld = self._kld(qz_params=qz_params)
        x_rec = self.decoder.__call__(sample)
        return x_rec, kld

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


class LadderVariationalAutoencoder(VariationalAutoencoder):
    encoder: List[LadderEncoder]  # type: ignore
    decoder: List[LadderDecoder]  # type: ignore
    reconstruction: Decoder

    def __init__(self, dims: Tuple[int, Tuple[int, ...], Tuple[int, ...]],
                 LadderEncode: Type[LadderEncoder]=LadderEncoder,
                 LadderDecode: Type[LadderDecoder]=LadderDecoder,
                 Decode: Type[Decoder]=Decoder):
        """
        Ladder Variational Autoencoder as described by
        [Sønderby 2016]. Adds several stochastic
        layers to improve the log-likelihood estimate.

        :param dims: x, z and hidden dimensions of the networks
        """
        x_dim, z_dim, h_dim = dims
        super(LadderVariationalAutoencoder, self).__init__((x_dim, z_dim[0], h_dim))

        neurons: List[int] = [x_dim, *h_dim]
        encoder_layers = [LadderEncode((neurons[i - 1], neurons[i], z_dim[i - 1])) for i in range(1, len(neurons))]
        decoder_layers = [LadderDecode((z_dim[i - 1], h_dim[i - 1], z_dim[i])) for i in range(1, len(h_dim))][::-1]

        # noinspection PyTypeChecker
        self.encoder = nn.ModuleList(encoder_layers)
        # noinspection PyTypeChecker
        self.decoder = nn.ModuleList(decoder_layers)
        self.reconstruction = Decode((z_dim[0], h_dim, x_dim))

    def forward_(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
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

        # If at top, encoder == decoder, use prior for KL:
        l_μ, l_log_σ = latents[0]
        kld, z = self._kld(z, (l_μ, l_log_σ))
        kl_divergence = kld

        if self._save_qz_params:
            self._qz_params = (z, (l_μ, l_log_σ))

        # Perform downward merge of information:
        for i, decoder in enumerate(self.decoder):
            l_μ, l_log_σ = latents[i + 1]
            h, (q_z, q_params, p_params) = decoder.__call__(z, l_μ, l_log_σ)
            kld, _ = self._kld(q_z, q_params, p_params, unmod_kld=True)
            kl_divergence += kld
            z = h

        x_rec = self.reconstruction.__call__(z)
        return x_rec, kl_divergence

    def sample(self, z: Tensor, y: Tensor=None) -> Tensor:  # pylint: disable=unused-argument
        h: Tensor
        for decoder in self.decoder:
            h, _, _ = decoder.sample2(z)
            z = h
        return self.reconstruction.__call__(z)
