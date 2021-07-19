from typing import List, Tuple, Callable, Type
import torch as tr
from torch import Tensor, nn

from beta_tcvae_typed import Verbose, XYDictStrZ, KLDLoss
from .vae import (Encoder, Decoder, LadderEncoder, LadderDecoder, VariationalAutoencoder, DecoderPassthrough,
                  LadderKLDLoss, Perceptron, PostInit)
from .inference import neg_log_standard_categorical
from .dgm_types import ModuleXOptYToXY

δ = 1e-8


class Classifier(ModuleXOptYToXY, PostInit):
    perceptron: Perceptron
    dims: Tuple[int, Tuple[int, ...], int]

    def __init__(self, dims: Tuple[int, Tuple[int, ...], int], activation_fn: Callable[[Tensor], Tensor]=tr.selu):
        """
        ``dims`` = x_dim, (h_dim1, ...), y_dim

        Hidden layer classifier with softmax output (by default VAE subclasses create single hidden layer).
        """
        super(Classifier, self).__init__()
        x_dim, h_dims, y_dim = dims
        self.dims = dims
        self.perceptron = Perceptron(dims=(x_dim, *h_dims, y_dim), activation_fn=activation_fn,
                                     output_activation=None)
        self.__final_init__()

    @staticmethod
    def cross_entropy(probs: Tensor, target: Tensor=None) -> Tensor:
        """ If y is None then entropy is returned. """
        y = target if (target is not None) else probs
        return -(y * tr.log(probs + δ)).sum(dim=-1)

    @staticmethod
    def neg_log_py(probs: Tensor) -> Tensor:
        return neg_log_standard_categorical(probs)

    @staticmethod
    def accuracy(logits_or_probs: Tensor, target: Tensor) -> Tensor:
        return (logits_or_probs.max(dim=1)[1] == target.max(dim=1)[1]).to(dtype=logits_or_probs.dtype)

    def get_logits(self, x: Tensor) -> Tensor:
        return self.perceptron.__call__(x)

    @staticmethod
    def transform_y(y: Tensor) -> Tensor:
        return y

    def forward_(self, x: Tensor, y: Tensor=None) -> Tuple[Tensor, Tensor]:
        """ :return: (probs, cross_entropy). If y is None then entropy is returned. """
        probs = tr.softmax(self.get_logits(x), dim=-1)
        return probs, self.cross_entropy(probs, y)


class DeepGenerativeModel(VariationalAutoencoder):
    encoder: Encoder
    decoder: Decoder
    classifier: Classifier

    def __init__(self, dims: Tuple[int, int, int, Tuple[int, ...]],
                 Encode: Type[Encoder]=Encoder,
                 Decode: Type[Decoder]=Decoder,
                 Classify: Type[Classifier]=Classifier,
                 kld: KLDLoss=None):
        """
        M2 code replication from the paper
        'Semi-Supervised Learning with Deep Generative Models'
        (Kingma 2014) in PyTorch.

        The "Generative semi-supervised model" is a probabilistic
        model that incorporates label information in both
        inference and generation.

        Initialise a new generative model
        :param dims: dimensions of x, y, z and hidden layers.
        """
        x_dim, self.y_dim, z_dim, h_dim = dims
        with self.disable_final_init():
            super(DeepGenerativeModel, self).__init__((x_dim, z_dim, h_dim), kld=kld)

        self.encoder = Encode((x_dim + self.y_dim, h_dim, z_dim))
        self.decoder = Decode((z_dim + self.y_dim, tuple(reversed(h_dim)), x_dim))
        self.classifier = Classify((x_dim, (h_dim[0],), self.y_dim))
        self.__final_init__()

    def encode(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        """
        :param x: input data
        :param y: target
        :return: (flow_qz_x(z), qz_params)
        """
        z, qz_params = self.encoder.__call__(tr.cat([x, y], dim=1))
        return self.kld.flow_qz_x(z), qz_params

    def forward_(self, x: Tensor, y: Tensor) -> XYDictStrZ:
        # Add label and data and generate latent variable
        z, qz_params = self.encoder.__call__(tr.cat([x, y], dim=1))
        kld, z, verb = self.kld.__call__(z, qz_params)

        # Reconstruct data point from latent data and label
        x_rec = self.decoder.__call__(tr.cat([z, y], dim=1))
        return x_rec, kld, verb

    def classify(self, x: Tensor, y: Tensor=None) -> Tuple[Tensor, Tensor]:
        """ :return: (probs, cross_entropy). If y is None then entropy is returned. """
        return self.classifier.__call__(x, y)

    def classify_deterministic(self, x: Tensor, y: Tensor=None) -> Tuple[Tensor, Tensor]:
        """ :return: (probs, cross_entropy). If y is None then entropy is returned. """
        return self.classify(x, y)

    # noinspection PyMethodOverriding
    def sample(self, z: Tensor, y: Tensor, use_pz_flow: bool=True) -> Tensor:  # pylint: disable=signature-differs
        """
        Samples from the Decoder to generate an x.
        :param z: latent normal variable
        :param y: label (one-hot encoded)
        :param use_pz_flow:
        :return: x
        """
        z = self.kld.flow_pz(z) if use_pz_flow else z
        x = self.decoder.__call__(tr.cat([z, y], dim=1))
        return x


class StackedDeepGenerativeModel(DeepGenerativeModel):
    features: VariationalAutoencoder
    q_params_μ_first: bool

    def __init__(self, dims: Tuple[int, int, int, Tuple[int, ...]], features: VariationalAutoencoder,
                 q_params_μ_first: bool=True,
                 Encode: Type[Encoder]=Encoder,
                 Decode: Type[Decoder]=Decoder,
                 Classify: Type[Classifier]=Classifier,
                 kld: KLDLoss=None):
        """
        M1+M2 model as described in [Kingma 2014].

        Initialise a new stacked generative model
        :param dims: dimensions of x, y, z and hidden layers
        :param features: a pretrained M1 model of class `VariationalAutoencoder`
            trained on the same dataset.
        """
        x_dim, y_dim, z_dim, h_dim = dims
        with self.disable_final_init():
            super(StackedDeepGenerativeModel, self).__init__((features.z_dim, y_dim, z_dim, h_dim), Encode=Encode,
                                                             Decode=Decode, Classify=Classify, kld=kld)

        # Be sure to reconstruct with the same dimensions
        in_features = self.decoder.reconstruction.in_features
        self.decoder.reconstruction = nn.Linear(in_features, x_dim)

        # Make vae feature model untrainable by freezing parameters
        self.features = features
        self.features.train(False)
        self.q_params_μ_first = q_params_μ_first

        param: nn.Parameter
        for param in self.features.parameters():
            param.requires_grad = False
        self.__final_init__()

    def forward_(self, x: Tensor, y: Tensor) -> XYDictStrZ:
        # Sample a new latent x from the M1 model
        x_sample, _ = self.features.encoder.__call__(x)

        # Use the sample as new input to M2
        return super(StackedDeepGenerativeModel, self).forward(x_sample, y)

    def classify(self, x: Tensor, y: Tensor=None) -> Tuple[Tensor, Tensor]:
        """ :return: (probs, cross_entropy). If y is None then entropy is returned. """
        fz_, fqz_params = self.features.encoder.__call__(x)
        fz = fqz_params[0] if self.q_params_μ_first else fz_
        return self.classifier.__call__(fz, y)


class AuxiliaryDeepGenerativeModel(DeepGenerativeModel, Verbose):
    aux_encoder: Encoder
    aux_decoder: Encoder
    classifier: Classifier
    encoder: Encoder
    decoder: Decoder

    def __init__(self, dims: Tuple[int, int, int, int, Tuple[int, ...]],
                 Encode: Type[Encoder]=Encoder,
                 Decode: Type[Decoder]=Decoder,
                 AuxEncoder: Type[Encoder]=Encoder,
                 AuxDecoder: Type[Encoder]=Encoder,
                 Classify: Type[Classifier]=Classifier,
                 kld: KLDLoss=None,
                 kld_a: KLDLoss=None):
        """
        Auxiliary Deep Generative Models [Maaløe 2016]
        code replication. The ADGM introduces an additional
        latent variable 'a', which enables the model to fit
        more complex variational distributions.

        When calculates KLD_a ``self._kld(..., unmod_kld=True)`` is called.

        KLD = λ * KLD_a + KLD_z

        :param dims: dimensions of x, y, z, a, hidden layers.
        """
        x_dim, y_dim, z_dim, a_dim, h_dim = dims
        with self.disable_final_init():
            super(AuxiliaryDeepGenerativeModel, self).__init__((x_dim, y_dim, z_dim, h_dim), kld=kld)
        Verbose.__init__(self=self)

        self.aux_encoder = AuxEncoder((x_dim, h_dim, a_dim))
        self.aux_decoder = AuxDecoder((x_dim + y_dim + z_dim, tuple(reversed(h_dim)), a_dim))

        self.classifier = Classify((x_dim + a_dim, (h_dim[0],), y_dim))

        self.encoder = Encode((a_dim + y_dim + x_dim, h_dim, z_dim))
        self.decoder = Decode((y_dim + z_dim, tuple(reversed(h_dim)), x_dim))
        self.kld_a = kld_a if (kld_a is not None) else KLDLoss()
        self.__final_init__()

    def classify(self, x: Tensor, y: Tensor=None) -> Tuple[Tensor, Tensor]:
        """ :return: (probs, cross_entropy). If y is None then entropy is returned. """
        # Auxiliary inference q(a|x)
        a_x, _ = self.aux_encoder.__call__(x)

        # Classification q(y|a,x)
        return self.classifier.__call__(tr.cat((x, a_x), dim=1), y)

    def classify_deterministic(self, x: Tensor, y: Tensor=None) -> Tuple[Tensor, Tensor]:
        """ :return: (probs, cross_entropy). If y is None then entropy is returned. """
        _, (aμ_x, _) = self.aux_encoder.__call__(x)
        return self.classifier.__call__(tr.cat((x, aμ_x), dim=1), y)

    def encode(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        """
        :param x: features
        :param y: labels
        :return: (z_axy, qz_axy_params)
        """
        a_x, qa_x_params = self.aux_encoder.__call__(x)
        z_axy, qz_axy_params = self.encoder.__call__(tr.cat((x, y, a_x), dim=1))
        return self.kld.flow_qz_x(z_axy), qz_axy_params + qa_x_params

    def forward_(self, x: Tensor, y: Tensor) -> XYDictStrZ:
        """
        Forward through the model
        :param x: features
        :param y: labels
        :return: reconstruction
        """
        # Auxiliary inference q(a|x)
        a_x, qa_x_params = self.aux_encoder.__call__(x)

        # Latent inference q(z|a,y,x)
        z_axy, qz_axy_params = self.encoder.__call__(tr.cat((x, y, a_x), dim=1))

        kld_z, z_axy, verb = self.kld.__call__(z_axy, qz_axy_params)

        # Generative p(x|z,y)
        x_zy = self.decoder.__call__(tr.cat((z_axy, y), dim=1))

        # Generative p(a|x,y,z)
        _, pa_xyz_params = self.aux_decoder.__call__(tr.cat((x, y, z_axy), dim=1))
        # kld_a = q(a|x) - p(a|x,y,z)
        kld_a, _, _ = self.kld_a.__call__(a_x, qa_x_params, pa_xyz_params)

        kld = kld_a + kld_z
        if self._verbose:
            verb['adgm_kld_z'] = kld_z
            verb['adgm_kld_a'] = kld_a
        return x_zy, kld, verb


class DGMPassthrough(DeepGenerativeModel):
    decoder: DecoderPassthrough

    def __init__(self, dims: Tuple[int, int, int, Tuple[int, ...]],
                 Encode: Type[Encoder] = Encoder,
                 Decode: Type[DecoderPassthrough] = DecoderPassthrough,
                 Classify: Type[Classifier] = Classifier,
                 kld: KLDLoss=None) -> None:
        super(DGMPassthrough, self).__init__(dims=dims, Encode=Encode, Decode=Decode, Classify=Classify, kld=kld)

    def forward_(self, x: Tensor, y: Tensor) -> XYDictStrZ:
        self.decoder.set_passthr_x(x)
        return super(DGMPassthrough, self).forward_(x=x, y=y)


class DGMPassthroughSeparateClassifier(DGMPassthrough):
    def __init__(self, dims: Tuple[int, int, int, int, int, Tuple[int, ...]],
                 Encode: Type[Encoder] = Encoder,
                 Decode: Type[DecoderPassthrough] = DecoderPassthrough,
                 Classify: Type[Classifier] = Classifier,
                 kld: KLDLoss=None) -> None:
        """
        M2 code replication from the paper
        'Semi-Supervised Learning with Deep Generative Models'
        (Kingma 2014) in PyTorch.

        The "Generative semi-supervised model" is a probabilistic
        model that incorporates label information in both
        inference and generation.

        Initialise a new generative model
        :param dims: dimensions of x, y, z, classifier_dim, x_out_dim, hidden_layers_dims.
        """
        (x_dim, y_dim, z_dim, cls_dim, x_out_dim, h_dims) = dims
        with self.disable_final_init():
            super(DGMPassthroughSeparateClassifier, self).__init__(dims=(x_dim, y_dim, z_dim, h_dims),
                                                                   Encode=Encode, Decode=Decode, Classify=Classify,
                                                                   kld=kld)
        self.classifier = Classify((cls_dim, (h_dims[0],), y_dim))
        self.decoder = Decode((z_dim + self.y_dim, tuple(reversed(h_dims)), x_out_dim))
        self.__final_init__()


class ADGMPassthrough(AuxiliaryDeepGenerativeModel):
    decoder: DecoderPassthrough

    def __init__(self, dims: Tuple[int, int, int, int, Tuple[int, ...]],
                 Encode: Type[Encoder]=Encoder,
                 Decode: Type[DecoderPassthrough]=DecoderPassthrough,
                 AuxEncoder: Type[Encoder]=Encoder,
                 AuxDecoder: Type[Encoder]=Encoder,
                 Classify: Type[Classifier]=Classifier,
                 kld: KLDLoss=None, kld_a: KLDLoss=None) -> None:
        super(ADGMPassthrough, self).__init__(dims=dims, Encode=Encode, Decode=Decode, AuxEncoder=AuxEncoder,
                                              AuxDecoder=AuxDecoder, Classify=Classify, kld=kld, kld_a=kld_a)

    def forward_(self, x: Tensor, y: Tensor) -> XYDictStrZ:
        self.decoder.set_passthr_x(x)
        return super(ADGMPassthrough, self).forward_(x=x, y=y)


class LadderDeepGenerativeModel(DeepGenerativeModel):
    encoder: List[LadderEncoder]  # type: ignore
    decoder: List[LadderDecoder]  # type: ignore
    reconstruction: Decoder
    classifier: Classifier

    def __init__(self, dims: Tuple[int, int, Tuple[int, ...], Tuple[int, ...]],
                 LadderEncode: Type[LadderEncoder]=LadderEncoder,
                 LadderDecode: Type[LadderDecoder]=LadderDecoder,
                 Classify: Type[Classifier]=Classifier,
                 Decode: Type[Decoder]=Decoder,
                 kld: LadderKLDLoss=None):
        """
        Ladder version of the Deep Generative Model.
        Uses a hierarchical representation that is
        trained end-to-end to give very nice disentangled
        representations.

        :param dims: dimensions of x, y, z layers and h layers
            note that len(z) == len(h).
        """
        x_dim, y_dim, z_dim, h_dim = dims
        with self.disable_final_init():
            super(LadderDeepGenerativeModel, self).__init__((x_dim, y_dim, z_dim[0], h_dim), kld=kld)

        neurons: List[int] = [x_dim, *h_dim]
        encoder_layers = [LadderEncode((neurons[i - 1], neurons[i], z_dim[i - 1])) for i in range(1, len(neurons))]

        e = encoder_layers[-1]
        encoder_layers[-1] = LadderEncode((e.in_features + y_dim, e.out_features, e.z_dim))

        decoder_layers = [LadderDecode((z_dim[i - 1], h_dim[i - 1], z_dim[i])) for i in range(1, len(h_dim))][::-1]

        self.classifier = Classify((x_dim, (h_dim[0],), y_dim))

        # noinspection PyTypeChecker
        self.encoder = nn.ModuleList(encoder_layers)  # type: ignore
        # noinspection PyTypeChecker
        self.decoder = nn.ModuleList(decoder_layers)  # type: ignore
        self.reconstruction = Decode((z_dim[0] + y_dim, h_dim, x_dim))
        self.__final_init__()

    def set_kld(self, kld: KLDLoss):
        assert isinstance(kld, LadderKLDLoss) and kld.closed_form
        self._kld = kld

    def forward_ldgm_to_z(self, x: Tensor, y: Tensor) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        # Gather latent representation
        # from encoders along with final z.
        latents: List[Tuple[Tensor, Tensor]] = []
        for i, encoder in enumerate(self.encoder):
            if i == (len(self.encoder) - 1):
                x, (z, (μ, log_σ)) = encoder.__call__(tr.cat([x, y], dim=1))
            else:
                x, (z, (μ, log_σ)) = encoder.__call__(x)
            latents.append((μ, log_σ))
        # noinspection PyUnboundLocalVariable
        z = z

        latents = list(reversed(latents))
        return z, latents

    def forward_ldgm_to_x(self, z: Tensor, latents: List[Tuple[Tensor, Tensor]], y: Tensor) -> XYDictStrZ:
        # If at top, encoder == decoder, use prior for KL:
        l_μ, l_log_σ = latents[0]
        kld, z, _ = self.kld.__call__(z, (l_μ, l_log_σ))
        kl_divergence = kld

        # Perform downward merge of information:
        h: Tensor
        for i, decoder in enumerate(self.decoder):
            l_μ, l_log_σ = latents[i + 1]
            h, (q_z, q_params, p_params) = decoder.__call__(z, l_μ, l_log_σ)
            kld, _, _ = self.kld.__call__(q_z, q_params, p_params)
            kl_divergence += kld
            z = h

        x_rec = self.reconstruction.__call__(tr.cat([z, y], dim=1))
        return x_rec, kl_divergence, {}

    def encode(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        z, latents = self.forward_ldgm_to_z(x, y)
        return self.kld.flow_qz_x(z), latents[0]

    def forward_(self, x: Tensor, y: Tensor) -> XYDictStrZ:
        z, latents = self.forward_ldgm_to_z(x, y)
        return self.forward_ldgm_to_x(z, latents, y)

    def sample(self, z: Tensor, y: Tensor, use_pz_flow: bool=True) -> Tensor:
        z = self.kld.flow_pz(z) if use_pz_flow else z
        h: Tensor
        for decoder in self.decoder:
            h, _, _ = decoder.sample2(z)
            z = h
        return self.reconstruction.__call__(tr.cat([z, y], dim=1))
