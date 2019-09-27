from typing import List, Tuple, Callable, Type
import torch as tr
from torch import Tensor

from kiwi_bugfix_typechecker import nn
from beta_tcvae_typed import Verbose
from .vae_types import ModuleXToX
from .vae import Encoder, Decoder, LadderEncoder, LadderDecoder, VariationalAutoencoder, Act


class Classifier(ModuleXToX):
    dense: nn.Linear
    logits: nn.Linear
    activation_fn: Act

    def __init__(self, dims: Tuple[int, int, int], activation_fn: Callable[[Tensor], Tensor]=tr.relu):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(Classifier, self).__init__()
        x_dim, h_dim, y_dim = dims
        self.dense = nn.Linear(x_dim, h_dim)
        self.logits = nn.Linear(h_dim, y_dim)
        self.activation_fn = Act(activation_fn)

    def forward_(self, x: Tensor) -> Tensor:
        x = self.activation_fn.a(self.dense.__call__(x))
        x = tr.softmax(self.logits.__call__(x), dim=-1)
        return x


class DeepGenerativeModel(VariationalAutoencoder):
    encoder: Encoder
    decoder: Decoder
    classifier: Classifier

    def __init__(self, dims: Tuple[int, int, int, Tuple[int, ...]],
                 Encode: Type[Encoder]=Encoder,
                 Decode: Type[Decoder]=Decoder,
                 Classify: Type[Classifier]=Classifier):
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
        super(DeepGenerativeModel, self).__init__((x_dim, z_dim, h_dim))

        self.encoder = Encode((x_dim + self.y_dim, h_dim, z_dim))
        self.decoder = Decode((z_dim + self.y_dim, tuple(reversed(h_dim)), x_dim))
        self.classifier = Classify((x_dim, h_dim[0], self.y_dim))

    def forward_(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        # Add label and data and generate latent variable
        z, qz_params = self.encoder.__call__(tr.cat([x, y], dim=1))

        kld, z = self._kld(z, qz_params)

        if self._save_qz_params:
            self._qz_params = (z, qz_params)

        # Reconstruct data point from latent data and label
        x_rec = self.decoder.__call__(tr.cat([z, y], dim=1))
        return x_rec, kld

    def classify(self, x: Tensor) -> Tensor:
        """ returns probs """
        probs = self.classifier.__call__(x)
        return probs

    # noinspection PyMethodOverriding
    def sample(self, z: Tensor, y: Tensor) -> Tensor:  # pylint: disable=signature-differs
        """
        Samples from the Decoder to generate an x.
        :param z: latent normal variable
        :param y: label (one-hot encoded)
        :return: x
        """
        x = self.decoder.__call__(tr.cat([z, y], dim=1))
        return x


class StackedDeepGenerativeModel(DeepGenerativeModel):
    features: VariationalAutoencoder

    def __init__(self, dims: Tuple[int, int, int, Tuple[int, ...]], features: VariationalAutoencoder):
        """
        M1+M2 model as described in [Kingma 2014].

        Initialise a new stacked generative model
        :param dims: dimensions of x, y, z and hidden layers
        :param features: a pretrained M1 model of class `VariationalAutoencoder`
            trained on the same dataset.
        """
        x_dim, y_dim, z_dim, h_dim = dims
        super(StackedDeepGenerativeModel, self).__init__((features.z_dim, y_dim, z_dim, h_dim))

        # Be sure to reconstruct with the same dimensions
        in_features = self.decoder.reconstruction.in_features
        self.decoder.reconstruction = nn.Linear(in_features, x_dim)

        # Make vae feature model untrainable by freezing parameters
        self.features = features
        self.features.train(False)

        param: nn.Parameter
        for param in self.features.parameters():
            param.requires_grad = False

    def forward_(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        # Sample a new latent x from the M1 model
        x_sample, _ = self.features.encoder.__call__(x)

        # Use the sample as new input to M2
        return super(StackedDeepGenerativeModel, self).forward(x_sample, y)

    def classify(self, x: Tensor) -> Tensor:
        """ returns probs """
        fz, fqz_params = self.features.encoder.__call__(x)
        probs = self.classifier.__call__(fqz_params[0] if self.features.q_params_μ_first else fz)
        return probs


class AuxiliaryDeepGenerativeModel(DeepGenerativeModel, Verbose):
    aux_encoder: Encoder
    aux_decoder: Encoder
    classifier: Classifier
    encoder: Encoder
    decoder: Decoder
    λ: float
    stored: List[Tuple[Tensor, Tensor]]  # type: ignore

    def __init__(self, dims: Tuple[int, int, int, int, Tuple[int, ...], Tuple[int, ...]],
                 λ: float=1,
                 Encode: Type[Encoder]=Encoder,
                 Decode: Type[Decoder]=Decoder,
                 Classify: Type[Classifier]=Classifier):
        """
        Auxiliary Deep Generative Models [Maaløe 2016]
        code replication. The ADGM introduces an additional
        latent variable 'a', which enables the model to fit
        more complex variational distributions.

        When calculates KLD_a ``self._kld(..., unmod_kld=True)`` is called.

        KLD = λ * KLD_a + KLD_z

        :param dims: dimensions of x, y, z, a, hidden layers, aux hidden layers (from z_dim+y_dim to a_dim).
        :param λ: multiplier for KLD_a
        """
        x_dim, y_dim, z_dim, a_dim, h_dim, ha_dim = dims
        super(AuxiliaryDeepGenerativeModel, self).__init__((x_dim, y_dim, z_dim, h_dim))
        Verbose.__init__(self=self)

        self.aux_encoder = Encode((x_dim, h_dim, a_dim))
        self.aux_decoder = Encode((z_dim + y_dim, ha_dim, a_dim))

        self.classifier = Classify((x_dim + a_dim, h_dim[0], y_dim))

        self.encoder = Encode((a_dim + y_dim + x_dim, h_dim, z_dim))
        self.decoder = Decode((y_dim + z_dim, tuple(reversed(h_dim)), x_dim))
        self.λ = λ

    def stored_pop(self) -> None:
        """
        ``self.stored.pop()`` returns (kld_z, kld_a)
        """
        raise RuntimeError

    def set_λ(self, λ: float):
        self.λ = λ

    def classify(self, x: Tensor) -> Tensor:
        """ returns probs """
        # Auxiliary inference q(a|x)
        a, _ = self.aux_encoder.__call__(x)

        # Classification q(y|a,x)
        probs = self.classifier.__call__(tr.cat((x, a), dim=1))
        return probs

    def forward_(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
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

        kld_z, z_axy = self._kld(z_axy, qz_axy_params)

        if self._save_qz_params:
            self._qz_params = (z_axy, qz_axy_params)

        # Generative p(x|z,y)
        x_zy = self.decoder.__call__(tr.cat((z_axy, y), dim=1))

        # Generative p(a|z,y)
        _, pa_zy_params = self.aux_decoder.__call__(tr.cat((z_axy, y), dim=1))
        kld_a, _ = self._kld(a_x, qa_x_params, pa_zy_params, unmod_kld=True)

        if self.λ == 1:
            kld = kld_a + kld_z
        else:
            kld = kld_a * self.λ + kld_z
        if self._verbose:
            self.stored.append((kld_z, kld_a))
        return x_zy, kld


class LadderDeepGenerativeModel(DeepGenerativeModel):
    encoder: List[LadderEncoder]  # type: ignore
    decoder: List[LadderDecoder]  # type: ignore
    reconstruction: Decoder
    classifier: Classifier

    def __init__(self, dims: Tuple[int, int, Tuple[int, ...], Tuple[int, ...]],
                 LadderEncode: Type[LadderEncoder] = LadderEncoder,
                 LadderDecode: Type[LadderDecoder] = LadderDecoder,
                 Classify: Type[Classifier] = Classifier,
                 Decode: Type[Decoder] = Decoder):
        """
        Ladder version of the Deep Generative Model.
        Uses a hierarchical representation that is
        trained end-to-end to give very nice disentangled
        representations.

        :param dims: dimensions of x, y, z layers and h layers
            note that len(z) == len(h).
        """
        x_dim, y_dim, z_dim, h_dim = dims
        super(LadderDeepGenerativeModel, self).__init__((x_dim, y_dim, z_dim[0], h_dim))

        neurons: List[int] = [x_dim, *h_dim]
        encoder_layers = [LadderEncode((neurons[i - 1], neurons[i], z_dim[i - 1])) for i in range(1, len(neurons))]

        e = encoder_layers[-1]
        encoder_layers[-1] = LadderEncode((e.in_features + y_dim, e.out_features, e.z_dim))

        decoder_layers = [LadderDecode((z_dim[i - 1], h_dim[i - 1], z_dim[i])) for i in range(1, len(h_dim))][::-1]

        self.classifier = Classify((x_dim, h_dim[0], y_dim))

        # noinspection PyTypeChecker
        self.encoder = nn.ModuleList(encoder_layers)
        # noinspection PyTypeChecker
        self.decoder = nn.ModuleList(decoder_layers)
        self.reconstruction = Decode((z_dim[0] + y_dim, h_dim, x_dim))

    def forward_(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        # Gather latent representation
        # from encoders along with final z.
        latents: List[Tuple[Tensor, Tensor]] = []
        for i, encoder in enumerate(self.encoder):
            if i == len(self.encoder)-1:
                x, (z, (μ, log_σ)) = encoder.__call__(tr.cat([x, y], dim=1))
            else:
                x, (z, (μ, log_σ)) = encoder.__call__(x)
            latents.append((μ, log_σ))
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
        h: Tensor
        for i, decoder in enumerate(self.decoder):
            l_μ, l_log_σ = latents[i + 1]
            h, (q_z, q_params, p_params) = decoder.__call__(z, l_μ, l_log_σ)
            kld, _ = self._kld(q_z, q_params, p_params, unmod_kld=True)
            kl_divergence += kld
            z = h

        x_rec = self.reconstruction.__call__(tr.cat([z, y], dim=1))
        return x_rec, kl_divergence

    def sample(self, z: Tensor, y: Tensor) -> Tensor:
        h: Tensor
        for decoder in self.decoder:
            h, _, _ = decoder.sample2(z)
            z = h
        return self.reconstruction.__call__(tr.cat([z, y], dim=1))
