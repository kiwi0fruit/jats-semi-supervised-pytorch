from typing import Tuple, Type, Optional as Opt
from abc import abstractmethod
import torch as tr
from torch import Tensor

from beta_tcvae_typed import KLDLoss
from .svi import SVI
from .vae import Encoder, Decoder, VariationalAutoencoder, DecoderPassthrough, VAEPassthrough
from .dgm import Classifier


class VAEClassifyMeta:
    classifier: Classifier
    svi: Opt[SVI]
    use_svi: bool
    α: float = 0.4  # 0.4

    def __init__(self, classifier: Classifier) -> None:
        self.classifier = classifier
        self.svi = None
        self.use_svi = False

    def set_svi(self, svi: SVI=None) -> None:
        self.svi = svi

    @abstractmethod
    def get_z_μ(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def classify(self, x: Tensor, y: Tensor=None) -> Tuple[Tensor, Tensor]:
        """ returns probs, cross_entropy """
        z, μ = self.get_z_μ(x)
        if (self.svi is not None) and self.use_svi:
            probs_z, cross_entropy_rnd = self.svi.model.classify(z, y)
            _, cross_entropy_det = self.svi.model.classify(μ, y)
        else:
            probs_z, cross_entropy_rnd = self.classifier.__call__(z, y)
            _, cross_entropy_det = self.classifier.__call__(μ, y)
        return probs_z, cross_entropy_det * self.α + cross_entropy_rnd * (1 - self.α)

    def classify_deterministic(self, x: Tensor, y: Tensor=None) -> Tuple[Tensor, Tensor]:
        """ returns probs, cross_entropy """
        _, μ = self.get_z_μ(x)
        if (self.svi is not None) and self.use_svi:
            return self.svi.model.classify(μ, y)
        return self.classifier.__call__(μ, y)


class VAEClassify(VariationalAutoencoder, VAEClassifyMeta):
    def __init__(self, dims: Tuple[int, int, Tuple[int, ...], int, Tuple[int, ...]],
                 Encode: Type[Encoder]=Encoder,
                 Decode: Type[Decoder]=Decoder,
                 Classify: Type[Classifier]=Classifier,
                 kld: KLDLoss=None):
        """ x_dim, z_dim, h_dim, y_dim, hy_dims = dims """
        x_dim, z_dim, h_dim, y_dim, hy_dims = dims
        super(VAEClassify, self).__init__(dims=(x_dim, z_dim, h_dim), Encode=Encode, Decode=Decode, kld=kld)
        VAEClassifyMeta.__init__(self=self, classifier=Classify((z_dim, hy_dims, y_dim)))

    def get_z_μ(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z, (μ, _) = self.encoder.__call__(x)
        return self.kld.inv_flow_pz(self.kld.flow_qz_x(z)), self.kld.inv_flow_pz(self.kld.flow_qz_x(μ))


class VAEPassthroughClassify(VAEPassthrough, VAEClassifyMeta):
    def __init__(self, dims: Tuple[int, int, Tuple[int, ...], int, Tuple[int, ...]],
                 Encode: Type[Encoder]=Encoder,
                 Decode: Type[DecoderPassthrough]=DecoderPassthrough,
                 Classify: Type[Classifier]=Classifier,
                 kld: KLDLoss=None):
        """ x_dim, z_dim, h_dim, y_dim, hy_dims = dims """
        x_dim, z_dim, h_dim, y_dim, hy_dims = dims
        with self.disable_final_init():
            super(VAEPassthroughClassify, self).__init__(dims=(x_dim, z_dim, h_dim), Encode=Encode, Decode=Decode,
                                                         kld=kld)
        VAEClassifyMeta.__init__(self=self, classifier=Classify((z_dim + self.decoder.passthr_dim, hy_dims, y_dim)))
        self.__final_init__()

    def get_z_μ(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z, (μ, _) = self.encoder.__call__(x)
        s = x[:, :self.decoder.passthr_dim]
        z, μ = self.kld.inv_flow_pz(self.kld.flow_qz_x(z)), self.kld.inv_flow_pz(self.kld.flow_qz_x(μ))
        return tr.cat((s, z), dim=1), tr.cat((s, μ), dim=1)
