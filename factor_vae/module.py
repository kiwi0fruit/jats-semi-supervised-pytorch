from typing import Tuple, Callable
import torch as tr
from torch import Tensor
from torch.nn import Module
from kiwi_bugfix_typechecker import test_assert
from semi_supervised_typed import Perceptron, VariationalAutoencoder
from .types import BaseDiscriminator

δ = 1e-8
test_assert()


class Discriminator(BaseDiscriminator):
    perceptron: Perceptron

    def __init__(self, dims: Tuple[int, Tuple[int, ...]], activation_fn: Callable[[Tensor], Tensor]=tr.relu):
        """
        Discriminator from [1] to compute the Factor-VAE loss as per Algorithm 2 of [1].

        ``dims`` = z_dim, (h_dim1, ...)

        Hidden layer classifier with softmax output.

        [1] References:
            Hyunjik Kim, Andriy Mnih; Disentangling by Factorising (2018)
            https://arxiv.org/abs/1802.05983
        """
        super(Discriminator, self).__init__()
        z_dim, h_dims = dims
        self.perceptron = Perceptron(dims=(z_dim, *h_dims, 2), activation_fn=activation_fn, output_activation=None)

    @staticmethod
    def permute_dims(z: Tensor) -> Tensor:
        """
        Implementation of Algorithm 1 in ref [1]. Randomly permutes the sample from
        q(z) (latent_dist) across the batch for each of the latent dimensions (mean
        and log_σ).

        :param z:
            sample from the latent dimension using the reparameterisation trick
            shape: (batch_size, z_dim).
        """
        batch_size, z_dim = z.shape
        permuted_dims = []
        for i in range(z_dim):
            permuted_dims.append(z[:, i][tr.randperm(batch_size)])
        z_perm = tr.stack(permuted_dims, dim=1)
        return z_perm

    def forward_(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """ See ``self.__call__`` """
        logits_d_z = self.perceptron.__call__(z)
        d_z = tr.softmax(logits_d_z, dim=-1)
        tc = logits_d_z[:, 0] - logits_d_z[:, 1]
        return tc, d_z

    def tc_discr(self, z2: Tensor, d_z1: Tensor) -> Tensor:
        """
        :return: tc_discr of shape (batch_size,)
        """
        assert z2.shape[0] == d_z1.shape[0]
        logits_d_z2_perm = self.perceptron.__call__(self.permute_dims(z2))
        d_z2_perm = tr.softmax(logits_d_z2_perm, dim=-1)

        return (tr.log(d_z1[:, 0] + δ) + tr.log(d_z2_perm[:, 1] + δ)) * (-0.5)

    @staticmethod
    def skip_iter(x1: Tensor, x2: Tensor) -> bool:
        return x1.shape[0] != x2.shape[0]


class FactorVAEContainer(Module):  # pylint: disable=abstract-method
    def __init__(self, vae: VariationalAutoencoder, discriminator: Discriminator):
        super(FactorVAEContainer, self).__init__()
        self.vae = vae
        self.discriminator = discriminator
