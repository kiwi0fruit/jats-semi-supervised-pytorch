import math
from torch import Tensor
from torch.nn import Module
import torch as tr


def kernel(x: Tensor, y: Tensor) -> Tensor:
    """ Returns tensor of size (x.shape[0], y.shape[0]). """
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]

    x = x.view(x_size, 1, dim)
    y = y.view(1, y_size, dim)

    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)

    kernel_input: Tensor = ((tiled_x - tiled_y)**2).mean(dim=2) * (1. / dim)
    return tr.exp(-kernel_input)


def mmd(x: Tensor, y: Tensor) -> Tensor:
    return kernel(x, x).mean() + kernel(y, y).mean() - kernel(x, y).mean() * 2


class MMDNormalLoss(Module):
    mu: Tensor
    log_sigma: Tensor
    batch_size: int

    def __init__(self, batch_size: int = 128, mu: float = 0, sigma: float = 1):
        """
        RSamples from a Normal distribution using the reparameterization trick.
        ``self.forward(...)`` returns fully reduced scalar MMD loss.
        """
        super(MMDNormalLoss, self).__init__()
        self.batch_size = batch_size
        self.register_buffer('mu', tr.tensor([float(mu)]))
        self.register_buffer('log_sigma', tr.tensor([math.log(sigma)]))

    def rsample(self, z: Tensor) -> Tensor:
        batch_size, z_dim = z.shape
        batch_size = max(batch_size, self.batch_size)
        eps = tr.randn((batch_size, z_dim)).to(dtype=z.dtype, device=z.device)
        return tr.exp(self.log_sigma) * eps + self.mu

    def forward(self, z: Tensor) -> Tensor:
        return mmd(self.rsample(z), z)

    def extra_repr(self) -> str:
        return f'min_mmd_batch_size={self.batch_size}, ' + 'mu={:.3f}, sigma={:.3f}'.format(
            self.mu[0].item(), self.log_sigma.exp()[0].item())
