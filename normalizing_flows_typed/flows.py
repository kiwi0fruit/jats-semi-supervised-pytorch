"""
Code by Tony Duan was forked from https://github.com/tonyduan/normalizing-flows

MIT License

Copyright (c) 2019 Tony Duan, 2019 Peter Zagubisalo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from typing import Dict, Callable, Tuple, Type, List, Optional as Opt
import math
import numpy as np
import scipy as sp
import torch as tr
from torch import Tensor
from torch.nn import init  # type: ignore
from kiwi_bugfix_typechecker import nn, func
from .utils import unconstrained_rqs
from .flow import Flow
from .types import ModuleXToX

ε = 1e-8
# supported non-linearities: note that the function must be invertible


def dtanhxdx(x: Tensor) -> Tensor:
    return -tr.pow(tr.tanh(x), 2) + 1


def dleakyreluxdx(x: Tensor) -> Tensor:
    return (x > 0).to(dtype=tr.float) + (x < 0).to(dtype=tr.float) * -0.01


def deluxdx(x: Tensor) -> Tensor:
    return (x > 0).type(tr.float) + (x < 0).type(tr.float) * tr.exp(x)


XToX = Callable[[Tensor], Tensor]
functional_derivatives: Dict[XToX, XToX] = {tr.tanh: dtanhxdx, func.leaky_relu: dleakyreluxdx, func.elu: deluxdx}


class Planar(Flow):
    """
    Planar flow.

        z_out = f(z) = z + u h(wᵀz + b)

    [Rezende and Mohamed, 2015]
    """
    def __init__(self, dim: int, nonlinearity: XToX=tr.tanh):
        super().__init__()
        self.dim = dim
        self.h = nonlinearity
        self.w = nn.Parameter(tr.zeros(dim))
        self.u = nn.Parameter(tr.zeros(dim))
        self.b = nn.Parameter(tr.zeros(1))
        self.reset_parameters(dim)

    def reset_parameters(self, dim: int):
        init.uniform_(self.w, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.u, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.b, -math.sqrt(1/dim), math.sqrt(1/dim))

    def forward_(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Given z, returns z_out and the log-determinant log|df/dx|.

        Returns
        -------
        """
        w: Tensor = self.w
        u: Tensor = self.u
        if self.h in (func.elu, func.leaky_relu):
            pass
        elif self.h == tr.tanh:
            scal = tr.log(1 + tr.exp(w @ u)) - w @ u - 1
            u = u + scal * w / func.norm(w)
        else:
            raise NotImplementedError("This non-linearity is not supported.")
        lin = tr.unsqueeze(z @ w, 1) + self.b
        z_out = z + u * self.h(lin)
        ϕ = functional_derivatives[self.h](lin) * w
        log_det = tr.log(tr.abs(1 + ϕ @ u) + ε)
        return z_out, log_det

    def backward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError("Planar flow has no algebraic inverse.")


class Radial(Flow):
    """
    Radial flow.

        z_out = f(z) = z + β h(α, r)(z − z0)

    [Rezende and Mohamed 2015]
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.x0 = nn.Parameter(tr.zeros(dim))
        self.log_α = nn.Parameter(tr.zeros(1))
        self.β = nn.Parameter(tr.zeros(1))

    def reset_parameters(self, dim: int):
        init.uniform_(self.x0, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.log_α, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.β, -math.sqrt(1/dim), math.sqrt(1/dim))

    def forward_(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Given z, returns z_out and the log-determinant log|df/dx|.
        """
        log_α, x0 = self.log_α, self.x0

        _, n = z.shape
        r = func.norm(z - x0)
        h = (tr.exp(log_α) + r)**-1
        β = -tr.exp(log_α) + tr.log(1 + tr.exp(self.β))
        z_out = z + β * h * (z - x0)
        log_det = (n - 1) * tr.log(1 + β * h) + tr.log(1 + β * h - β * r / (tr.exp(log_α) + r)**2)
        return z_out, log_det

    def backward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class FCNN(ModuleXToX):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward_(self, x: Tensor) -> Tensor:
        return self.network.__call__(x)


class RealNVP(Flow):
    """
    Non-volume preserving flow.

    [Dinh et. al. 2017]
    """
    def __init__(self, dim: int, hidden_dim: int=8, base_network: Type[FCNN]=FCNN):
        super().__init__()
        self.dim = dim
        self.t1 = base_network(dim // 2, dim // 2, hidden_dim)
        self.s1 = base_network(dim // 2, dim // 2, hidden_dim)
        self.t2 = base_network(dim // 2, dim // 2, hidden_dim)
        self.s2 = base_network(dim // 2, dim // 2, hidden_dim)

    def forward_(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]
        t1_transformed = self.t1.__call__(lower)
        s1_transformed = self.s1.__call__(lower)
        upper = t1_transformed + upper * tr.exp(s1_transformed)
        t2_transformed = self.t2.__call__(upper)
        s2_transformed = self.s2.__call__(upper)
        lower = t2_transformed + lower * tr.exp(s2_transformed)
        z_out = tr.cat([lower, upper], dim=1)
        log_det = tr.sum(s1_transformed, dim=1) + tr.sum(s2_transformed, dim=1)
        return z_out, log_det

    def backward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]
        t2_transformed = self.t2.__call__(upper)
        s2_transformed = self.s2.__call__(upper)
        lower = (lower - t2_transformed) * tr.exp(-s2_transformed)
        t1_transformed = self.t1.__call__(lower)
        s1_transformed = self.s1.__call__(lower)
        upper = (upper - t1_transformed) * tr.exp(-s1_transformed)
        x = tr.cat([lower, upper], dim=1)
        log_det = tr.sum(-s1_transformed, dim=1) + tr.sum(-s2_transformed, dim=1)
        return x, log_det


class MAF(Flow):
    """
    Masked auto-regressive flow.

    [Papamakarios et al. 2018]
    """
    layers: List[FCNN]

    def __init__(self, dim: int, hidden_dim: int=8, base_network: Type[FCNN]=FCNN):
        super().__init__()
        self.dim = dim
        # noinspection PyTypeChecker
        self.layers = nn.ModuleList()
        self.initial_param = nn.Parameter(tr.zeros(2))
        for i in range(1, dim):
            self.layers += [base_network(i, 2, hidden_dim)]
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.initial_param, -math.sqrt(0.5), math.sqrt(0.5))

    def forward_(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        z_out = tr.zeros_like(z).to(dtype=z.dtype, device=z.device)
        log_det = tr.zeros(z_out.shape[0]).to(dtype=z.dtype, device=z.device)
        for i in range(self.dim):
            if i == 0:
                μ, α = self.initial_param[0], self.initial_param[1]
            else:
                out = self.layers[i - 1].__call__(z[:, :i])
                μ, α = out[:, 0], out[:, 1]
            z_out[:, i] = (z[:, i] - μ) / tr.exp(α)
            log_det -= α
        return z_out.flip(dims=(1,)), log_det

    def backward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        z_out = tr.zeros_like(z).to(dtype=z.dtype, device=z.device)
        log_det = tr.zeros(z.shape[0]).to(dtype=z.dtype, device=z.device)
        z = z.flip(dims=(1,))
        for i in range(self.dim):
            if i == 0:
                μ, α = self.initial_param[0], self.initial_param[1]
            else:
                out = self.layers[i - 1].__call__(z_out[:, :i])
                μ, α = out[:, 0], out[:, 1]
            z_out[:, i] = μ + tr.exp(α) * z[:, i]
            log_det += α
        return z_out, log_det


class ActNorm(Flow):
    """
    ActNorm layer.

    [Kingma and Dhariwal, 2018.]
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.μ = nn.Parameter(tr.zeros(dim, dtype=tr.float))
        self.log_σ = nn.Parameter(tr.zeros(dim, dtype=tr.float))

    def forward_(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        z_out = z * tr.exp(self.log_σ) + self.μ
        log_det = tr.sum(self.log_σ)
        return z_out, log_det

    def backward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        z_out = (z - self.μ) / tr.exp(self.log_σ)
        log_det = -tr.sum(self.log_σ)
        return z_out, log_det


class OneByOneConv(Flow):
    """
    Invertible 1x1 convolution.

    [Kingma and Dhariwal, 2018.]
    """
    P: Tensor
    W_inv: Opt[Tensor]

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        W, _ = sp.linalg.qr(np.random.randn(dim, dim))
        P, L, U = sp.linalg.lu(W)
        self.register_buffer('P', tr.tensor(P, dtype=tr.float))
        self.L = nn.Parameter(tr.tensor(L, dtype=tr.float))
        self.S = nn.Parameter(tr.tensor(np.diag(U), dtype=tr.float))
        self.U = nn.Parameter(tr.triu(tr.tensor(U, dtype=tr.float), diagonal=1))
        self.W_inv = None

    def forward_(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        L = tr.tril(self.L, diagonal=-1) + tr.diag(tr.ones(self.dim))
        U = tr.triu(self.U, diagonal=1)
        z_out = z @ self.P @ L @ (U + tr.diag(self.S))
        log_det = tr.sum(tr.log(tr.abs(self.S)))
        return z_out, log_det

    def backward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        if self.W_inv is None:
            L = tr.tril(self.L, diagonal=-1) + tr.diag(tr.ones(self.dim))
            U = tr.triu(self.U, diagonal=1)
            W = self.P @ L @ (U + tr.diag(self.S))
            W_inv = tr.inverse(W)
            self.W_inv = W_inv
        else:
            W_inv = self.W_inv
        z_out = z @ W_inv
        log_det = -tr.sum(tr.log(tr.abs(self.S)))
        return z_out, log_det


class NSFAR(Flow):
    """
    Neural spline flow, auto-regressive.

    [Durkan et al. 2019]
    """
    layers: List[FCNN]

    def __init__(self, dim: int, K: int=5, B: int=3, hidden_dim: int=8, base_network: Type[FCNN]=FCNN):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        # noinspection PyTypeChecker
        self.layers = nn.ModuleList()
        self.init_param = nn.Parameter(tr.zeros(3 * K - 1))
        for i in range(1, dim):
            self.layers += [base_network(i, 3 * K - 1, hidden_dim)]
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.init_param, - 1 / 2, 1 / 2)

    def forward_(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        z_out = tr.zeros_like(z).to(dtype=z.dtype, device=z.device)
        log_det = tr.zeros(z_out.shape[0]).to(dtype=z.dtype, device=z.device)
        for i in range(self.dim):
            if i == 0:
                init_param = self.init_param.expand(z.shape[0], 3 * self.K - 1)
                W, H, D = init_param.split(self.K, dim=1)
            else:
                out = self.layers[i - 1].__call__(z[:, :i])
                W, H, D = out.split(self.K, dim=1)
            W, H = tr.softmax(W, dim=1), tr.softmax(H, dim=1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = func.softplus(D)
            z_out[:, i], ld = unconstrained_rqs(
                z[:, i], W, H, D, inverse=False, tail_bound=self.B)
            log_det += ld
        return z_out, log_det

    def backward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        z_out = tr.zeros_like(z).to(dtype=z.dtype, device=z.device)
        log_det = tr.zeros(z_out.shape[0]).to(dtype=z.dtype, device=z.device)
        for i in range(self.dim):
            if i == 0:
                init_param = self.init_param.expand(z_out.shape[0], 3 * self.K - 1)
                W, H, D = init_param.split(self.K, dim=1)
            else:
                out = self.layers[i - 1].__call__(z_out[:, :i])
                W, H, D = out.split(self.K, dim=1)
            W, H = tr.softmax(W, dim=1), tr.softmax(H, dim=1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = func.softplus(D)
            z_out[:, i], ld = unconstrained_rqs(
                z[:, i], W, H, D, inverse=True, tail_bound=self.B)
            log_det += ld
        return z_out, log_det


class NSFCL(Flow):
    """
    Neural spline flow, coupling layer.

    [Durkan et al. 2019]
    """
    def __init__(self, dim: int, K: int=5, B: int=3, hidden_dim: int=8, base_network: Type[FCNN]=FCNN):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.f1 = base_network(dim // 2, (3 * K - 1) * dim // 2, hidden_dim)
        self.f2 = base_network(dim // 2, (3 * K - 1) * dim // 2, hidden_dim)

    def forward_(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        log_det = tr.zeros(z.shape[0]).to(dtype=z.dtype, device=z.device)
        lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]
        out = self.f1.__call__(lower).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = out.split(self.K, dim=2)
        W, H = tr.softmax(W, dim=2), tr.softmax(H, dim=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = func.softplus(D)
        upper, ld = unconstrained_rqs(
            upper, W, H, D, inverse=False, tail_bound=self.B)
        log_det += tr.sum(ld, dim=1)
        out = self.f2.__call__(upper).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = out.split(self.K, dim=2)
        W, H = tr.softmax(W, dim=2), tr.softmax(H, dim=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = func.softplus(D)
        lower, ld = unconstrained_rqs(
            lower, W, H, D, inverse=False, tail_bound=self.B)
        log_det += tr.sum(ld, dim=1)
        return tr.cat([lower, upper], dim=1), log_det

    def backward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        log_det = tr.zeros(z.shape[0]).to(dtype=z.dtype, device=z.device)
        lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]
        out = self.f2.__call__(upper).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = out.split(self.K, dim=2)
        W, H = tr.softmax(W, dim=2), tr.softmax(H, dim=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = func.softplus(D)
        lower, ld = unconstrained_rqs(
            lower, W, H, D, inverse=True, tail_bound=self.B)
        log_det += tr.sum(ld, dim=1)
        out = self.f1.__call__(lower).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = out.split(self.K, dim=2)
        W, H = tr.softmax(W, dim=2), tr.softmax(H, dim=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = func.softplus(D)
        upper, ld = unconstrained_rqs(
            upper, W, H, D, inverse=True, tail_bound=self.B)
        log_det += tr.sum(ld, dim=1)
        return tr.cat([lower, upper], dim=1), log_det
