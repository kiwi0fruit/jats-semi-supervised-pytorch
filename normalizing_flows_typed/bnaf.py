"""
Code by Nicola De Cao was forked from https://github.com/nicola-decao/BNAF

MIT License

Copyright (c) 2019 Nicola De Cao, 2019 Peter Zagubisalo

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
from typing import Tuple, Optional as Opt, Union, Sequence as Seq, Dict, List
import math
import numpy as np
import torch as tr
from torch import Tensor
from torch.nn import init  # type: ignore
from kiwi_bugfix_typechecker import func, nn
from .flow import SequentialFlow
from .types import ModuleZToXY, SequentialZToXY, ModuleZOptJToXY


class Permutation(ModuleZToXY):
    p: Union[Seq[int], Tensor]
    zero: Tensor

    def __init__(self, in_features: int, p: Union[Seq[int], Tensor, str]=None):
        """
        Module that outputs a permutation of its input.

        Parameters
        ----------
        in_features :
            The number of input features.
        p :
            The list of indeces that indicate the permutation. When ``p`` is not a
            list, tuple or Tensor: if ``p = 'flip'`` the tensor is reversed, if ``p=None`` a random
            permutation is applied.
        """
        super(Permutation, self).__init__()
        self.in_features = in_features
        self.register_buffer('zero', tr.zeros(1))

        if p is None:
            self.p = [int(s) for s in np.random.permutation(in_features)]
        elif p == 'flip':
            self.p = list(reversed(range(in_features)))
        elif isinstance(p, (tuple, list)):
            self.p = [int(s) for s in p]
        elif isinstance(p, Tensor) and (tuple(p.size()) == (in_features,)):
            self.p = p.long()
        else:
            raise ValueError

    def forward_(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :return: The permuted tensor and the log-det-Jacobian of this permutation.
        """
        return z[:, self.p], self.zero

    def __repr__(self):
        return 'Permutation(in_features={}, p={})'.format(self.in_features, self.p)


class BNAF(SequentialZToXY):
    gate: Opt[nn.Parameter]
    _modules: Dict[str, ModuleZOptJToXY]

    def __init__(self, *args: ModuleZOptJToXY, res: str=None):
        """
        Class that extends ``torch.nn.Sequential`` for constructing a Block Neural
        Normalizing Flow.

        ``res=None`` is no residual connection, ``res='normal'`` is ``x + f(x)``
        and ``res='gated'`` is ``a * x + (1 - a) * f(x)`` where ``a`` is a learnable parameter.

        :param args: modules to use.
        :param res: Which kind of residual connection to use.
        """
        super(BNAF, self).__init__(*args)

        self.res = res

        if res == 'gated':
            self.gate = nn.Parameter(init.normal_(tr.empty(1)))
        else:
            self.gate = None

    def forward_(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :return: The output tensor and the log-det-Jacobian of this transformation.
        """
        z_out = z
        j: Opt[Tensor] = None

        for module in self._modules.values():
            z_out, j = module.__call__(z_out, j)
            j = j if len(j.shape) == 4 else j.view(j.shape + (1, 1))
        if j is not None:
            grad = j
        else:
            raise ValueError('Presumably empty Sequential')

        if z.shape[-1] != z_out.shape[-1]:
            raise AssertionError

        if self.res == 'normal':
            return z + z_out, func.softplus(grad.squeeze()).sum(-1)
        if (self.res == 'gated') and (self.gate is not None):
            return (
                self.gate.sigmoid() * z_out + (-self.gate.sigmoid() + 1) * z,
                (func.softplus(grad.squeeze() + self.gate) - func.softplus(self.gate)).sum(-1)
            )
        return z_out, grad.squeeze().sum(-1)

    def _get_name(self):
        return 'BNAF(res={})'.format(self.res)


class MaskedWeight(ModuleZOptJToXY):
    mask_d: Tensor
    mask_o: Tensor

    def __init__(self, in_features: int, out_features: int, dim: int, bias: bool=True):
        """
        Module that implements a linear layer with block matrices with positive diagonal blocks.
        Moreover, it uses Weight Normalization (https://arxiv.org/abs/1602.07868) for stability.

        Parameters
        ----------
        in_features :
            The number of input features per each dimension ``dim``.
        out_features :
            The number of output features per each dimension ``dim``.
        dim :
            The number of dimensions of the input of the flow.
        bias :
            Whether to add a parametrizable bias.
        """
        super(MaskedWeight, self).__init__()
        self.in_features, self.out_features, self.dim = in_features, out_features, dim

        weight = tr.zeros(out_features, in_features)
        for i in range(dim):
            weight[
                i * out_features // dim:(i + 1) * out_features // dim,
                0:(i + 1) * in_features // dim
            ] = init.xavier_uniform_(tr.empty(out_features // dim, (i + 1) * in_features // dim))

        self._weight = nn.Parameter(weight)
        self._diag_weight = nn.Parameter(init.uniform_(tr.empty(out_features, 1)).log())

        self.bias = nn.Parameter(init.uniform_(
            tr.empty(out_features), -1 / math.sqrt(out_features), 1 / math.sqrt(out_features)
        )) if bias else 0

        mask_d = tr.zeros_like(weight)
        for i in range(dim):
            mask_d[i * (out_features // dim):(i + 1) * (out_features // dim),
                   i * (in_features // dim):(i + 1) * (in_features // dim)] = 1

        self.register_buffer('mask_d', mask_d)

        mask_o = tr.ones_like(weight)
        for i in range(dim):
            mask_o[i * (out_features // dim):(i + 1) * (out_features // dim),
                   i * (in_features // dim):] = 0

        self.register_buffer('mask_o', mask_o)

    def get_weights(self) -> Tuple[Tensor, Tensor]:
        """
        Computes the weight matrix using masks and weight normalization.
        It also compute the log diagonal blocks of it.
        """
        w = tr.exp(self._weight) * self.mask_d + self._weight * self.mask_o
        w_squared_norm = (w ** 2).sum(-1, keepdim=True)

        w = self._diag_weight.exp() * w / w_squared_norm.sqrt()
        wpl = self._diag_weight + self._weight - 0.5 * tr.log(w_squared_norm)

        return (
            w.t(),
            wpl.t()[self.mask_d.byte().t()].view(
                self.dim, self.in_features // self.dim, self.out_features // self.dim)
        )

    def forward_(self, z: Tensor, j: Tensor=None) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        z :
            ...
        j :
            The log diagonal block of the partial Jacobian of previous transformations.

        Returns
        -------
        ret :
            The output tensor and the log diagonal blocks of the partial log-Jacobian of previous
            transformations combined with this transformation.
        """
        w, wpl = self.get_weights()
        grad = wpl.transpose(-2, -1).unsqueeze(0).repeat(z.shape[0], 1, 1, 1)

        return (
            z.matmul(w) + self.bias,
            tr.logsumexp(grad.unsqueeze(-2) + j.transpose(-2, -1).unsqueeze(-3), -1) if (j is not None) else grad
        )

    def __repr__(self):
        return 'MaskedWeight(in_features={}, out_features={}, dim={}, bias={})'.format(
            self.in_features, self.out_features, self.dim, not isinstance(self.bias, int))


class Tanh(ModuleZOptJToXY, nn.Tanh):  # type: ignore
    """
    Class that extends ``torch.nn.Tanh`` additionally computing the log diagonal
    blocks of the Jacobian.
    """
    def forward_(self, z: Tensor, j: Tensor=None) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        z :
            ...
        j :
            The log diagonal blocks of the partial Jacobian of previous transformations.

        Returns
        -------
        ret :
            The output tensor and the log diagonal blocks of the partial log-Jacobian of previous
            transformations combined with this transformation.
        """

        grad = -2 * (z - math.log(2) + func.softplus(-2 * z))
        return tr.tanh(z), (grad.view(j.shape) + j) if (j is not None) else grad


class BNAFs(SequentialFlow):
    def __init__(self, dim: int, hidden_dim: int=10, flows_n: int=16, layers_n: int=1, res: str= 'gated'):
        """
        ``res=None`` is no residual connection, ``res='normal'`` is ``x + f(x)``
        and ``res='gated'`` is ``a * x + (1 - a) * f(x)`` where ``a`` is a learnable parameter.

        :param res: Which kind of residual connection to use.
        """
        flows = []
        for f in range(flows_n):
            layers: List[ModuleZOptJToXY] = []
            for _ in range(layers_n - 1):
                layers.append(MaskedWeight(dim * hidden_dim, dim * hidden_dim, dim=dim))
                layers.append(Tanh())

            layers.append(MaskedWeight(dim * hidden_dim, dim, dim=dim))
            flows.append(BNAF(*(
                    [MaskedWeight(dim, dim * hidden_dim, dim=dim)] + [Tanh()] + layers
                ), res=res if f < (flows_n - 1) else None
            ))

            if f < (flows_n - 1):
                flows.append(Permutation(dim, 'flip'))

        super(BNAFs, self).__init__(*flows)

    def backward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError
