"""
Code by Jesper Wohlert was forked from https://github.com/wohlert/semi-supervised-pytorch
Code by Nicola De Cao was forked from https://github.com/nicola-decao/BNAF

MIT License

Copyright (c) 2017 Jesper Wohlert, 2019 Nicola De Cao, 2019 Peter Zagubisalo

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
from typing import Tuple, List, Dict, Union
from abc import abstractmethod
import torch as tr
from torch import Tensor

from kiwi_bugfix_typechecker import func, nn
from .types import ModuleZToXY, SequentialZToXY

δ = 1e-8
UModuleZToXY = Union[SequentialZToXY, ModuleZToXY]


# noinspection PyAbstractClass
class Flow(ModuleZToXY):
    @abstractmethod
    def backward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class NormalizingFlows(Flow):
    flows: List[Flow]

    def __init__(self, dim: int=None, flows: List[Flow]=None):
        """
        Presents a sequence of normalizing flows as a ``torch.nn.Module``.

        default value is [PlanarNormalizingFlow(dim) for i in range(16)]

        Forked from github.com/wohlert/semi-supervised-pytorch
        """
        super(NormalizingFlows, self).__init__()

        if (flows is None) and (dim is not None):
            flows_ = [PlanarNormalizingFlow(dim=dim) for _ in range(16)]
        elif flows:
            flows_ = flows
        else:
            raise ValueError('Either dim or non empty flows list should be provided.')

        flows__ = nn.ModuleList(flows_)
        # noinspection PyTypeChecker
        self.flows = flows__

    def forward_(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        z, sum_log_abs_det_jacobian = self.flows[0].__call__(z)

        for flow in self.flows[1:]:
            z, j = flow.__call__(z)
            sum_log_abs_det_jacobian = sum_log_abs_det_jacobian + j

        return z, sum_log_abs_det_jacobian

    def backward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        z, sum_log_abs_det_jacobian = self.flows[-1].backward(z)

        for flow in reversed(self.flows[:-1]):
            z, j = flow.backward(z)
            sum_log_abs_det_jacobian = sum_log_abs_det_jacobian + j

        return z, sum_log_abs_det_jacobian


class SequentialFlow(SequentialZToXY):
    _modules: Dict[str, UModuleZToXY]

    def __init__(self, *args: UModuleZToXY):  # pylint: disable=useless-super-delegation
        """
        Class that extends ``torch.nn.Sequential`` for computing the output of
        the function alongside with the log-det-Jacobian of such transformation.

        Forked from github.com/nicola-decao/BNAF
        """
        super(SequentialFlow, self).__init__(*args)

    def forward_(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :return: The output tensor and the log-det-Jacobian of this transformation.
        """
        log_det_jacobian: Union[int, Tensor] = 0
        for module in self._modules.values():
            fz, log_det_jacobian_ = module.__call__(z)
            log_det_jacobian = log_det_jacobian_ + log_det_jacobian
            z = fz
        if not isinstance(log_det_jacobian, int):
            return z, log_det_jacobian
        raise RuntimeError('Presumably empty Sequential')

    @abstractmethod
    def backward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class PlanarNormalizingFlow(Flow):
    def __init__(self, dim: int):
        """
        Planar normalizing flow [Rezende & Mohamed 2015].
        Provides a tighter bound on the ELBO by giving more expressive
        power to the approximate distribution, such as by introducing
        covariance between terms.

        Forked from github.com/wohlert/semi-supervised-pytorch
        """
        super(PlanarNormalizingFlow, self).__init__()
        self.dim = dim
        self.u = nn.Parameter(tr.randn(dim))
        self.w = nn.Parameter(tr.randn(dim))
        self.b = nn.Parameter(tr.ones(1))

    def forward_(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        u, w, b = self.u, self.w, self.b

        # Create uhat such that it is parallel to w
        uw = tr.dot(u, w)
        u_hat = u + (func.softplus(uw) - 1 - uw) * tr.transpose(w, 0, -1) / tr.sum(w**2)
        # m(uw) == softplus(uw) - 1 == log(1 + exp(uw)) - 1

        # Equation 21 - Transform z
        zw__b = tr.mv(z, vec=w) + b  # z @ w + b
        fz = z + (u_hat.view(1, -1) * tr.tanh(zw__b).view(-1, 1))

        # Compute the Jacobian using the fact that
        # tanh(x) dx = 1 - tanh(x)**2
        ψ = (-tr.tanh(zw__b)**2 + 1).view(-1, 1) * w.view(1, -1)
        ψu = tr.mv(ψ, vec=u_hat)  # ψ @ u_hat

        # Return the transformed output along with log determninant of J
        logabs_detjacobian = tr.log(tr.abs(ψu + 1) + δ)

        return fz, logabs_detjacobian

    def backward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError
