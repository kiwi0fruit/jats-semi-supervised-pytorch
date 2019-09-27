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
from typing import List, Tuple
import torch as tr
from torch import Tensor
from kiwi_bugfix_typechecker import nn
from kiwi_bugfix_typechecker.distrib import Distribution
from .flows import Flow
from .types import ModuleXToXYZ


class NormalizingFlowModel(ModuleXToXYZ):
    flows: List[Flow]

    def __init__(self, prior: Distribution, flows: List[Flow]):
        super().__init__()
        self.prior = prior
        flows_ = nn.ModuleList(flows)
        # noinspection PyTypeChecker
        self.flows = flows_

    def forward_(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        m, _ = x.shape
        log_det = tr.zeros(m).to(dtype=x.dtype, device=x.device)
        for flow in self.flows:
            x, ld = flow.__call__(x)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x)
        return z, prior_logprob, log_det

    def backward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        m, _ = z.shape
        log_det = tr.zeros(m).to(dtype=z.dtype, device=z.device)
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples: int) -> Tensor:
        z = self.prior.sample((n_samples,))
        x, _ = self.backward(z)
        return x
