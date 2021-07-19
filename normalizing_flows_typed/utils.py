"""
Code by Tony Duan was forked from https://github.com/tonyduan/normalizing-flows
Implementation of rational-quadratic splines in this file is taken from https://github.com/bayesiains/nsf

MIT License

Copyright (c) 2019 Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios, 2019 Tony Duan, 2019 Peter Zagubisalo

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
from typing import Tuple
import numpy as np
import torch as tr
from torch import Tensor
from torch.nn import functional as func
from kiwi_bugfix_typechecker import func as func_

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def searchsorted(bin_locations: Tensor, inputs: Tensor, ε=1e-6) -> Tensor:
    bin_locations[..., -1] += ε
    return tr.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1


def unconstrained_rqs(inputs: Tensor, unnormalized_widths: Tensor, unnormalized_heights: Tensor,
                      unnormalized_derivatives: Tensor, inverse: bool=False,
                      tail_bound: float=1., min_bin_width: float=DEFAULT_MIN_BIN_WIDTH,
                      min_bin_height: float=DEFAULT_MIN_BIN_HEIGHT,
                      min_derivative: float=DEFAULT_MIN_DERIVATIVE) -> Tuple[Tensor, Tensor]:

    inside_intvl_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_intvl_mask

    outputs = tr.zeros_like(inputs)
    logabsdet = tr.zeros_like(inputs)

    unnormalized_derivatives = func.pad(unnormalized_derivatives, pad=[1, 1])
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0

    outputs[inside_intvl_mask], logabsdet[inside_intvl_mask] = rqs(
        inputs=inputs[inside_intvl_mask],
        unnormalized_widths=unnormalized_widths[inside_intvl_mask, :],
        unnormalized_heights=unnormalized_heights[inside_intvl_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_intvl_mask, :],
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )
    return outputs, logabsdet


def rqs(inputs: Tensor, unnormalized_widths: Tensor, unnormalized_heights: Tensor,
        unnormalized_derivatives: Tensor, inverse: bool=False, left: float=0., right: float=1.,
        bottom: float=0., top: float=1., min_bin_width: float=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height: float=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative: float=DEFAULT_MIN_DERIVATIVE) -> Tuple[Tensor, Tensor]:

    if tr.min(inputs) < left or tr.max(inputs) > right:
        raise ValueError("Input outside domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = func.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = tr.cumsum(widths, dim=-1)
    cumwidths = func.pad(cumwidths, pad=[1, 0], mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + func_.softplus(unnormalized_derivatives)

    heights = func.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = tr.cumsum(heights, dim=-1)
    cumheights = func.pad(cumheights, pad=[1, 0], mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    Δ = heights / widths
    input_Δ = Δ.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)
    input_derivatives_plus_one = input_derivatives_plus_one[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = ((inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_Δ)
             + input_heights * (input_Δ - input_derivatives))
        b = (input_heights * input_derivatives - (inputs - input_cumheights)
             * (input_derivatives + input_derivatives_plus_one - 2 * input_Δ))
        c = - input_Δ * (inputs - input_cumheights)

        discriminant = b**2 - 4 * a * c
        if not (discriminant >= 0).all():
            raise AssertionError

        root = (2 * c) / (-b - tr.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        θ_1minθ = root * (-root + 1)
        denominator = input_Δ + ((input_derivatives + input_derivatives_plus_one - 2 * input_Δ) * θ_1minθ)
        derivative_numerator = (input_Δ ** 2) * (
                input_derivatives_plus_one * root**2 + 2 * input_Δ * θ_1minθ + input_derivatives * (-root + 1)**2)
        logabsdet = tr.log(derivative_numerator) - 2 * tr.log(denominator)
        return outputs, -logabsdet

    θ = (inputs - input_cumwidths) / input_bin_widths
    θ_1minθ = θ * (-θ + 1)

    numerator = input_heights * (input_Δ * θ**2 + input_derivatives * θ_1minθ)
    denominator = input_Δ + ((input_derivatives + input_derivatives_plus_one - 2 * input_Δ) * θ_1minθ)
    outputs = input_cumheights + numerator / denominator

    derivative_numerator = (input_Δ ** 2) * (
            input_derivatives_plus_one * θ**2 + 2 * input_Δ * θ_1minθ + input_derivatives * (-θ + 1)**2)
    logabsdet = tr.log(derivative_numerator) - 2 * tr.log(denominator)
    return outputs, logabsdet
