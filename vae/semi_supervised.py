from typing import Callable, Tuple, Type
import torch as tr
from torch import Tensor
from semi_supervised_typed.layers import GaussianSample, BaseSample, GaussianSampleTrim
from semi_supervised_typed import Perceptron, Encoder, Decoder, LadderDecoder, LadderEncoder, Classifier


class PerceptronSELU(Perceptron):
    # noinspection PyUnusedLocal
    def __init__(self, dims: Tuple[int, ...], output_activation: Callable[[Tensor], Tensor]=None):
        super(PerceptronSELU, self).__init__(dims=dims, activation_fn=tr.selu, output_activation=None)


class EncoderSELU(Encoder):
    def __init__(self, dims: Tuple[int, Tuple[int, ...], int], sample_layer: Type[BaseSample]=GaussianSample):
        super(EncoderSELU, self).__init__(dims=dims, sample_layer=sample_layer, activation_fn=tr.selu)


class EncoderSELUTrim(EncoderSELU):
    def __init__(self, dims: Tuple[int, Tuple[int, ...], int]):
        super(EncoderSELUTrim, self).__init__(dims=dims, sample_layer=GaussianSampleTrim)


class DecoderSELU(Decoder):
    def __init__(self, dims: Tuple[int, Tuple[int, ...], int]):
        super(DecoderSELU, self).__init__(dims=dims, activation_fn=tr.selu, output_activation=None)


class LadderEncoderSELU(LadderEncoder):
    def __init__(self, dims: Tuple[int, int, int]):
        super(LadderEncoderSELU, self).__init__(dims=dims)

    def act(self, x: Tensor) -> Tensor:
        return tr.selu(x)


class LadderDecoderSELU(LadderDecoder):
    def __init__(self, dims: Tuple[int, int, int]):
        super(LadderDecoderSELU, self).__init__(dims=dims)

    def act1(self, z: Tensor) -> Tensor:
        return tr.selu(z)

    def act2(self, z: Tensor) -> Tensor:
        return tr.selu(z)


class ClassifierSELU(Classifier):
    def __init__(self, dims: Tuple[int, int, int]):
        super(ClassifierSELU, self).__init__(dims=dims, activation_fn=tr.selu)
