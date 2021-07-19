from abc import abstractmethod
from typing import Tuple
import torch as tr
from torch import Tensor
from torch.nn import Module


class BaseDiscriminator(Module):
    # noinspection PyMissingConstructor
    def __init__(self): ...
    @abstractmethod
    def forward_(self, z: Tensor) -> Tuple[Tensor, Tensor]: ...
    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]: ...  # type: ignore
    def __call__(self, z: Tensor) -> Tuple[Tensor, Tensor]: # type: ignore
        """
        returns TC = log[D(z) / (1 - D(z))] where D(z) is probability of sampled ~q(z)

        let P_q be probability of sampled ~q(z)
        let P_Πqi be probability of sampled ~Πqi(z)

        P_q = e^logit_q / (e^logit_q + e^logit_Πqi)
        P_Πqi = e^logit_Πqi / (e^logit_q + e^logit_Πqi)
        hence: log(P_q / P_Πqi) = logit_q - logit_Πqi

        :return: (tc, d_z);
            tc is of shape (batch_size,);
            d_z is of shape (batch_size, 2)
        """
        ...
