from torch import Tensor


def einsum(equation: str, *operands: Tensor) -> Tensor: ...
