import math
import torch as tr
from torch import Tensor


class ImportanceWeightedSampler:
    """
    Importance weighted sampler [Burda 2015] to
    be used in conjunction with SVI.
    """
    def __init__(self, mc: int=1, iw: int=1) -> None:
        """
        Initialise a new sampler.

        :param mc: number of Monte Carlo samples
        :param iw: number of Importance Weighted samples
        """
        self._mc = mc
        self._iw = iw

    def resample(self, x: Tensor) -> Tensor:
        return x.repeat(self._mc * self._iw, 1)

    def __call__(self, elbo: Tensor) -> Tensor:
        elbo = elbo.view(self._mc, self._iw, -1)
        elbo = tr.mean(tr.logsumexp(elbo, dim=1) - math.log(elbo.shape[1]), dim=0)
        return elbo.view(-1)


class DeterministicWarmup:
    _inc: float
    _t: float
    _t_end: float

    def __init__(self, n: int=100, t_start: float=0, t_end: float=1) -> None:
        """
        With default values it's a linear deterministic warm-up
        as described in [Sønderby 2016].

        :param n: number of iterations to transit from t_start to t_end.
            With DeterministicWarmup(n=2) it would be: 0.0, 0.5, 1.0. Then yield t_end infinitely.
            If n=0 then t_start is ignored and t_end is yielded infinitely.
        """
        if n < 0:
            raise ValueError('n < 0')
        self._t_end = t_end
        if n == 0:
            self._inc = 0
            self._t = t_end
        else:
            self._inc = (t_end - t_start) / n
            self._t = t_start - self._inc
        self._cap = min if (t_end > t_start) else max

    def __iter__(self):
        return self

    def __next__(self) -> float:
        t = self._t + self._inc
        self._t = self._cap(t, self._t_end)
        return self._t
