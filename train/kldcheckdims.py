from typing import Tuple
from torch import Tensor
from torch.nn import Module


class CheckKLDDims(Module):
    def __init__(self, switch_from_min_to_max_thr=False, thr: float = 0, subset: Tuple[int, ...] = (),
                 check_interv: Tuple[int, int] = ()):
        super(CheckKLDDims, self).__init__()
        self.thr = thr
        self.subset = subset
        self.check_interv = check_interv
        # if check_interv[0] > check_interv[1]: raise ValueError
        self.switch_from_min_to_max_thr = switch_from_min_to_max_thr

    def forward(self, epoch: int, *kldvecs: Tensor) -> bool:
        if self.check_interv:
            if (epoch < self.check_interv[0]) or (epoch > self.check_interv[1]):
                return True
        ret = kldvecs != ()
        for kldvec in kldvecs:
            ret = ret and self._check(kldvec)
        return ret

    def _check(self, kldvec: Tensor) -> bool:
        ret = True
        kldvec_ = [s for i, s in enumerate(kldvec) if i in self.subset] if self.subset else kldvec
        if self.thr > 1e-8:
            if self.switch_from_min_to_max_thr:
                ret = ret and (sum(kl.item() > self.thr for kl in kldvec_) == 0)
            else:
                ret = ret and (sum(kl.item() < self.thr for kl in kldvec_) == 0)
        return ret

    def extra_repr(self) -> str:
        return (f'thr={self.thr}, switch_from_min_to_max_thr={self.switch_from_min_to_max_thr}, ' +
                f'subset={self.subset}, check_epochs={self.check_epochs}')
