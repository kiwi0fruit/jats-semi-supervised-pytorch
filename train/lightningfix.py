from abc import ABCMeta
from typing import Optional as Opt, Tuple, Iterable, Dict, List, Union, Callable, Any
import os
from os import path
import math
import re
import git
import numpy as np
from torch import Tensor
from torch.nn import Module
import pytorch_lightning as pl
from scipy.interpolate import interp1d


def get_last_checkpoint(logger_log_dir: str) -> Tuple[Opt[str], int, int]:
    """
    Returns prev. ``checkpoint`` filepath, last ``epoch`` and last ``step``.
    Use ``MAX_EPOCHS + epoch + 1`` to resume training for the same number of epochs.
    If ``logger_log_dir`` doesn't exist returns (None, 0).
    """
    dir_ = path.join(logger_log_dir, 'checkpoints')
    if not path.isdir(dir_):
        return None, 0, 0

    epochsraw = [(s.split('-')[0].split('=')[1], s.split('-')[1].split('=')[1].split('.')[0], s)
                 for s in os.listdir(dir_)
                 if path.isfile(path.join(dir_, s))]
    epochsplus = [(int(i), int(j), s) for i, j, s in epochsraw if i.isdigit() and j.isdigit()]
    epoch = max([i for i, j, s in epochsplus])
    stepsfilenames = [(j, s) for i, j, s in epochsplus if i == epoch]
    if not stepsfilenames: raise ValueError(f"Directory doesn't have expected files: {logger_log_dir}")
    checkpoint = path.join(dir_, stepsfilenames[-1][1])
    return checkpoint, epoch, stepsfilenames[-1][0]


def get_git_repo_sha(path_: str):
    repo = git.Repo(path=path_)
    repo.git.add(A=True)
    sha = repo.head.object.hexsha
    if repo.head.commit.diff():
        raise RuntimeError(
            f'Repository has changed: {path_}.' +
            f'\nMake a commit. Or to wipe changes run "git reset --hard {sha}"."'
        )
    return sha


class DeterministicWarmup(Module):
    def __init__(self, xscale: float, xloc: float, *named_points: Dict[str, Iterable]) -> None:
        """
        Don't forget to additionally log batch size as it affects warmup operating with steps.
        It might be useful to have it later in the logs.

        :param xscale: X axes is multiplied by scale.
        :param xloc: X axes is summed with xloc (after multiplication).
        :param named_points: several dictionaries WITH SINGLE key-value pairs.
          Value contains iterables of points for linear interpolation.
          Should be of shape (N, 2): X, Y = points[:, 0], points[:, 1].
          Should be: X[0] == 0 and X[-1] are equal for every variable.
          Class assumes that we start from the 0 step and end at
          the last_step: X would be stretched accordingly.
        """
        super(DeterministicWarmup, self).__init__()

        self.names = [keys[0] for dic in named_points
                      for keys in [list(dic.keys())] if len(keys) == 1]
        if len(self.names) != len(named_points): raise ValueError('Dicts should only contain SINGE key-value pair')
        if len(set(self.names)) != len(self.names): raise ValueError('Duplicate keys.')
        self.lists = [list(list(dic.values())[0]) for dic in named_points]

        self.interp1d: List[interp1d] = []
        _first_y: List[float] = []
        _last_y: List[float] = []

        lastx_set = set()
        for points_ in self.lists:
            points = np.array(points_).astype(float)
            if (len(points.shape) != 2) or (points.shape[1] != 2):
                raise ValueError('Iterable should be of shape (N, 2)')
            x, y = points[:, 0], points[:, 1]
            lastx_set.add(x[-1])
            if x[0] != 0.: raise ValueError
            if np.sum((x < 0) | (x >= math.inf)) != 0: raise ValueError("X should be finite positive or zero.")
            if np.sum((y <= -math.inf) | (y >= math.inf)) != 0: raise ValueError('Y should be finite.')
            _first_y.append(float(y[0]))
            _last_y.append(float(y[-1]))
            self.interp1d.append(interp1d(np.round(x * xscale + xloc), y))
        if len(lastx_set) != 1: raise ValueError(f'Some X[-1] are not equal: {lastx_set}')
        lastx = float(list(lastx_set)[0])
        self._xscale = xscale
        self._xloc = xloc

        self.first_step = round(xloc)
        self.last_step = round(lastx * xscale + xloc)
        self.first_y = tuple(_first_y)
        self.last_y = tuple(_last_y)
        self.first_log = {f'warmup_{s}': y for s, y in zip(self.names, self.first_y)}
        self.last_log = {f'warmup_{s}': y for s, y in zip(self.names, self.last_y)}

    def __call__(self, step: int) -> Tuple[Tuple[float, ...], Dict[str, float]]:
        if step <= self.first_step:
            return self.first_y, self.first_log
        if step >= self.last_step:
            return self.last_y, self.last_log
        ret = tuple(float(f(step)) for f in self.interp1d)
        log = {f'warmup_{s}': x for s, x in zip(self.names, ret)}
        return ret, log

    def extra_repr(self) -> str:
        return (f'xscale={self._xscale}, xloc={self._xloc},' +
                ','.join(f'dict({n}={l})' for n, l in zip(self.names, self.lists)))


class GitDirsSHACallback(pl.Callback):
    def __init__(self, *dirs: str):
        super(GitDirsSHACallback, self).__init__()
        self.dirs = dirs
        self.basenames = tuple(path.basename(d) for d in dirs)
        if len(dirs) != len(set(self.basenames)): raise ValueError('Dirs basenames should differ.')
        for basename in self.basenames:
            if len(basename) != len(re.sub(r"\W", "", basename)): raise ValueError(r'Basenames should only contain \w.')

    def on_train_start(self, trainer, pl_module):
        for b, d in zip(self.basenames, self.dirs):
            pl_module.logger.log_metrics({f'{b}_{get_git_repo_sha(d)}': pl_module.current_epoch},
                                         step=pl_module.global_step)


def tensor_to_dict(flat_tensor: Tensor, prefix='t') -> Dict[str, float]:
    if len(tuple(flat_tensor.shape)) != 1: raise ValueError(flat_tensor.shape)
    return dict(**{f'{prefix}{i}': float(kl.item()) for i, kl in enumerate(flat_tensor)})


class LightningModule2(pl.LightningModule, metaclass=ABCMeta):
    successful_run = True
    dummy_validation = False

    def logg(self, logname: Union[str, Tuple[str, str]], metric,
             f: Callable[[Tensor], Any] = lambda m: m) -> Tensor:
        met = metric.compute()
        metric.reset()
        if isinstance(logname, str):
            args = (logname, f(met))
        else:
            args = (logname[0], {logname[1]: f(met)})
        if self.dummy_validation:
            return met
        self.logger.experiment.add_scalars(*args, global_step=self.global_step)
        return met

    def set_dummy_validation(self, outputs):
        self.dummy_validation = len(outputs) < 5
