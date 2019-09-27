from typing import Union, Tuple, Dict, List, Iterator, Any, Iterable, Callable, Optional as Opt
from dataclasses import dataclass, asdict
from abc import abstractmethod
# noinspection PyPep8Naming
from numpy import ndarray as Array
import torch as tr
from torch import Tensor
from torch.optim.optimizer import Optimizer
from kiwi_bugfix_typechecker.nn import Module
from semi_supervised_typed.inference import DeterministicWarmup
from .tools import load_model
from .display import Log
from .utils import OutputChecker

DO_EXCEPTIONS = ('id', 'len', 'start', 'end')

IntArg = Union[int, List[int]]
FloatArg = Union[float, List[float]]


@dataclass
class Consts:
    α: float
    β: float
    γ: float
    λ: float
    τ: float
    η: float


@dataclass
class ConstsIterators:
    α: Iterator[float]
    β: Iterator[float]
    γ: Iterator[float]
    λ: Iterator[float]
    τ: Iterator[float]
    η: Iterator[float]

    def get_consts(self):
        return Consts(α=next(self.α), β=next(self.β), γ=next(self.γ), λ=next(self.λ),
                      τ=next(self.τ), η=next(self.η))


@dataclass
class Do:
    """
    Single value should not be a list.
    List given would always mean series of values.
    Non-list values (even tuples) would be converted to [value].
    """
    id: int
    len: int
    batch_size: IntArg
    epochs: IntArg
    iters_per_epoch: IntArg
    anneal_epochs: IntArg = 0
    start: Opt[int] = None
    end: Opt[int] = None
    α0: Opt[FloatArg] = None
    α: FloatArg = 1
    β0: Opt[FloatArg] = None
    β: FloatArg = 1
    γ0: Opt[FloatArg] = None
    γ: FloatArg = 1
    λ0: Opt[FloatArg] = None
    λ: FloatArg = 1
    τ0: Opt[FloatArg] = None
    τ: FloatArg = 1
    η0: Opt[FloatArg] = None
    η: FloatArg = 1
    max_failure: IntArg = 15

    def __post_init__(self):
        if self.α0 is None:
            self.α0 = self.α
        if self.β0 is None:
            self.β0 = self.β
        if self.γ0 is None:
            self.γ0 = self.γ
        if self.λ0 is None:
            self.λ0 = self.λ
        if self.τ0 is None:
            self.τ0 = self.τ
        if self.η0 is None:
            self.η0 = self.η

    @staticmethod
    def int_(x: IntArg) -> int:
        if isinstance(x, int):
            return x
        raise RuntimeError("Presumably Guide.get_spec(...) wasn't run.")

    @staticmethod
    def float_(x: Opt[FloatArg]) -> float:
        if isinstance(x, (float, int)):
            return x
        raise RuntimeError("Presumably Guide.get_spec(...) wasn't run.")

    def get_iterators(self) -> ConstsIterators:
        nt, flt = self.int_, self.float_
        n = nt(self.anneal_epochs) * nt(self.iters_per_epoch)
        return ConstsIterators(
            α=DeterministicWarmup(n=n, t_start=flt(self.α0), t_end=flt(self.α)),
            β=DeterministicWarmup(n=n, t_start=flt(self.β0), t_end=flt(self.β)),
            γ=DeterministicWarmup(n=n, t_start=flt(self.γ0), t_end=flt(self.γ)),
            λ=DeterministicWarmup(n=n, t_start=flt(self.λ0), t_end=flt(self.λ)),
            τ=DeterministicWarmup(n=n, t_start=flt(self.τ0), t_end=flt(self.τ)),
            η=DeterministicWarmup(n=n, t_start=flt(self.η0), t_end=flt(self.η))
        )


class Guide:
    id_pref: str
    models_idxs: Tuple[int, ...]
    successful_models_n: int
    to_do: Tuple[int, ...]
    do_specs: Dict[int, Do]
    successful_models: List[int]
    dtype: tr.dtype
    device: tr.device
    model: Module
    optimizer: Optimizer
    print_every_epoch: int
    glb_spec: Dict[str, Any]
    log_stats: List[Dict[str, Any]]
    _train: bool
    logger: Log
    output_checker: OutputChecker
    epoch: int
    iteration: int
    _get_loader: Tuple[Callable[[], Iterable]]

    def __init__(self, model: Module, optimizer: Optimizer, get_loader: Callable[[], Iterable],
                 id_pref: Union[str, int], models_idxs: Tuple[int, ...],
                 successful_models_n: int, to_do: Tuple[int, ...], do_specs: Tuple[Do, ...], logger: Log,
                 glb_spec: Dict[str, Any], output_checker: OutputChecker=OutputChecker(), dtype: tr.dtype=tr.float,
                 device: tr.device=tr.device('cuda') if tr.cuda.is_available() else tr.device('cpu'),
                 print_every_epoch: int=10, train: bool=True):
        self.id_pref = str(id_pref)
        self.models_idxs = models_idxs
        self.successful_models_n = successful_models_n
        self.to_do = to_do
        if len(set(do.id for do in do_specs)) != len(do_specs):
            raise ValueError
        self.do_specs = {do.id: self.all_lists(do) for do in do_specs}
        self._train = train
        self.successful_models = []
        self.logger = logger
        self.dtype = dtype
        self.device = device
        model = model.to(device=device, dtype=dtype)
        self.model = model
        self.optimizer = optimizer
        self.print_every_epoch = print_every_epoch
        self.glb_spec = glb_spec
        self.log_stats = []
        self.output_checker = output_checker
        self.epoch = 0
        self.iteration = 0
        self._get_loader = (get_loader,)

    @property
    def get_loader(self) -> Callable[[], Iterable]:
        return self._get_loader[0]

    @abstractmethod
    def step(self, spec: Do, item: Any, consts: Consts) -> Tuple[Tensor, Dict[str, float]]:
        """
        Method that is called from train loop inside ``self.train`` per iteration.
        It should return loss scalar to backpropagate (``loss.backward()``) and dict of other losses of interest.

        :return: loss (or None if to skip iteration), dict
        """
        raise NotImplementedError

    @abstractmethod
    def print(self, epoch: int, iteration: int, spec: Do, item: Any, consts: Consts, losses: Dict[str, float]) -> None:
        """ This function is called from ``self.train`` with ``with tr.no_grad():`` context
            when ``self.print_every_epoch`` is satisfied. """
        raise NotImplementedError

    @abstractmethod
    def save_model_output(self, spec: Do, epoch: int, run_id: str) -> None:
        """ Method that is called with ``with tr.no_grad():`` context after successful model train.
            It can be overridden to save needed results. """
        raise NotImplementedError

    @staticmethod
    def all_lists(do: Do) -> Do:
        kwargs = asdict(do)
        for k, v in kwargs.items():
            if not (k in DO_EXCEPTIONS) and not isinstance(v, list):
                kwargs[k] = [v]
        return type(do)(**kwargs)

    def get_spec(self, do_id: int, i: int) -> Do:
        do = self.do_specs[do_id]
        kwargs = asdict(do)
        for k, v in kwargs.items():
            if not (k in DO_EXCEPTIONS):
                kwargs[k] = v[min(i, len(v) - 1)]
        spec = type(do)(**kwargs)
        return spec

    @staticmethod
    def mean_losses(total_losses: Dict[str, float], batches: int) -> Dict[str, float]:
        return {k: v / batches for k, v in total_losses.items()}

    def train(self, spec: Do, epoch_0: int, iteration_0: int) -> Tuple[int, int, Dict[str, float]]:
        """
        If fresh model then ``iteration_0 == -1``.

        :return: (epoch, iteration, save_dict)
        """
        consts_iterators = spec.get_iterators()
        iteration, printed_epoch = iteration_0, -1

        losses: Dict[str, float]
        epoch, losses, batches = epoch_0, dict(), 0  # for type checker
        item_last_good: Any = None

        for epoch in range(epoch_0 + 1, epoch_0 + 1 + spec.int_(spec.epochs)):
            self.model.train()
            self.optimizer.zero_grad()
            losses, batches = dict(), 0
            item: Any = None
            for item in self.get_loader():
                consts = consts_iterators.get_consts()
                loss, losses_i = self.step(spec=spec, item=item, consts=consts)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                for k, v in losses_i.items():
                    v0 = losses.get(k, 0.)
                    losses[k] = v0 + v

                batches += 1
                iteration += 1

                if iteration == iteration_0 + 1:
                    with tr.no_grad():
                        self.print(epoch=epoch, iteration=iteration, spec=spec, item=item, consts=consts,
                                   losses=dict())
                item_last_good = item
            if item is None:
                raise RuntimeError('Empty DataLoader')

            if epoch % self.print_every_epoch == 0:
                # noinspection PyUnboundLocalVariable
                consts = consts
                with tr.no_grad():
                    self.print(epoch=epoch, iteration=iteration, spec=spec, item=item_last_good, consts=consts,
                               losses=self.mean_losses(losses, batches))
                printed_epoch = epoch
        if epoch == epoch_0:
            raise RuntimeError('No epochs')

        if printed_epoch != epoch:
            # noinspection PyUnboundLocalVariable
            consts = consts
            with tr.no_grad():
                self.print(epoch=epoch, iteration=iteration, spec=spec, item=item_last_good, consts=consts,
                           losses=self.mean_losses(losses, batches))
        return epoch, iteration, self.mean_losses(losses, batches)

    def set_output_checker(self, output_checker: OutputChecker):
        self.output_checker = output_checker

    def tensor(self, x: Union[Array, Tensor], dtype: tr.dtype=None, device: tr.device=None,
               requires_grad: bool=None, non_blocking: bool=None) -> Tensor:
        """
        Uses default or doesn't change values except for ``dtype`` and ``device``
        that use ``self.dtype`` and ``self.device`` as defaults.
        """
        device = device if (device is not None) else self.device
        dtype = dtype if (dtype is not None) else self.dtype
        if isinstance(x, Tensor):
            ret = x.to(device=device, dtype=dtype)
        elif isinstance(x, Array):
            ret = tr.tensor(x, dtype=dtype, device=device)
        else:
            raise TypeError
        if non_blocking is not None:
            ret = ret.to(non_blocking=non_blocking)
        if requires_grad is not None:
            ret.requires_grad = requires_grad
        return ret

    def load_model(self, spec: Do) -> None:  # pylint: disable=unused-argument
        """
        If you override this method you can more flexibly load the model.
        Like changing ``self.model`` before or after loading.

        >>> '''
        >>> class Guide1(Guide):
        >>>     def load_model(self, spec: Do):
        >>>         # do something to the self.model before loading it
        >>>         super(Guide1, self).load_model(spec)
        >>>         # do something to the self.model after loading it:
        >>>         del self.model.qz_flow
        >>> '''
        """
        self.epoch, self.iteration, self.log_stats = load_model(
            self.logger.checkpoint(), self.model, self.optimizer)

    def run_guide(self):
        for model_idx in self.models_idxs:
            run_id = f'{self.id_pref}-{model_idx}'
            for do_id in self.to_do:
                do = self.do_specs[do_id]
                double_break = False
                for i in list(range(do.len))[do.start:do.end]:
                    spec = self.get_spec(do_id=do_id, i=i)
                    for _ in range(abs(int(spec.max_failure)) + 1):
                        self.logger.set_run_id(run_id)
                        self.load_model(spec=spec)
                        if self._train:
                            self.epoch, self.iteration, total_losses = self.train(
                                spec=spec, epoch_0=self.epoch, iteration_0=self.iteration)
                            success = self.output_checker.check(model=self.model, log_stats=self.log_stats)
                            if success:
                                save_dict = dict(
                                    model_state_dict=self.model.state_dict(),
                                    optimizer_state_dict=self.optimizer.state_dict(),
                                    epoch=self.epoch,
                                    iteration=self.iteration,
                                    log_stats=self.log_stats,
                                    do_spec=asdict(spec),
                                    dtype=repr(self.dtype),
                                    device=repr(self.device),
                                    total_losses=total_losses,
                                    glb_spec=self.glb_spec,
                                )
                                tr.save(save_dict, self.logger.checkpoint())
                                tr.save(save_dict, self.logger.checkpoint(epoch=self.epoch))
                                break
                        else:
                            break
                    else:
                        double_break = True
                        break
                    with tr.no_grad():
                        self.save_model_output(spec=spec, epoch=self.epoch, run_id=run_id)

                if double_break:
                    break
            else:
                self.successful_models.append(model_idx)
            if len(self.successful_models) >= self.successful_models_n:
                break
