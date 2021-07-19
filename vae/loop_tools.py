from typing import Union, Tuple, Dict, List, Iterator, Any, Iterable, Callable, Optional as Opt
from dataclasses import dataclass, asdict
from abc import abstractmethod
from os import path
import math
# noinspection PyPep8Naming
from numpy import ndarray as Array
import torch as tr
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.nn import Module
from kiwi_bugfix_typechecker import test_assert
from semi_supervised_typed.inference import DeterministicWarmup
from .display import Log

DO_EXCEPTIONS = ('id', 'len', 'start', 'end')

IntArg = Union[int, List[int]]
FloatArg = Union[float, List[float]]
StrArg = Union[str, List[str]]
TupleIntArg = Union[Tuple[int, ...], List[Tuple[int, ...]]]
test_assert()


class OutputChecker:
    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def check(self, guide: Any) -> bool:  # pylint: disable=unused-argument,no-self-use
        return True


@dataclass
class Consts:
    α: float
    β: float
    γ: float
    δ: float
    ε: float
    η: float
    λ: float
    μ: float
    ρ: float
    τ: float
    ω: float


@dataclass
class ConstsIterators:
    α: Iterator[float]
    β: Iterator[float]
    γ: Iterator[float]
    δ: Iterator[float]
    ε: Iterator[float]
    η: Iterator[float]
    λ: Iterator[float]
    μ: Iterator[float]
    ρ: Iterator[float]
    τ: Iterator[float]
    ω: Iterator[float]

    def get_consts(self):
        return Consts(α=next(self.α), β=next(self.β), γ=next(self.γ), δ=next(self.δ), ε=next(self.ε), η=next(self.η),
                      λ=next(self.λ), μ=next(self.μ), ρ=next(self.ρ), τ=next(self.τ), ω=next(self.ω))


@dataclass
class Do:
    """
    Single value should not be a list.
    List given would always mean series of values.
    Non-list values (even tuples) would be converted to [value].
    """
    id: int = -1
    epochs: IntArg = 40
    batch_size: IntArg = 64
    iters_per_epoch: IntArg = -1
    len: int = 1
    formula: StrArg = ''
    anneal_epochs: IntArg = 0
    start: Opt[int] = None
    end: Opt[int] = None
    α0: Opt[FloatArg] = None
    α: FloatArg = 1
    β0: Opt[FloatArg] = None
    β: FloatArg = 1
    γ0: Opt[FloatArg] = None
    γ: FloatArg = 1
    δ0: Opt[FloatArg] = None
    δ: FloatArg = 1
    ε0: Opt[FloatArg] = None
    ε: FloatArg = 1
    η0: Opt[FloatArg] = None
    η: FloatArg = 1
    λ0: Opt[FloatArg] = None
    λ: FloatArg = 1
    μ0: Opt[FloatArg] = None
    μ: FloatArg = 1
    ρ0: Opt[FloatArg] = None
    ρ: FloatArg = 1
    τ0: Opt[FloatArg] = None
    τ: FloatArg = 1
    ω0: Opt[FloatArg] = None
    ω: FloatArg = 1
    max_failure: IntArg = 15
    basis_strip: TupleIntArg = ()
    post_load: bool = False
    model_load_skip: Tuple[str, ...] = ()
    optimizer_load_skip: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.α0 is None:
            self.α0 = self.α
        if self.β0 is None:
            self.β0 = self.β
        if self.γ0 is None:
            self.γ0 = self.γ
        if self.δ0 is None:
            self.δ0 = self.δ
        if self.ε0 is None:
            self.ε0 = self.ε
        if self.η0 is None:
            self.η0 = self.η
        if self.λ0 is None:
            self.λ0 = self.λ
        if self.μ0 is None:
            self.μ0 = self.μ
        if self.ρ0 is None:
            self.ρ0 = self.ρ
        if self.τ0 is None:
            self.τ0 = self.τ
        if self.ω0 is None:
            self.ω0 = self.ω
        assert self.id >= 0
        if isinstance(self.iters_per_epoch, list):
            assert sum(int(it <= 0) for it in self.iters_per_epoch) == 0
        else:
            assert self.iters_per_epoch > 0

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
            δ=DeterministicWarmup(n=n, t_start=flt(self.δ0), t_end=flt(self.δ)),
            ε=DeterministicWarmup(n=n, t_start=flt(self.ε0), t_end=flt(self.ε)),
            η=DeterministicWarmup(n=n, t_start=flt(self.η0), t_end=flt(self.η)),
            λ=DeterministicWarmup(n=n, t_start=flt(self.λ0), t_end=flt(self.λ)),
            μ=DeterministicWarmup(n=n, t_start=flt(self.μ0), t_end=flt(self.μ)),
            ρ=DeterministicWarmup(n=n, t_start=flt(self.ρ0), t_end=flt(self.ρ)),
            τ=DeterministicWarmup(n=n, t_start=flt(self.τ0), t_end=flt(self.τ)),
            ω=DeterministicWarmup(n=n, t_start=flt(self.ω0), t_end=flt(self.ω)),
        )


class Guide:
    """ context kwarg should be provides as context=Guide.ctx(...) """

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
    context: Dict[str, Any]
    _get_loader: Tuple[Callable[[Do], Iterable]]
    _get_model: Tuple[Callable[[Do, Opt[Module]], Opt[Module]]]
    _get_optimizer: Tuple[Callable[[Do, Module], Opt[Optimizer]]]
    batch_size: int = -1
    last_consts: Consts
    _total_losses: Dict[str, float] = {'loss': 0.}

    def __init__(self, get_model: Callable[[Do, Opt[Module]], Opt[Module]],
                 get_optimizer: Callable[[Do, Module], Opt[Optimizer]],
                 get_loader: Callable[[Do], Iterable], id_pref: Union[str, int], models_idxs: Tuple[int, ...],
                 successful_models_n: int, to_do: Tuple[int, ...], do_specs: Tuple[Do, ...], logger: Log,
                 glb_spec: Dict[str, Any], context: Dict[str, Any],
                 output_checker: OutputChecker=OutputChecker(), dtype: tr.dtype=tr.float,
                 device: tr.device=tr.device('cuda') if tr.cuda.is_available() else tr.device('cpu'),
                 print_every_epoch: int=10, train: bool=True):
        """ context kwarg should be provides as context=Guide.ctx(...) """
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
        self.print_every_epoch = print_every_epoch
        self.glb_spec = glb_spec
        self.log_stats = []
        self.output_checker = output_checker
        self.epoch = 0
        self.iteration = 0
        self.context = context

        self._get_loader = (get_loader,)
        self._get_model = (get_model,)
        self._get_optimizer = (get_optimizer,)
        self.__post_init__()

    @staticmethod
    def ctx() -> Dict[str, Any]:
        """ ctx method doesn't hold Liskov substitution principle """
        return dict()

    def __post_init__(self) -> None:
        pass

    def modify_model_pre_load(self) -> None:
        pass

    def set_model(self, spec: Do) -> None:
        get_model = self._get_model[0]

        if spec.post_load:
            model = get_model(spec, self.model)
            if model is not None:
                self.model = model.to(device=self.device, dtype=self.dtype)
            return

        model = get_model(spec, None)
        assert model is not None
        self.model = model
        self.modify_model_pre_load()
        self.model = self.model.to(device=self.device, dtype=self.dtype)

    def set_optimizer(self, spec: Do) -> None:
        get_optimizer = self._get_optimizer[0]
        optimizer = get_optimizer(spec, self.model)
        if spec.post_load and (optimizer is None):
            return
        assert optimizer is not None
        self.optimizer = optimizer

    @abstractmethod
    def step(self, spec: Do, item: Any, consts: Consts) -> Tuple[Tensor, Dict[str, float], bool]:
        """
        Method that is called from train loop inside ``self.train`` per iteration.
        It should return loss scalar to backpropagate (``loss.backward()``) and dict of other losses of interest.

        :return: loss (or None if to skip iteration), dict, skip
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
        kwargs: dict = asdict(do)
        for k, v in kwargs.items():
            if not (k in DO_EXCEPTIONS) and not isinstance(v, list):
                kwargs[k] = [v]
        # noinspection PyArgumentList
        return type(do)(**kwargs)

    def get_spec(self, do_id: int, i: int) -> Do:
        do = self.do_specs[do_id]
        kwargs: dict = asdict(do)
        for k, v in kwargs.items():
            if not (k in DO_EXCEPTIONS):
                kwargs[k] = v[min(i, len(v) - 1)]
        # noinspection PyArgumentList
        spec = type(do)(**kwargs)
        return spec

    @staticmethod
    def mean_losses(total_losses: Dict[str, float], batches: int) -> Dict[str, float]:
        denominator = dict(_max=1, _min=1)
        return {k: v / denominator.get(k[-4:], batches) for k, v in total_losses.items()}

    def train_step(self, spec: Do, item: Any, consts: Consts) -> Tuple[Dict[str, float], bool]:
        loss, losses_i, skip = self.step(spec=spec, item=item, consts=consts)
        if skip:
            return losses_i, skip
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return losses_i, skip

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
            losses, batches = dict(), 0
            item: Any = None
            get_loader = self._get_loader[0]
            for item in get_loader(spec):
                consts = consts_iterators.get_consts()
                losses_i, skip = self.train_step(spec=spec, item=item, consts=consts)
                if skip:
                    continue

                for k, v in losses_i.items():
                    if k.endswith('_max'):
                        v0 = losses.get(k, -math.inf)
                        losses[k] = max(v0, v)
                    elif k.endswith('_min'):
                        v0 = losses.get(k, math.inf)
                        losses[k] = min(v0, v)
                    else:
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

        # noinspection PyUnboundLocalVariable
        consts = consts
        if printed_epoch != epoch:
            with tr.no_grad():
                self.print(epoch=epoch, iteration=iteration, spec=spec, item=item_last_good, consts=consts,
                           losses=self.mean_losses(losses, batches))
        self.last_consts = consts
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

        * If ``optimizer_load_skip=('RESET',)`` kwarg is set then optimizer doesn't load.
        * If ``optimizer_load_skip=`` kwarg is set to non empty str then optimizer merges loaded into defaults
        (skipping empty string '' names).
        * Same is for loading model.

        >>> '''
        >>> class Guide1(Guide):
        >>>     def load_model(self, spec: Do):
        >>>         # do something to the self.model before loading it
        >>>         super(Guide1, self).load_model(spec)
        >>>         # do something to the self.model after loading it:
        >>>         del self.model.qz_flow
        >>> '''
        """
        assert isinstance(spec.batch_size, int)
        self.batch_size = spec.batch_size

        checkpoint = self.logger.checkpoint()
        spec.post_load = False
        self.set_model(spec)
        self.set_optimizer(spec)

        self.epoch, self.iteration, self.log_stats = 0, -1, []
        if path.isfile(checkpoint):
            _checkpoint = tr.load(checkpoint)

            model_state_dict_default = self.model.state_dict()
            model_state_dict_loaded: Dict = _checkpoint['model_state_dict']
            if spec.model_load_skip == ('RESET',):
                model_state_dict_loaded = model_state_dict_default
            elif spec.model_load_skip:
                model_state_dict_upd = model_state_dict_loaded
                for key in spec.model_load_skip:
                    if key != '':
                        del model_state_dict_upd[key]
                model_state_dict_loaded = model_state_dict_default
                model_state_dict_loaded.update(model_state_dict_upd)

            optimizer_state_dict_default = self.optimizer.state_dict()
            optimizer_state_dict_loaded: Dict = _checkpoint['optimizer_state_dict']
            if spec.optimizer_load_skip == ('RESET',):
                optimizer_state_dict_loaded = optimizer_state_dict_default
            elif spec.optimizer_load_skip:
                optimizer_state_dict_upd = optimizer_state_dict_loaded
                for key in spec.optimizer_load_skip:
                    if key != '':
                        del optimizer_state_dict_upd[key]
                optimizer_state_dict_loaded = optimizer_state_dict_default
                optimizer_state_dict_loaded.update(optimizer_state_dict_upd)

            self.model.load_state_dict(model_state_dict_loaded)
            try:
                self.optimizer.load_state_dict(optimizer_state_dict_loaded)
            except ValueError:
                self.optimizer.load_state_dict(optimizer_state_dict_default)
            self.model.eval()
            self.epoch = _checkpoint.get('epoch', self.epoch)
            self.iteration = _checkpoint.get('iteration', self.iteration)
            self.log_stats = _checkpoint.get('log_stats', self.log_stats)
            if not (isinstance(self.log_stats, list)
                    and isinstance(self.epoch, int) and isinstance(self.iteration, int)):
                raise RuntimeError('Loaded log_stats should be a list, epoch and iteration should be int.')

        spec.post_load = True
        self.set_model(spec)
        self.set_optimizer(spec)
        spec.post_load = False

    # noinspection PyMethodMayBeStatic
    def do_post_train(self, spec: Do) -> None:
        pass

    def run_guide(self):
        for model_idx in self.models_idxs:
            run_id = f'{self.id_pref}-{model_idx}'
            for do_id in self.to_do:
                do = self.do_specs[do_id]
                double_break = False
                for i in list(range(do.len))[do.start:do.end]:
                    spec = self.get_spec(do_id=do_id, i=i)
                    assert isinstance(spec.max_failure, int)
                    for _ in range(abs(int(spec.max_failure)) + 1):
                        self.logger.set_run_id(run_id)
                        self.load_model(spec=spec)
                        if self._train:
                            self.epoch, self.iteration, self._total_losses = self.train(
                                spec=spec, epoch_0=self.epoch, iteration_0=self.iteration)
                            self.do_post_train(spec=spec)
                            success = self.output_checker.check(guide=self)
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
                                    total_losses=self._total_losses,
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

    def save(self):
        do_id = self.to_do[-1]
        do = self.do_specs[do_id]
        spec = self.get_spec(do_id=do_id, i=list(range(do.len))[do.start:do.end][-1])
        save_dict = dict(
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            epoch=self.epoch,
            iteration=self.iteration,
            log_stats=self.log_stats,
            do_spec=asdict(spec),
            dtype=repr(self.dtype),
            device=repr(self.device),
            total_losses=self._total_losses,
            glb_spec=self.glb_spec,
        )
        self.logger.set_run_id(f'{self.id_pref}-{self.models_idxs[-1]}')
        tr.save(save_dict, self.logger.checkpoint())
        tr.save(save_dict, self.logger.checkpoint(epoch=self.epoch))
