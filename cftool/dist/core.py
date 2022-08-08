import os
import dill
import time
import pprint
import random
import signal
import inspect
import logging
import platform

from tqdm import tqdm
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from typing import Callable
from pathos.pools import ProcessPool
from multiprocessing import Process
from multiprocessing.managers import SyncManager

from ..misc import timestamp
from ..misc import grouped_into
from ..misc import print_warning
from ..misc import LoggingMixin
from ..misc import PureLoggingMixin
from ..manage import PCManager
from ..manage import GPUManager
from ..manage import ResourceManager


LINUX = platform.system() == "Linux"
dill._dill._reverse_typemap["ClassType"] = type


class Parallel(PureLoggingMixin):
    """
    Util class which can help running tasks in parallel.

    Warnings
    ----------
        On platforms other than Linux, functions are dramatically reduced because
        only Linux system well supports pickling. In this occasion, `Parallel`
        will simply leverage `pathos` to do the jobs.

    Parameters
    ----------
    num_jobs : int, number of jobs run in parallel.
    sleep : float, idle duration of new jobs.
    use_tqdm: bool, whether show progress bar (with tqdm) or not.
    use_cuda: bool, whether tasks need CUDA or not.
    name : str, summary name of these tasks.
    meta_name : str, name of the meta information.
    logging_folder : str, where the logging will be placed.
    task_names : List[str], names of each task.
    resource_config : Dict[str, Any], config used in `ResourceManager`.

    Examples
    ----------
    >>> def add_one(x):
    >>>     import time
    >>>     time.sleep(1)
    >>>     return x + 1
    >>>
    >>> print(Parallel(10)(add_one, list(range(10))).parallel_results)

    """

    class _ParallelError(Exception):
        pass

    def __init__(
        self,
        num_jobs: int,
        *,
        sleep: float = 1.0,
        use_tqdm: bool = True,
        use_cuda: bool = False,
        name: Optional[str] = None,
        meta_name: Optional[str] = None,
        logging_folder: Optional[str] = None,
        task_names: Optional[List[str]] = None,
        tqdm_config: Optional[Dict[str, Any]] = None,
        resource_config: Optional[Dict[str, Any]] = None,
        warn_num_jobs: bool = True,
    ):
        self._rs = None
        self._use_tqdm, self._use_cuda = use_tqdm, use_cuda
        self._num_jobs, self._sleep = num_jobs, sleep
        if tqdm_config is None:
            tqdm_config = {}
        if resource_config is None:
            resource_config = {}
        if logging_folder is None:
            logging_folder = os.path.join(os.getcwd(), "_parallel_", "logs")
        self._tqdm_config = tqdm_config
        self._resource_config = resource_config
        self._name, self._meta_name = name, meta_name
        self._logging_folder, self._task_names = logging_folder, task_names
        self._refresh_patience = resource_config.setdefault("refresh_patience", 10)
        self._init_logger(self.meta_log_name)
        self._warn_num_jobs = warn_num_jobs

    def __call__(self, f: Callable, *args_list: Any) -> "Parallel":
        # if f returns a dict with 'terminate' key, Parallel can be terminated at
        # early stage by setting 'terminate' key to True
        n_tasks = len(args_list[0])
        n_jobs = min(self._num_jobs, n_tasks)
        if self._task_names is None:
            self._task_names = [None] * n_tasks
        if not LINUX or n_jobs <= 1:
            if LINUX and self._warn_num_jobs:
                print_warning(
                    "Detected Linux system but "
                    f"n_jobs={n_jobs}, functions will be dramatically reduced.\n"
                    "* It is recommended to set n_jobs to a larger value"
                )
            results = []
            task_names = list(map(self._get_task_name, range(n_tasks)))
            if n_jobs <= 1:
                iterator = (f(*args) for args in zip(*args_list))
            else:
                p = ProcessPool(ncpus=n_jobs)
                iterator = p.imap(f, *args_list)
            if self._use_tqdm:
                iterator = tqdm(iterator, total=n_tasks, **self._tqdm_config)
            for result in iterator:
                results.append(result)
            self._rs = dict(zip(task_names, results))
            return self
        self._func, self._args_list = f, args_list
        self._cursor, self._all_task_indices = 0, list(range(n_jobs, n_tasks))
        self._log_meta_msg("initializing sync manager")
        self._sync_manager = SyncManager()
        self._sync_manager.start(lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
        meta = {"n_jobs": n_jobs, "n_tasks": n_tasks, "terminated": False}
        self._rs = self._sync_manager.dict(
            {
                "__meta__": meta,
                "__exceptions__": {},
            }
        )
        self._overwritten_task_info = {}
        self._pid2task_idx = None
        self._log_meta_msg("initializing resource manager")
        self._resource_manager = ResourceManager(
            self._resource_config, self._get_task_name, self._refresh_patience
        )
        self._log_meta_msg("registering PC manager")
        pc_manager = PCManager()
        ram_methods = {
            "get_pid_usage_dict": None,
            "get_pid_usage": pc_manager.get_pid_ram_usage,
            "get_available_dict": lambda: {"total": pc_manager.get_available_ram()},
        }
        self._resource_manager.register("RAM", ram_methods)
        gpu_config = self._resource_config.setdefault("gpu_config", {})
        default_cuda_list = None if self._use_cuda else []
        available_cuda_list = gpu_config.setdefault(
            "available_cuda_list", default_cuda_list
        )
        if available_cuda_list is None or available_cuda_list:
            self._log_meta_msg("registering GPU manager")
            if available_cuda_list is not None:
                available_cuda_list = list(map(int, available_cuda_list))
            gpu_manager = GPUManager(available_cuda_list)
            gpu_methods = {
                "get_pid_usage": None,
                "get_pid_usage_dict": gpu_manager.get_pid_usages,
                "get_available_dict": gpu_manager.get_gpu_frees,
            }
            self._resource_manager.register("GPU", gpu_methods)
        self._resource_manager.register_logging(self._init_logger, self)
        self._log_meta_msg("initializing with refreshing")
        self._refresh(skip_check_finished=True)
        self._working_processes = None
        if not self._use_tqdm:
            self._tqdm_bar = None
        else:
            self._tqdm_bar = tqdm(list(range(n_tasks)), **self._tqdm_config)
        try:
            self._log_meta_msg("initializing processes")
            init_task_indices = list(range(n_jobs))
            init_processes = [
                self._get_process(i, start=False) for i in init_task_indices
            ]
            if self.terminated:
                self._user_terminate()
            init_failed_slots, init_failed_task_indices = [], []
            for i, (task_idx, process) in enumerate(
                zip(init_task_indices, init_processes)
            ):
                if process is None:
                    init_failed_slots.append(i)
                    init_failed_task_indices.append(task_idx)
                    task_name = self._get_task_name(task_idx)
                    self._log_with_meta(
                        task_name,
                        "initialization failed, it may due to lack of resources",
                        msg_level=logging.WARNING,
                    )
            if init_failed_slots:
                for slot in init_failed_slots:
                    init_task_indices[slot] = None
                    init_processes[slot] = [None] * 4
                self._all_task_indices = (
                    init_failed_task_indices + self._all_task_indices
                )
            self._working_task_indices = init_task_indices
            self._working_processes, task_info = map(list, zip(*init_processes))
            self._log_meta_msg("starting all initial processes")
            tuple(
                map(
                    lambda p_: None if p_ is None else p_.start(),
                    self._working_processes,
                )
            )
            tuple(
                map(
                    self._record_process,
                    self._working_task_indices,
                    self._working_processes,
                    task_info,
                )
            )
            self._resource_manager.initialize_running_usages()
            self._log_meta_msg("entering parallel main loop")
            while True:
                self._log_meta_msg("waiting for finished slot")
                self._wait_and_handle_finish(wait_until_finish=True)
                if not self._add_new_processes():
                    break
        except KeyboardInterrupt:
            self.exception(self.meta_log_name, f"keyboard interrupted")
            exceptions = self.exceptions
            exceptions["base"] = self._ParallelError("Keyboard Interrupted")
            self._rs["__exceptions__"] = exceptions
        except Exception as err:
            self.exception(self.meta_log_name, f"exception occurred, {err}")
            exceptions = self.exceptions
            exceptions["base"] = err
            self._rs["__exceptions__"] = exceptions
        finally:
            self._log_meta_msg("joining processes left behind")
            if self._working_processes is not None:
                for process in self._working_processes:
                    if process is None:
                        continue
                    process.join()
            if self._tqdm_bar is not None:
                self._tqdm_bar.close()
            self._log_meta_msg("casting parallel results to Python dict")
            self._rs = dict(self._rs)
            self._log_meta_msg("shutting down sync manager")
            self._sync_manager.shutdown()
            self.log_block_msg(
                self.meta_log_name,
                "parallel results",
                pprint.pformat(self._rs, compact=True),
            )
        return self

    def grouped(self, f: Callable, *args_list: Any) -> "Parallel":
        num_jobs = min(len(args_list[0]), self._num_jobs)
        grouped_args_list = [grouped_into(args, num_jobs) for args in args_list]

        def _grouped_f(i: int, *args_list_: Tuple[Any], cuda: Any = None) -> List[Any]:
            results: List[Any] = []
            kwargs = {} if not self._use_cuda else {"cuda": cuda}
            for args in tqdm(
                zip(*args_list_),
                total=len(args_list_[0]),
                position=i + 1,
                leave=False,
            ):
                results.append(f(*args, **kwargs))
            return results

        return self(_grouped_f, list(range(num_jobs)), *grouped_args_list)

    @property
    def meta(self) -> Dict[str, Any]:
        return self._rs["__meta__"]

    @property
    def exceptions(self) -> Dict[str, Any]:
        return self._rs["__exceptions__"]

    @property
    def terminated(self) -> bool:
        return self.meta["terminated"]

    @property
    def parallel_results(self) -> Dict[str, Any]:
        return self._rs

    @property
    def ordered_results(self) -> List[Any]:
        return [None if key is None else self._rs[key] for key in self._task_names]

    def __sleep(self, skip_check_finished: bool) -> None:
        time.sleep(self._sleep + random.random())
        self._refresh(skip_check_finished=skip_check_finished)

    def __wait(self, wait_until_finished: bool) -> List[int]:
        try:
            while True:
                task_names = ", ".join(
                    map(
                        self._get_task_name,
                        filter(bool, self._working_task_indices),
                    )
                )
                self._log_meta_msg(
                    f"waiting for slots (working tasks : {task_names})",
                    msg_level=logging.DEBUG,
                )
                finished_slots = []
                for i, (task_idx, process) in enumerate(
                    zip(self._working_task_indices, self._working_processes)
                ):
                    if process is None:
                        self._log_meta_msg(f"pending on slot {i}")
                        finished_slots.append(i)
                        continue
                    task_name = self._get_task_name(task_idx)
                    if not process.is_alive():
                        msg = f"in slot {i} is found finished"
                        self._log_with_meta(task_name, msg)
                        finished_slots.append(i)
                if not wait_until_finished or finished_slots:
                    return finished_slots
                self.__sleep(skip_check_finished=True)
        except KeyboardInterrupt:
            self._set_terminate(scope="wait")
            raise self._ParallelError("Keyboard Interrupted")

    def _init_logger(self, task_name: str) -> None:
        logging_folder = os.path.join(self._logging_folder, task_name)
        os.makedirs(logging_folder, exist_ok=True)
        logging_path = os.path.join(logging_folder, f"{timestamp()}.log")
        self._setup_logger(task_name, logging_path)

    def _refresh(self, skip_check_finished: bool) -> None:
        if self._pid2task_idx is None:
            self._pid2task_idx = self._resource_manager.pid2task_idx
        if not self._resource_manager.inference_usages_initialized:
            self._resource_manager.initialize_inference_usages()
        if not self._resource_manager.checkpoint_initialized:
            return
        self._resource_manager.log_pid_usages_and_inference_frees()
        self._resource_manager.check()
        if not skip_check_finished:
            self._wait_and_handle_finish(wait_until_finish=False)

    def _wait_and_handle_finish(self, wait_until_finish: bool) -> None:
        finished_slots = self.__wait(wait_until_finish)
        if not finished_slots:
            return
        if self.terminated:
            self._user_terminate()
        finished_bundle = [[], []]
        for finished_slot in finished_slots[::-1]:
            if self._tqdm_bar is not None:
                self._tqdm_bar.update()
            tuple(
                map(
                    list.append,
                    finished_bundle,
                    map(
                        list.pop,
                        [self._working_task_indices, self._working_processes],
                        [finished_slot] * 2,
                    ),
                )
            )
        for task_idx, process in zip(*finished_bundle):
            task_name = self._resource_manager.handle_finish(task_idx, process)
            if task_name is None:
                continue
            self.del_logger(task_name)

    def _add_new_processes(self) -> bool:
        n_working = len(self._working_processes)
        n_new_jobs = self._num_jobs - n_working
        n_res = len(self._all_task_indices) - self._cursor
        if n_res > 0:
            n_new_jobs = min(n_new_jobs, n_res)
            for _ in range(n_new_jobs):
                new_task_idx = self._all_task_indices[self._cursor]
                self._working_processes.append(self._get_process(new_task_idx))
                self._working_task_indices.append(new_task_idx)
                self._cursor += 1
            return True
        return n_working > 0

    def _user_terminate(self) -> None:
        self._log_meta_msg(
            "`_user_terminate` method hit, joining processes",
            logging.ERROR,
        )
        for process in self._working_processes:
            if process is None:
                continue
            process.join()
        self._log_meta_msg(
            "processes joined, raising self._ParallelError",
            logging.ERROR,
        )
        recorded_exceptions = self.exceptions
        if not recorded_exceptions:
            raise self._ParallelError("Parallel terminated by user action")
        else:
            raise self._ParallelError("Parallel terminated by unexpected errors")

    def _set_terminate(self, **kwargs) -> None:
        meta = self.meta
        meta["terminated"] = True
        self._rs["__meta__"] = meta
        if not kwargs:
            suffix = ""
        else:
            suffix = f" ({' ; '.join(f'{k}: {v}' for k, v in kwargs.items())})"
        self._log_meta_msg(f"`_set_terminate` method hit{suffix}", logging.ERROR)

    def _get_task_name(self, task_idx: int) -> Optional[str]:
        if task_idx is None:
            return
        if self._task_names[task_idx] is None:
            self._task_names[task_idx] = f"task_{task_idx}"
        task_name = f"{self._task_names[task_idx]}{self.name_suffix}"
        self._init_logger(task_name)
        return task_name

    def _f_wrapper(self, task_idx: int, cuda: int = None) -> Callable:
        task_name = self._get_task_name(task_idx)
        logger = self._loggers_[task_name]

        def log_method(msg, msg_level=logging.INFO, frame=None):
            if frame is None:
                frame = inspect.currentframe().f_back
            self.log_msg(logger, msg, msg_level, frame)
            return logger

        def _inner(*args):
            if self.terminated:
                return
            try:
                log_method("task started", logging.DEBUG)
                kwargs = {}
                f_wants_cuda = f_wants_log_method = False
                f_signature = inspect.signature(self._func)
                for name, param in f_signature.parameters.items():
                    if param.kind is inspect.Parameter.VAR_KEYWORD:
                        f_wants_cuda = f_wants_log_method = True
                        break
                    if name == "cuda":
                        f_wants_cuda = True
                        continue
                    if name == "log_method":
                        f_wants_log_method = True
                        continue
                if not f_wants_cuda:
                    if self._use_cuda:
                        log_method(
                            "task function doesn't want cuda but cuda is used",
                            logging.WARNING,
                        )
                else:
                    log_method("task function wants cuda")
                    kwargs["cuda"] = cuda
                if not f_wants_log_method:
                    msg = "task function doesn't want log_method"
                    log_method(msg, logging.WARNING)
                else:
                    log_method("task function wants log_method")
                    kwargs["log_method"] = log_method
                self._rs[task_name] = rs = self._func(*args, **kwargs)
                terminate = isinstance(rs, dict) and rs.get("terminate", False)
                if not terminate:
                    log_method("task finished", logging.DEBUG)
            except KeyboardInterrupt:
                log_method("key board interrupted", logging.ERROR)
                return
            except Exception as err:
                logger.exception(
                    f"exception occurred, {err}",
                    extra={"func_prefix": LoggingMixin._get_func_prefix(None)},
                )
                terminate = True
                exceptions = self.exceptions
                self._rs[task_name] = rs = err
                exceptions[task_name] = rs
                self._rs["__exceptions__"] = exceptions
            if terminate:
                self._set_terminate(scope="f_wrapper", task=task_name)
                log_method("task terminated", logging.ERROR)

        return _inner

    def _get_process(
        self,
        task_idx: int,
        start: bool = True,
    ) -> Optional[Union[Tuple[Process, Dict[str, Any]], Process]]:
        rs = self._resource_manager.get_process(
            task_idx,
            lambda: self.__sleep(skip_check_finished=False),
            start,
        )
        task_name = rs["__task_name__"]
        if not rs["__create_process__"]:
            return
        if not self._use_cuda or "GPU" not in rs:
            args = (task_idx,)
        else:
            args = (task_idx, rs["GPU"]["tgt_resource_id"])
        target = self._f_wrapper(*args)
        process = Process(
            target=target, args=tuple(args[task_idx] for args in self._args_list)
        )
        self._log_with_meta(task_name, "process created")
        if start:
            process.start()
            self._log_with_meta(task_name, "process started")
            self._record_process(task_idx, process, rs)
            return process
        return process, rs

    def _record_process(
        self,
        task_idx: int,
        process: Optional[Process],
        rs: Dict[str, Any],
    ) -> None:
        if process is None:
            return
        self._resource_manager.record_process(task_idx, process, rs)


__all__ = ["Parallel"]
