import io
import os
import sys
import dill
import json
import math
import time
import errno
import random
import shutil
import inspect
import logging
import hashlib
import zipfile
import datetime
import operator
import threading
import unicodedata

import numpy as np
import matplotlib.pyplot as plt

from typing import *
from PIL import Image
from functools import reduce
from collections import Counter
from abc import abstractmethod

dill._dill._reverse_typemap["ClassType"] = type


# util functions

def timestamp(simplify: bool = False,
              ensure_different: bool = False) -> str:
    """
    Return current timestamp

    Parameters
    ----------
    simplify : bool. If True, format will be simplified to 'year-month-day'
    ensure_different : bool. If True, format will include millisecond

    Returns
    -------
    timestamp : str

    """

    now = datetime.datetime.now()
    if simplify:
        return now.strftime("%Y-%m-%d")
    if ensure_different:
        return now.strftime("%Y-%m-%d_%H-%M-%S-%f")
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def prod(iterable: Iterable) -> float:
    """ return cumulative production of an iterable """

    return float(reduce(operator.mul, iterable, 1))


def hash_code(code: str) -> str:
    """ return hash code for a string """

    code = code.encode()
    return hashlib.md5(code).hexdigest()[:8]


def prefix_dict(d: Dict[str, Any], prefix: str):
    """ prefix every key in dict `d` with `prefix` """

    return {f"{prefix}_{k}": v for k, v in d.items()}


def shallow_copy_dict(d: dict) -> dict:
    d = d.copy()
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = shallow_copy_dict(v)
    return d


def update_dict(src_dict: dict, tgt_dict: dict) -> dict:
    """
    Update tgt_dict with src_dict
    * Notice that changes will happen only on keys which src_dict holds

    Parameters
    ----------
    src_dict : dict
    tgt_dict : dict

    Returns
    -------
    tgt_dict : dict

    """

    for k, v in src_dict.items():
        tgt_v = tgt_dict.get(k)
        if tgt_v is None:
            tgt_dict[k] = v
        elif not isinstance(v, dict):
            tgt_dict[k] = v
        else:
            update_dict(v, tgt_v)
    return tgt_dict


def fix_float_to_length(num: float, length: int) -> str:
    """ change a float number to string format with fixed length """

    str_num = f"{num:f}"
    if str_num == "nan":
        return f"{str_num:^{length}s}"
    length = max(length, str_num.find("."))
    return str_num[:length].ljust(length, "0")


def truncate_string_to_length(string: str, length: int) -> str:
    """ truncate a string to make sure its length not exceeding a given length """

    if len(string) <= length:
        return string
    half_length = int(0.5 * length) - 1
    return string[:half_length] + "." * (length - 2 * half_length) + string[-half_length:]


def grouped(iterable: Iterable, n: int, *, keep_tail=False) -> List[tuple]:
    """ group an iterable every `n` elements """

    if not keep_tail:
        return list(zip(*[iter(iterable)] * n))
    with batch_manager(object(), iterable, batch_size=n) as manager:
        return [tuple(batch) for batch in manager]


def is_numeric(s: str) -> bool:
    """ check whether `s` is a number """

    try:
        s = float(s)
        return True
    except (TypeError, ValueError):
        try:
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            return False


def get_one_hot(feature: Union[list, np.ndarray], dim: int) -> np.ndarray:
    """
    Get one-hot representation

    Parameters
    ----------
    feature : array-like, source data of one-hot representation
    dim : int, dimension of the one-hot representation

    Returns
    -------
    one_hot : np.ndarray, one-hot representation of `feature`

    """

    one_hot = np.zeros([len(feature), dim], np.int64)
    one_hot[range(len(one_hot)), np.asarray(feature, np.int64).ravel()] = 1
    return one_hot


def show_or_save(export_path: str,
                 fig: plt.figure = None, **kwargs) -> None:
    """
    Utility function to deal with figure

    Parameters
    ----------
    export_path : {None, str}
        * None : the figure will be shown
        * str  : it represents the path where the figure should be saved to
    fig : {None, plt.Figure}
        * None       : default figure contained in plt will be executed
        * plt.Figure : it will be executed

    """

    if export_path is None:
        fig.show(**kwargs) if fig is not None else plt.show(**kwargs)
    else:
        fig.savefig(export_path) if fig is not None else plt.savefig(export_path, **kwargs)
    plt.close()


def show_or_return(return_canvas: bool) -> Union[None, np.ndarray]:
    """
    Utility function to deal with current plt

    Parameters
    ----------
    return_canvas : bool, whether return canvas or not

    """

    if not return_canvas:
        plt.show()
        return

    buffer_ = io.BytesIO()
    plt.savefig(buffer_, format="png")
    plt.close()
    buffer_.seek(0)
    image = Image.open(buffer_)
    canvas = np.asarray(image)[..., :3]
    buffer_.close()
    return canvas


def get_indices_from_another(base: np.ndarray,
                             segment: np.ndarray) -> np.ndarray:
    """
    Get `segment` elements' indices in `base`

    Warnings
    ----------
    All elements in segment should appear in base to ensure validity

    Parameters
    ----------
    base : np.ndarray, base array
    segment : np.ndarray, segment array

    Returns
    -------
    indices : np.ndarray, positions where elements in `segment` appear in `base`

    Examples
    -------
    >>> import numpy as np
    >>> base, segment = np.arange(100), np.random.permutation(100)[:10]
    >>> assert np.allclose(get_indices_from_another(base, segment), segment)

    """
    base_sorted_args = np.argsort(base)
    positions = np.searchsorted(base[base_sorted_args], segment)
    return base_sorted_args[positions]


class UniqueIndices(NamedTuple):
    """
    unique           : np.ndarray, unique values of the given array (`arr`)
    unique_cnt       : np.ndarray, counts of each unique value
    sorting_indices  : np.ndarray, indices which can (stably) sort the given array by its value
    split_arr        : np.ndarray, array which can split the `sorting_indices` to make sure that
                       each portion of the split indices belong & only belong to one of the unique values
    """
    unique: np.ndarray
    unique_cnt: np.ndarray
    sorting_indices: np.ndarray
    split_arr: np.ndarray

    @property
    def split_indices(self):
        return np.split(self.sorting_indices, self.split_arr)


def get_unique_indices(arr: np.ndarray) -> UniqueIndices:
    """
    Get indices for unique values of an array

    Parameters
    ----------
    arr : np.ndarray, target array which we wish to find indices of each unique value

    Returns
    -------
    UniqueIndices

    Examples
    -------
    >>> import numpy as np
    >>> arr = np.array([1, 2, 3, 2, 4, 1, 0, 1], np.int64)
    >>> # UniqueIndices(
    >>> #   unique          = array([0, 1, 2, 3, 4], dtype=int64),
    >>> #   unique_cnt      = array([1, 3, 2, 1, 1], dtype=int64),
    >>> #   sorting_indices = array([6, 0, 5, 7, 1, 3, 2, 4], dtype=int64),
    >>> #   split_arr       = array([1, 4, 6, 7], dtype=int64))
    >>> #   split_indices   = [array([6], dtype=int64), array([0, 5, 7], dtype=int64), array([1, 3], dtype=int64),
    >>> #                      array([2], dtype=int64), array([4], dtype=int64)]
    >>> print(get_unique_indices(arr))

    """
    unique, unique_inv, unique_cnt = np.unique(arr, return_inverse=True, return_counts=True)
    sorting_indices, split_arr = np.argsort(unique_inv, kind="mergesort"), np.cumsum(unique_cnt)[:-1]
    return UniqueIndices(unique, unique_cnt, sorting_indices, split_arr)


def get_counter_from_arr(arr: np.ndarray) -> Counter:
    """
    Get `Counter` of an array

    Parameters
    ----------
    arr : np.ndarray, target array which we wish to get `Counter` from

    Returns
    -------
    Counter

    Examples
    -------
    >>> import numpy as np
    >>> arr = np.array([1, 2, 3, 2, 4, 1, 0, 1], np.int64)
    >>> # Counter({1: 3, 2: 2, 0: 1, 3: 1, 4: 1})
    >>> print(get_counter_from_arr(arr))

    """
    if isinstance(arr, np.ndarray):
        arr = dict(zip(*np.unique(arr, return_counts=True)))
    return Counter(arr)


def allclose(*arrays: np.ndarray, **kwargs) -> bool:
    """
    Perform `np.allclose` to `arrays` one by one

    Parameters
    ----------
    arrays : np.ndarray, target arrays
    **kwargs : keyword arguments which will be passed into `np.allclose`

    Returns
    -------
    allclose : bool

    """
    for i, arr in enumerate(arrays[:1]):
        if not np.allclose(arr, arrays[i + 1], **kwargs):
            return False
    return True


def register_core(name: str,
                  global_dict: Dict[str, type], *,
                  before_register: callable = None,
                  after_register: callable = None):
    def _register(cls):
        if before_register is not None:
            before_register(cls)
        registered = global_dict.get(name)
        if registered is not None:
            print(f"~~~ [warning] '{name}' has already registered "
                  f"in the given global dict ({global_dict})")
            return cls
        global_dict[name] = cls
        if after_register is not None:
            after_register(cls)
        return cls
    return _register


# util modules

class Incrementer:
    """
    Util class which can calculate running mean & running std efficiently

    Parameters
    ----------
    window_size : {int, None}, window size of running statistics
        * None : then all history records will be used for calculation

    Examples
    ----------
    >>> incrementer = Incrementer(window_size=5)
    >>> for i in range(10):
    >>>     incrementer.update(i)
    >>>     if i >= 4:
    >>>         print(incrementer.mean)  # will print 2.0, 3.0, ..., 6.0, 7.0

    """

    def __init__(self, window_size: int = None):
        if window_size is not None:
            if not isinstance(window_size, int):
                raise ValueError(f"window size should be integer, {type(window_size)} found")
            if window_size < 2:
                raise ValueError(f"window size should be greater than 2, {window_size} found")
        self._window_size = window_size
        self._n_record = self._running_sum = self._running_square_sum = self._previous = None

    @property
    def mean(self):
        return self._running_sum / self._n_record

    @property
    def std(self):
        return math.sqrt(max(self._running_square_sum / self._n_record - self.mean ** 2, 0.))

    @property
    def n_record(self):
        return self._n_record

    def update(self, new_value):
        if self._n_record is None:
            self._n_record, self._running_sum, self._running_square_sum = 1, new_value, new_value ** 2
        else:
            self._n_record += 1
            self._running_sum += new_value
            self._running_square_sum += new_value ** 2
        if self._window_size is not None:
            if self._previous is None:
                self._previous = [new_value]
            else:
                self._previous.append(new_value)
            if self._n_record == self._window_size + 1:
                self._n_record -= 1
                previous = self._previous.pop(0)
                self._running_sum -= previous
                self._running_square_sum -= previous ** 2


class _Formatter(logging.Formatter):
    """ formatter for logging, which supports millisecond """

    converter = datetime.datetime.fromtimestamp

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s.%03d" % (t, record.msecs)
        return s


class LoggingMixin:
    """
    Mixin class to provide logging methods for base class

    Attributes
    ----------
    _triggered_ : bool
        If not `_triggered_`, log file will not be created

    _verbose_level_ : int
        Preset verbose level of the whole logging process

    Methods
    ----------
    log_msg(self, body, prefix="", verbose_level=1)
        Log something either through console or to a file
        * body : str
            main logging message
        * prefix : str
            prefix added to `body` when logging message goes through console
        * verbose_level : int
            if `self._verbose_level_` >= verbose_level, then logging message will go through console

    log_block_msg(self, body, prefix="", title="", verbose_level=1)
        Almost the same as `log_msg`, except adding `title` on top of `body`

    """

    _triggered_ = False
    _initialized_ = False
    _logging_path_ = None
    _logger_ = _verbose_level_ = None
    _date_format_string_ = "%Y-%m-%d %H:%M:%S.%f"
    _formatter_ = _Formatter(
        "[ {asctime:s} ] [ {levelname:^8s} ] {func_prefix:s} {message:s}",
        _date_format_string_, style="{"
    )
    _timing_dict_, _time_cache_dict_ = {}, {}

    info_prefix = "~~~  [ info ] "
    warning_prefix = "~~~ [warning] "
    error_prefix = "~~~ [ error ] "

    @property
    def logging_path(self):
        if self._logging_path_ is None:
            folder = os.path.join(os.getcwd(), "_logging", type(self).__name__)
            os.makedirs(folder, exist_ok=True)
            self._logging_path_ = os.path.join(folder, f"{timestamp()}.log")
        return self._logging_path_

    @property
    def console_handler(self):
        if self._logger_ is None:
            return
        for handler in self._logger_.handlers:
            if isinstance(handler, logging.StreamHandler):
                return handler

    @staticmethod
    def _get_func_prefix(frame=None, return_prefix=True):
        if frame is None:
            frame = inspect.currentframe().f_back.f_back
        if not return_prefix:
            return frame
        frame_info = inspect.getframeinfo(frame)
        file_name = truncate_string_to_length(os.path.basename(frame_info.filename), 16)
        func_name = truncate_string_to_length(frame_info.function, 24)
        func_prefix = f"[ {func_name:^24s} ] [ {file_name:>16s}:{frame_info.lineno:<4d} ]"
        return func_prefix

    @staticmethod
    def _release_handlers(logger):
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    def _init_logging(self,
                      verbose_level: Union[int, None] = 2,
                      trigger: bool = True):
        if LoggingMixin._initialized_:
            return self
        LoggingMixin._initialized_ = True
        logger_name = getattr(self, "_logger_name_", "root")
        logger = LoggingMixin._logger_ = logging.getLogger(logger_name)
        LoggingMixin._verbose_level_ = verbose_level
        if not trigger:
            return self
        LoggingMixin._triggered_ = True
        config = getattr(self, "config", {})
        self._logging_path_ = config.get("_logging_path_")
        if self._logging_path_ is None:
            self._logging_path_ = config["_logging_path_"] = self.logging_path
        os.makedirs(os.path.dirname(self.logging_path), exist_ok=True)
        file_handler = logging.FileHandler(self.logging_path)
        file_handler.setFormatter(self._formatter_)
        file_handler.setLevel(logging.DEBUG)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(_Formatter("{custom_prefix:s}{message:s}", style="{"))
        logger.setLevel(logging.DEBUG)
        self._release_handlers(logger)
        logger.addHandler(console)
        logger.addHandler(file_handler)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.CRITICAL)
        self.log_block_msg(sys.version, title="system version", verbose_level=None)
        return self

    def log_msg(self,
                body: str,
                prefix: str = "",
                verbose_level: Union[int, None] = 1,
                msg_level: int = logging.INFO,
                frame=None):
        preset_verbose_level = getattr(self, "_verbose_level", None)
        if preset_verbose_level is not None:
            self._verbose_level_ = preset_verbose_level
        elif self._verbose_level_ is None:
            self._verbose_level_ = 0
        console_handler = self.console_handler
        if verbose_level is None or self._verbose_level_ < verbose_level:
            do_print, console_level = False, msg_level + 10
        else:
            do_print, console_level = not LoggingMixin._triggered_, msg_level
        if console_handler is not None:
            console_handler.setLevel(console_level)
        if do_print:
            print(prefix + body)
        elif LoggingMixin._triggered_:
            func_prefix = self._get_func_prefix(frame)
            self._logger_.log(msg_level, body, extra={"func_prefix": func_prefix, "custom_prefix": prefix})
        if console_handler is not None:
            console_handler.setLevel(logging.INFO)

    def log_block_msg(self,
                      body: str,
                      prefix: str = "",
                      title: str = "",
                      verbose_level: Union[int, None] = 1,
                      msg_level: int = logging.INFO,
                      frame=None):
        frame = self._get_func_prefix(frame, False)
        self.log_msg(f"{title}\n{body}\n", prefix, verbose_level, msg_level, frame)

    def exception(self, body, frame=None):
        self._logger_.exception(body, extra={
            "custom_prefix": self.error_prefix,
            "func_prefix": LoggingMixin._get_func_prefix(frame)
        })

    @staticmethod
    def log_with_external_method(body, prefix, log_method, *args, **kwargs):
        if log_method is None:
            print(prefix + body)
        else:
            kwargs["frame"] = LoggingMixin._get_func_prefix(kwargs.pop("frame", None), False)
            log_method(body, prefix, *args, **kwargs)

    @staticmethod
    def merge_logs_by_time(*log_files, tgt_file):
        tgt_folder = os.path.dirname(tgt_file)
        date_str_len = len(datetime.datetime.today().strftime(LoggingMixin._date_format_string_)) + 4
        with lock_manager(tgt_folder, [tgt_file], clear_stuffs_after_exc=False):
            msg_dict, msg_block, last_searched = {}, [], None
            for log_file in log_files:
                with open(log_file, "r") as f:
                    for line in f:
                        date_str = line[:date_str_len]
                        if date_str[:2] == "[ " and date_str[-2:] == " ]":
                            searched_time = datetime.datetime.strptime(
                                date_str[2:-2], LoggingMixin._date_format_string_)
                        else:
                            msg_block.append(line)
                            continue
                        if last_searched is not None:
                            msg_dict.setdefault(last_searched, []).append("".join(msg_block))
                        last_searched = searched_time
                        msg_block = [line]
                    if msg_block:
                        msg_dict.setdefault(last_searched, []).append("".join(msg_block))
            with open(tgt_file, "w") as f:
                f.write("".join(["".join(msg_dict[key]) for key in sorted(msg_dict)]))

    @classmethod
    def start_timer(cls, name):
        cls._time_cache_dict_[name] = time.time()

    @classmethod
    def end_timer(cls, name):
        start_time = cls._time_cache_dict_.pop(name, None)
        if start_time is None:
            raise KeyError(f"'{name}' was not found in time cache dict, "
                           "this may cause by not calling `start_timer` method")
        incrementer = cls._timing_dict_.setdefault(name, Incrementer())
        incrementer.update(time.time() - start_time)

    def log_timing(self):
        timing_str_list = ["=" * 138]
        for name in sorted(self._timing_dict_.keys()):
            incrementer = self._timing_dict_[name]
            timing_str_list.append(
                f"|   {name:<82s}   | "
                f"{fix_float_to_length(incrementer.mean, 10)} ± "
                f"{fix_float_to_length(incrementer.std, 10)} | "
                f"{incrementer.n_record:>12d} hits   |"
            )
            timing_str_list.append("-" * 138)
        self.log_block_msg(
            "\n".join(timing_str_list), title="timing",
            verbose_level=None, msg_level=logging.DEBUG
        )
        return self


class PureLoggingMixin:
    """
    Mixin class to provide (pure) logging method for base class

    Attributes
    ----------
    _loggers_ : dict(int, logging.Logger)
        Recorded all loggers initialized

    _formatter_ : _Formatter
        Formatter for all loggers

    Methods
    ----------
    log_msg(self, name, msg, msg_level=logging.INFO)
        Log something to a file, with logger initialized by `name`

    log_block_msg(self, name, title, body, msg_level=logging.INFO)
        Almost the same as `log_msg`, except adding `title` on top of `body`

    """

    _name = _meta_name = None

    _formatter_ = LoggingMixin._formatter_
    _loggers_: Dict[str, logging.Logger] = {}
    _logger_paths_: Dict[str, str] = {}
    _timing_dict_ = {}

    @property
    def meta_suffix(self):
        return "" if self._meta_name is None else self._meta_name

    @property
    def name_suffix(self):
        return "" if self._name is None else f"-{self._name}"

    @property
    def meta_log_name(self):
        return f"__meta__{self.meta_suffix}{self.name_suffix}"

    @staticmethod
    def get_logging_path(logger):
        logging_path = None
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                logging_path = handler.baseFilename
                break
        if logging_path is None:
            raise ValueError(f"No FileHandler was found in given logger '{logger}'")
        return logging_path

    def _get_logger_info(self, name):
        logger = name if isinstance(name, logging.Logger) else self._loggers_.get(name)
        if logger is None:
            raise ValueError(f"logger for '{name}' is not defined, please call `_setup_logger` first")
        if isinstance(name, str):
            logging_path = self._logger_paths_[name]
        else:
            logging_path = self.get_logging_path(logger)
        return logger, os.path.dirname(logging_path), logging_path

    def _setup_logger(self, name, logging_path, level=logging.DEBUG):
        if name in self._loggers_:
            return
        console = logging.StreamHandler()
        console.setLevel(logging.CRITICAL)
        console.setFormatter(self._formatter_)
        file_handler = logging.FileHandler(logging_path)
        file_handler.setFormatter(self._formatter_)
        file_handler.setLevel(level)
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        LoggingMixin._release_handlers(logger)
        logger.addHandler(console)
        logger.addHandler(file_handler)
        PureLoggingMixin._loggers_[name] = logger
        PureLoggingMixin._logger_paths_[name] = logging_path
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.CRITICAL)
        self.log_block_msg(name, "system version", sys.version)

    def _log_meta_msg(self, msg, msg_level=logging.INFO, frame=None):
        if frame is None:
            frame = inspect.currentframe().f_back
        self.log_msg(self.meta_log_name, msg, msg_level, frame)

    def _log_with_meta(self, task_name, msg, msg_level=logging.INFO, frame=None):
        if frame is None:
            frame = inspect.currentframe().f_back
        self._log_meta_msg(f"{task_name} {msg}", msg_level, frame)
        self.log_msg(task_name, f"current task {msg}", msg_level, frame)

    def log_msg(self, name, msg, msg_level=logging.INFO, frame=None):
        logger, logging_folder, logging_path = self._get_logger_info(name)
        with lock_manager(logging_folder, [logging_path], clear_stuffs_after_exc=False):
            logger.log(msg_level, msg, extra={"func_prefix": LoggingMixin._get_func_prefix(frame)})
        return logger

    def log_block_msg(self, name, title, body, msg_level=logging.INFO, frame=None):
        frame = LoggingMixin._get_func_prefix(frame, False)
        self.log_msg(name, f"{title}\n{body}\n", msg_level, frame)

    def exception(self, name, msg, frame=None):
        logger, logging_folder, logging_path = self._get_logger_info(name)
        with lock_manager(logging_folder, [logging_path], clear_stuffs_after_exc=False):
            logger.exception(msg, extra={"func_prefix": LoggingMixin._get_func_prefix(frame)})

    def del_logger(self, name):
        logger = self.log_msg(name, f"clearing up logger information of '{name}'")
        del self._loggers_[name], self._logger_paths_[name]
        LoggingMixin._release_handlers(logger)
        del logger


class SavingMixin(LoggingMixin):
    """
    Mixin class to provide logging & saving method for base class

    Methods
    ----------
    save(self, folder)
        Save `self` to folder

    def load(self, folder)
        Load from folder

    """

    @property
    @abstractmethod
    def data_tuple_base(self) -> Union[None, Type[NamedTuple]]:
        pass

    @property
    @abstractmethod
    def data_tuple_attributes(self) -> Union[None, List[str]]:
        pass

    @property
    def cache_excludes(self):
        return set()

    @property
    def lock_verbose(self):
        verbose_level = getattr(self, "_verbose_level", None)
        if verbose_level is None:
            return False
        return verbose_level >= 5

    def _data_tuple_context(self, *, is_saving):
        return data_tuple_saving_controller(self, is_saving=is_saving)

    def save(self, folder: str, *, compress: bool = True):
        with self._data_tuple_context(is_saving=True):
            Saving.save_instance(self, folder, self.log_msg)
        if compress:
            abs_folder = os.path.abspath(folder)
            base_folder = os.path.dirname(abs_folder)
            with lock_manager(base_folder, [folder]):
                Saving.compress(abs_folder, remove_original=True)
        return self

    def load(self, folder: str, *, compress: bool = True):
        base_folder = os.path.dirname(os.path.abspath(folder))
        with lock_manager(base_folder, [folder]):
            with Saving.compress_loader(self, folder, compress, remove_extracted=True):
                with self._data_tuple_context(is_saving=False):
                    Saving.load_instance(self, folder, log_method=self.log_msg)
        return self


class Saving(LoggingMixin):
    """
    Util class for saving instances

    Methods
    ----------
    save_instance(instance, folder, log_method=None)
        Save instance to `folder`
        * instance : object, instance to save
        * folder : str, folder to save to
        * log_method : {None, function}, used as `log_method` parameter in
        `log_with_external_method` method of `LoggingMixin`

    load_instance(instance, folder, log_method=None)
        Load instance from `folder`
        * instance : object, instance to load, need to be initialized
        * folder : str, folder to load from
        * log_method : {None, function}, used as `log_method` parameter in
        `log_with_external_method` method of `LoggingMixin`

    """

    delim = "^_^"
    dill_suffix = ".pkl"
    array_sub_folder = "__arrays"

    @staticmethod
    def _check_core(elem):
        if isinstance(elem, dict):
            if not Saving._check_dict(elem):
                return False
        if isinstance(elem, (list, tuple)):
            if not Saving._check_list_and_tuple(elem):
                return False
        if not Saving._check_elem(elem):
            return False
        return True

    @staticmethod
    def _check_elem(elem):
        if isinstance(elem, (type, np.generic, np.ndarray)):
            return False
        if callable(elem):
            return False
        return True

    @staticmethod
    def _check_list_and_tuple(arr: Union[list, tuple]):
        for elem in arr:
            if not Saving._check_core(elem):
                return False
        return True

    @staticmethod
    def _check_dict(d: dict):
        for v in d.values():
            if not Saving._check_core(v):
                return False
        return True

    @staticmethod
    def save_dict(d: dict, name: str, folder: str):
        if Saving._check_dict(d):
            kwargs = {}
            suffix, method, mode = ".json", json.dump, "w"
        else:
            kwargs = {"recurse": True}
            suffix, method, mode = Saving.dill_suffix, dill.dump, "wb"
        with open(os.path.join(folder, f"{name}{suffix}"), mode) as f:
            method(d, f, **kwargs)

    @staticmethod
    def load_dict(name: str, folder: str = None):
        if folder is None:
            folder, name = os.path.split(name)
        name, suffix = os.path.splitext(name)
        if not suffix:
            json_file = os.path.join(folder, f"{name}.json")
            if os.path.isfile(json_file):
                with open(json_file, "r") as f:
                    return json.load(f)
            dill_file = os.path.join(folder, f"{name}{Saving.dill_suffix}")
            if os.path.isfile(dill_file):
                with open(dill_file, "rb") as f:
                    return dill.load(f)
        else:
            assert_msg = f"suffix should be either 'json' or 'pkl', {suffix} found"
            assert suffix in {".json", ".pkl"}, assert_msg
            name = f"{name}{suffix}"
            file = os.path.join(folder, name)
            if os.path.isfile(file):
                if suffix == ".json":
                    mode, load_method = "r", json.load
                else:
                    mode, load_method = "rb", dill.load
                with open(file, mode) as f:
                    return load_method(f)
        raise ValueError(f"config '{name}' is not found under '{folder}' folder")

    @staticmethod
    def deep_copy_dict(d: dict):
        tmp_folder = os.path.join(os.getcwd(), "___tmp_dict_cache___")
        if os.path.isdir(tmp_folder):
            shutil.rmtree(tmp_folder)
        os.makedirs(tmp_folder)
        dict_name = "deep_copy"
        Saving.save_dict(d, dict_name, tmp_folder)
        loaded_dict = Saving.load_dict(dict_name, tmp_folder)
        shutil.rmtree(tmp_folder)
        return loaded_dict

    @staticmethod
    def get_cache_file(instance):
        return f"{type(instance).__name__}.pkl"

    @staticmethod
    def save_instance(instance, folder, log_method=None):
        instance_str = str(instance)
        Saving.log_with_external_method(
            f"saving '{instance_str}' to '{folder}'", Saving.info_prefix, log_method, 5)
        def _record_array(k, v):
            extension = ".npy" if isinstance(v, np.ndarray) else ".lst"
            array_attribute_dict[f"{k}{extension}"] = v
        def _check_array(attr_key_, attr_value_, depth=0):
            if isinstance(attr_value_, dict):
                for k in list(attr_value_.keys()):
                    v = attr_value_[k]
                    extended_k = f"{attr_key_}{delim}{k}"
                    if isinstance(v, dict):
                        _check_array(extended_k, v, depth + 1)
                    elif isinstance(v, array_types):
                        _record_array(extended_k, v)
                        attr_value_.pop(k)
            if isinstance(attr_value_, array_types):
                _record_array(attr_key_, attr_value_)
                if depth == 0:
                    cache_excludes.add(attr_key_)
        main_file, instance_dict = Saving.get_cache_file(instance), shallow_copy_dict(instance.__dict__)
        verbose, cache_excludes = map(
            getattr, [instance] * 2,
            ["lock_verbose", "cache_excludes"], [False, set()]
        )
        if os.path.isdir(folder):
            if verbose:
                prefix = Saving.warning_prefix
                msg = f"'{folder}' will be cleaned up when saving '{instance_str}'"
                Saving.log_with_external_method(msg, prefix, log_method, msg_level=logging.WARNING)
            shutil.rmtree(folder)
        save_path = os.path.join(folder, main_file)
        array_folder = os.path.join(folder, Saving.array_sub_folder)
        tuple(map(lambda folder_: os.makedirs(folder_, exist_ok=True), [folder, array_folder]))
        sorted_attributes, array_attribute_dict = sorted(instance_dict), {}
        delim, array_types = Saving.delim, (list, np.ndarray)
        for attr_key in sorted_attributes:
            if attr_key in cache_excludes:
                continue
            attr_value = instance_dict[attr_key]
            _check_array(attr_key, attr_value)
        cache_excludes.add("_verbose_level_")
        with lock_manager(folder, [os.path.join(folder, main_file)], name=instance_str):
            with open(save_path, "wb") as f:
                dill.dump({k: v for k, v in instance_dict.items() if k not in cache_excludes}, f, recurse=True)
        if array_attribute_dict:
            sorted_array_files = sorted(array_attribute_dict)
            sorted_array_files_full_path = list(map(lambda f_: os.path.join(array_folder, f_), sorted_array_files))
            with lock_manager(array_folder, sorted_array_files_full_path, name=f"{instance_str} (arrays)"):
                for array_file, array_file_full_path in zip(sorted_array_files, sorted_array_files_full_path):
                    array_value = array_attribute_dict[array_file]
                    if array_file.endswith(".npy"):
                        np.save(array_file_full_path, array_value)
                    elif array_file.endswith(".lst"):
                        with open(array_file_full_path, "wb") as f:
                            np.save(f, array_value)
                    else:
                        raise ValueError(f"unrecognized file type '{array_file}' occurred")

    @staticmethod
    def load_instance(instance, folder, *, log_method=None):
        Saving.log_with_external_method(
            f"loading '{instance}' from '{folder}'", Saving.info_prefix, log_method, 5)
        with open(os.path.join(folder, Saving.get_cache_file(instance)), "rb") as f:
            instance.__dict__.update(dill.load(f))
        delim, array_folder = Saving.delim, os.path.join(folder, Saving.array_sub_folder)
        for array_file in os.listdir(array_folder):
            attr_name, attr_ext = os.path.splitext(array_file)
            if attr_ext == ".npy":
                load_method = np.load
            elif attr_ext == ".lst":
                def load_method(path):
                    return np.load(path).tolist()
            else:
                raise ValueError(f"unrecognized file type '{array_file}' occurred")
            array_value = load_method(os.path.join(array_folder, array_file))
            attr_hierarchy = attr_name.split(delim)
            if len(attr_hierarchy) == 1:
                instance.__dict__[attr_name] = array_value
            else:
                hierarchy_dict = instance.__dict__
                for attr in attr_hierarchy[:-1]:
                    hierarchy_dict = hierarchy_dict.setdefault(attr, {})
                hierarchy_dict[attr_hierarchy[-1]] = array_value

    @staticmethod
    def prepare_folder(instance, folder):
        if os.path.isdir(folder):
            instance.log_msg(
                f"'{folder}' already exists, it will be cleared up to save our model",
                instance.warning_prefix, msg_level=logging.WARNING
            )
            shutil.rmtree(folder)
        os.makedirs(folder)

    @staticmethod
    def compress(abs_folder, remove_original=True):
        shutil.make_archive(abs_folder, "zip", abs_folder)
        if remove_original:
            shutil.rmtree(abs_folder)

    @staticmethod
    def compress_loader(instance, folder, is_compress, *, remove_extracted=True):
        class _manager(context_error_handler):
            def __enter__(self):
                if is_compress:
                    if os.path.isdir(folder):
                        instance.log_msg(
                            f"'{folder}' already exists, it will be cleared up to load our model",
                            instance.warning_prefix, msg_level=logging.WARNING
                        )
                        shutil.rmtree(folder)
                    with zipfile.ZipFile(f"{folder}.zip", "r") as zip_ref:
                        zip_ref.extractall(folder)

            def _normal_exit(self, exc_type, exc_val, exc_tb):
                if is_compress and remove_extracted:
                    shutil.rmtree(folder)
        return _manager()


class Sampler:
    def __init__(self,
                 method: str,
                 probabilities: np.ndarray):
        self.method = method
        self.p = probabilities
        self._p_shape = list(self.p.shape)

    def __str__(self):
        return f"Sampler({self.method})"

    __repr__ = __str__

    def _reshape(self, n: int, samples: np.ndarray) -> np.ndarray:
        return samples.reshape([n] + self._p_shape[:-1]).astype(np.int64)

    def sample(self, n: int) -> np.ndarray:
        return getattr(self, self.method)(n)

    def multinomial(self, n: int) -> np.ndarray:
        samples = np.random.multinomial(n, self.p.ravel())
        sampled_indices = np.repeat(np.arange(n), samples)
        return self._reshape(n, sampled_indices)


# contexts

class context_error_handler:
    """ Util class which provides exception handling when using context manager """

    @property
    def exception_suffix(self):
        return ""

    def _normal_exit(self, exc_type, exc_val, exc_tb):
        pass

    def _exception_exit(self, exc_type, exc_val, exc_tb):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:
            self._normal_exit(exc_type, exc_val, exc_tb)
        else:
            self._exception_exit(exc_type, exc_val, exc_tb)


class timeit(context_error_handler):
    """
    Timing context manager

    Examples
    --------
    >>> with timeit("something"):
    >>>     # do something here
    >>> # will print "~~~  [ info ] timing for    something     : x.xxxx"

    """

    def __init__(self, msg, precision=6):
        self._msg = msg
        self._p = precision

    def __enter__(self):
        self._t = time.time()

    def _normal_exit(self, exc_type, exc_val, exc_tb):
        prefix = LoggingMixin.info_prefix
        print(f"{prefix}timing for {self._msg:^16s} : {time.time() - self._t:{self._p}.{self._p-2}f}")


class _lock_file_refresher(threading.Thread):
    def __init__(self, lock_file, delay=1, refresh=0.01):
        super().__init__()
        self.__stop_event = threading.Event()
        self._lock_file, self._delay, self._refresh = lock_file, delay, refresh
        with open(lock_file, "r") as f:
            self._lock_file_contents = f.read()

    def run(self) -> None:
        counter = 0
        while True:
            counter += 1
            time.sleep(self._refresh)
            if counter * self._refresh >= self._delay:
                counter = 0
                with open(self._lock_file, "w") as f:
                    prefix = "\n\n"
                    add_line = f"{prefix}refreshed at {timestamp()}"
                    f.write(self._lock_file_contents + add_line)
            if self.__stop_event.is_set():
                break

    def stop(self):
        self.__stop_event.set()


class lock_manager(context_error_handler, LoggingMixin):
    """
    Util class to make simultaneously-write process safe with some hacked (ugly) tricks

    Examples
    --------
    >>> import dill
    >>> workplace = "_cache"
    >>> target_write_files_full_path = [
    >>>     os.path.join(workplace, "file1.pkl"),
    >>>     os.path.join(workplace, "file2.pkl")
    >>> ]
    >>> with lock_manager(workplace, target_write_files_full_path):
    >>>     for write_file_full_path in target_write_files_full_path:
    >>>         with open(write_file_full_path, "wb") as wf:
    >>>             dill.dump(..., wf)

    """

    delay = 0.01
    __lock__ = "__lock__"

    def __init__(self, workplace, stuffs, verbose_level=None,
                 set_lock=True, clear_stuffs_after_exc=True, name=None, wait=1000):
        self._workplace = workplace
        self._verbose_level = verbose_level
        self._name, self._wait = name, wait
        os.makedirs(workplace, exist_ok=True)
        self._stuffs, self._set_lock = stuffs, set_lock
        self._clear_stuffs = clear_stuffs_after_exc
        self._is_locked = False

    def __enter__(self):
        frame = inspect.currentframe().f_back
        self.log_msg(
            f"waiting for lock at {self.lock_file}",
            self.info_prefix, 5, logging.DEBUG, frame
        )
        enter_time = file_modify = None
        while True:
            try:
                fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                self.log_msg("lock acquired", self.info_prefix, 5, logging.DEBUG, frame)
                if not self._set_lock:
                    self.log_msg(
                        "releasing lock since set_lock=False",
                        self.info_prefix, 5, logging.DEBUG, frame
                    )
                    os.unlink(self.lock_file)
                    self.__refresher = None
                else:
                    self.log_msg("writing info to lock file", self.info_prefix, 5, logging.DEBUG, frame)
                    with os.fdopen(fd, "a") as f:
                        f.write(f"name      : {self._name}\n"
                                f"timestamp : {timestamp()}\n"
                                f"workplace : {self._workplace}\n"
                                f"stuffs    :\n{self.cache_stuffs_str}")
                    self.__refresher = _lock_file_refresher(self.lock_file)
                    self.__refresher.start()
                break
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                try:
                    if file_modify is None:
                        enter_time = time.time()
                        file_modify = os.path.getmtime(self.lock_file)
                    else:
                        new_file_modify = os.path.getmtime(self.lock_file)
                        if new_file_modify != file_modify:
                            enter_time = time.time()
                            file_modify = new_file_modify
                        else:
                            wait_time = time.time() - enter_time
                            if wait_time >= self._wait:
                                raise ValueError(
                                    f"'{self.lock_file}' has been waited for too long ({wait_time})")
                    time.sleep(random.random() * self.delay + self.delay)
                except ValueError:
                    self.exception(f"lock_manager was blocked by dead lock ({self.lock_file})")
                    raise
                except FileNotFoundError:
                    pass
        self.log_block_msg(
            self.cache_stuffs_str, title="start processing following stuffs:",
            verbose_level=5, msg_level=logging.DEBUG, frame=frame
        )
        self._is_locked = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__refresher is not None:
            self.__refresher.stop()
            self.__refresher.join()
        if self._set_lock:
            super().__exit__(exc_type, exc_val, exc_tb)

    def _normal_exit(self, exc_type, exc_val, exc_tb, frame=None):
        if self._set_lock:
            os.unlink(self.lock_file)
        if frame is None:
            frame = inspect.currentframe().f_back.f_back.f_back
        self.log_msg("lock released", self.info_prefix, 5, logging.DEBUG, frame)

    def _exception_exit(self, exc_type, exc_val, exc_tb):
        frame = inspect.currentframe().f_back.f_back.f_back
        if self._clear_stuffs:
            for stuff in self._stuffs:
                if os.path.isfile(stuff):
                    self.log_msg(f"clearing cached file: {stuff}", "~~~~~~ ", 5, logging.ERROR, frame)
                    os.remove(stuff)
                elif os.path.isdir(stuff):
                    self.log_msg(f"clearing cached directory: {stuff}", "~~~~~~ ", 5, logging.ERROR, frame)
                    shutil.rmtree(stuff)
        self._normal_exit(exc_type, exc_val, exc_tb, frame)

    @property
    def locked(self):
        return self._is_locked

    @property
    def available(self):
        return not os.path.isfile(self.lock_file)

    @property
    def cache_stuffs_str(self):
        return "\n".join([f"~~~~~~ {stuff}" for stuff in self._stuffs])

    @property
    def exception_suffix(self):
        return f", clearing caches for safety{self.logging_suffix}"

    @property
    def lock_file(self):
        return os.path.join(self._workplace, self.__lock__)

    @property
    def logging_suffix(self):
        return "" if self._name is None else f" - {self._name}"


class batch_manager(context_error_handler):
    """
    Process data in batch

    Parameters
    ----------
    inputs : tuple(np.ndarray), auxiliary array inputs.
    n_elem : {int, float}, indicates how many elements will be processed in a batch
    batch_size : int, indicates the batch_size; if None, batch_size will be calculated by n_elem

    Examples
    --------
    >>> instance = type("test", (object,), {})()
    >>> with batch_manager(instance, np.arange(5), np.arange(1, 6), batch_size=2) as manager:
    >>>     for arr, tensor in manager:
    >>>         print(arr, tensor)
    >>>         # Will print:
    >>>         #   [0 1], [1 2]
    >>>         #   [2 3], [3 4]
    >>>         #   [4]  , [5]

    """

    def __init__(self, *inputs, n_elem=1e6, batch_size=None, max_batch_size=1024):
        if not inputs:
            raise ValueError("inputs should be provided in general_batch_manager")
        input_lengths = list(map(len, inputs))
        self._n, self._rs, self._inputs = input_lengths[0], [], inputs
        assert all(length == self._n for length in input_lengths), "inputs should be of same length"
        if batch_size is not None:
            self._batch_size = batch_size
        else:
            n_elem = int(n_elem)
            self._batch_size = int(n_elem / sum(map(lambda arr: prod(arr.shape[1:]), inputs)))
        self._batch_size = min(max_batch_size, min(self._n, self._batch_size))
        self._n_epoch = int(self._n / self._batch_size)
        self._n_epoch += int(self._n_epoch * self._batch_size < self._n)

    def __enter__(self):
        return self

    def __iter__(self):
        self._start, self._end = 0, self._batch_size
        return self

    def __next__(self):
        if self._start >= self._n:
            raise StopIteration
        batched_data = tuple(map(lambda arr: arr[self._start:self._end], self._inputs))
        self._start, self._end = self._end, self._end + self._batch_size
        if len(batched_data) == 1:
            return batched_data[0]
        return batched_data

    def __len__(self):
        return self._n_epoch


class timing_context(context_error_handler):
    """
    Wrap codes in any base class of `LoggingMixin` with this timing context to timeit

    Parameters
    ----------
    logging_mixin : LoggingMixin, arbitrary base classes of LoggingMixin
    name : str, explain what the wrapped codes are doing

    Examples
    --------
    >>> import time
    >>> import random
    >>> instance = type(
    >>>    "test", (LoggingMixin,),
    >>>    {"config": {}, "_verbose_level": 2}
    >>> )()._init_logging(2, True)
    >>> for _ in range(50):
    >>>     with timing_context(instance, "random sleep"):
    >>>         time.sleep(random.random() * 0.1)
    >>> instance.log_timing()

    """

    def __init__(self, logging_mixin, name):
        self._cls, self._name = logging_mixin, name

    @property
    def timer_name(self):
        return f"[{type(self._cls).__name__:^24s}] {self._name}"

    def __enter__(self):
        self._cls.start_timer(self.timer_name)

    def _normal_exit(self, exc_type, exc_val, exc_tb):
        self._cls.end_timer(self.timer_name)


class data_tuple_saving_controller(context_error_handler):
    """
    Help saving DataTuple of SavingMixin

    Parameters
    ----------
    instance : SavingMixin, instance whose DataTuples are to be saved / loaded
    is_saving : bool, whether it is a saving context or not
    """

    __prefix__ = "_Data_Tuple__"

    def __init__(self,
                 instance: SavingMixin,
                 *, is_saving: bool):
        self._instance = instance
        self._is_saving = is_saving
        self._data_tuple_base = instance.data_tuple_base
        self._data_tuple_attributes = instance.data_tuple_attributes
        if self.trigger and is_saving:
            self._data_tuples: List[NamedTuple] = [
                instance.__dict__.pop(attr) for attr in self._data_tuple_attributes]

    def __enter__(self):
        if self.trigger and self._is_saving:
            self.__tmp_attr_list = []
            for attr, data_tuple in zip(self._data_tuple_attributes, self._data_tuples):
                local_attr_list = self._get_attr(attr, data_tuple)
                for local_attr, data in zip(local_attr_list, data_tuple):
                    setattr(self._instance, local_attr, data)
                self.__tmp_attr_list += local_attr_list

    @property
    def trigger(self):
        return self._data_tuple_base is not None and self._data_tuple_attributes is not None

    def _normal_exit(self, exc_type, exc_val, exc_tb):
        if self.trigger:
            if self._is_saving:
                for attr, data_tuple in zip(self._data_tuple_attributes, self._data_tuples):
                    setattr(self._instance, attr, self._data_tuple_base(*data_tuple))
                for attr in self.__tmp_attr_list:
                    self._instance.__dict__.pop(attr)
                del self._data_tuples
            else:
                attr_pool_map = self._get_attr()
                for core_attr, attr_dict in attr_pool_map.items():
                    local_attr_values = []
                    for idx in range(len(attr_dict)):
                        local_attr = attr_dict[idx]
                        local_attr_values.append(self._instance.__dict__.pop(local_attr))
                    setattr(self._instance, core_attr, self._data_tuple_base(*local_attr_values))

    def _get_attr(self,
                  attr: str = None,
                  data_tuple: NamedTuple = None) -> Union[None, List[str], Dict[str, Dict[int, str]]]:
        prefix = self.__prefix__
        if self._is_saving:
            if data_tuple is None:
                raise ValueError("data tuple should be provided in saving context")
            assert isinstance(attr, str), "attr should be string in saving context"
            return [f"{prefix}{attr}_{i}" for i in range(len(data_tuple))]
        attr_pool = list(filter(lambda attr_: attr_.startswith(prefix), self._instance.__dict__.keys()))
        attr_pool_split = [attr_[len(prefix):].split("_") for attr_ in attr_pool]
        attr_pool_map = {}
        for attr, attr_split in zip(attr_pool, attr_pool_split):
            core_attr, idx = "_".join(attr_split[:-1]), int(attr_split[-1])
            local_map = attr_pool_map.setdefault(core_attr, {})
            local_map[idx] = attr
        loaded_attr_list = sorted(attr_pool_map)
        preset_attr_list = sorted(self._data_tuple_attributes)
        assert_msg = f"loaded attributes ({loaded_attr_list}) " \
                     f"are not identical with preset attributes ({preset_attr_list})"
        assert loaded_attr_list == preset_attr_list, assert_msg
        return attr_pool_map


__all__ = [
    "get_indices_from_another", "get_unique_indices", "get_one_hot", "hash_code", "prefix_dict",
    "timestamp", "fix_float_to_length", "truncate_string_to_length", "grouped", "is_numeric", "update_dict",
    "show_or_save", "show_or_return", "Saving", "LoggingMixin", "PureLoggingMixin", "SavingMixin", "Sampler",
    "context_error_handler", "timeit", "lock_manager", "batch_manager", "timing_context", "prod",
    "get_counter_from_arr", "allclose", "shallow_copy_dict", "register_core", "data_tuple_saving_controller"
]
