import psutil
import logging
import subprocess

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from collections import defaultdict

from .misc import PureLoggingMixin


class PCManager:
    """Util class which can check PC status (unit: MB)."""

    @staticmethod
    def get_available_ram() -> float:
        return psutil.virtual_memory().available / 1024**2

    @staticmethod
    def get_pid_ram_usage(pid: int) -> float:
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return 0.0
        return process.memory_info().rss / 1024**2


class GPUManager:
    """
    Util class which can check CUDA usages.

    Parameters
    ----------
    available_cuda_list : {None, list}, indicates CUDAs which are available.
    * If None, then all CUDAs will be available.
    reuse : bool, indicates whether one CUDA could be used multiple times.
    * If `available_cuda_list` is None, then `reuse` will have no effect.

    Examples
    --------
    >>> available_cuda_list = list(range(8))  # indicates that CUDA-0, ..., CUDA-7 is available
    >>> cuda = GPUManager(available_cuda_list).choose()  # `cuda` will be the CUDA id with most memory

    """

    def __init__(
        self,
        available_cuda_list: Optional[List[Union[str, int]]] = None,
        *,
        reuse: bool = True,
    ):
        self._reuse = reuse
        if available_cuda_list is None:
            self._available_cuda = None
        else:
            self._available_cuda = set(map(int, available_cuda_list))

    @staticmethod
    def _query_gpu() -> List[Dict[str, int]]:
        q_args = ["index", "gpu_name", "memory.free", "memory.total"]
        cmd = f"nvidia-smi --query-gpu={','.join(q_args)} --format=csv,noheader"
        results = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            close_fds=True,
        ).communicate()[0]
        results = list(filter(bool, results.decode("utf-8").strip().split("\n")))

        def _parse(line):
            numeric_args = ["memory.free", "memory.total"]
            power_manage_enable = lambda v: ("Not Support" not in v)
            to_numeric = lambda v: float(
                v.upper().strip().replace("MIB", "").replace("W", "")
            )
            process = lambda k, v: (
                (int(to_numeric(v)) if power_manage_enable(v) else 1)
                if k in numeric_args
                else v.strip()
            )
            rs = {k: process(k, v) for k, v in zip(q_args, line.strip().split(","))}
            rs["index"] = int(rs["index"])
            return rs

        return list(map(_parse, results))

    @staticmethod
    def _query_pid() -> List[Tuple[int, int]]:
        cmd = "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"
        results = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            close_fds=True,
        ).communicate()[0]
        results = list(filter(bool, results.decode("utf-8").strip().split("\n")))
        pid_list = []
        for line in results:
            key, value = map(int, line.replace("MiB", "").strip().split(", "))
            pid_list.append((key, value))
        return pid_list

    @staticmethod
    def _sort_by_memory(
        gpus: List[Dict[str, int]],
        by_size: bool = False,
    ) -> List[Dict[str, int]]:
        if by_size:
            return sorted(gpus, key=lambda d: d["memory.free"], reverse=True)
        return sorted(
            gpus,
            key=lambda d: float(d["memory.free"]) / d["memory.total"],
            reverse=True,
        )

    def _gpu_filter(self, gpu: Dict[str, int]):
        return self._available_cuda is None or gpu["index"] in self._available_cuda

    def choose(self) -> int:
        if isinstance(self._available_cuda, set) and not self._available_cuda:
            raise ValueError("No more CUDAs are available")
        chosen_gpu = int(
            next(
                filter(
                    self._gpu_filter,
                    self._sort_by_memory(self._query_gpu(), True),
                )
            )["index"]
        )
        if not self._reuse and self._available_cuda is not None:
            self._available_cuda.remove(chosen_gpu)
        return chosen_gpu

    def get_gpu_frees(self) -> Dict[int, int]:
        return {
            gpu["index"]: gpu["memory.free"]
            for gpu in filter(self._gpu_filter, self._query_gpu())
        }

    def get_pid_usages(self) -> Dict[int, int]:
        usages = defaultdict(int)
        for key, usage in self._query_pid():
            usages[key] += usage
        return usages


class ResourceManager:
    """
    Util class which can monitor & manage resources.
    * It utilizes `PCManager` & `GPUManager` defined above.
    * It is currently used in cftool.dist.core.Parallel only.

    """

    def __init__(
        self,
        config: Dict[str, Any],
        get_task_name: Callable[[int], Optional[str]],
        refresh_patience: int,
    ):
        self._resources, self._info_dict, self._overwritten_task_info = [], {}, {}
        self._init_logger = self._meta_log_name = None
        self._log_msg = self._log_block_msg = None
        self._log_meta_msg = self._log_with_meta = None
        self.pid2task_idx, self._get_task_name = {}, get_task_name
        self.config, self._refresh_patience = config, refresh_patience

    def register(self, resource_name: str, methods: Dict[str, Callable]) -> None:
        self._resources.append(resource_name)
        resource_config = self.config.setdefault(f"{resource_name.lower()}_config", {})
        preset_usages = resource_config.setdefault("preset_usages", {})
        minimum_resource = resource_config.setdefault(
            "minimum_resource",
            preset_usages.setdefault("__default__", 1024),
        )
        counter_threshold = resource_config.setdefault("counter_threshold", 4)
        warning_threshold = resource_config.setdefault("warning_threshold", 1024)
        info = self._info_dict.setdefault(
            resource_name,
            {
                "preset_usages": preset_usages,
                "minimum_resource": minimum_resource,
                "counter_threshold": counter_threshold,
                "warning_threshold": warning_threshold,
            },
        )
        get_pid_usage, get_pid_usage_dict, get_available_dict = map(
            methods.get,
            ["get_pid_usage", "get_pid_usage_dict", "get_available_dict"],
        )
        if get_pid_usage is None and get_pid_usage_dict is None:
            raise ValueError(
                "either get_pid_usage or get_pid_usage_dict "
                "should be provided in methods"
            )
        if get_available_dict is None:
            raise ValueError("get_available_dict should be provided in methods")
        info["get_pid_usage"] = get_pid_usage
        info["get_pid_usage_dict"] = get_pid_usage_dict
        info["get_available_dict"] = get_available_dict
        info["inference_frees"], info["checkpoint_pid_usages"] = {}, {}
        info["running_pid_usages"], info["running_pid_counters"] = {}, {}
        info["inference_usages_initialized"] = False
        info["pid2resource_id"] = {}

    def register_logging(
        self,
        init_logger: Callable[[str], None],
        mixin: PureLoggingMixin,
    ):
        self._init_logger, self._meta_log_name = init_logger, mixin.meta_log_name
        self._log_msg, self._log_block_msg = mixin.log_msg, mixin.log_block_msg
        self._log_meta_msg = mixin._log_meta_msg
        self._log_with_meta = mixin._log_with_meta

    @property
    def inference_usages_initialized(self) -> bool:
        for info in self._info_dict.values():
            if not info["inference_usages_initialized"]:
                return False
        return True

    @property
    def checkpoint_initialized(self) -> bool:
        for info in self._info_dict.values():
            if not info["checkpoint_pid_usages"]:
                return False
        return True

    @staticmethod
    def _get_all_relevant_processes(pid: int) -> List[psutil.Process]:
        parent = psutil.Process(pid)
        processes = [parent]
        for process in parent.children(recursive=True):
            processes.append(process)
        return processes

    @staticmethod
    def _get_pid_usages(info: Dict[str, Any]) -> Dict[str, int]:
        get_pid_usage_dict = info["get_pid_usage_dict"]
        if get_pid_usage_dict is not None:
            return get_pid_usage_dict()
        pid_usages, get_pid_usage = {}, info["get_pid_usage"]
        for pid in info["running_pid_usages"].keys():
            try:
                processes = ResourceManager._get_all_relevant_processes(pid)
                pid_usages[pid] = int(
                    sum(get_pid_usage(process.pid) for process in processes)
                )
            except psutil.NoSuchProcess:
                pid_usages[pid] = 0
        return pid_usages

    @staticmethod
    def get_dict_block_msg(d: Dict[str, int]) -> str:
        keys = sorted(d.keys())
        values = [d[key] for key in keys]
        header = " | ".join(map(lambda pid: f"{pid:^12s}", map(str, keys)))
        body = " | ".join(map(lambda usage: f"{int(usage):^10d}MB", values))
        len_header = len(header)
        above = f"{'=' * len_header}\n{header}\n{'-' * len_header}"
        return f"{above}\n{body}\n{'-' * len_header}"

    def default_usage(self, resource: str) -> int:
        return int(self._info_dict[resource]["preset_usages"]["__default__"])

    def initialize_running_usages(self) -> None:
        for info in self._info_dict.values():
            info["running_pid_usages"] = info["checkpoint_pid_usages"].copy()

    def initialize_inference_usages(self) -> None:
        for info in self._info_dict.values():
            info["inference_frees"] = info["get_available_dict"]()
            info["inference_usages_initialized"] = True

    def log_pid_usages_and_inference_frees(self) -> None:
        for resource in self._resources:
            info = self._info_dict[resource]
            self._log_block_msg(
                self._meta_log_name,
                f"current pid {resource} usages",
                self.get_dict_block_msg(self._get_pid_usages(info)),
                logging.DEBUG,
            )
            self._log_block_msg(
                self._meta_log_name,
                f"current inference {resource} frees",
                self.get_dict_block_msg(info["inference_frees"]),
                logging.DEBUG,
            )

    def check(self) -> None:
        for resource in self._resources:
            info = self._info_dict[resource]
            checkpoint_pid_usages, running_pid_usages, running_pid_counters = map(
                info.get,
                [
                    "checkpoint_pid_usages",
                    "running_pid_usages",
                    "running_pid_counters",
                ],
            )
            get_pid_usage, get_pid_usage_dict = map(
                info.get,
                ["get_pid_usage", "get_pid_usage_dict"],
            )
            if get_pid_usage is None:
                pid_usage_dict = get_pid_usage_dict()
                get_pid_usage = lambda pid_: pid_usage_dict.get(pid_, 0)
            for pid, checkpoint_usage in checkpoint_pid_usages.items():
                task_name = self._get_task_name(self.pid2task_idx[pid])
                self._log_msg(
                    task_name,
                    f"checkpoint {resource} usage : {checkpoint_usage} MB",
                    logging.DEBUG,
                )
                children_pid_usages = {}
                try:
                    processes = self._get_all_relevant_processes(pid)
                except psutil.NoSuchProcess as err:
                    self._log_with_meta(
                        task_name,
                        f"already finished, {err}",
                        logging.INFO,
                    )
                    continue
                current_usage = get_pid_usage(pid)
                for process in processes[1:]:
                    process_usage = get_pid_usage(process.pid)
                    children_pid_usages[process.pid] = process_usage
                    current_usage += process_usage
                if children_pid_usages:
                    self._log_block_msg(
                        task_name,
                        f"children {resource} usages",
                        self.get_dict_block_msg(children_pid_usages),
                        logging.DEBUG,
                    )
                self._log_msg(
                    task_name,
                    f"current {resource} usage : {current_usage} MB",
                    logging.DEBUG,
                )
                if current_usage == 0:
                    continue
                running_usage = running_pid_usages[pid]
                self._log_msg(
                    task_name,
                    f"running {resource} usage : {running_usage} MB",
                    logging.DEBUG,
                )
                if running_usage > checkpoint_usage:
                    running_pid_counters[pid] = self._refresh_patience
                    self._log_msg(
                        task_name,
                        f"increasing {resource} counter directly to "
                        f"{self._refresh_patience}",
                        logging.DEBUG,
                    )
                elif abs(running_usage - current_usage) <= info["counter_threshold"]:
                    running_pid_counters[pid] += 1
                    self._log_msg(
                        task_name,
                        f"increasing {resource} counter to "
                        f"{running_pid_counters[pid]}",
                        logging.DEBUG,
                    )
                else:
                    running_pid_counters[pid] = 0
                    msg = f"reset {resource} counter"
                    self._log_msg(task_name, msg, logging.DEBUG)
                running_pid_usages[pid] = current_usage
                if running_pid_counters[pid] >= self._refresh_patience:
                    d_usage = current_usage - checkpoint_usage
                    self._log_msg(
                        task_name,
                        f"delta {resource} usage : {d_usage} MB",
                        logging.DEBUG,
                    )
                    checkpoint_pid_usages[pid] = current_usage
                    inference_frees = info["inference_frees"]
                    actual_frees = info["get_available_dict"]()
                    inference_frees[info["pid2resource_id"][pid]] -= d_usage
                    self._log_block_msg(
                        task_name,
                        f"inference {resource} frees after updating delta usage",
                        self.get_dict_block_msg(inference_frees),
                        logging.DEBUG,
                    )
                    self._log_block_msg(
                        task_name,
                        f"actual {resource} frees",
                        self.get_dict_block_msg(actual_frees),
                        logging.DEBUG,
                    )
                    running_pid_counters[pid] = 0
                    msg = f"reset {resource} counter"
                    self._log_msg(task_name, msg, logging.DEBUG)

    def get_process(
        self,
        task_idx: int,
        sleep_method: Callable,
        start: bool,
    ) -> Dict[str, Any]:
        task_name = self._get_task_name(task_idx)
        results: Dict[str, Any] = {
            "__task_name__": task_name,
            "__create_process__": True,
        }
        for resource in self._resources:
            info = self._info_dict[resource]
            local_results = results.setdefault(resource, {})
            preset_usage = info["preset_usages"].setdefault(
                task_idx,
                self.default_usage(resource),
            )
            self._log_with_meta(
                task_name,
                f"preset {resource} usage : {preset_usage} MB; "
                f"minimum {resource} memory needed : {info['minimum_resource']} MB",
            )
            try:
                while True:
                    inference_frees, minimum_resource = map(
                        info.get,
                        ["inference_frees", "minimum_resource"],
                    )
                    self._log_msg(task_name, f"checking {resource}")
                    if len(inference_frees) == 1:
                        tgt_resource_id = next(iter(inference_frees.keys()))
                        free_memory = inference_frees[tgt_resource_id]
                    else:
                        tgt_resource_id, free_memory = sorted(
                            inference_frees.items(), key=lambda kv: kv[1]
                        )[-1]
                    self._log_msg(
                        task_name,
                        f"best choice : {resource} {tgt_resource_id} with "
                        f"{free_memory} MB free memory",
                        logging.DEBUG,
                    )
                    local_results["preset_usage"] = preset_usage
                    local_results["tgt_resource_id"] = tgt_resource_id
                    if preset_usage < free_memory - minimum_resource:
                        self._log_msg(
                            task_name,
                            f"acceptable, {resource} checked",
                            logging.DEBUG,
                        )
                        break
                    if not start:
                        self._log_msg(
                            task_name,
                            "not acceptable, also break out because it's initializing",
                            logging.DEBUG,
                        )
                        results["__create_process__"] = False
                        break
                    self._log_msg(task_name, "not acceptable, waiting", logging.DEBUG)
                    sleep_method()
                if results["__create_process__"]:
                    inference_frees[tgt_resource_id] -= preset_usage
            except KeyboardInterrupt:
                results["__create_process__"] = False
                return results
        return results

    def record_process(
        self,
        task_idx: int,
        process: psutil.Process,
        rs: Dict[str, Any],
    ) -> None:
        pid = process.pid
        task_name = self._get_task_name(task_idx)
        rs_task_name = rs["__task_name__"]
        assert_msg = f"internal error occurred, {task_name} != {rs_task_name}"
        assert task_name == rs_task_name, assert_msg
        task_info, overwritten = None, False
        self.pid2task_idx[pid] = task_idx
        for resource in self._resources:
            local_rs = rs[resource]
            info = self._info_dict[resource]
            pid2resource_id = info["pid2resource_id"]
            checkpoint_pid_usages, running_pid_usages, running_pid_counters = map(
                info.get,
                [
                    "checkpoint_pid_usages",
                    "running_pid_usages",
                    "running_pid_counters",
                ],
            )
            if pid in pid2resource_id:
                self._log_with_meta(
                    task_name,
                    f"pid ({pid}) collision started for {resource}",
                    logging.WARNING,
                )
                if not overwritten:
                    overwritten = True
                    task_info = self._overwritten_task_info.setdefault(task_name, {})
                    task_info["pid"], task_info["task_idx"] = (
                        pid,
                        self.pid2task_idx[pid],
                    )
                task_resource_info = task_info.setdefault(f"{resource}_info", {})
                task_resource_info["ckpt_usage"] = checkpoint_pid_usages[pid]
                task_resource_info["running_usage"] = running_pid_usages[pid]
                task_resource_info["resource_id"] = pid2resource_id[pid]
            running_pid_counters[pid] = 0
            resource_id = pid2resource_id[pid] = local_rs["tgt_resource_id"]
            usage = local_rs["preset_usage"]
            checkpoint_pid_usages[pid] = running_pid_usages[pid] = usage
            self._log_with_meta(
                task_name,
                f"record : using {resource} {resource_id} with {usage} MB "
                f"memory usage (pid : {pid})",
            )

    def handle_finish(self, task_idx: int, process: psutil.Process) -> Optional[str]:
        if process is None:
            return
        pid = process.pid
        task_name = self._get_task_name(task_idx)
        self._init_logger(task_name)
        task_info = None
        if task_name not in self._overwritten_task_info:
            pid_task_idx = self.pid2task_idx.pop(pid)
        else:
            task_info = self._overwritten_task_info.pop(task_name)
            recorded_pid, pid_task_idx = map(task_info.get, ["pid", "task_idx"])
            assert_msg = f"internal error occurred ({pid} != {recorded_pid})"
            assert pid == recorded_pid, assert_msg
        msg = "task_idx should be identical with pid_task_idx, internal error occurred"
        assert task_idx == pid_task_idx, msg
        for resource in self._resources:
            resource_info = self._info_dict[resource]
            if task_info is not None:
                task_resource_info = task_info[f"{resource}_info"]
                resource_id = task_resource_info["resource_id"]
                pid_ckpt_usage, pid_running_usage = map(
                    task_resource_info.get, ["ckpt_usage", "running_usage"]
                )
                self._log_with_meta(
                    task_name,
                    f"pid ({pid}) collision ended for {resource}",
                    logging.WARNING,
                )
            else:
                pid_ckpt_usage, pid_running_usage, _ = map(
                    dict.pop,
                    [
                        resource_info["checkpoint_pid_usages"],
                        resource_info["running_pid_usages"],
                        resource_info["running_pid_counters"],
                    ],
                    [pid] * 3,
                )
                resource_id = resource_info["pid2resource_id"].pop(pid)
            pid_usage = pid_running_usage
            if pid_running_usage != pid_ckpt_usage:
                self._log_with_meta(
                    task_name,
                    f"running pid {resource} usage ({pid_running_usage}) != "
                    f"checkpoint pid {resource} usage ({pid_ckpt_usage}), "
                    f"checkpoint {resource} usage will be used to inference "
                    f"{resource} free",
                    logging.WARNING,
                )
                pid_usage = pid_ckpt_usage
            preset_usage = resource_info["preset_usages"][task_idx]
            if pid_running_usage >= preset_usage + resource_info["warning_threshold"]:
                self._log_with_meta(
                    task_name,
                    f"running pid {resource} usage ({pid_running_usage}) exceeded "
                    f"preset cuda {resource} usage ({preset_usage}) so much, "
                    f"it may cause {resource} out of memory",
                    logging.WARNING,
                )
            self._log_with_meta(
                task_name,
                f"finished, releasing {pid_ckpt_usage} MB (inference) memory "
                f"from {resource} {resource_id}",
            )
            inference_frees = resource_info["inference_frees"]
            inference_frees[resource_id] += pid_usage
            for name in [task_name, self._meta_log_name]:
                self._log_block_msg(
                    name,
                    f"inference {resource} frees after releasing",
                    self.get_dict_block_msg(inference_frees),
                    logging.DEBUG,
                )
                self._log_block_msg(
                    name,
                    f"actual {resource} frees",
                    self.get_dict_block_msg(resource_info["get_available_dict"]()),
                    logging.DEBUG,
                )
        return task_name


__all__ = ["PCManager", "GPUManager", "ResourceManager"]
