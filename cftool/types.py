import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple

try:
    import torch
    import torchvision
    import torch.nn.functional as F
except:

    class _torch(NamedTuple):
        device: Any
        Tensor: Any
        from_numpy: Callable

    class _torchvision_utils(NamedTuple):
        make_grid: Callable
        save_image: Callable

    class _torchvision(NamedTuple):
        utils: _torchvision_utils

    torch = _torch(None, None, lambda: None)
    torchvision = _torchvision_utils(lambda: None, lambda: None)
    F = None


general_config_type = Optional[Union[str, Dict[str, Any]]]
configs_type = Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]

arr_type = Union[np.ndarray, torch.Tensor]
np_dict_type = Dict[str, Union[np.ndarray, Any]]
tensor_dict_type = Dict[str, Union[torch.Tensor, Any]]

TNumberPair = Optional[Union[int, Tuple[int, int]]]
