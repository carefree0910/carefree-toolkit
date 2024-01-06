import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from dataclasses import dataclass


try:
    import torch
    import torchvision
    import torch.nn.functional as F
except:

    @dataclass
    class torch:
        device: Any = None
        Tensor: Any = None
        from_numpy: Callable = None

    @dataclass
    class torchvision_utils:
        make_grid: Callable = None
        save_image: Callable = None

    @dataclass
    class torchvision:
        utils: torchvision_utils

    F = None


general_config_type = Optional[Union[str, Dict[str, Any]]]
configs_type = Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]

arr_type = Union[np.ndarray, torch.Tensor]
np_dict_type = Dict[str, Union[np.ndarray, Any]]
tensor_dict_type = Dict[str, Union[torch.Tensor, Any]]

TNumberPair = Optional[Union[int, Tuple[int, int]]]
