import os

import numpy as np

from typing import List

from ..misc import LoggingMixin

try:
    import SharedArray as sa
except:
    sa = None


class SharedArray:
    def __init__(self,
                 name: str,
                 shape: List[int] = None,
                 dtype: np.dtype = np.float32,
                 *,
                 base_folder: str = None,
                 overwrite: bool = False):
        self.name = name

        def _check_shape():
            if shape is None:
                msg = "`shape` should be provided when creating a new array"
                raise ValueError(msg)

        if sa is not None:
            try:
                self.array = sa.attach(self.sa_key)
            except FileNotFoundError:
                _check_shape()
                self.array = sa.create(self.sa_key, shape, dtype)
        else:
            if base_folder is None:
                base_folder = self.default_base_folder
            self.base_folder = base_folder
            if os.path.isfile(self.np_path) and not overwrite:
                print(
                    f"{LoggingMixin.warning_prefix}POSIX is not available "
                    "so read-only array is returned. "
                    "Call `to_mutable` to make it mutable."
                )
                self.array = np.load(self.np_path)
                self.array.flags.writeable = False
            else:
                _check_shape()
                self.array = np.zeros(shape, dtype)
                np.save(self.np_path, self.array)

    def __str__(self) -> str:
        return self.array.__str__()

    def __repr__(self) -> str:
        return self.array.__repr__()

    @property
    def sa_key(self) -> str:
        return f"shm://{self.name}"

    @property
    def np_path(self) -> str:
        return os.path.join(self.base_folder, f"{self.name}.npy")

    @property
    def default_base_folder(self) -> str:
        home = os.path.expanduser("~")
        folder = os.path.join(home, ".carefree-toolkit", ".shared_array")
        os.makedirs(folder, exist_ok=True)
        return folder

    # api

    def to_mutable(self) -> "SharedArray":
        self.array.flags.writeable = True
        return self

    def save(self) -> "SharedArray":
        if sa is not None:
            print(
                f"{LoggingMixin.warning_prefix}`save` method will take not effect "
                "when POSIX is available"
            )
            return self
        np.save(self.np_path, self.array)
        return self

    def delete(self) -> None:
        if sa is not None:
            sa.delete(self.sa_key)
        else:
            os.remove(self.np_path)
        del self.array


__all__ = ["SharedArray"]
