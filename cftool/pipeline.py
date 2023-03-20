import os
import shutil

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import TypeVar
from typing import Optional
from typing import ContextManager
from zipfile import ZipFile
from tempfile import mkdtemp

from .misc import WithRegister
from .misc import DataClassBase
from .misc import ISerializable


TBlock = TypeVar("TBlock", bound="IBlock")
TConfig = TypeVar("TConfig", bound="DataClassBase")
TPipeline = TypeVar("TPipeline", bound="IPipeline")

pipelines: Dict[str, Type["IPipeline"]] = {}
pipeline_blocks: Dict[str, Type["IBlock"]] = {}


def get_req_choices(req: TBlock) -> List[str]:
    return [r.strip() for r in req.__identifier__.split("|")]


def check_requirement(block: "IBlock", previous: Dict[str, "IBlock"]) -> None:
    for req in block.requirements:
        choices = get_req_choices(req)
        if all(c != "none" and c not in previous for c in choices):
            raise ValueError(
                f"'{block.__identifier__}' requires '{req}', "
                "but none is provided in the previous blocks"
            )


def get_workspace(folder: str, *, force_new: bool = False) -> ContextManager:
    class _:
        tmp_folder: Optional[str]

        def __init__(self) -> None:
            self.tmp_folder = None

        def __enter__(self) -> str:
            if os.path.isdir(folder):
                if not force_new:
                    return folder
                self.tmp_folder = mkdtemp()
                shutil.copytree(folder, self.tmp_folder, dirs_exist_ok=True)
                return self.tmp_folder
            path = f"{folder}.zip"
            if not os.path.isfile(path):
                raise ValueError(f"neither '{folder}' nor '{path}' exists")
            self.tmp_folder = mkdtemp()
            with ZipFile(path, "r") as ref:
                ref.extractall(self.tmp_folder)
            return self.tmp_folder

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            if self.tmp_folder is not None:
                shutil.rmtree(self.tmp_folder)

    return _()


class IBlock(WithRegister["IBlock"], metaclass=ABCMeta):
    d = pipeline_blocks

    """
    This property should be injected by the `IPipeline`.
    > In runtime (i.e. executing the `run` method), this property will represent ALL `IBlock`s used in the `IPipeline`.
    """
    previous: Dict[str, TBlock]

    @abstractmethod
    def build(self, config: TConfig) -> None:
        """This method can modify the `config` inplace, which will affect the following blocks"""

    @property
    def requirements(self) -> List[Type[TBlock]]:
        return []

    def try_get_previous(self, block: Union[str, Type[TBlock]]) -> Optional[TBlock]:
        if not isinstance(block, str):
            block = block.__identifier__
        return self.previous.get(block)

    def get_previous(self, block: Union[str, Type[TBlock]]) -> TBlock:
        b = self.try_get_previous(block)
        if b is None:
            raise ValueError(f"cannot find '{block}' in `previous`")
        return b


class IPipeline(ISerializable["IPipeline"], metaclass=ABCMeta):
    d = pipelines

    config: TConfig
    blocks: List[TBlock]

    def __init__(self) -> None:
        self.blocks = []

    # abstract

    @classmethod
    @abstractmethod
    def init(cls: Type[TPipeline], config: TConfig) -> TPipeline:
        pass

    # optional callbacks

    def before_block_build(self, block: TBlock) -> None:
        pass

    def after_block_build(self, block: TBlock) -> None:
        pass

    # api

    @property
    def block_mappings(self) -> Dict[str, TBlock]:
        return {b.__identifier__: b for b in self.blocks}

    def try_get_block(self, block: Union[str, Type[TBlock]]) -> Optional[TBlock]:
        if not isinstance(block, str):
            block = block.__identifier__
        return self.block_mappings.get(block)

    def get_block(self, block: Union[str, Type[TBlock]]) -> TBlock:
        b = self.try_get_block(block)
        if b is None:
            raise ValueError(f"cannot find '{block}' in `previous`")
        return b

    def remove(self, *block_names: str) -> None:
        pop_indices = []
        for i, block in enumerate(self.blocks):
            if block.__identifier__ in block_names:
                pop_indices.append(i)
        for i in pop_indices[::-1]:
            self.blocks.pop(i)

    def build(self, *blocks: TBlock) -> None:
        previous: Dict[str, TBlock] = self.block_mappings
        for block in blocks:
            check_requirement(block, previous)
            block.previous = previous
            self.before_block_build(block)
            block.build(self.config)
            self.after_block_build(block)
            previous[block.__identifier__] = block
            self.blocks.append(block)


__all__ = [
    "IBlock",
    "IPipeline",
    "TPipeline",
    "get_workspace",
    "get_req_choices",
]
