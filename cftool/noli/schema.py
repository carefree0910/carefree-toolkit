from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from typing import Generator
from pydantic import BaseModel

from ..geometry import Matrix2D


class SingleNodeType(str, Enum):
    POLYGON = "polygon"
    ELLIPSE = "ellipse"
    RECTANGLE = "rectangle"
    STAR = "star"
    LINE = "line"
    PATH = "path"
    SVG = "svg"
    TEXT = "text"
    IMAGE = "image"
    NOLI_FRAME = "noliFrame"
    NOLI_TEXT_FRAME = "noliTextFrame"


class GroupType(str, Enum):
    GROUP = "group"
    FRAME = "frame"


INodeType = Union[SingleNodeType, GroupType]


class LayerParams(BaseModel):
    z: float


class RenderType(str, Enum):
    FIT = "fit"
    FILL = "fill"
    COVER = "cover"


class RenderParams(BaseModel):
    src: Optional[str]
    renderType: Optional[RenderType]
    cropFields: Optional[Matrix2D]


class SingleNode(BaseModel):
    type: SingleNodeType
    alias: str
    bboxFields: Matrix2D
    layerParams: LayerParams
    params: Dict[str, Any]
    renderParams: Optional[RenderParams]


class Group(BaseModel):
    type: GroupType
    alias: str
    transform: Matrix2D
    nodes: List["INode"]
    params: Optional[Dict[str, Any]] = None


class INode(BaseModel):
    type: INodeType
    alias: str
    bboxFields: Optional[Matrix2D]  # only for single node
    layerParams: Optional[LayerParams]  # only for single node
    params: Optional[Dict[str, Any]]  # only for single node
    renderParams: Optional[RenderParams]  # only for single node
    nodes: Optional[List["INode"]]  # only for group
    transform: Optional[Matrix2D]  # only for group

    def dict(self, **kwargs: Any) -> Dict[str, Any]:
        d = super().dict(**kwargs)
        d.pop("type")
        return dict(className=get_class_name(self.type), info=d)


Group.update_forward_refs()


class Graph(BaseModel):
    root_nodes: List[INode]

    def get(self, alias: str) -> INode:
        def _search(nodes: List[INode]) -> Optional[INode]:
            for node in nodes:
                if node.alias == alias:
                    return node
                if node.nodes is not None:
                    result = _search(node.nodes)
                    if result is not None:
                        return result
            return None

        node = _search(self.root_nodes)
        if node is None:
            raise ValueError(f"node {alias} not found")
        return node

    @property
    def all_single_nodes(self) -> Generator[SingleNode, None, None]:
        def _generate(nodes: List[INode]) -> Generator[SingleNode, None, None]:
            for node in nodes:
                if node.type in SingleNodeType:
                    yield node
                elif node.type in GroupType:
                    if node.nodes is None:
                        raise ValueError(f"`Group` '{node.alias}' has no nodes")
                    yield from _generate(node.nodes)

        yield from _generate(self.root_nodes)

    @property
    def bg_node(self) -> Optional[SingleNode]:
        for node in self.all_single_nodes:
            if node.params.get("isBackground"):
                return node
        return None


_class_name2type = {
    "PolygonShapeNode": SingleNodeType.POLYGON,
    "EllipseShapeNode": SingleNodeType.ELLIPSE,
    "RectangleShapeNode": SingleNodeType.RECTANGLE,
    "StarShapeNode": SingleNodeType.STAR,
    "LineNode": SingleNodeType.LINE,
    "PathNode": SingleNodeType.PATH,
    "SVGNode": SingleNodeType.SVG,
    "TextNode": SingleNodeType.TEXT,
    "ImageNode": SingleNodeType.IMAGE,
    "NoliFrameNode": SingleNodeType.NOLI_FRAME,
    "NoliTextFrameNode": SingleNodeType.NOLI_TEXT_FRAME,
    "Group": GroupType.GROUP,
    "Frame": GroupType.FRAME,
}
_type2class_name = {v: k for k, v in _class_name2type.items()}


def get_node_type(class_name: str) -> INodeType:
    return _class_name2type[class_name]


def get_class_name(node_type: INodeType) -> str:
    return _type2class_name[node_type]


__all__ = [
    "SingleNodeType",
    "GroupType",
    "INodeType",
    "LayerParams",
    "RenderType",
    "RenderParams",
    "SingleNode",
    "Group",
    "INode",
    "Graph",
    "get_node_type",
    "get_class_name",
]
