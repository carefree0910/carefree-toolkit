from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from typing import Generator
from pydantic import BaseModel

from .geometry import Matrix2D


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
        return dict(className=type2class_name[self.type], info=d)


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


class_name2type = {
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
type2class_name = {v: k for k, v in class_name2type.items()}


def _parse_single_node(info: Dict[str, Any]) -> SingleNode:
    core_info = info["info"]
    return SingleNode(type=class_name2type[info["className"]], **core_info)


def _parse_group(info: Dict[str, Any]) -> Group:
    core_info = info["info"]
    return Group(
        type=GroupType.GROUP,
        alias=core_info["alias"],
        transform=Matrix2D(**core_info["transform"]),
        nodes=list(map(parse_node, core_info["nodes"])),
    )


def parse_node(info: Dict[str, Any]) -> INode:
    class_name = info["className"]
    if class_name == "Group":
        return _parse_group(info)
    return _parse_single_node(info)


def parse_graph(render_info_list: List[Dict[str, Any]]) -> Graph:
    return Graph(root_nodes=list(map(parse_node, render_info_list)))


# specific utility functions

DEFAULT_RENDER_TYPE = RenderType.COVER


def get_img_transform(w: float, h: float, render_params: RenderParams) -> Matrix2D:
    img_transform = Matrix2D.scale_matrix(w, h)
    crop_transform = render_params.cropFields
    if crop_transform is None:
        return img_transform
    return img_transform @ crop_transform


def get_img_render_transform(
    bbox: Matrix2D,
    img_w: float,
    img_h: float,
    render_params: RenderParams,
) -> Matrix2D:
    img_transform = get_img_transform(img_w, img_h, render_params)
    render_transform = img_transform.inverse
    render_type = render_params.renderType or DEFAULT_RENDER_TYPE
    if render_type == RenderType.FILL:
        return render_transform
    w, h = bbox.wh
    h = max(1.0e-8, abs(h))
    area_wh_ratio = w / h
    img_wh_ratio = img_w / img_h
    ratio = img_wh_ratio / area_wh_ratio
    is_fit = render_type == RenderType.FIT
    if (ratio >= 1.0 and not is_fit) or (ratio < 1.0 and is_fit):
        w_scale, h_scale = ratio, 1.0
    else:
        w_scale, h_scale = 1.0, 1.0 / ratio
    scale_center = Matrix2D.identical().center
    render_transform = render_transform.scale(w_scale, h_scale, scale_center)
    return render_transform


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
    "parse_node",
    "parse_graph",
    "get_img_transform",
    "get_img_render_transform",
]
