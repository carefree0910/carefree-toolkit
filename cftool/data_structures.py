import io
import gc
import time

from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Generic
from typing import TypeVar
from typing import Callable
from typing import Iterator
from typing import Optional
from typing import NamedTuple
from pydantic import Field
from pydantic import BaseModel

from .misc import print_info
from .misc import sort_dict_by_value
from .constants import TIME_FORMAT

try:
    import networkx as nx
except:
    nx = None
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except:
    plt = None
    mpatches = None
try:
    from PIL import Image

    TImage = Image.Image
except:
    Image = None
    TImage = None


TItemData = TypeVar("TItemData")
TTypes = TypeVar("TTypes")
TItem = TypeVar("TItem")


class Item(Generic[TItemData]):
    def __init__(self, key: str, data: TItemData) -> None:
        self.key = key
        self.data = data


class Bundle(Generic[TItemData]):
    def __init__(self, *, no_mapping: bool = False) -> None:
        """
        * use mapping is fast at the cost of doubled memory.
        * for the `queue` use case, mapping is not needed because all operations
        focus on the first item.

        Details
        -------
        * no_mapping = False
            * get    : O(1)
            * push   : O(1)
            * remove : O(1) (if not found) / O(n)
        * no_mapping = True
            * get    : O(n)
            * push   : O(1)
            * remove : O(n)
        * `queue` (both cases, so use no_mapping = False to save memory)
            * get    : O(1)
            * push   : O(1)
            * remove : O(1)
        """

        self._items: List[Item[TItemData]] = []
        self._mapping: Optional[Dict[str, Item[TItemData]]] = None if no_mapping else {}

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Item[TItemData]]:
        return iter(self._items)

    @property
    def first(self) -> Optional[Item[TItemData]]:
        if self.is_empty:
            return None
        return self._items[0]

    @property
    def last(self) -> Optional[Item[TItemData]]:
        if self.is_empty:
            return None
        return self._items[-1]

    @property
    def is_empty(self) -> bool:
        return not self._items

    def get(self, key: str) -> Optional[Item[TItemData]]:
        if self._mapping is not None:
            return self._mapping.get(key)
        for item in self._items:
            if key == item.key:
                return item
        return None

    def get_index(self, index: int) -> Item[TItemData]:
        return self._items[index]

    def push(self, item: Item[TItemData]) -> None:
        if self.get(item.key) is not None:
            raise ValueError(f"item '{item.key}' already exists")
        self._items.append(item)
        if self._mapping is not None:
            self._mapping[item.key] = item

    def remove(self, key: str) -> Optional[Item[TItemData]]:
        if self._mapping is None:
            for i, item in enumerate(self._items):
                if key == item.key:
                    self._items.pop(i)
                    return item
            return None
        item = self._mapping.pop(key, None)  # type: ignore
        if item is not None:
            for i, _item in enumerate(self._items):
                if key == _item.key:
                    self._items.pop(i)
                    break
        return item


class QueuesInQueue(Generic[TItemData]):
    def __init__(self, *, no_mapping: bool = True) -> None:
        self._cursor = 0
        self._queues: Bundle[Bundle[TItemData]] = Bundle(no_mapping=no_mapping)

    def __iter__(self) -> Iterator[Item[Bundle[TItemData]]]:
        return iter(self._queues)

    @property
    def is_empty(self) -> bool:
        return self.num_items == 0

    @property
    def num_queues(self) -> int:
        return len(self._queues)

    @property
    def num_items(self) -> int:
        return sum(len(q.data) for q in self._queues)

    def get(self, queue_id: str) -> Optional[Item[Bundle[TItemData]]]:
        return self._queues.get(queue_id)

    def push(self, queue_id: str, item: Item[TItemData]) -> None:
        queue_item = self._queues.get(queue_id)
        if queue_item is not None:
            queue = queue_item.data
        else:
            queue = Bundle()
            self._queues.push(Item(queue_id, queue))
        queue.push(item)

    def next(self) -> Tuple[Optional[str], Optional[Item[TItemData]]]:
        if self._queues.is_empty:
            return None, None
        self._cursor %= len(self._queues)
        queue = self._queues.get_index(self._cursor)
        item = queue.data.first
        if item is None:
            self._queues.remove(queue.key)
            return self.next()
        self._cursor += 1
        return queue.key, item

    def remove(self, queue_id: str, item_key: str) -> None:
        queue_item = self._queues.get(queue_id)
        if queue_item is None:
            return
        queue_item.data.remove(item_key)
        if queue_item.data.is_empty:
            self._queues.remove(queue_id)

    def get_pending(self, item_key: str) -> Optional[List[Item[TItemData]]]:
        if self._queues.is_empty:
            return None
        layer = 0
        searched = False
        pending: List[Item[TItemData]] = []
        finished_searching = [False] * len(self._queues)

        init = (self._cursor + len(self._queues) - 1) % len(self._queues)
        cursor = init
        while not all(finished_searching):
            if not finished_searching[cursor]:
                queue = self._queues.get_index(cursor)
                if layer >= len(queue.data):
                    finished_searching[cursor] = True
                else:
                    item = queue.data.get_index(layer)
                    if item.key == item_key:
                        searched = True
                        break
                    pending.append(item)
            cursor = (cursor + 1) % len(self._queues)
            if cursor == init:
                layer += 1

        return pending if searched else None


class Types(Generic[TTypes]):
    def __init__(self) -> None:
        self._types: Dict[str, Type[TTypes]] = {}

    def __iter__(self) -> Iterator[str]:
        return iter(self._types)

    def __setitem__(self, key: str, value: Type[TTypes]) -> None:
        self._types[key] = value

    def make(self, key: str, *args: Any, **kwargs: Any) -> Optional[TTypes]:
        t = self._types.get(key)
        return None if t is None else t(*args, **kwargs)

    def items(self) -> Iterator[Tuple[str, Type[TTypes]]]:
        return self._types.items()  # type: ignore

    def values(self) -> Iterator[Type[TTypes]]:
        return self._types.values()  # type: ignore


class ILoadableItem(Generic[TItem]):
    _item: Optional[TItem]

    def __init__(
        self,
        init_fn: Callable[[], TItem],
        *,
        init: bool = False,
        force_keep: bool = False,
    ):
        self.init_fn = init_fn
        self.load_time = time.time()
        self.force_keep = force_keep
        self._item = init_fn() if init or force_keep else None

    def load(self, **kwargs: Any) -> TItem:
        self.load_time = time.time()
        if self._item is None:
            self._item = self.init_fn()
        return self._item

    def unload(self) -> None:
        self._item = None
        gc.collect()


class ILoadablePool(Generic[TItem]):
    pool: Dict[str, ILoadableItem]
    activated: Dict[str, ILoadableItem]

    # set `limit` to negative values to indicate 'no limit'
    def __init__(self, limit: int = -1):
        self.pool = {}
        self.activated = {}
        self.limit = limit
        if limit == 0:
            raise ValueError(
                "limit should either be negative "
                "(which indicates 'no limit') or be positive"
            )

    def __contains__(self, key: str) -> bool:
        return key in self.pool

    @property
    def num_activated(self) -> int:
        return len([v for v in self.activated.values() if not v.force_keep])

    def register(self, key: str, init_fn: Callable[[bool], ILoadableItem]) -> None:
        if key in self.pool:
            raise ValueError(f"key '{key}' already exists")
        init = self.limit < 0 or self.num_activated < self.limit
        loadable_item = init_fn(init)
        self.pool[key] = loadable_item
        if init or loadable_item.force_keep:
            self.activated[key] = loadable_item

    def get(self, key: str, **kwargs: Any) -> TItem:
        loadable_item = self.pool.get(key)
        if loadable_item is None:
            raise ValueError(f"key '{key}' does not exist")
        item = loadable_item.load(**kwargs)
        if key in self.activated:
            return item
        load_times = {
            key: item.load_time
            for key, item in self.activated.items()
            if not item.force_keep
        }
        print("> activated", self.activated)
        print("> load_times", load_times)
        earliest_key = list(sort_dict_by_value(load_times).keys())[0]
        self.activated.pop(earliest_key).unload()
        self.activated[key] = loadable_item
        time_format = "-".join(TIME_FORMAT.split("-")[:-1])
        print_info(
            f"'{earliest_key}' is unloaded to make room for '{key}' "
            f"(last updated: {time.strftime(time_format, time.localtime(loadable_item.load_time))})"
        )
        return item


class InjectionPack(BaseModel):
    index: Optional[int]
    field: str


class WorkNode(BaseModel):
    key: str = Field(
        ...,
        description="Key of the node, should be identical within the same workflow",
    )
    endpoint: str = Field(..., description="Algorithm endpoint of the node")
    injections: Dict[str, Union[InjectionPack, List[InjectionPack]]] = Field(
        ...,
        description=(
            "Injection map, maps 'key' from other `WorkNode` (A) to 'index' of A's results  & "
            "'field' of the algorithm's field. In runtime, we'll collect "
            "the (list of) results from the depedencies (other `WorkNode`) and "
            "inject the specific result (based on 'index') to the algorithm's field.\n"
            "> If external caches is provided, the 'key' could be the key of the external cache.\n"
            "> Hierarchy injection is also supported, you just need to set 'field' to:\n"
            ">> `a.b.c` to inject the result to data['a']['b']['c']\n"
            ">> `a.0.b` to inject the first result to data['a'][0]['b']\n"
        ),
    )
    data: Dict[str, Any] = Field(..., description="Algorithm's data")

    def to_item(self) -> Item["WorkNode"]:
        return Item(self.key, self)

    class Config:
        smart_union = True


class ToposortResult(NamedTuple):
    in_edges: Dict[str, Set[str]]
    hierarchy: List[List[Item[WorkNode]]]
    edge_labels: Dict[Tuple[str, str], str]


class Workflow(Bundle[WorkNode]):
    def copy(self) -> "Workflow":
        return Workflow.from_json(self.to_json())

    def push(self, node: WorkNode) -> None:
        return super().push(node.to_item())

    def toposort(self) -> ToposortResult:
        in_edges = {item.key: set() for item in self}
        out_degrees = {item.key: 0 for item in self}
        edge_labels: Dict[Tuple[str, str], str] = {}
        for item in self:
            for dep, packs in item.data.injections.items():
                in_edges[dep].add(item.key)
                out_degrees[item.key] += 1
                if not isinstance(packs, list):
                    packs = [packs]
                for pack in packs:
                    label_key = (item.key, dep)
                    existing_label = edge_labels.get(label_key)
                    if existing_label is None:
                        edge_labels[label_key] = pack.field
                    else:
                        edge_labels[label_key] = f"{existing_label}, {pack.field}"

        ready = [k for k, v in out_degrees.items() if v == 0]
        result = []
        while ready:
            layer = ready.copy()
            result.append(layer)
            ready.clear()
            for dep in layer:
                for node in in_edges[dep]:
                    out_degrees[node] -= 1
                    if out_degrees[node] == 0:
                        ready.append(node)

        if len(self) != sum(map(len, result)):
            raise ValueError("cyclic dependency detected")

        hierarchy = [list(map(self.get, layer)) for layer in result]
        return ToposortResult(in_edges, hierarchy, edge_labels)

    def get_dependency_path(self, target: str) -> ToposortResult:
        def dfs(key: str) -> None:
            if key in reachable:
                return
            reachable.add(key)
            for dep_key in self.get(key).data.injections:
                dfs(dep_key)

        reachable = set()
        dfs(target)
        in_edges, raw_hierarchy, edge_labels = self.toposort()
        hierarchy = []
        for raw_layer in raw_hierarchy:
            layer = []
            for item in raw_layer:
                if item.key in reachable:
                    layer.append(item)
            if layer:
                hierarchy.append(layer)
        return ToposortResult(in_edges, hierarchy, edge_labels)

    def to_json(self) -> List[Dict[str, Any]]:
        return [node.data.dict() for node in self]

    @classmethod
    def from_json(cls, data: List[Dict[str, Any]]) -> "Workflow":
        workflow = cls()
        for json in data:
            workflow.push(WorkNode(**json))
        return workflow

    def inject_caches(self, caches: Dict[str, Any]) -> "Workflow":
        for k in caches:
            self.push(WorkNode(key=k, endpoint="", injections={}, data={}))
        return self

    def render(
        self,
        *,
        target: Optional[str] = None,
        caches: Optional[Dict[str, Any]] = None,
        fig_w_ratio: int = 4,
        fig_h_ratio: int = 3,
        dpi: int = 200,
        node_size: int = 2000,
        node_shape: str = "s",
        node_color: str = "lightblue",
        layout: str = "multipartite_layout",
    ) -> TImage:
        if nx is None:
            raise ValueError("networkx is required for `render`")
        if plt is None or mpatches is None:
            raise ValueError("matplotlib is required for `render`")
        if Image is None:
            raise ValueError("PIL is required for `render`")
        # setup workflow
        workflow = self.copy()
        if caches is not None:
            workflow.inject_caches(caches)
        # setup graph
        G = nx.DiGraph()
        if target is None:
            target = self.last.key
        in_edges, hierarchy, edge_labels = workflow.get_dependency_path(target)
        # setup plt
        fig_w = max(fig_w_ratio * len(hierarchy), 8)
        fig_h = fig_h_ratio * max(map(len, hierarchy))
        plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        box = plt.gca().get_position()
        plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # map key to indices
        key2idx = {}
        for layer in hierarchy:
            for node in layer:
                key2idx[node.key] = len(key2idx)
        # add nodes
        for i, layer in enumerate(hierarchy):
            for node in layer:
                G.add_node(key2idx[node.key], subset=f"layer_{i}")
        # add edges
        for dep, links in in_edges.items():
            for link in links:
                if dep not in key2idx or link not in key2idx:
                    continue
                label = edge_labels[(link, dep)]
                G.add_edge(key2idx[dep], key2idx[link], label=label)
        # calculate positions
        layout_fn = getattr(nx, layout, None)
        if layout_fn is None:
            raise ValueError(f"unknown layout: {layout}")
        pos = layout_fn(G)
        # draw the nodes
        nodes_styles = dict(
            node_size=node_size,
            node_shape=node_shape,
            node_color=node_color,
        )
        nx.draw_networkx_nodes(G, pos, **nodes_styles)
        node_labels_styles = dict(
            font_size=18,
        )
        nx.draw_networkx_labels(G, pos, **node_labels_styles)
        # draw the edges
        nx_edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edges(
            G,
            pos,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=16,
            node_size=nodes_styles["node_size"],
            node_shape=nodes_styles["node_shape"],
        )
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx_edge_labels)
        # draw captions
        patches = [
            mpatches.Patch(color=node_color, label=f"{idx}: {key}")
            for key, idx in key2idx.items()
        ]
        plt.legend(handles=patches, bbox_to_anchor=(1, 0.5), loc="center left")
        # render
        plt.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return Image.open(buf)
