import math

import numpy as np

from enum import Enum
from typing import List
from typing import Tuple
from typing import Union
from typing import TypeVar
from typing import Optional
from pydantic import BaseModel
from dataclasses import dataclass


class PivotType(str, Enum):
    LT = "lt"
    TOP = "top"
    RT = "rt"
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    LB = "lb"
    BOTTOM = "bottom"
    RB = "rb"


# 从左上角开始，「顺时针排布」的八个角点，加上最后的中心点
outer_pivots: List[PivotType] = [
    PivotType.LT,
    PivotType.TOP,
    PivotType.RT,
    PivotType.RIGHT,
    PivotType.RB,
    PivotType.BOTTOM,
    PivotType.LB,
    PivotType.LEFT,
]
all_pivots: List[PivotType] = outer_pivots + [PivotType.CENTER]
# 从左上角开始，「顺时针排布」的四个角点
corner_pivots: List[PivotType] = [
    PivotType.LT,
    PivotType.RT,
    PivotType.RB,
    PivotType.LB,
]
edge_pivots: List[PivotType] = [
    PivotType.TOP,
    PivotType.RIGHT,
    PivotType.BOTTOM,
    PivotType.LEFT,
]
mid_pivots: List[PivotType] = edge_pivots + [PivotType.CENTER]


def is_close(a: float, b: float, *, atol: float = 1.0e-6, rtol: float = 1.0e-4) -> bool:
    diff = abs(a - b)
    a = max(a, 1.0e-8)
    b = max(b, 1.0e-8)
    if diff >= atol or abs(diff / a) >= rtol or abs(diff / b) >= rtol:
        return False
    return True


@dataclass
class Point:
    x: float
    y: float

    @property
    def tuple(self) -> Tuple[float, float]:
        return self.x, self.y

    def __rmatmul__(self, other: "Matrix2D") -> "Point":
        x, y = self.x, self.y
        a, b, c, d, e, f = other.tuple
        return Point(x=a * x + c * y + e, y=b * x + d * y + f)

    def inside(self, box: "Matrix2D") -> bool:
        x, y = (box.inverse @ self).tuple
        return 0 <= x <= 1 and 0 <= y <= 1

    @classmethod
    def origin(cls) -> "Point":
        return cls(x=0.0, y=0.0)


@dataclass
class Line:
    start: Point
    end: Point

    def intersect(self, other: "Line", extendable: bool = False) -> Optional[Point]:
        x1, y1 = self.start.tuple
        x2, y2 = self.end.tuple
        x3, y3 = other.start.tuple
        x4, y4 = other.end.tuple
        x13 = x1 - x3
        x21 = x2 - x1
        x43 = x4 - x3
        y13 = y1 - y3
        y21 = y2 - y1
        y43 = y4 - y3
        denom = y43 * x21 - x43 * y21
        if is_close(denom, 0):
            return None
        uA = (x43 * y13 - y43 * x13) / denom
        uB = (x21 * y13 - y21 * x13) / denom
        if extendable or (0 <= uA <= 1 and 0 <= uB <= 1):
            return Point(x1 + uA * (x2 - x1), y1 + uA * (y2 - y1))
        return None

    def distance_to(self, target_line: "Line") -> float:
        x1, y1 = self.start.tuple
        x2, y2 = self.end.tuple
        x4, y4 = target_line.end.tuple
        dy = y1 - y2 or 10e-10
        k = (x1 - x2) / dy
        d = (k * (y2 - y4) + x4 - x2) / math.sqrt(1 + k**2)
        return d


class Matrix2DProperties(BaseModel):
    x: float
    y: float
    w: float
    h: float
    theta: float
    skew_x: float
    skew_y: float


TMatMul = TypeVar("TMatMul", bound=Union[Point, "Matrix2D"])


class Matrix2D(BaseModel):
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    def __matmul__(self, other: TMatMul) -> TMatMul:
        if isinstance(other, Point):
            return other.__rmatmul__(self)
        if isinstance(other, Matrix2D):
            a1, b1, c1, d1, e1, f1 = self.tuple
            a2, b2, c2, d2, e2, f2 = other.tuple
            return Matrix2D(
                a=a1 * a2 + c1 * b2,
                b=b1 * a2 + d1 * b2,
                c=a1 * c2 + c1 * d2,
                d=b1 * c2 + d1 * d2,
                e=a1 * e2 + c1 * f2 + e1,
                f=b1 * e2 + d1 * f2 + f1,
            )
        msg = f"unsupported operand type(s) for @: 'Matrix2D' and '{type(other)}'"
        raise TypeError(msg)

    @property
    def tuple(self) -> Tuple[float, float, float, float, float, float]:
        return self.a, self.b, self.c, self.d, self.e, self.f

    @property
    def x(self) -> float:
        return self.e

    @property
    def y(self) -> float:
        return self.f

    @property
    def position(self) -> Point:
        return Point(self.e, self.f)

    @property
    def w(self) -> float:
        return math.sqrt(self.a**2 + self.b**2)

    @property
    def h(self) -> float:
        return (self.a * self.d - self.b * self.c) / max(self.w, 1.0e-12)

    @property
    def wh(self) -> Tuple[float, float]:
        w = self.w
        h = (self.a * self.d - self.b * self.c) / max(w, 1.0e-12)
        return w, h

    @property
    def area(self) -> float:
        w, h = self.wh
        return w * abs(h)

    @property
    def theta(self) -> float:
        return -math.atan2(self.b, self.a)

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.a, self.c, self.e], [self.b, self.d, self.f]])

    @property
    def inverse(self) -> "Matrix2D":
        a, b, c, d, e, f = self.tuple
        ad = a * d
        bc = b * c
        return Matrix2D(
            a=d / (ad - bc),
            b=b / (bc - ad),
            c=c / (bc - ad),
            d=a / (ad - bc),
            e=(d * e - c * f) / (bc - ad),
            f=(b * e - a * f) / (ad - bc),
        )

    @property
    def lt(self) -> Point:
        return Point(self.e, self.f)

    @property
    def top(self) -> Point:
        return Point(0.5 * self.a + self.e, 0.5 * self.b + self.f)

    @property
    def rt(self) -> Point:
        return Point(self.a + self.e, self.b + self.f)

    @property
    def right(self) -> Point:
        return self @ Point(1.0, 0.5)

    @property
    def rb(self) -> Point:
        return self @ Point(1.0, 1.0)

    @property
    def bottom(self) -> Point:
        return self @ Point(0.5, 1.0)

    @property
    def lb(self) -> Point:
        return Point(self.c + self.e, self.d + self.f)

    @property
    def left(self) -> Point:
        return Point(0.5 * self.c + self.e, 0.5 * self.d + self.f)

    @property
    def center(self) -> Point:
        return self @ Point(0.5, 0.5)

    def pivot(self, pivot: PivotType) -> Point:
        return getattr(self, pivot.value)

    # lt -> rt -> rb -> lb
    @property
    def corner_points(self) -> List[Point]:
        return [self.pivot(pivot) for pivot in corner_pivots]

    # top -> right -> bottom -> left -> center
    @property
    def mid_points(self) -> List[Point]:
        return [self.pivot(pivot) for pivot in mid_pivots]

    # lt -> top -> rt -> right -> rb -> bottom -> lb -> left -> center
    @property
    def all_points(self) -> List[Point]:
        return [self.pivot(pivot) for pivot in all_pivots]

    # top -> right -> bottom -> left
    @property
    def edges(self) -> List[Line]:
        corners = self.corner_points
        return [Line(corner, corners[(i + 1) % 4]) for i, corner in enumerate(corners)]

    def decompose(self) -> Matrix2DProperties:
        w, h = self.wh
        a, b, c, d, e, f = self.tuple
        return Matrix2DProperties(
            x=e,
            y=f,
            w=w,
            h=h,
            theta=self.theta,
            skew_x=math.atan2(a * c + b * d, w**2),
            skew_y=0.0,
        )

    @classmethod
    def skew_matrix(
        cls,
        skew_x: float,
        skew_y: float,
        center: Optional[Point] = None,
    ) -> "Matrix2D":
        center = center or Point.origin()
        tx = math.tan(skew_x)
        ty = math.tan(skew_y)
        return cls(a=1, b=ty, c=tx, d=1, e=-tx * center.y, f=-ty * center.x)

    @classmethod
    def scale_matrix(
        cls,
        w: float,
        h: float,
        center: Optional[Point] = None,
    ) -> "Matrix2D":
        center = center or Point.origin()
        return cls(a=w, b=0, c=0, d=h, e=center.x * (1 - w), f=center.y * (1 - h))

    @classmethod
    def rotation_matrix(
        cls,
        theta: float,
        center: Optional[Point] = None,
    ) -> "Matrix2D":
        center = center or Point.origin()
        sin = math.sin(theta)
        cos = math.cos(theta)
        return cls(
            a=cos,
            b=-sin,
            c=sin,
            d=cos,
            e=(1.0 - cos) * center.x - center.y * sin,
            f=(1.0 - cos) * center.y + center.x * sin,
        )

    @classmethod
    def move_matrix(cls, x: float, y: float) -> "Matrix2D":
        return cls(a=1, b=0, c=0, d=1, e=x, f=y)

    @classmethod
    def from_properties(cls, properties: Matrix2DProperties) -> "Matrix2D":
        return (
            cls.move_matrix(properties.x, properties.y)
            @ cls.rotation_matrix(properties.theta)
            @ cls.scale_matrix(properties.w, properties.h)
            @ cls.skew_matrix(properties.skew_x, properties.skew_y)
        )


class HitTest:
    @staticmethod
    def line_line(a: Line, b: Line) -> bool:
        return a.intersect(b) is not None

    @staticmethod
    def line_box(a: Line, b: Matrix2D) -> bool:
        edges = b.edges
        for edge in edges:
            if HitTest.line_line(a, edge):
                return True
        return False

    @staticmethod
    def box_box(a: Matrix2D, b: Matrix2D) -> bool:
        b_edges = b.edges
        for b_edge in b_edges:
            if HitTest.line_box(b_edge, a):
                return True
        if a.position.inside(b):
            return True
        if b.position.inside(a):
            return True
        return False


__all__ = [
    "PivotType",
    "Point",
    "Line",
    "Matrix2D",
    "HitTest",
]
