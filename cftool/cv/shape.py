import cv2
import math

import numpy as np

from typing import *
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from .utils import *
from ..misc import LoggingMixin, timing_context


class BBox(NamedTuple):
    # x, y 是中心点的坐标
    # angle 是以平面坐标系中 x 轴为 0°，y 轴为 90° 计的
    x: int
    y: int
    w: int
    h: int
    angle: float = 0.

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def center(self) -> Tuple[int, int]:
        return self.x, self.y

    @property
    def left_top(self) -> Tuple[int, int]:
        hw, hh = 0.5 * self.w, 0.5 * self.h
        ltx = -(hw * math.cos(self.angle) + hh * math.sin(self.angle))
        lty = hw * math.sin(self.angle) - hh * math.cos(self.angle)
        ltx, lty = map(int, [ltx + self.x, lty + self.y])
        return ltx, lty

    @property
    def right_bottom(self) -> Tuple[float, float]:
        hw, hh = 0.5 * self.w, 0.5 * self.h
        rbx = hw * math.cos(self.angle) + hh * math.sin(self.angle)
        rby = -(hw * math.sin(self.angle) - hh * math.cos(self.angle))
        rbx, rby = map(int, [rbx + self.x, rby + self.y])
        return rbx, rby

    def move(self, dx, dy) -> "BBox":
        new_x, new_y = self.x + dx, self.y + dy
        return BBox(new_x, new_y, self.w, self.h, self.angle)

    @classmethod
    def from_contour(cls, contour) -> "BBox":
        (x, y), (w, h), angle = cv2.minAreaRect(contour)
        return cls(*map(int, [x, y, w, h]), angle)

    @classmethod
    def from_contour_simplified(cls, contour) -> "BBox":
        x, y, w, h = cv2.boundingRect(contour)
        x, y, w, h = map(int, [x + 0.5 * w, y + 0.5 * h, w, h])
        return BBox(x, y, w, h, 0.)


class ShapeObject(LoggingMixin):
    def __init__(self,
                 gray: np.ndarray,
                 contour: np.ndarray = None,
                 *,
                 force_hull: bool = False):
        self.gray = gray
        self._mask = None
        self._mask_area = None
        self._edge = None
        self._contour = contour
        self._force_hull = force_hull
        self._contour_center = None
        self._contour_area = None
        self._bbox = self._bbox_middle = self._bbox_simplified = None

    @property
    def wh(self) -> Tuple[int, int]:
        if len(self.gray.shape) == 2:
            h, w = self.gray.shape
        else:
            h, w, channels = self.gray.shape
        return w, h

    @property
    def canvas_area(self) -> float:
        w, h = self.wh
        return w * h

    @property
    def bbox(self) -> BBox:
        if self._bbox is None:
            self._bbox = BBox.from_contour(self.contour)
        return self._bbox

    @property
    def bbox_middle(self) -> BBox:
        if self._bbox_middle is None:
            center_y = self.contour_center[1]
            y_start, y_end = center_y - 2, center_y + 2
            middle_cropped = self.gray[y_start:y_end, ...]
            cropped_bbox = ShapeObject(middle_cropped).bbox_simplified
            self._bbox_middle = cropped_bbox.move(0, y_start)
        return self._bbox_middle

    @property
    def bbox_simplified(self) -> BBox:
        if self._bbox_simplified is None:
            self._bbox_simplified = BBox.from_contour_simplified(self.contour)
        return self._bbox_simplified

    @property
    def edge(self) -> np.ndarray:
        if self._edge is None:
            self._edge = cv2.Canny(self.gray, 50, 200)
        return self._edge

    @property
    def mask(self) -> np.ndarray:
        if self._mask is None:
            self._mask = self.gray >= 127
        return self._mask

    @property
    def mask_area(self) -> float:
        if self._mask_area is None:
            self._mask_area = self.mask.sum().item()
        return self._mask_area

    @property
    def contour(self) -> Union[np.ndarray, None]:
        if self._contour is None:
            self._contour = self.get_contour(self.gray, self._force_hull)
        return self._contour

    @property
    def contour_center(self) -> Tuple[float, float]:
        if self._contour_center is None:
            self._contour_center = self.get_contour_center(self.contour)
        return self._contour_center

    @property
    def contour_area(self) -> float:
        if self._contour_area is None:
            if self.contour is None or len(self.contour) == 0:
                self._contour_area = 0.
            else:
                self._contour_area = self.get_contour_area(self.contour)
        return self._contour_area

    @property
    def is_valid(self) -> bool:
        return self.contour is not None and self.contour_area > 0. and self.mask_area < self.canvas_area

    def copy(self) -> "ShapeObject":
        return ShapeObject(self.gray.copy(), self.contour.copy())

    def move(self, dx, dy) -> "ShapeObject":
        """
        平移移动图形，
        ------------> dx
        |
        |    坐标系
        |
        v dy

        :param dx: 在横轴的移动长度，dx为正的时候是右移，dx为负的时候为左移
        :param dy: 在纵轴的移动长度，dy为正的时候是下移，dy为负的时候为上移
        :return:
        """
        moved = self.move_img(self.gray, dx, dy)
        return ShapeObject(moved, (self.contour + [dx, dy]).astype(np.int))

    def to_tight(self, *,
                 padding_ratio: float = 0.) -> "ShapeObject":
        ltx, lty = self.bbox_simplified.left_top
        rbx, rby = self.bbox_simplified.right_bottom
        cropped = self.gray[lty:rby, ltx:rbx]
        cropped_contour = self.contour - [ltx, lty]
        if padding_ratio <= 0.:
            return ShapeObject(cropped, cropped_contour)
        w, h = rbx - ltx, rby - lty
        x_padding = int(padding_ratio * w)
        y_padding = int(padding_ratio * h)
        new_w, new_h = w + 2 * x_padding, h + 2 * y_padding
        canvas = np.zeros([new_h, new_w], np.uint8)
        canvas[y_padding:y_padding+h, x_padding:x_padding+w] = cropped
        canvas_contour = cropped_contour + [x_padding, y_padding]
        return ShapeObject(canvas, canvas_contour)

    def to_center(self) -> "ShapeObject":
        w, h = self.wh
        current_center = self.contour_center
        target_center = w // 2, h // 2
        dx, dy = [tc - cc for tc, cc in zip(target_center, current_center)]
        return self.move(dx, dy)

    def match_center(self,
                     template: "ShapeObject",
                     *,
                     use_bbox: bool = False,
                     return_dxy: bool = False) -> Union["ShapeObject", Tuple[float, float]]:
        if not use_bbox:
            current_center = self.contour_center
            target_center = template.contour_center
        else:
            current_center = self.bbox_simplified.center
            target_center = template.bbox_simplified.center
        dx, dy = [tc - cc for tc, cc in zip(target_center, current_center)]
        if return_dxy:
            return dx, dy
        return self.move(dx, dy)

    def separate(self,
                 *,
                 min_area_threshold: float = 0.01,
                 **kwargs) -> List["ShapeObject"]:
        fg = self.gray >= 127
        bgx, bgy = np.vstack(np.nonzero(~fg))[..., 0]
        n_areas, areas_map = cv2.connectedComponents(self.gray)
        area_threshold = min_area_threshold * fg.sum()
        masks = [areas_map == i for i in range(n_areas) if i != areas_map[bgx, bgy]]
        thresholds = list(map(
            lambda mask: mask.astype(np.uint8) * 255,
            filter(lambda mask: mask.sum() > area_threshold, masks)
        ))
        return [ShapeObject(threshold, **kwargs) for threshold in thresholds]

    def get_scaled_img(self, scale: float, align_bottom: bool):
        return self.scale_img(
            self.gray, self.contour_center, scale,
            bbox=self.bbox_simplified, align_bottom=align_bottom
        )

    def get_scaled_contour(self, w_scale, h_scale=None, *, canvas_changed=False) -> np.ndarray:
        if h_scale is None:
            h_scale = w_scale
        cx, cy = self.contour_center
        cnt_norm = self.contour - [cx, cy]
        cnt_scaled = cnt_norm * [w_scale, h_scale]
        if canvas_changed:
            cx, cy = cx * w_scale, cy * h_scale
        return (cnt_scaled + [cx, cy]).astype(np.int)

    @staticmethod
    def _fetch_match_shape_scores(src, tgt):
        scores = {}
        for method_num in [1, 2, 3]:
            score = cv2.matchShapes(src, tgt, method_num, 0.0)
            scores[str(method_num)] = max(0., 10. - score)
        return scores

    def get_match_scores(self,
                         template: "ShapeObject",
                         matching_method: Union[str, Callable, List[Union[str, Callable]]] = "contour_all",
                         **kwargs) -> Dict[str, float]:

        if kwargs.get("trigger_logging", False):
            self._init_logging()

        def _raise():
            raise NotImplementedError(f"matching_method '{matching_method}' not defined")

        scores = defaultdict(float)
        matching_methods = matching_method
        if not isinstance(matching_method, (list, tuple)):
            matching_methods = [matching_method]

        for matching_method in matching_methods:
            with timing_context(self, matching_method):
                local_scores = {}
                if callable(matching_method):
                    local_scores["mean"] = matching_method(self.contour, template.contour)
                elif "contour" in matching_method:
                    if matching_method == "contour_all":
                        local_scores.update(self._fetch_match_shape_scores(self.contour, template.contour))
                    elif matching_method == "contour_assignment":
                        with timing_context(self, "template resize"):
                            template_resized = template.resize(self.wh)
                        with timing_context(self, "get contour"):
                            self_contour, resized_contour = self.contour, template_resized.contour
                        with timing_context(self, "diff matrix"):
                            diff_matrix = self_contour - resized_contour.transpose([1, 0, 2])
                            diff_matrix = np.linalg.norm(diff_matrix, axis=-1)
                        with timing_context(self, "linear sum assignment"):
                            row_ind, col_ind = linear_sum_assignment(diff_matrix)
                        with timing_context(self, "diff from assignment"):
                            diff = diff_matrix[row_ind, col_ind].mean()
                            diff /= math.sqrt(self.wh[0] ** 2 + self.wh[1] ** 2)
                        local_scores["mean"] = 1. - diff
                    else:
                        _raise()
                elif "filled" in matching_method:
                    resized, template_resized = map(ShapeObject.resize, [self, template])
                    gray, template_gray = resized.gray, template_resized.gray
                    if matching_method == "filled_shape_all":
                        local_scores.update(self._fetch_match_shape_scores(gray, template_gray))
                    elif matching_method == "filled_iou":
                        local_scores["mean"] = 1. - self.iou_distance(gray, template_gray)
                    else:
                        _raise()
                elif matching_method == "edge":
                    # TODO : 通过缓存优化这里
                    num = kwargs.get("edge_scale_num", 1)
                    floor = kwargs.get("edge_scale_floor", 0.8)
                    ceiling = kwargs.get("edge_scale_ceiling", 0.8)
                    resize_floor = kwargs.get("edge_resize_floor", 0.5)
                    resized_template, resized_res = template.resize_to(self, False, tight=True, area_definition="bbox")
                    if resized_res["resize_scale"] <= resize_floor:
                        local_scores["mean"] = -math.inf
                    else:
                        for scale in np.linspace(floor, ceiling, num):
                            scaled_template = resized_template.rescale(scale, tight=True, scale_contour=False)
                            match_score = cv2.matchTemplate(self.edge, scaled_template.edge, cv2.TM_CCOEFF_NORMED).max()
                            local_scores[f"{scale:6.4f}"] = match_score
                else:
                    _raise()
                for k, v in local_scores.items():
                    if k == "mean":
                        k = matching_method
                    scores[k] += v

        no_max, no_mean = "max" not in scores, "mean" not in scores
        if no_max or no_mean:
            max_score = max(scores.values())
            mean_score = sum(scores.values()) / len(scores)
            if no_max:
                scores["max"] = max_score
            if no_mean:
                scores["mean"] = mean_score
        return scores

    def get_min_distance(self, template: "ShapeObject") -> float:
        return np.min(np.linalg.norm(self.contour - template.contour[..., 0, :][None, ...], axis=-1))

    def resize(self,
               size: Tuple[float, float] = (300, 300),
               *,
               tight: bool = True,
               keep_aspect_ratio: bool = True) -> "ShapeObject":
        sw, sh = size
        if not tight:
            w, h = self.wh
            target = self
        else:
            tight = self.to_tight()
            w, h = tight.wh
            target = tight

        w_scale, h_scale = sw / w, sh / h
        if not keep_aspect_ratio:
            resized = cv2.resize(target.gray, size)
            resized_contour = target.get_scaled_contour(w_scale, h_scale, canvas_changed=True)
        else:
            if w_scale < h_scale:
                adjust_x = False
                target_scale = w_scale
            else:
                adjust_x = True
                target_scale = h_scale
            resized_contour = target.get_scaled_contour(target_scale, canvas_changed=True)
            target_size = tuple(map(int, map(round, [w * target_scale, h * target_scale])))
            resized_raw = cv2.resize(target.gray, target_size)
            resized = np.zeros(size, dtype=np.uint8)
            tx, ty = target_size
            if adjust_x:
                dx = int(0.5 * (sw - tx))
                resized[:ty, dx:dx + tx] = resized_raw
                resized_contour += [dx, 0]
            else:
                dy = int(0.5 * (sh - ty))
                resized[dy:dy + ty, :tx] = resized_raw
                resized_contour += [0, dy]
        return ShapeObject(resized, resized_contour)

    def rescale(self,
                scale: float,
                *,
                tight: bool = False,
                align_bottom: bool = False,
                scale_contour: bool = True) -> "ShapeObject":
        dy, scaled = self.get_scaled_img(scale, align_bottom)
        contour = None if not scale_contour else self.get_scaled_contour(scale)
        if contour is not None and align_bottom:
            contour += [0, dy]
        scaled_obj = ShapeObject(scaled, contour)
        if tight:
            scaled_obj = scaled_obj.to_tight()
        return scaled_obj

    def resize_to(self,
                  template: "ShapeObject",
                  match_center: bool,
                  *,
                  tight: bool = False,
                  scale_ceiling: int = None,
                  area_definition: str = "contour") -> Union[None, Tuple["ShapeObject", Dict[str, float]]]:
        """
        将 self.contour 缩放，使得缩放后与 template.contour 的面积相同
        :param template：
        :param match_center：是否将返回的 ShapeObject 的中心点与 template 重合
        :param tight：是否将返回的 ShapeObject 的画布贴紧主体物
        :param scale_ceiling：放缩的最大比例，None 代指无限制
        :param area_definition：["contour", "bbox"]，面积的定义
        :return:
        """

        (w, h), (wt, ht) = self.wh, template.wh
        w_scale, h_scale = wt / w, ht / h
        resized = cv2.resize(self.gray, (wt, ht))
        new_obj = ShapeObject(resized, self.get_scaled_contour(w_scale, h_scale, canvas_changed=True))
        if area_definition == "contour":
            template_area, new_area = template.contour_area, new_obj.contour_area
        elif area_definition == "bbox":
            template_area, new_area = template.bbox.area, new_obj.bbox.area
        else:
            raise NotImplementedError(f"area_definition '{area_definition}' is not defined")
        scale = (template_area / new_area) ** 0.5
        if scale_ceiling is not None and scale >= scale_ceiling:
            return
        _, scaled = new_obj.get_scaled_img(scale, False)
        scaled_obj = ShapeObject(scaled, new_obj.get_scaled_contour(scale))
        if tight:
            if match_center:
                print(f"Warning : match_center will not take effect because tight=True")
            scaled_obj = scaled_obj.to_tight()
        elif match_center:
            scaled_obj = scaled_obj.match_center(template)
        return scaled_obj, {"resize_scale": scale}

    def rotate_to(self,
                  template: "ShapeObject",
                  rotated_per_itr: int = 10,
                  distance_choice: str = "iou",
                  match_center: bool = True,
                  theta_floor: int = 0,
                  theta_ceiling: int = 360,
                  scale_ceiling: int = None) -> Union[None, Tuple["ShapeObject", Dict[str, float]]]:

        # 可以选择不同的distance function来定义匹配效果的好坏
        distance_function = {
            "iou": self.iou_distance
        }

        # 将 self.gray 缩放并将中心位置移到 template_img 的中心位置
        template_img = template.gray
        resize_result = self.resize_to(template, match_center, scale_ceiling=scale_ceiling)
        if resize_result is None:
            return
        resized, resize_res = resize_result
        center_pos = resized.contour_center

        if len(template_img.shape) == 2:
            rows, cols = template_img.shape
        else:
            rows, cols, channels = template_img.shape

        # 这里的iou_distance 是用1-iou_score后的，越小越好
        min_loss = distance_function[distance_choice](resized.gray, template_img)
        min_theta = 0
        rotated_img = resized.gray
        for theta in range(theta_floor, theta_ceiling, rotated_per_itr):
            m = cv2.getRotationMatrix2D(center_pos, theta, 1.0)
            # logger.info(type(M), M.shape)
            # 第三个参数是输出图像的尺寸中心
            tmp = cv2.warpAffine(resized.gray, m, (rows, cols))
            cur_loss = distance_function[distance_choice](template_img, tmp)
            if cur_loss < min_loss:
                min_loss = cur_loss
                min_theta = theta
                rotated_img = tmp
        res = {"theta": min_theta, "loss": min_loss, "resize_scale": resize_res["resize_scale"]}
        return ShapeObject(rotated_img), res

    def save_img(self, saving_path):
        cv2.imwrite(saving_path, self.gray)

    def save_edge(self, saving_path):
        cv2.imwrite(saving_path, self.edge)

    def save_contour(self, saving_path):
        w, h = self.wh
        draw_contours(w, h, [self.contour], saving_path=saving_path)

    @staticmethod
    def get_contour(gray, force_hull):
        """
        返回 gray 的轮廓，如果是单连通图像返回其外轮廓，如果是多个图形组成的返回其最小凸包
        :param gray: 512*512 的二值化图片
        :param force_hull: 是否强制使用最小凸包
        :return: 外形轮廓
        """
        contours, hierarchy = cv2.findContours(gray, mode=0, method=1)
        if not contours:
            return
        if len(contours) == 1 and not force_hull:
            hull = contours[0]
        else:
            all_cont = np.concatenate(contours, axis=0)
            hull = cv2.convexHull(all_cont)
        return hull

    @staticmethod
    def get_contour_area(contour):
        return cv2.contourArea(contour)

    @staticmethod
    def get_contour_center(contour):
        """
        获得形状轮廓的重心坐标
        :param contour:
        :return: 重心坐标
        """
        m = cv2.moments(contour)
        m00 = max(m['m00'], 1e-5)
        cx = int(m['m10'] / m00)
        cy = int(m['m01'] / m00)
        return cx, cy

    @staticmethod
    def scale_img(img, center, scale,
                  *,
                  bbox: BBox = None,
                  align_bottom: bool = False):
        """
        对输入的图像，在其中心位置进行缩放
        :param img:
        :param center:
        :param scale:
        :param bbox: 若 align_bottom 为 True，则要提供一个 bbox
        :param align_bottom: 是否让 scale 后的 bbox 的 bottom 和原 bbox 的 bottom 对齐
        :return:
        """
        m = cv2.getRotationMatrix2D(center, angle=0, scale=scale)
        h, w = img.shape[:2]
        scaled = cv2.warpAffine(img, m, (w, h))
        if not align_bottom:
            return 0, scaled
        original_bottom_y = bbox.right_bottom[1]
        scaled_y = ShapeObject(scaled).bbox_simplified.right_bottom[1]
        dy = int(round(original_bottom_y - scaled_y))
        moved = ShapeObject.move_img(scaled, 0, dy)
        return dy, moved

    @staticmethod
    def move_img(img, dx, dy):
        h, w = img.shape[:2]
        m = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float)
        return cv2.warpAffine(img, m, (w, h))

    @staticmethod
    def iou_distance(gray1, gray2):
        m1, m2 = gray1 >= 127, gray2 >= 127
        return 1 - (m1 & m2).sum() / (m1 | m2).sum()

    @classmethod
    def from_file(cls, img_path, **kwargs):
        reverse = kwargs.pop("reverse", False)
        gray = Reader(img_path).to_gray(reverse=reverse)
        if gray is None:
            return
        threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
        return cls(threshold, **kwargs)


__all__ = ["BBox", "ShapeObject"]
