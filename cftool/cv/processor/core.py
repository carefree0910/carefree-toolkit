import cv2
import math

import numpy as np
import matplotlib.pyplot as plt

from typing import *
from scipy.ndimage import rotate

from ..utils import Reader
from ...misc import check


filters = {
    "roberts": {1: np.array([[1, 0], [0, -1]]), 2: np.array([[0, 1], [-1, 0]])},
    "laplace": {
        1: np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
        2: np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]),
        3: np.array([[2, -1, 2], [-1, -4, -1], [2, -1, 2]]),
        4: np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]),
    },
    "prewitt": {
        1: np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
        2: np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]),
        3: np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        4: np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]),
        5: np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
        6: np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]]),
        7: np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
        8: np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]]),
    },
    "sobel": {
        1: np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
        2: np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]),
        3: np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        4: np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]),
        5: np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        6: np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]]),
        7: np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
        8: np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]]),
    },
    "robinson": {
        1: np.array([[1, 1, 1], [1, -2, 1], [-1, -1, -1]]),
        2: np.array([[1, 1, 1], [-1, -2, 1], [-1, -1, 1]]),
        3: np.array([[-1, 1, 1], [-1, -2, 1], [-1, 1, 1]]),
        4: np.array([[-1, -1, 1], [-1, -2, 1], [1, 1, 1]]),
        5: np.array([[-1, -1, -1], [1, -2, 1], [1, 1, 1]]),
        6: np.array([[1, -1, -1], [1, -2, -1], [1, 1, 1]]),
        7: np.array([[1, 1, -1], [1, -2, -1], [1, 1, -1]]),
        8: np.array([[1, 1, 1], [1, -2, -1], [1, -1, -1]]),
    },
    "kirsch": {
        1: np.array([[3, 3, 3], [3, 0, 3], [-5, -5, -5]]),
        2: np.array([[3, 3, 3], [-5, 0, 3], [-5, -5, 3]]),
        3: np.array([[-5, 3, 3], [-5, 0, 3], [-5, 3, 3]]),
        4: np.array([[-5, -5, 3], [-5, 0, 3], [3, 3, 3]]),
        5: np.array([[-5, -5, -5], [3, 0, 3], [3, 3, 3]]),
        6: np.array([[3, -5, -5], [3, 0, -5], [3, 3, 3]]),
        7: np.array([[3, 3, -5], [3, 0, -5], [3, 3, -5]]),
        8: np.array([[3, 3, 3], [3, 0, -5], [3, -5, -5]]),
    },
}


class Processor:
    def __init__(
        self,
        *,
        rgb: np.ndarray = None,
        bgr: np.ndarray = None,
        img_path: str = None,
    ):
        if rgb is not None or bgr is not None:
            img_array = bgr if bgr is not None else rgb[..., ::-1]
            img_array = img_array.copy()
            gray = img_array.mean(-1).astype(np.uint8)
        else:
            reader = Reader(img_path)
            if not reader.is_valid:
                raise ValueError(f"'{img_path}' is not a valid image")
            img_array = reader.img_array
            gray = reader.to_gray()
        self.img_gray = gray
        self.result = self.img_array = img_array
        self._in_pipeline = False

    def enter_pipeline(self) -> "Processor":
        self.result = self.img_array
        self._in_pipeline = True
        return self

    def exit_pipeline(self) -> "Processor":
        self._in_pipeline = False
        return self

    @property
    def result_is_gray(self) -> bool:
        return len(self.result.shape) == 2

    @property
    def current_array(self) -> np.ndarray:
        return self.result if self._in_pipeline else self.img_array

    @property
    def current_gray(self) -> np.ndarray:
        if not self._in_pipeline:
            return self.img_gray
        if self.result_is_gray:
            return self.result
        return cv2.cvtColor(self.result, cv2.COLOR_RGB2GRAY)

    # smoothen methods

    @check({"ksize": "int"})
    def average(self, *, ksize: int = 3) -> "Processor":
        self.result = cv2.blur(self.current_array, (ksize, ksize))
        return self

    @check({"ksize": "int"})
    def median(self, *, ksize: int = 3) -> "Processor":
        self.result = cv2.medianBlur(self.current_array, ksize)
        return self

    @check({"ksize": ["int", "odd"], "sigma_x": "float", "sigma_y": "float"})
    def gaussian(
        self, *, ksize: int = 3, sigma_x: float = 0.0, sigma_y: float = 0.0
    ) -> "Processor":
        self.result = cv2.GaussianBlur(
            self.current_array, (ksize, ksize), sigma_x, sigmaY=sigma_y
        )
        return self

    # sharpen & edge detection methods

    def _filter(self, name: str, i_type: int, is_edge: bool) -> np.ndarray:
        kernel = filters[name][i_type].copy()
        if not is_edge:
            kernel *= -1
            kernel[1][1] += 1
        return cv2.filter2D(self.current_array, -1, kernel)

    @check({"i_type": ["choices", list(range(1, 3))]})
    def roberts(self, *, i_type: int = 1, is_edge: bool = False):
        self.result = self._filter("roberts", i_type, is_edge)
        return self

    @check({"i_type": ["choices", list(range(1, 5))]})
    def laplace(self, *, i_type: int = 1, is_edge: bool = False):
        self.result = self._filter("laplace", i_type, is_edge)
        return self

    @check({"i_type": ["choices", list(range(1, 9))]})
    def prewitt(self, *, i_type: int = 1, is_edge: bool = False):
        self.result = self._filter("prewitt", i_type, is_edge)
        return self

    @check({"i_type": ["choices", list(range(1, 9))]})
    def sobel(self, *, i_type: int = 1, is_edge: bool = False):
        self.result = self._filter("sobel", i_type, is_edge)
        return self

    @check({"i_type": ["choices", list(range(1, 9))]})
    def robinson(self, *, i_type: int = 1, is_edge: bool = False):
        self.result = self._filter("robinson", i_type, is_edge)
        return self

    @check({"i_type": ["choices", list(range(1, 9))]})
    def kirsch(self, *, i_type: int = 1, is_edge: bool = False):
        self.result = self._filter("kirsch", i_type, is_edge)
        return self

    # corner detection methods

    @check(
        {
            "ksize": ["int", "odd"],
            "k": "float",
            "threshold": "float",
            "block_size": "int",
        }
    )
    def harris(
        self,
        *,
        ksize: int = 3,
        k: float = 0.04,
        threshold: float = 0.01,
        block_size: int = 2,
    ) -> "Processor":
        img = self.current_array.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dst = cv2.cornerHarris(gray, block_size, ksize, k)
        dst = cv2.dilate(dst, None)
        img[dst > threshold * dst.max()] = [0, 0, 255]
        self.result = img
        return self

    # degradation methods

    @check({"mean": "float", "sigma": "float"})
    def gaussian_noise(self, mean: float = 0.0, sigma: float = 51.0) -> "Processor":
        noise = np.random.normal(mean, sigma, self.current_array.shape)
        self.result = np.clip(self.current_array + noise, 0.0, 255.0).astype(np.uint8)
        return self

    @check({"length": "int", "angle": "float"})
    def motion(self, length: int = 8, angle: float = 0.0) -> "Processor":
        h, w, c = self.current_array.shape
        kernel = np.zeros([h, w])
        kernel[
            int(h / 2) : int(h / 2 + 1),
            int(w / 2 - length / 2) : int(w / 2 + length / 2),
        ] = 1
        kernel = rotate(kernel, angle, reshape=False)
        kernel /= kernel.sum()
        self.result = cv2.filter2D(self.current_array, -1, kernel)
        return self

    # thresholding methods

    @check({"max_val": "int", "eps": "float"})
    def global_thresh(
        self,
        max_val: int = 255,
        *,
        eps: float = 1.0,
        return_info: bool = False,
    ) -> Union["Processor", Any]:
        img = self.current_gray
        threshold = img.mean()
        while True:
            mask = img >= threshold
            g1, g2 = img[mask], img[~mask]
            m1, m2 = g1.mean(), g2.mean()
            new_threshold = 0.5 * (m1 + m2)
            if abs(threshold - new_threshold) <= eps:
                threshold = new_threshold
                break
            threshold = new_threshold
        if return_info:
            return g1, g2, m1, m2
        self.result = cv2.threshold(img, threshold, max_val, cv2.THRESH_BINARY)[1]
        return self

    @check(
        {
            "max_val": "int",
            "method": ["choices", ["mean", "gaussian"]],
            "block_size": ["int", "odd"],
            "c": "float",
        }
    )
    def adaptive_thresh(
        self,
        max_val: int = 255,
        *,
        method: str = "gaussian",
        block_size: int = 11,
        c: int = 2,
    ) -> "Processor":
        method = (
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if method == "gaussian"
            else cv2.ADAPTIVE_THRESH_MEAN_C
        )
        self.result = cv2.adaptiveThreshold(
            self.current_gray,
            max_val,
            method,
            cv2.THRESH_BINARY,
            block_size,
            c,
        )
        return self

    @check(
        {
            "max_val": "int",
            "ksize": "int",
            "method": ["choices", ["otsu", "gaussian"]],
            "eps": "float",
        }
    )
    def optimal_thresh(
        self,
        max_val: int = 255,
        *,
        ksize: int = 5,
        method: str = "otsu",
    ):
        if method == "otsu":
            blur = cv2.GaussianBlur(self.current_gray, (ksize, ksize), 0.0)
            self.result = cv2.threshold(
                blur, 0, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]
        elif method == "gaussian":
            g1, g2, m1, m2 = self.global_thresh(return_info=True)
            ng1, ng2 = map(len, [g1, g2])
            q, s1 = min(ng1 / ng2, ng2 / ng1), g1.std()
            threshold = 0.5 * (m1 + m2) + s1 ** 2 / (m2 - m1) * math.log(q / (1.0 - q))
            self.result = cv2.threshold(
                self.current_gray, threshold, max_val, cv2.THRESH_BINARY
            )[1]
        return self

    # visualization

    def visualize(self, saving_path: str = None) -> "Processor":
        if saving_path is not None:
            cv2.imwrite(saving_path, self.result)
        else:
            img = self.result if self.result_is_gray else self.result[..., ::-1]
            kwargs = {"cmap": "gray"} if self.result_is_gray else {}
            plt.imshow(img, **kwargs)
            plt.show()
        return self


__all__ = ["Processor"]
