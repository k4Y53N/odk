import numpy as np
from numpy.typing import NDArray

__all__ = [
    'xywh_to_xyxy',
    'xyxy_to_xywh',
]


def xywh_to_xyxy(bboxes: NDArray[np.float32]) -> NDArray[np.float32]:
    """[x, y, w, h] -> [x1, y1, x2, y2]"""
    bboxes[..., 0] -= bboxes[..., 2] / 2
    bboxes[..., 1] -= bboxes[..., 3] / 2
    bboxes[..., 2] += bboxes[..., 0]
    bboxes[..., 3] += bboxes[..., 1]

    return bboxes


def xyxy_to_xywh(bboxes: NDArray[np.float32]) -> NDArray[np.float32]:
    """[x1, y1, x2, y2] -> [x, y, w, h]"""
    bboxes[..., 2] -= bboxes[..., 0]
    bboxes[..., 3] -= bboxes[..., 1]
    bboxes[..., 0] += bboxes[..., 2] / 2
    bboxes[..., 1] += bboxes[..., 3] / 2

    return bboxes
