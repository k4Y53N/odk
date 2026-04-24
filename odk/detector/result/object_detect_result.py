from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    'ObjectInfo',
    'ObjectDetectResult',
]


@dataclass(slots=True)
class ObjectInfo:
    bbox: NDArray[np.float32]
    class_id: int
    score: float
    label: str


@dataclass(slots=True)
class ObjectDetectResult:
    bboxes: NDArray[np.float32]
    classes: NDArray[np.uint16]
    scores: NDArray[np.float32]
    class_label: list[str]
