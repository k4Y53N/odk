from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    'Tracker',
]


@dataclass(slots=True)
class Tracker(ABC):
    timeout: int = 10

    @abstractmethod
    def update(
        self,
        bboxes: NDArray[np.float32],
        classes: NDArray[np.uint16],
        scores: NDArray[np.float32],
    ) -> NDArray[np.uint64]:
        """Associate detections with existing tracks and assign persistent track IDs.

        Args:
            bboxes (NDArray[np.float32]): Bounding boxes of shape ``(N, 4)`` in
            ``[left, top, right, bottom]`` pixel coordinates.
            classes (NDArray[np.uint16]): Class IDs of shape ``(N,)`` for each
                detection.
            scores (NDArray[np.float32]): Scores of shape ``(N,)`` for
                each detection.

        Returns:
            NDArray[np.uint64]: Track IDs of shape ``(N,)``.
        """
