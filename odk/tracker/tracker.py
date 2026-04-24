from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from ..detector.result import ObjectDetectResult

__all__ = [
    'Tracker',
]


class Tracker(ABC):
    @abstractmethod
    def update(self, result: ObjectDetectResult) -> NDArray[np.uint64]:
        """Update tracker state with new detections and return assigned track IDs.

        Associates the given detection results with existing tracks or creates
        new tracks, returning a unique track ID for each detection.

        Args:
            result (ObjectDetectResult): Detection result containing bounding
                boxes, class IDs, scores, and class labels for the current frame.

        Returns:
            NDArray[np.uint64]: Array of track IDs with shape ``(N,)``, one per
                detection in *result*, where ``N = len(result)``.
        """
