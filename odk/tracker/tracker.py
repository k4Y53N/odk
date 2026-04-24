from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from ..detector.result import ObjectDetectResult

__all__ = [
    'Tracker',
]


@dataclass(slots=True)
class Tracker(ABC):
    """Abstract base class for object trackers.

    Subclasses must implement :meth:`update` to associate detections across frames
    and assign persistent track IDs.

    Attributes:
        timeout (int): Number of frames a track can remain unmatched before it is
            removed. Defaults to ``10``.
    """

    timeout: int = 10

    @abstractmethod
    def update(self, result: ObjectDetectResult) -> NDArray[np.uint64]:
        """Update tracker state with new detections and return assigned track IDs.

        Associates the given detection results with existing tracks or creates new
        tracks, returning a unique track ID for each detection.

        Args:
            result (ObjectDetectResult): Detection result containing bounding boxes,
                class IDs, scores, and class labels for the current frame.

        Returns:
            NDArray[np.uint64]: Array of track IDs with shape ``(N,)``, one per
                detection in *result*, where ``N = len(result)``.
        """
