from abc import ABC, abstractmethod

from ..detector.result import ObjectDetectResult
from .result import ObjectTrackResult

__all__ = [
    'Tracker',
]


class Tracker(ABC):
    @abstractmethod
    def update(self, result: ObjectDetectResult) -> ObjectTrackResult:
        """Update the tracker with a new frame's detection result and return tracked
        objects.

        Args:
            result (ObjectDetectResult): Detection result containing bounding boxes,
                class IDs, scores, and class labels for the current frame.

        Returns:
            ObjectTrackResult: Tracking result with bounding boxes, assigned track IDs,
                class IDs, scores, and class labels for each tracked object.
        """
