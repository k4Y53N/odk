from abc import ABC, abstractmethod

from ..detector.result import ObjectDetectResult
from .result import ObjectTrackResult

__all__ = [
    'Tracker',
]


class Tracker(ABC):
    @abstractmethod
    def update(self, result: ObjectDetectResult) -> ObjectTrackResult: ...
