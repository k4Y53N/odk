from typing import Protocol

from ..detector import ObjectDetector, ObjectDetectResult
from ..image import Image
from .option import TrackOption
from .result import ObjectTrackResult
from .tracker import Tracker

__all__ = [
    'ObjectTracker',
]


class ObjectDetectAPI(Protocol):
    def __call__(
        self,
        image: Image,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        nms_mix_classes: bool = True,
    ) -> ObjectDetectResult: ...


class ObjectTracker:
    def __init__(self, fn: ObjectDetectAPI, option: TrackOption | None = None):
        if option is None:
            option = TrackOption()

        self._fn: ObjectDetectAPI = fn
        self._tracker: Tracker = option.create()

    @classmethod
    def from_config_path(
        cls,
        path: str,
        option: TrackOption | None = None,
    ) -> 'ObjectTracker':
        """Create an ObjectTracker from an object detector configuration file path.

        Args:
            path (str): Path to the detector configuration file.
            option (TrackOption | None, optional): Tracking options. Defaults to None.

        Returns:
            ObjectTracker: An instance of ObjectTracker initialized with the detector
                and options.
        """
        detector = ObjectDetector.from_config_path(path)
        return ObjectTracker(detector.detect, option)

    @classmethod
    def from_object_detector(
        cls,
        detector: ObjectDetector,
        option: TrackOption | None = None,
    ) -> 'ObjectTracker':
        """Create an ObjectTracker from an existing ObjectDetector instance.

        Args:
            detector (ObjectDetector): An instance of ObjectDetector.
            option (TrackOption | None, optional): Tracking options. Defaults to None.

        Returns:
            ObjectTracker: An instance of ObjectTracker initialized with the detector
                and options.
        """
        return ObjectTracker(detector.detect, option)

    def track(
        self,
        image: Image,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        nms_mix_classes: bool = True,
        class_mask: list[int] | None = None,
    ) -> ObjectTrackResult:
        """Run object detection and tracking on an image.

        Args:
            image (Image): The input image to process.
            score_threshold (float, optional): Minimum score threshold for detections.
                Defaults to 0.5.
            iou_threshold (float, optional): IOU threshold for non-maximum suppression.
                Defaults to 0.5.
            nms_mix_classes (bool, optional): Whether to mix classes in NMS.
                Defaults to True.
            class_mask (list[int] | None, optional): List of class indices to filter
                detections. Defaults to None.

        Returns:
            ObjectTrackResult: The result of object tracking.
        """
        result = self._fn(
            image=image,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            nms_mix_classes=nms_mix_classes,
        )

        if class_mask:
            result = result.class_filter(class_mask)

        return self._tracker.track(result)

    def detect(
        self,
        image: Image,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        nms_mix_classes: bool = True,
        class_mask: list[int] | None = None,
    ) -> ObjectTrackResult:
        """Detect and track objects in an image (alias for track).

        Args:
            image (Image): The input image to process.
            score_threshold (float, optional): Minimum score threshold for detections.
                Defaults to 0.5.
            iou_threshold (float, optional): IOU threshold for non-maximum suppression.
                Defaults to 0.5.
            nms_mix_classes (bool, optional): Whether to mix classes in NMS.
                Defaults to True.
            class_mask (list[int] | None, optional): List of class indices to filter detections.
                Defaults to None.

        Returns:
            ObjectTrackResult: The result of object tracking.
        """
        return self.track(
            image=image,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            nms_mix_classes=nms_mix_classes,
            class_mask=class_mask,
        )
