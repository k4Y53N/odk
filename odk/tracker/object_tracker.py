from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .configer import ObjectTrackConfiger
from .result import ObjectTrackResult
from .tracker import Tracker

__all__ = [
    'ObjectTracker',
]


class Image(Protocol):
    data: NDArray[np.uint8]


class ObjectDetectResult(Protocol):
    bboxes: NDArray[np.float32]
    classes: NDArray[np.uint16]
    scores: NDArray[np.float32]
    class_label: list[str]

    def class_filter(self, classes: NDArray[np.int_]) -> 'ObjectDetectResult': ...


class ObjectDetectFunction(Protocol):
    def __call__(
        self,
        image: Image,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        nms_mix_classes: bool = True,
    ) -> ObjectDetectResult: ...


class ObjectTracker:
    def __init__(
        self,
        fn: ObjectDetectFunction | None = None,
        configer: ObjectTrackConfiger | None = None,
    ):
        """Create an ObjectTracker with an optional detection function.

        Args:
            fn (ObjectDetectFunction | None, optional): Detection function used for
                automatic tracking via `track()`. If None, only manual `update()` is
                available. Defaults to None.
            configer (ObjectTrackConfiger | None, optional): Tracking configuration. If
                None, default ObjectTrackConfiger settings are used. Defaults to None.
        """
        if configer is None:
            configer = ObjectTrackConfiger()

        self._fn: ObjectDetectFunction | None = fn
        self._tracker: Tracker = configer.create()

    @classmethod
    def manual(cls, config: ObjectTrackConfiger | None = None) -> 'ObjectTracker':
        """Create an ObjectTracker without a detection function.

        Use this when you want to feed detection results manually via `update()`.

        Args:
            config (ObjectTrackConfiger | None, optional): Tracking configuration. If
                None, default ObjectTrackConfiger settings are used. Defaults to None.

        Returns:
            ObjectTracker: A tracker instance that only supports manual `update()`
                calls.
        """
        return ObjectTracker(None, config)

    @classmethod
    def from_config_path(
        cls,
        path: str,
        configer: ObjectTrackConfiger | None = None,
    ) -> 'ObjectTracker':
        """Create an ObjectTracker from a detector configuration file.

        Loads an ObjectDetector from the given config path and uses its
        `detect` method as the tracking detection function.

        Args:
            path (str): Path to the detector configuration JSON file.
            configer (ObjectTrackConfiger | None, optional): Tracking configuration.If
                None, default ObjectTrackConfiger settings are used. Defaults to None.

        Returns:
            ObjectTracker: A tracker instance with automatic detection via `track()`.
        """
        from ..detector import ObjectDetector

        detector = ObjectDetector.from_config_path(path)
        return ObjectTracker(detector.detect, configer)

    @classmethod
    def from_detect_fn(
        cls,
        fn: ObjectDetectFunction,
        configer: ObjectTrackConfiger | None = None,
    ) -> 'ObjectTracker':
        """Create an ObjectTracker from an existing detection function.

        Args:
            fn (ObjectDetectFunction): Detection function to use for automatic
                tracking via `track()`.
            configer (ObjectTrackConfiger | None, optional): Tracking configuration.If
                None, default ObjectTrackConfiger settings are used. Defaults to None.

        Returns:
            ObjectTracker: A tracker instance with automatic detection via `track()`.
        """
        return ObjectTracker(fn, configer)

    def update(self, result: ObjectDetectResult) -> ObjectTrackResult:
        """Assign track IDs to pre-computed detection results.

        Args:
            result (ObjectDetectResult): Detection results containing bboxes,
                classes, scores, and class labels.

        Returns:
            ObjectTrackResult: Detection results with assigned track IDs.
        """
        track_ids = self._tracker.update(result.bboxes, result.classes, result.scores)
        return ObjectTrackResult(
            bboxes=result.bboxes,
            track_ids=track_ids,
            classes=result.classes,
            scores=result.scores,
            class_label=result.class_label,
        )

    def track(
        self,
        image: Image,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        nms_mix_classes: bool = True,
        class_mask: NDArray[np.int_] | None = None,
    ) -> ObjectTrackResult:
        """Detect objects in an image and track them across frames.

        Runs the detection function on the image and assigns persistent track IDs to
        detected objects. Requires a detection function to have been provided at
        construction time.

        Args:
            image (Image): Input image to run detection on.
            score_threshold (float, optional): Minimum confidence score for
                detections. Defaults to 0.5.
            iou_threshold (float, optional): IoU threshold for non-maximum
                suppression. Defaults to 0.5.
            nms_mix_classes (bool, optional): Whether to apply NMS across all
                classes together. Defaults to True.
            class_mask (NDArray[np.int_] | None, optional): Class indices to
                keep. If provided, filters detections to only these classes.
                Defaults to None.

        Raises:
            RuntimeError: If no detection function was provided.

        Returns:
            ObjectTrackResult: Detection results with assigned track IDs.
        """
        if self._fn is None:
            raise RuntimeError(
                "Cannot track without a detector. "
                "Provide one via from_config_path() or from_detect_fn()."
            )

        result = self._fn(
            image=image,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            nms_mix_classes=nms_mix_classes,
        )

        if class_mask:
            result = result.class_filter(class_mask)

        return self.update(result)
