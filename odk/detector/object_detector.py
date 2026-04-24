from itertools import islice
from typing import Iterable, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray

from ..image import Image
from .configer import ObjectDetectConfiger, Version
from .decoder import Decoder, yolo_decoder
from .detector import Detector
from .encoder import ImageEncoder
from .option import ObjectDetectOption
from .result import ObjectDetectResult

__all__ = [
    'ObjectDetector',
]

BASE = Detector[ObjectDetectConfiger, ObjectDetectOption, list[ObjectDetectResult]]
DECODER_MAP: dict[Version, Decoder[ObjectDetectOption, list[ObjectDetectResult]]] = {
    Version.V4: yolo_decoder.Yolov4Decoder,
    Version.V7: yolo_decoder.Yolov7Decoder,
    Version.V8: yolo_decoder.Yolov8Decoder,
    Version.V9: yolo_decoder.Yolov9Decoder,
    Version.V11: yolo_decoder.Yolov11Decoder,
}
T = TypeVar('T')


def batched(iterable: Iterable[T], size: int) -> Iterable[tuple[T, ...]]:
    if size < 1:
        raise ValueError('size must be at least one')

    it = iter(iterable)

    while batch := tuple(islice(it, size)):
        yield batch


class ObjectDetector(BASE):
    def __init__(self, configer):
        super().__init__(configer)
        self.class_label = configer.class_label

    @classmethod
    def get_configer_class(cls):
        return ObjectDetectConfiger

    @classmethod
    def get_encoder_class(cls, configer):
        return ImageEncoder

    @classmethod
    def get_decoder_class(cls, configer):
        return DECODER_MAP[configer.version]

    def detect(
        self,
        image: Image,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        nms_mix_classes: bool = True,
    ) -> ObjectDetectResult:
        """Run object detection on a single image.

        Convenience wrapper around ``batch_detect`` for single-image inference.

        Args:
            image (Image): The input image to run detection on.
            score_threshold (float, optional): Minimum confidence score for a detection
                to be kept. Defaults to 0.5.
            iou_threshold (float, optional): IoU threshold used for non-maximum
                suppression. Defaults to 0.5.
            nms_mix_classes (bool, optional): If True, NMS is applied across all
                classes jointly; otherwise per-class. Defaults to True.

        Returns:
            ObjectDetectResult: Detection result for the input image.
        """
        return self.batch_detect(
            [image.data],
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            nms_mix_classes=nms_mix_classes,
        )[0]

    def batch_detect(
        self,
        images: Sequence[NDArray[np.uint8]],
        score_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        nms_mix_classes: bool = True,
        batch_size: int = 4,
    ) -> list[ObjectDetectResult]:
        """Run object detection on a batch of images.

        Args:
            images (Sequence[NDArray[np.uint8]]): Batch of input images as uint8 NumPy
                arrays.
            score_threshold (float, optional): Minimum confidence score for a detection
                to be kept. Defaults to 0.5.
            iou_threshold (float, optional): IoU threshold used for non-maximum
                suppression. Defaults to 0.5.
            nms_mix_classes (bool, optional): If True, NMS is applied across all
                classes jointly; otherwise per-class. Defaults to True.
            batch_size (int, optional): Maximum number of images to process per
                inference call. Set to 0 or negative to disable batching and process
                all images at once. Defaults to 4.

        Returns:
            list[ObjectDetectResult]: Detection results, one per input image.
        """
        if not len(images):
            return []

        option = ObjectDetectOption(
            class_label=self.class_label,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            nms_mix_classes=nms_mix_classes,
        )

        if batch_size > 0 and len(images) > batch_size:
            results = list[ObjectDetectResult]()

            for batch_images in batched(images, batch_size):
                results.extend(self.infer(origin=batch_images, option=option))

            return results

        return self.infer(origin=images, option=option)
