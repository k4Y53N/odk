from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..engine import Engine
from ..option import ObjectDetectOption
from ..result import ObjectDetectResult
from .decoder import Decoder
from .nms import NMS, batch_nms

__all__ = [
    'YoloDecoder',
    'Yolov4Decoder',
    'Yolov7Decoder',
    'Yolov8Decoder',
    'Yolov9Decoder',
    'Yolov11Decoder',
    'YolovXDecoder',
]


def batch_mask_output(
    batch_bboxes: NDArray[np.float32],
    batch_scores: NDArray[np.float32],
    batch_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    batch_size = batch_mask.shape[0]
    num_classes = batch_scores.shape[2]
    batch_bboxes = [bboxes[mask] for bboxes, mask in zip(batch_bboxes, batch_mask)]
    batch_scores = [scores[mask] for scores, mask in zip(batch_scores, batch_mask)]
    max_num_bboxes = np.max([bboxes.shape[0] for bboxes in batch_bboxes])
    candidate_bboxes = np.zeros((batch_size, max_num_bboxes, 4), dtype=np.float32)
    candidate_scores = np.full(
        (batch_size, max_num_bboxes, num_classes),
        -np.inf,
        dtype=np.float32,
    )

    for i, (bboxes, scores) in enumerate(zip(batch_bboxes, batch_scores)):
        candidate_bboxes[i, : len(bboxes)] = bboxes
        candidate_scores[i, : len(scores)] = scores

    return candidate_bboxes, candidate_scores


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


class YoloDecoder(Decoder[ObjectDetectOption, list[ObjectDetectResult]], ABC):
    def __init__(self, height: int, width: int):
        self.height: int = height
        self.width: int = width

    @classmethod
    def from_engine(cls, engine: Engine) -> 'YoloDecoder':
        height = engine.input_shapes[0][2]
        width = engine.input_shapes[0][3]

        return cls(height, width)

    def decode(
        self,
        origin_input: Sequence[NDArray],
        model_output: Sequence[NDArray],
        option: ObjectDetectOption,
    ):
        batch_bboxes, batch_scores = self._get_bboxes_and_scores(
            model_output=model_output,
            score_threshold=option.score_threshold,
        )
        nmses = batch_nms(
            batch_bboxes=batch_bboxes,
            batch_scores=batch_scores,
            score_threshold=option.score_threshold,
            iou_threshold=option.iou_threshold,
            nms_mix_clsses=option.nms_mix_classes,
        )

        return self._decode_nms(
            nmses=nmses,
            origin_input=origin_input,
            class_label=option.class_label,
        )

    @abstractmethod
    def _get_bboxes_and_scores(
        self,
        model_output: Sequence[NDArray],
        score_threshold: float,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Extract bounding boxes and class scores from raw model output.

        Parses the YOLO model's raw output tensors into structured bounding boxes
        and per-class confidence scores. Implementations may optionally use
        ``score_threshold`` to pre-filter candidates for efficiency, but this
        is not required — NMS will handle filtering regardless.

        Args:
            model_output (Sequence[NDArray]): Raw output tensors from the YOLO
                model inference engine.
            score_threshold (float): Minimum confidence score.

        Returns:
            tuple[NDArray[np.float32], NDArray[np.float32]]: A tuple of
                ``(batch_bboxes, batch_scores)`` where:

                - **batch_bboxes** has shape ``(batch, N, 4)`` in
                  ``(x, y, w, h)`` format (centre-based, pixel coordinates
                  relative to the model input size).
                - **batch_scores** has shape ``(batch, N, num_classes)`` with
                  per-class confidence scores.
        """

    def _decode_nms(
        self,
        nmses: list[NMS],
        origin_input: Sequence[NDArray[np.uint8]],
        class_label: list[str],
    ) -> list[ObjectDetectResult]:
        results = list[ObjectDetectResult]()

        for nms, image in zip(nmses, origin_input):
            height, width = image.shape[:2]
            width_factor = width / self.width
            height_factor = height / self.height
            nms.bboxes = xywh_to_xyxy(nms.bboxes)
            nms.bboxes[..., [0, 2]] *= width_factor
            nms.bboxes[..., [1, 3]] *= height_factor
            result = ObjectDetectResult(
                bboxes=nms.bboxes,
                scores=nms.scores,
                classes=nms.classes,
                class_label=class_label,
            )
            results.append(result)

        return results


class Yolov4Decoder(YoloDecoder):
    def _get_bboxes_and_scores(
        self,
        model_output: Sequence[NDArray],
        score_threshold: float,
    ):
        # bboxes = [batch size, num_boxes, 1, 4]
        # scores = [batch_size, num_boxes, num_classes]
        # anchor ~= [0, 1]
        # left, top, right, bottom
        batch_bboxes, batch_scores = model_output
        batch_bboxes = batch_bboxes[:, :, 0, :]
        batch_mask = np.max(batch_scores, axis=2) >= score_threshold
        batch_bboxes, batch_scores = batch_mask_output(
            batch_bboxes,
            batch_scores,
            batch_mask,
        )
        batch_bboxes[..., [0, 2]] *= self.width
        batch_bboxes[..., [1, 3]] *= self.height
        batch_bboxes = xyxy_to_xywh(batch_bboxes)

        return batch_bboxes, batch_scores


class Yolov7Decoder(YoloDecoder):
    def _get_bboxes_and_scores(
        self,
        model_output: Sequence[NDArray],
        score_threshold: float,
    ):
        # [batch size, num_boxes, num_classes + 5]
        # [x, y, w, h, confidence, ...classes scores]
        # anchor ~= [0, model input size]
        # center x, center y, width, height
        anchor = model_output[0]
        batch_bboxes = anchor[..., :4]
        batch_confidence = anchor[..., 4]
        batch_scores = anchor[..., 5:]
        batch_mask = batch_confidence >= score_threshold
        batch_bboxes, batch_scores = batch_mask_output(
            batch_bboxes,
            batch_scores,
            batch_mask,
        )

        return batch_bboxes, batch_scores


class YolovXDecoder(YoloDecoder):
    def _get_bboxes_and_scores(
        self,
        model_output: Sequence[NDArray],
        score_threshold: float,
    ):
        # [batch size, num_class + 4, num_boxes]
        # [x, y, w, h, ...classes scores]
        # anchor ~= [0, model input size]
        # center x, center y, width, height
        infer = model_output[-1]
        infer = np.transpose(infer, (0, 2, 1))
        batch_bboxes = infer[..., :4]
        batch_scores = infer[..., 4:]

        return batch_bboxes, batch_scores


class Yolov8Decoder(YolovXDecoder): ...


class Yolov9Decoder(YolovXDecoder): ...


class Yolov11Decoder(YolovXDecoder): ...
