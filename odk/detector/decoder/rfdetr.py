from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ..params import ObjectDetectParams
from ..result import ObjectDetectResult
from .decoder import Decoder
from .util import xywh_to_xyxy

__all__ = [
    'RFDetrDecoder',
]


BASE = Decoder[
    Sequence[NDArray[np.uint8]],
    list[ObjectDetectResult],
    ObjectDetectParams,
]


class RFDetrDecoder(BASE):
    @classmethod
    def from_engine(cls, engine):
        return RFDetrDecoder()

    def decode(self, input, output, params):
        # bbox = [batch size, N, 4]
        # scores = [batch size, N, num classes + 1 (last is background)]
        batch_bboxes, batch_scores = output
        batch_size = batch_scores.shape[0]
        num_anchors = batch_scores.shape[1]
        num_classes = batch_scores.shape[2]
        batch_classes = np.argmax(batch_scores, axis=2)
        batch_index = np.arange(batch_size)[:, None]  # (batch size, 1)
        anchor_index = np.arange(num_anchors)[None, :]  # (1, N)
        # (batch size, N)
        batch_scores = batch_scores[batch_index, anchor_index, batch_classes]
        batch_mask = batch_classes != num_classes - 1
        batch_mask &= batch_scores >= params.score_threshold
        results = list[ObjectDetectResult]()

        for i, image in enumerate(input):
            height, width = image.shape[:2]
            mask = batch_mask[i]
            bboxes: NDArray = batch_bboxes[i, mask]
            classes: NDArray = batch_classes[i, mask]
            scores: NDArray = batch_scores[i, mask]
            bboxes = xywh_to_xyxy(bboxes)
            bboxes[..., [0, 2]] *= width
            bboxes[..., [1, 3]] *= height
            results.append(
                ObjectDetectResult(
                    bboxes=bboxes.astype(np.float32),
                    classes=classes.astype(np.uint16),
                    scores=scores.astype(np.float32),
                    class_label=params.class_label,
                )
            )

        return results
