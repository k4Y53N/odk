from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

__all__ = [
    'NMS',
    'batch_nms',
]


@dataclass(slots=True)
class NMS:
    """Non-Maximum Suppression result for a single image.

    Attributes:
        bboxes (NDArray[np.float32]): Filtered bounding boxes of shape (N, 4) in
            (x, y, w, h) format.
        classes (NDArray[np.uint16]): Class indices of shape (N,) for each kept
            detection.
        scores (NDArray[np.float32]): Confidence scores of shape (N,) for each kept
            detection.
    """

    bboxes: NDArray[np.float32]
    classes: NDArray[np.uint16]
    scores: NDArray[np.float32]

    def __post_init__(self):
        if self.bboxes.shape[0] == 0:
            self.bboxes = self.bboxes.reshape((-1, 4))


def batch_nms(
    batch_bboxes: NDArray[np.float32],
    batch_scores: NDArray[np.float32],
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    nms_mix_clsses: bool = True,
) -> list[NMS]:
    """Perform batched Non-Maximum Suppression on detection outputs.

    Args:
        batch_bboxes (NDArray[np.float32]): Bounding boxes of shape (batch, N, 4) in
            (x, y, w, h) format.
        batch_scores (NDArray[np.float32]): Class scores of shape
            (batch, N, num_classes).
        score_threshold (float, optional): Minimum score to keep a detection. Defaults
            to 0.5.
        iou_threshold (float, optional): IoU threshold for suppressing overlapping
            boxes. Defaults to 0.5.
        nms_mix_clsses (bool, optional): If True, apply NMS across all classes
            together; if False, apply NMS per class independently. Defaults to True.

    Returns:
        list[NMS]: One NMS result per batch item, each containing filtered bboxes,
            classes, and scores.
    """
    batch_mask = np.max(batch_scores, axis=2) >= score_threshold
    nmses = list[NMS]()

    for bboxes, scores, mask in zip(batch_bboxes, batch_scores, batch_mask):
        bboxes = bboxes[mask]
        scores = scores[mask]
        classes = np.argmax(scores, axis=1)
        scores = scores[np.arange(classes.size), classes]
        nmsed_bboxes: list[NDArray[np.float32]] | NDArray[np.float32] = []
        nmsed_classes: list[NDArray[np.uint16]] | NDArray[np.uint16] = []
        nmsed_scores: list[NDArray[np.float32]] | NDArray[np.float32] = []

        if nms_mix_clsses:
            index = cv2.dnn.NMSBoxes(
                bboxes=bboxes,
                scores=scores,
                score_threshold=0,
                nms_threshold=iou_threshold,
            )

            if len(index) > 0:
                nmsed_bboxes = bboxes[index]
                nmsed_scores = scores[index]
                nmsed_classes = classes[index]
        else:
            for class_id in np.unique(classes):
                mask = classes == class_id
                class_bboxes = bboxes[mask]
                class_scores = scores[mask]
                index = cv2.dnn.NMSBoxes(
                    bboxes=class_bboxes,
                    scores=class_scores,
                    score_threshold=0,
                    nms_threshold=iou_threshold,
                )

                if len(index) == 0:
                    continue

                nmsed_bboxes.extend(class_bboxes[index])
                nmsed_scores.extend(class_scores[index])
                nmsed_classes.extend(np.full(len(index), class_id, dtype=np.uint16))

        nmses.append(
            NMS(
                bboxes=np.asarray(nmsed_bboxes, dtype=np.float32),
                classes=np.asarray(nmsed_classes, dtype=np.uint16),
                scores=np.asarray(nmsed_scores, dtype=np.float32),
            )
        )

    return nmses
