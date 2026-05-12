from dataclasses import dataclass

__all__ = [
    'ObjectDetectParams',
]


@dataclass(slots=True)
class ObjectDetectParams:
    class_label: list[str]
    score_threshold: float = 0.5
    iou_threshold: float = 0.5
    nms_mix_classes: bool = True
