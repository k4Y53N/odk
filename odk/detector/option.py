from dataclasses import dataclass

__all__ = [
    'ObjectDetectOption',
]


@dataclass(slots=True)
class ObjectDetectOption:
    class_label: list[str]
    score_threshold: float = 0.5
    iou_threshold: float = 0.5
    nms_mix_classes: bool = True
