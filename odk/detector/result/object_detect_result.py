from dataclasses import dataclass
from typing import Generator

import numpy as np
from numpy.typing import NDArray

__all__ = [
    'ObjectInfo',
    'ObjectDetectResult',
]


@dataclass(slots=True)
class ObjectInfo:
    bbox: NDArray[np.float32]
    class_id: int
    score: float
    label: str

    @property
    def left(self) -> float:
        return self.bbox[0]

    @property
    def top(self) -> float:
        return self.bbox[1]

    @property
    def right(self) -> float:
        return self.bbox[2]

    @property
    def bottom(self) -> float:
        return self.bbox[3]


@dataclass(slots=True)
class ObjectDetectResult:
    bboxes: NDArray[np.float32]
    classes: NDArray[np.uint16]
    scores: NDArray[np.float32]
    class_label: list[str]

    def __post_init__(self):
        if self.bboxes.shape[0] == 0:
            self.bboxes = self.bboxes.reshape((-1, 4))

    def __len__(self) -> int:
        return self.bboxes.shape[0]

    def __iter__(self) -> Generator[ObjectInfo, None, None]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index: int) -> ObjectInfo:
        class_id: int = self.classes[index]
        return ObjectInfo(
            bbox=self.bboxes[index],
            class_id=class_id,
            score=self.scores[index],
            label=self.class_label[class_id],
        )

    def copy(self) -> 'ObjectDetectResult':
        return ObjectDetectResult(
            bboxes=self.bboxes.copy(),
            classes=self.classes.copy(),
            scores=self.scores.copy(),
            class_label=self.class_label,
        )

    def filter(
        self,
        mask: NDArray[np.int_] | NDArray[np.bool_],
    ) -> 'ObjectDetectResult':
        return ObjectDetectResult(
            bboxes=self.bboxes[mask],
            classes=self.classes[mask],
            scores=self.scores[mask],
            class_label=self.class_label,
        )

    def class_filter(self, classes: NDArray[np.int_]) -> 'ObjectDetectResult':
        mask = np.isin(self.classes, classes)
        return self.filter(mask)

    def score_filter(self, score: float) -> 'ObjectDetectResult':
        mask = self.scores >= score
        return self.filter(mask)

    def add(self, x: float, y: float, inplace: bool = False) -> 'ObjectDetectResult':
        """Offset all bounding boxes by the given amounts.

        Args:
            x (float): Value to add to the horizontal (left/right) coordinates.
            y (float): Value to add to the vertical (top/bottom) coordinates.
            inplace (bool, optional): If True, modify in place; otherwise return a new
                copy. Defaults to False.

        Returns:
            ObjectDetectResult: The offset result.
        """
        instance = self

        if not inplace:
            instance = self.copy()

        instance.bboxes[..., [0, 2]] += x
        instance.bboxes[..., [1, 3]] += y

        return instance

    def subtract(
        self,
        x: float,
        y: float,
        inplace: bool = False,
    ) -> 'ObjectDetectResult':
        """Offset all bounding boxes by the given amounts.

        Args:
            x (float): Value to subtract from the horizontal (left/right) coordinates.
            y (float): Value to subtract from the vertical (top/bottom) coordinates.
            inplace (bool, optional): If True, modify in place; otherwise return a new
                copy. Defaults to False.

        Returns:
            ObjectDetectResult: The offset result.
        """
        return self.add(-x, -y, inplace)

    def plus(self, x: float, y: float, inplace: bool = False) -> 'ObjectDetectResult':
        """Alias for :meth:`add`."""
        return self.add(x, y, inplace)

    def minus(self, x: float, y: float, inplace: bool = False) -> 'ObjectDetectResult':
        """Alias for :meth:`subtract`."""
        return self.add(-x, -y, inplace)

    def multiply(
        self,
        x: float,
        y: float,
        inplace: bool = False,
    ) -> 'ObjectDetectResult':
        """Scale all bounding boxes by the given factors.

        Args:
            x (float): Scale factor for the horizontal (left/right) coordinates.
            y (float): Scale factor for the vertical (top/bottom) coordinates.
            inplace (bool, optional): If True, modify in place; otherwise return a new
                copy. Defaults to False.

        Returns:
            ObjectDetectResult: The scaled result.
        """
        instance = self

        if not inplace:
            instance = self.copy()

        instance.bboxes[..., [0, 2]] *= x
        instance.bboxes[..., [1, 3]] *= y

        return instance

    def divide(self, x: float, y: float, inplace: bool = False) -> 'ObjectDetectResult':
        """Divide all bounding box coordinates by the given factors.

        Args:
            x (float): Divisor for the horizontal (left/right) coordinates.
            y (float): Divisor for the vertical (top/bottom) coordinates.
            inplace (bool, optional): If True, modify in place; otherwise return a new
                copy. Defaults to False.

        Returns:
            ObjectDetectResult: The scaled result.
        """
        return self.multiply(1 / x, 1 / y, inplace)
