from collections.abc import Generator
from dataclasses import dataclass

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

    @property
    def width(self) -> float:
        return max(self.right - self.left, 0)

    @property
    def height(self) -> float:
        return max(self.bottom - self.top, 0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def center_point(self) -> tuple[float, float]:
        return self.center_x, self.center_y


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

    @property
    def left(self) -> NDArray[np.float32]:
        """Left (x-min) coordinates of all bounding boxes.


        Returns:
            NDArray[np.float32]: 1-D array of left edge values.
        """
        return self.bboxes[..., 0]

    @property
    def top(self) -> NDArray[np.float32]:
        """Top (y-min) coordinates of all bounding boxes.


        Returns:
            NDArray[np.float32]: 1-D array of top edge values.
        """
        return self.bboxes[..., 1]

    @property
    def right(self) -> NDArray[np.float32]:
        """Right (x-max) coordinates of all bounding boxes.


        Returns:
            NDArray[np.float32]: 1-D array of right edge values.
        """
        return self.bboxes[..., 2]

    @property
    def bottom(self) -> NDArray[np.float32]:
        """Bottom (y-max) coordinates of all bounding boxes.


        Returns:
            NDArray[np.float32]: 1-D array of bottom edge values.
        """
        return self.bboxes[..., 3]

    @property
    def width(self) -> NDArray[np.float32]:
        """Widths of all bounding boxes, clamped to a minimum of zero.


        Returns:
            NDArray[np.float32]: 1-D array of bounding box widths.
        """
        return np.clip(self.right - self.left, 0)

    @property
    def height(self) -> NDArray[np.float32]:
        """Heights of all bounding boxes, clamped to a minimum of zero.


        Returns:
            NDArray[np.float32]: 1-D array of bounding box heights.
        """
        return np.clip(self.bottom - self.top, 0)

    @property
    def area(self) -> NDArray[np.float32]:
        """Areas of all bounding boxes (width * height).


        Returns:
            NDArray[np.float32]: 1-D array of bounding box areas.
        """
        return self.width * self.height

    @property
    def center_x(self) -> NDArray[np.float32]:
        """Horizontal centre coordinates of all bounding boxes.

        Returns:
            NDArray[np.float32]: 1-D array of x-centre values ((left + right) / 2).
        """
        return np.sum(self.bboxes[..., [0, 2]], axis=-1) / 2

    @property
    def center_y(self) -> NDArray[np.float32]:
        """Vertical centre coordinates of all bounding boxes.

        Returns:
            NDArray[np.float32]: 1-D array of y-centre values ((top + bottom) / 2).
        """
        return np.sum(self.bboxes[..., [1, 3]], axis=-1) / 2

    @property
    def center_point(self) -> NDArray[np.float32]:
        """Centre points of all bounding boxes.

        Returns:
            NDArray[np.float32]: 2-D array of shape (N, 2) where each row is
                ``[center_x, center_y]`` for the corresponding bounding box.
        """
        points = np.empty((len(self), 2), dtype=np.float32)
        points[..., 0] = self.center_x
        points[..., 1] = self.center_y

        return points

    def copy(self) -> 'ObjectDetectResult':
        """Return a deep copy of this result.

        Returns:
            ObjectDetectResult: A new instance with copied arrays.
        """
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
        """Return a new result containing only the elements selected by *mask*.

        Args:
            mask (NDArray[np.int_] | NDArray[np.bool_]): Boolean or integer index
                array used to select detections.

        Returns:
            ObjectDetectResult: The filtered result.
        """
        return ObjectDetectResult(
            bboxes=self.bboxes[mask],
            classes=self.classes[mask],
            scores=self.scores[mask],
            class_label=self.class_label,
        )

    def class_filter(self, classes: NDArray[np.int_]) -> 'ObjectDetectResult':
        """Return a new result keeping only detections whose class is in *classes*.

        Args:
            classes (NDArray[np.int_]): Array of class IDs to keep.

        Returns:
            ObjectDetectResult: The filtered result.
        """
        mask = np.isin(self.classes, classes)
        return self.filter(mask)

    def score_filter(self, threshold: float) -> 'ObjectDetectResult':
        """Return a new result keeping only detections with score >= *threshold*.

        Args:
            threshold (float): Minimum confidence score to keep.

        Returns:
            ObjectDetectResult: The filtered result.
        """
        mask = self.scores >= threshold
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
