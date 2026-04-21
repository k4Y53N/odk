from dataclasses import dataclass
from typing import Generator

import numpy as np
from numpy.typing import NDArray

__all__ = [
    'ObjectTrackInfo',
    'ObjectTrackResult',
]


@dataclass(slots=True)
class ObjectTrackInfo:
    bbox: NDArray[np.float32]
    track_id: int
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
    def bbox_id(self) -> int:
        """Deprecated alias for :attr:`track_id`."""
        return self.track_id

    @bbox_id.setter
    def bbox_id(self, id: int):
        self.track_id = id


@dataclass(slots=True)
class ObjectTrackResult:
    bboxes: NDArray[np.float32]
    track_ids: NDArray[np.uint64]
    classes: NDArray[np.uint16]
    scores: NDArray[np.float32]
    class_label: list[str]

    def __post_init__(self):
        if self.bboxes.shape[0] == 0:
            self.bboxes = self.bboxes.reshape((-1, 4))

    def __len__(self):
        return self.bboxes.shape[0]

    def __iter__(self) -> Generator[ObjectTrackInfo, None, None]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index: int) -> ObjectTrackInfo:
        class_id = self.classes[index]
        return ObjectTrackInfo(
            bbox=self.bboxes[index],
            track_id=self.track_ids[index],
            class_id=self.classes[index],
            score=self.scores[index],
            label=self.class_label[class_id],
        )

    @property
    def bbox_ids(self):
        """Deprecated alias for :attr:`track_ids`."""
        return self.track_ids

    @bbox_ids.setter
    def bbox_ids(self, track_ids: NDArray[np.uint64]):
        self.track_ids = track_ids

    def copy(self) -> 'ObjectTrackResult':
        """Return a deep copy of this result.

        Returns:
            ObjectTrackResult: A new instance with copied arrays.
        """
        return ObjectTrackResult(
            bboxes=self.bboxes.copy(),
            track_ids=self.track_ids.copy(),
            classes=self.classes.copy(),
            scores=self.scores.copy(),
            class_label=self.class_label,
        )

    def filter(
        self,
        mask: NDArray[np.int_] | NDArray[np.bool_],
    ) -> 'ObjectTrackResult':
        """Return a new result containing only the elements selected by *mask*.

        Args:
            mask (NDArray[np.int_] | NDArray[np.bool_]): Boolean or integer index
                array used to select detections.

        Returns:
            ObjectTrackResult: The filtered result.
        """
        return ObjectTrackResult(
            bboxes=self.bboxes[mask],
            track_ids=self.track_ids[mask],
            classes=self.classes[mask],
            scores=self.scores[mask],
            class_label=self.class_label,
        )

    def id_filter(self, track_ids: NDArray[np.int_]) -> 'ObjectTrackResult':
        """Return a new result keeping only detections whose track ID is in
            *track_ids*.

        Args:
            track_ids (NDArray[np.int_]): Array of track IDs to keep.

        Returns:
            ObjectTrackResult: The filtered result.
        """
        mask = np.isin(self.track_ids, track_ids)
        return self.filter(mask)

    def class_filter(self, classes: NDArray[np.int_]) -> 'ObjectTrackResult':
        """Return a new result keeping only detections whose class is in *classes*.

        Args:
            classes (NDArray[np.int_]): Array of class IDs to keep.

        Returns:
            ObjectTrackResult: The filtered result.
        """
        mask = np.isin(self.classes, classes)
        return self.filter(mask)

    def score_filter(self, threshold: float) -> 'ObjectTrackResult':
        """Return a new result keeping only detections with score >= *threshold*.

        Args:
            threshold (float): Minimum confidence score to keep.

        Returns:
            ObjectTrackResult: The filtered result.
        """
        mask = self.scores >= threshold
        return self.filter(mask)

    def add(self, x: float, y: float, inplace: bool = False) -> 'ObjectTrackResult':
        """Offset all bounding boxes by the given amounts.

        Args:
            x (float): Value to add to the horizontal (left/right) coordinates.
            y (float): Value to add to the vertical (top/bottom) coordinates.
            inplace (bool, optional): If True, modify in place; otherwise return a new
                copy. Defaults to False.

        Returns:
            ObjectTrackResult: The offset result.
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
    ) -> 'ObjectTrackResult':
        """Offset all bounding boxes by the given amounts.

        Args:
            x (float): Value to subtract from the horizontal (left/right) coordinates.
            y (float): Value to subtract from the vertical (top/bottom) coordinates.
            inplace (bool, optional): If True, modify in place; otherwise return a new
                copy. Defaults to False.

        Returns:
            ObjectTrackResult: The offset result.
        """
        return self.add(-x, -y, inplace)

    def plus(self, x: float, y: float, inplace: bool = False) -> 'ObjectTrackResult':
        """Alias for :meth:`add`."""
        return self.add(x, y, inplace)

    def minus(self, x: float, y: float, inplace: bool = False) -> 'ObjectTrackResult':
        """Alias for :meth:`subtract`."""
        return self.add(-x, -y, inplace)

    def multiply(
        self,
        x: float,
        y: float,
        inplace: bool = False,
    ) -> 'ObjectTrackResult':
        """Scale all bounding boxes by the given factors.

        Args:
            x (float): Scale factor for the horizontal (left/right) coordinates.
            y (float): Scale factor for the vertical (top/bottom) coordinates.
            inplace (bool, optional): If True, modify in place; otherwise return a new
                copy. Defaults to False.

        Returns:
            ObjectTrackResult: The scaled result.
        """
        instance = self

        if not inplace:
            instance = self.copy()

        instance.bboxes[..., [0, 2]] *= x
        instance.bboxes[..., [1, 3]] *= y

        return instance

    def divide(self, x: float, y: float, inplace: bool = False) -> 'ObjectTrackResult':
        """Divide all bounding box coordinates by the given factors.

        Args:
            x (float): Divisor for the horizontal (left/right) coordinates.
            y (float): Divisor for the vertical (top/bottom) coordinates.
            inplace (bool, optional): If True, modify in place; otherwise return a new
                copy. Defaults to False.

        Returns:
            ObjectTrackResult: The scaled result.
        """
        return self.multiply(1 / x, 1 / y, inplace)
