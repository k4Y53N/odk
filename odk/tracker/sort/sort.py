from dataclasses import dataclass, field
from typing import Sequence

import lap
import numpy as np
from numpy.typing import NDArray

from ...detector.result import ObjectDetectResult
from ..tracker import Tracker
from .kalman_tracker import KalmanTrack

__all__ = [
    'SortTracker',
]

UINT64_MAX = 2**64 - 1


def batch_xysr_to_xyxy(xysr: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert bounding boxes from [x, y, s, r] to [left, top, right, bottom].

    Args:
        xysr (NDArray[np.float32]): [N, 4] in [x, y, s, r]
            where x, y are center coordinates, s is area, r is aspect ratio (w/h).

    Returns:
        NDArray[np.float32]: [N, 4] in [left, top, right, bottom]
    """
    width = np.sqrt(xysr[..., 2] * xysr[..., 3])
    height = xysr[..., 2] / width
    xysr[..., 0] -= width / 2
    xysr[..., 1] -= height / 2
    xysr[..., 2] = xysr[..., 0] + width
    xysr[..., 3] = xysr[..., 1] + height

    return xysr


def batch_xyxy_to_xysr(xyxy: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert bounding boxes from [left, top, right, bottom] to [x, y, s, r] in-place.

    Args:
        xyxy (NDArray[np.float32]): [N, 4] in [left, top, right, bottom]

    Returns:
        NDArray[np.float32]: [N, 4] in [x, y, s, r]
            where x, y are center coordinates, s is area, r is aspect ratio (w/h).
    """
    width = xyxy[..., 2] - xyxy[..., 0]
    height = xyxy[..., 3] - xyxy[..., 1]
    xyxy[..., 0] = (xyxy[..., 0] + xyxy[..., 2]) / 2
    xyxy[..., 1] = (xyxy[..., 1] + xyxy[..., 3]) / 2
    xyxy[..., 2] = width * height
    xyxy[..., 3] = width / height

    return xyxy


def linear_sum_assignment(
    iou_matrix: NDArray[np.float32],
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Solve the linear sum assignment problem using the Jonker-Volgenant algorithm.

    Finds the optimal assignment that minimizes the total cost in the given
    cost matrix. Used to match detections to tracked objects.

    Args:
        iou_matrix (NDArray[np.float32]): [N, M] cost matrix where N is the number
            of existing tracks and M is the number of new detections.

    Returns:
        tuple[NDArray[np.int_], NDArray[np.int_]]: A tuple of (row_indices, col_indices)
            representing the optimal assignment pairs.
    """
    _, x, _ = lap.lapjv(iou_matrix, extend_cost=True)
    row = np.where(x >= 0)[0]
    col = x[row]

    return row, col


def batch_iou(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
    """Computes IOU between two sets of bboxes in [x1, y1, x2, y2] format.

    Args:
        a (NDArray[np.float32]): [N, 4] bounding boxes.
        b (NDArray[np.float32]): [M, 4] bounding boxes.

    Returns:
        NDArray[np.float32]: [N, M] IOU matrix.
    """
    a_left, a_top, a_right, a_bottom = a.T
    b_left, b_top, b_right, b_bottom = b.T
    w = np.minimum(a_right[:, None], b_right) - np.maximum(a_left[:, None], b_left)
    h = np.minimum(a_bottom[:, None], b_bottom) - np.maximum(a_top[:, None], b_top)
    np.clip(w, 0, None, out=w)
    np.clip(h, 0, None, out=h)
    inter = w * h
    area_a = (a_right - a_left) * (a_bottom - a_top)
    area_b = (b_right - b_left) * (b_bottom - b_top)

    return inter / (area_a[:, None] + area_b[None, :] - inter)


@dataclass(slots=True)
class Track:
    track_id: int
    frame: int
    kalman_track: KalmanTrack

    @classmethod
    def from_xysr(cls, track_id: int, frame: int, xysr: NDArray[np.float32]) -> 'Track':
        return Track(
            track_id=track_id,
            frame=frame,
            kalman_track=KalmanTrack(xysr),
        )

    def predict(self) -> NDArray[np.float32]:
        self.kalman_track.predict()
        return self.kalman_track.project()

    def update(self, xysr: NDArray[np.float32]) -> NDArray[np.float32]:
        self.kalman_track.update(xysr)


@dataclass(slots=True)
class SortTracker(Tracker):
    _threshold: float = 0.3
    _frame: int = 0
    _track_id: int = 0
    _tracks: list[Track] = field(default_factory=list[Track])

    def update(self, result: ObjectDetectResult) -> NDArray[np.uint64]:
        self._frame += 1

        if not len(result):
            return self._when_detect_empty()

        if not len(self._tracks):
            return self._when_track_empty(result)

        self._remove_timeout()
        track_xysr = np.array([track.predict() for track in self._tracks])
        buff_ids = np.array([track.track_id for track in self._tracks], dtype=np.uint64)
        track_bboxes = batch_xysr_to_xyxy(track_xysr)
        iou = batch_iou(track_bboxes, result.bboxes)
        match_track, match_detect = linear_sum_assignment(-iou)
        mask = iou[match_track, match_detect] >= self._threshold
        match_track, match_detect = match_track[mask], match_detect[mask]
        not_match_detect = np.delete(np.arange(len(result)), match_detect)
        new_track_ids = self._extend_new_track(result, not_match_detect)
        self._assign_track(match_track, result, match_detect)
        track_ids = np.empty(len(result), dtype=np.uint64)
        track_ids[match_detect] = buff_ids[match_track]
        track_ids[not_match_detect] = new_track_ids

        return track_ids

    def _when_detect_empty(self) -> NDArray[np.uint64]:
        self._remove_timeout()
        return np.empty(0, dtype=np.uint64)

    def _when_track_empty(self, result: ObjectDetectResult) -> NDArray[np.uint64]:
        next_ids = [self._next_id() for _ in range(len(result))]
        bboxes = result.bboxes.copy()
        xysrs = batch_xyxy_to_xysr(bboxes)
        self._tracks.extend(
            Track.from_xysr(
                track_id=id,
                frame=self._frame,
                xysr=xysr,
            )
            for xysr, id in zip(xysrs, next_ids)
        )

        return np.array(next_ids, dtype=np.uint64)

    def _remove_timeout(self):
        expire_index = [
            i
            for i, track in enumerate(self._tracks)
            if (self._frame - track.frame) > self.timeout
        ]

        for index in expire_index[::-1]:
            self._tracks.pop(index)

    def _next_id(self) -> int:
        self._track_id = (self._track_id + 1) % UINT64_MAX
        return self._track_id

    def _assign_track(
        self,
        match_track: Sequence[int],
        result: ObjectDetectResult,
        match_detect: Sequence[int],
    ):
        bboxes = result.bboxes[match_detect]
        xysrs = batch_xyxy_to_xysr(bboxes)

        for index, xysr in zip(match_track, xysrs):
            track = self._tracks[index]
            track.update(xysr)
            track.frame = self._frame

    def _extend_new_track(
        self,
        result: ObjectDetectResult,
        mask: NDArray[np.int_],
    ) -> list[int]:
        bboxes = result.bboxes[mask]
        xysrs = batch_xyxy_to_xysr(bboxes)
        next_ids = [self._next_id() for _ in range(len(xysrs))]
        self._tracks.extend(
            Track.from_xysr(
                track_id=track_id,
                frame=self._frame,
                xysr=xysr,
            )
            for xysr, track_id in zip(xysrs, next_ids)
        )

        return next_ids
