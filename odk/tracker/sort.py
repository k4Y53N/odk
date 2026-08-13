from dataclasses import dataclass, field

import lap
import numpy as np
from numpy.typing import NDArray

from .tracker import Tracker

__all__ = [
    'SortTracker',
]

UINT64_MAX = 2**64 - 1

_dim_x = 7
_dim_z = 4
_F = np.array(
    [
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ],
    dtype=np.float32,
)
_H = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ],
    dtype=np.float32,
)
_P = np.eye(_dim_x, dtype=np.float32)
_Q = np.eye(_dim_x, dtype=np.float32)
_R = np.eye(_dim_z, dtype=np.float32)
_I = np.eye(_dim_x, dtype=np.float32)


_R[2:, 2:] *= 10.0
_P[4:, 4:] *= 1000.0
_P *= 10.0
_Q[-1, -1] *= 0.01
_Q[4:, 4:] *= 0.01


class KalmanTracker:
    """Fixed-capacity SORT Kalman states stored as a structure of arrays.

    The second axis of ``x`` identifies a track slot, so each state component
    is contiguous in memory. Slots are initialized, predicted, updated, and
    projected by their integer indices; the backing arrays never grow or move.
    """

    def __init__(self, capacity: int):
        self.x = np.zeros((_dim_x, capacity), dtype=np.float32)
        self.P = np.zeros((capacity, _dim_x, _dim_x), dtype=np.float32)

    def assign(self, indices: NDArray[np.int_], xysr: NDArray[np.float32]):
        """Initialize fixed slots from ``[x, y, scale, ratio]`` measurements."""
        self.x[:, indices] = 0.0
        self.x[:_dim_z, indices] = xysr.T
        self.P[indices] = _P

    def project(self, indices: NDArray[np.int_]) -> NDArray[np.float32]:
        """Return measurement-space states for the requested slots."""
        return self.x[:_dim_z, indices].T

    def predict(self, indices: NDArray[np.int_]):
        """Predict the state of the requested slots."""
        x = self.x[:, indices]
        invalid_scale = x[6] + x[2] <= 0
        x[6, invalid_scale] = 0.0
        # x = Fx
        x = _F @ x
        # P = FPF' + Q
        self.x[:, indices] = x
        self.P[indices] = _F @ self.P[indices] @ _F.T + _Q

    def update(self, indices: NDArray[np.int_], z: NDArray[np.float32]):
        """Update selected tracks with the corresponding measurement rows."""
        if not len(indices):
            return

        x = self.x[:, indices].T
        P = self.P[indices]
        # y = z - Hx (Residual between measurement and prediction)
        y = z - x[:, :_dim_z]
        PHT = P @ _H.T
        # S = HPH' + R (Project system uncertainty into measurement space)
        S = _H @ PHT + _R
        # K = PH'S^-1  (map system uncertainty into Kalman gain)
        K = np.linalg.solve(S, PHT.swapaxes(1, 2)).swapaxes(1, 2)
        # x = x + Ky  (predict new x with residual scaled by the Kalman gain)
        x += (K @ y[..., None])[..., 0]
        # P = (I-KH)P
        I_KH = _I - K @ _H
        P = I_KH @ P
        self.x[:, indices] = x.T
        self.P[indices] = P


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
    left, top, right, bottom = xyxy.T
    width = right - left
    height = bottom - top
    xyxy[..., 0] = (left + right) / 2
    xyxy[..., 1] = (top + bottom) / 2
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
class SortTracker(Tracker):
    threshold: float = 0.3
    capacity: int = 1024
    _frame: int = 0
    _track_id: int = 0
    _kalman: KalmanTracker = field(init=False)
    _active: NDArray[np.bool_] = field(init=False)
    _track_ids: NDArray[np.uint64] = field(init=False)
    _track_frames: NDArray[np.uint64] = field(init=False)

    def __post_init__(self):
        if self.capacity <= 0:
            raise ValueError('capacity must be positive')

        self._kalman = KalmanTracker(self.capacity)
        self._active = np.zeros(self.capacity, dtype=np.bool_)
        self._track_ids = np.zeros(self.capacity, dtype=np.uint64)
        self._track_frames = np.zeros(self.capacity, dtype=np.uint64)

    def __len__(self):
        """Return the number of currently active tracks."""
        return int(np.count_nonzero(self._active))

    def update(
        self,
        bboxes: NDArray[np.float32],
        classes: NDArray[np.uint16],
        scores: NDArray[np.float32],
    ) -> NDArray[np.uint64]:
        self._frame += 1
        detect_length = len(bboxes)

        if not detect_length:
            return self._when_detect_empty()

        self._remove_timeout()

        if np.all(~self._active):
            return self._when_track_empty(bboxes)

        active_slots = self._get_slots(True)
        self._kalman.predict(active_slots)
        track_xysrs = self._kalman.project(active_slots)
        buff_ids = self._track_ids[active_slots]

        track_bboxes = batch_xysr_to_xyxy(track_xysrs)
        iou = batch_iou(track_bboxes, bboxes)
        match_track, match_detect = linear_sum_assignment(-iou)
        mask = iou[match_track, match_detect] >= self.threshold
        match_track, match_detect = match_track[mask], match_detect[mask]
        not_match_mask = np.full(detect_length, True, dtype=np.bool_)
        not_match_mask[match_detect] = False
        xysrs = batch_xyxy_to_xysr(bboxes.copy())
        self._assign_track(active_slots[match_track], xysrs[match_detect])
        new_track_ids = self._extend_new_track(xysrs[not_match_mask])
        track_ids = np.empty(detect_length, dtype=np.uint64)
        track_ids[match_detect] = buff_ids[match_track]
        track_ids[not_match_mask] = new_track_ids

        return track_ids

    def _when_detect_empty(self) -> NDArray[np.uint64]:
        self._remove_timeout()
        return np.empty(0, dtype=np.uint64)

    def _when_track_empty(self, bboxes: NDArray[np.float32]) -> NDArray[np.uint64]:
        return self._extend_new_track(batch_xyxy_to_xysr(bboxes.copy()))

    def _remove_timeout(self):
        expired = self._active & ((self._frame - self._track_frames) > self.timeout)
        self._active[expired] = False

    def _get_slots(self, active: bool) -> NDArray[np.int_]:
        if active:
            return np.where(self._active)[0]

        return np.where(~self._active)[0]

    def _next_id(self, offset: int) -> NDArray[np.uint64]:
        ids = np.arange(offset, dtype=np.uint64) + np.uint64(self._track_id)
        self._track_id = (self._track_id + offset) % UINT64_MAX

        return ids

    def _assign_track(
        self,
        track_indices: NDArray[np.int_],
        xysrs: NDArray[np.float32],
    ):
        self._kalman.update(track_indices, xysrs)
        self._track_frames[track_indices] = self._frame

    def _extend_new_track(self, xysrs: NDArray[np.float32]) -> NDArray[np.uint64]:
        len_xysr = len(xysrs)

        if len_xysr > self.capacity:
            raise RuntimeError(
                f'Cannot assign {len_xysr} tracks to capacity {self.capacity}'
            )

        free_slots = self._get_slots(False)
        slots = free_slots[:len_xysr]
        missing = len_xysr - len(slots)

        if missing:
            active_slots = self._get_slots(True)
            oldest = np.argsort(self._track_frames[active_slots])[:missing]
            slots = np.concatenate((slots, active_slots[oldest]))

        next_ids = self._next_id(len_xysr)
        self._kalman.assign(slots, xysrs)
        self._active[slots] = True
        self._track_ids[slots] = next_ids
        self._track_frames[slots] = self._frame

        return next_ids
