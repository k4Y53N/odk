import numpy as np
from numpy.typing import NDArray

from .kalman_filter import KalmanFilter

__all__ = [
    'KalmanTrack',
]

F = np.array(
    [
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ]
)

H = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ]
)


class KalmanTrack:
    """This class represents the internal state of individual tracked objects observed as bounding boxes."""

    def __init__(self, xysr: NDArray[np.float32]):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = F.copy()
        self.kf.H = H.copy()
        self.kf.R[2:, 2:] *= 10.0
        # give high uncertainty to the unobservable initial velocities
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = xysr[:, None]

    def project(self) -> NDArray[np.float32]:
        return self.kf.x[:4, 0]

    def update(self, z: NDArray[np.float32]):
        self.kf.update(z[:, None])

    def predict(self):
        if self.kf.x[6, 0] + self.kf.x[2, 0] <= 0:
            self.kf.x[6, 0] *= 0.0

        self.kf.predict()
