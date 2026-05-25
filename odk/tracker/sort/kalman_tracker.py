from dataclasses import dataclass, field

import numpy as np
from numpy.linalg import inv
from numpy.typing import NDArray

__all__ = [
    'KalmanTrack',
]


_dim_x = 7
_dim_z = 4
_X = np.zeros((_dim_x, 1), dtype=np.float32)
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
_M = np.zeros((_dim_z, _dim_z), dtype=np.float32)
_I = np.eye(_dim_x, dtype=np.float32)


_R[2:, 2:] *= 10.0
_P[4:, 4:] *= 1000.0
_P *= 10.0
_Q[-1, -1] *= 0.01
_Q[4:, 4:] *= 0.01


@dataclass(slots=True)
class KalmanFilter:
    dim_x: int = _dim_x
    dim_z: int = _dim_z
    x: NDArray[np.float32] = field(default_factory=_X.copy)
    P: NDArray[np.float32] = field(default_factory=_P.copy)
    Q: NDArray[np.float32] = field(default_factory=_Q.copy)
    F: NDArray[np.float32] = field(default_factory=_F.copy)
    H: NDArray[np.float32] = field(default_factory=_H.copy)
    R: NDArray[np.float32] = field(default_factory=_R.copy)
    M: NDArray[np.float32] = field(default_factory=_M.copy)

    def predict(self):
        """Predict next state (prior) using the Kalman filter state propagation
        equations.
        """
        self.x = np.dot(self.F, self.x)  # x = Fx
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q  # P = FPF' + Q

    def update(self, z: NDArray):
        """At the time step k, this update step computes the posterior mean x and
        covariance P of the system state given a new measurement z.
        """
        # y = z - Hx (Residual between measurement and prediction)
        y = z - np.dot(self.H, self.x)
        PHT = np.dot(self.P, self.H.T)
        # S = HPH' + R (Project system uncertainty into measurement space)
        S = np.dot(self.H, PHT) + self.R
        # K = PH'S^-1  (map system uncertainty into Kalman gain)
        K = np.dot(PHT, inv(S))
        # x = x + Ky  (predict new x with residual scaled by the Kalman gain)
        self.x = self.x + np.dot(K, y)
        # P = (I-KH)P
        I_KH = _I - np.dot(K, self.H)
        self.P = np.dot(I_KH, self.P)


class KalmanTrack:
    """This class represents the internal state of individual tracked objects observed as bounding boxes."""

    def __init__(self, xysr: NDArray[np.float32]):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.x[:4] = xysr[:, None]

    def project(self) -> NDArray[np.float32]:
        return self.kf.x[:4, 0]

    def update(self, z: NDArray[np.float32]):
        self.kf.update(z[:, None])

    def predict(self):
        if self.kf.x[6, 0] + self.kf.x[2, 0] <= 0:
            self.kf.x[6, 0] *= 0.0

        self.kf.predict()
