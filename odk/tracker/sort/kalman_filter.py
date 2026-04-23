import numpy as np
from numpy.linalg import inv
from numpy.typing import NDArray

__all__ = [
    'KalmanFilter',
]


class KalmanFilter:
    """Kalman filtering, also known as linear quadratic estimation (LQE), is an
    algorithm that uses a series of measurements
    observed over time, containing statistical noise and other inaccuracies,
    and produces estimates of unknown variables that tend to be more accurate than those based on a single measurement
    alone, by estimating a joint probability distribution over the variables for each time frame.
    """

    def __init__(self, dim_x: int, dim_z: int):
        """Initialize the Kalman filter.

        Args:
            dim_x (int): Number of state variables. For example, if you are tracking
                position and velocity, dim_x would be 2.
            dim_z (int): Number of measurement variables. For example, if the sensor
                only provides position, dim_z would be 1.

        Attributes
            x (NDArray): (dim_x, 1) Current state estimate vector.
            P (NDArray): (dim_x, dim_x) Current state covariance matrix
                (uncertainty of the estimate).
            Q (NDArray): (dim_x, dim_x) Process noise covariance matrix.
            F (NDArray): (dim_x, dim_x) State transition matrix.
            H (NDArray): (dim_z, dim_x) Measurement function matrix.
            R (NDArray): (dim_z, dim_z) Measurement noise covariance matrix.
            M (NDArray): (dim_z, dim_z) Process-measurement cross-correlation matrix.
        """
        self.dim_x: int = dim_x
        self.dim_z: int = dim_z
        self.x = np.zeros((dim_x, 1))
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        self.F = np.eye(dim_x)
        self.H = np.zeros((dim_z, dim_x))
        self.R = np.eye(dim_z)
        self.M = np.zeros((dim_z, dim_z))
        # This helps the I matrix to always be compatible to the state vector's dim
        self._I = np.eye(dim_x)

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
        I_KH = self._I - np.dot(K, self.H)
        self.P = np.dot(I_KH, self.P)
