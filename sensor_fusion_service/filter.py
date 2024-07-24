import numpy as np
from scipy.signal import butter, filtfilt

XY_OBS_NOISE_STD = 5.0
FORWARD_VELOCITY_NOISE_STD = 0.3
YAW_RATE_NOISE_STD = 0.02


# Code adapted from:
# https://github.com/motokimura/kalman_filter_with_kitti/blob/master/src/kalman_filters/extended_kalman_filter.py
class ExtendedKalmanFilter:
    """Extended Kalman Filter
    for vehicle whose motion is modeled as eq. (5.9) in [1]
    and with observation of its 2d location (x, y)
    """

    def __init__(self, x, P, R, Q):
        """
        Args:
            x (numpy.array): state to estimate: [x_, y_, theta]^T
            P (numpy.array): estimation error covariance
        """
        self.x = x  #  [3,]
        self.P = P  #  [3, 3]
        self.R = R  #  [3, 3]
        self.Q = Q  #  [2, 2]

    def update(self, z, Q):
        """update x and P based on observation of (x_, y_)
        Args:
            z (numpy.array): observation for [x_, y_]^T
            Q (numpy.array): observation noise covariance
        """
        # compute Kalman gain
        H = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )  # Jacobian of observation function

        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + Q)

        # update state x
        x, y, theta = self.x
        z_ = np.array([x, y])  # expected observation from the estimated state
        self.x = self.x + K @ (z - z_)

        # update covariance P
        self.P = self.P - K @ H @ self.P

    def propagate(self, u, dt, R):
        """propagate x and P based on state transition model defined as eq. (5.9) in [1]
        Args:
            u (numpy.array): control input: [v, omega]^T
            dt (float): time interval in second
            R (numpy.array): state transition noise covariance
        """
        # propagate state x
        x, y, theta = self.x
        v, omega = u
        if omega == 0.0:
            omega = 1e-8  # avoid division by zero
        r = v / omega  # turning radius

        dtheta = omega * dt
        dx = -r * np.sin(theta) + r * np.sin(theta + dtheta)
        dy = +r * np.cos(theta) - r * np.cos(theta + dtheta)

        self.x += np.array([dx, dy, dtheta])

        # propagate covariance P
        G = np.array(
            [
                [1.0, 0.0, -r * np.cos(theta) + r * np.cos(theta + dtheta)],
                [0.0, 1.0, -r * np.sin(theta) + r * np.sin(theta + dtheta)],
                [0.0, 0.0, 1.0],
            ]
        )  # Jacobian of state transition function

        self.P = G @ self.P @ G.T + R

    def run_filter(
        self,
        forward_velocities,
        yaw_rates,
        measurement_x,
        measurement_y,
        time,
    ):
        """Written based off demo code, all input arrays must be the same length"""
        if (
            len(forward_velocities)
            != len(yaw_rates)
            != len(measurement_x)
            != len(measurement_y)
            != len(time)
        ):
            raise ValueError("All input arrays must be the same length")

        # init arrays that will hold position estimates
        est_x = [
            self.x[0],
        ]
        est_y = [
            self.x[1],
        ]
        est_theta = [
            self.x[2],
        ]

        # array to store estimated error variance of 2d pose
        var_est_x = [
            self.P[0, 0],
        ]
        var_est_y = [
            self.P[1, 1],
        ]
        var_est_theta = [
            self.P[2, 2],
        ]

        # variables for filter
        t_last = 0.0
        data_length = len(time)

        for t_idx in range(1, data_length):
            t = time[t_idx]
            dt = t - t_last

            # get control input `u = [v, omega] + noise`
            u = np.array([forward_velocities[t_idx], yaw_rates[t_idx]])

            # propagate!
            self.propagate(u, dt, self.R)
            z = np.array(
                [
                    measurement_x[t_idx],
                    measurement_y[t_idx],
                ]
            )

            self.update(z, self.Q)

            est_x.append(self.x[0])
            est_y.append(self.x[1])
            est_theta.append(normalize_angles(self.x[2]))

            t_last = t

        return est_x, est_y, est_theta


def butter_lowpass_filter(data, fs, cutoff=1, order=2):
    nyq = 0.5 * fs  # Define Nyquist Frequency
    normal_cutoff = cutoff / nyq  # Normalize cutoff frequency
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)
    return np.array(y)


############ Helper Functions ############
def normalize_angles(angles):
    """
    Args:
        angles (float or numpy.array): angles in radian (= [a1, a2, ...], shape of [n,])
    Returns:
        numpy.array or float: angles in radians normalized b/w/ -pi and +pi (same shape w/ angles)
    """
    angles = (angles + np.pi) % (2 * np.pi) - np.pi
    return angles
