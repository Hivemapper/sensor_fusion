from typing import List
import numpy as np
from scipy.interpolate import CubicSpline

from telemetryMath import extractAndSmoothImuData, extractGNSSData, calculateStationary
from conversions import lists_to_dicts, lla_to_enu, enu_to_lla
from sqliteInterface import IMUData, GNSSData
from filter import ExtendedKalmanFilter as EKF


##### Constants for Kalman Filter #####
XY_OBS_NOISE_STD = 3
FORWARD_VELOCITY_NOISE_STD = 1.5
YAW_RATE_NOISE_STD = 0.15
INITIAL_YAW_STD = np.pi
INITIAL_YAW = 0
#######################################


def processRawData(
    gnssData: List[GNSSData], imuData: List[IMUData], debug: bool = False
):
    """
    This function processes the raw IMU data and returns the processed data.
    Parameters:
    - data: A list of dictionaries, where each dictionary represents a row of raw IMU data.
    Returns:
    - list: A list of dictionaries, where each dictionary represents a row of processed IMU data.
    """

    ########### Filter/Extract Sensor Data ###########
    (
        acc_x,
        acc_y,
        acc_z,
        gyro_x,
        gyro_y,
        gyro_z,
        time,
        temperature,
        imu_session,
        row_id,
        imu_freq,
        imu_converted_time,
    ) = extractAndSmoothImuData(imuData)
    (
        lat,
        lon,
        alt,
        gnss_speed,
        gnss_heading,
        gnss_headingAccuracy,
        hdop,
        gdop,
        gnss_system_time,  ### time given in epoch format that should be used for calculations
        gnss_real_time,  ### time given by gnss model
        gnss_time_resolved,
        gnss_time,  ### Time that is in original format
        gnss_session,
        gnss_freq,
    ) = extractGNSSData(gnssData)

    ########### Process Sensor Data ###########
    stationary = calculateStationary(
        acc_x,
        acc_y,
        acc_z,
        gyro_x,
        gyro_y,
        gyro_z,
        imu_freq,
        imu_converted_time,
        debug,
    )

    ########### Package Processed Data ###########
    keys = [
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "stationary",
        "time",
        "temperature",
        "session",
        "row_id",
    ]
    processedIMUData = lists_to_dicts(
        keys,
        acc_x,
        acc_y,
        acc_z,
        gyro_x,
        gyro_y,
        gyro_z,
        stationary,
        time,
        temperature,
        imu_session,
        row_id,
    )
    if debug:
        return (
            processedIMUData,
            acc_x,
            acc_y,
            acc_z,
            gyro_x,
            gyro_y,
            gyro_z,
            imu_converted_time,
        )

    return processedIMUData


#### Orientation???????????????????????


def calculate_fused_position(
    orientation,
    acc_x,
    acc_y,
    acc_z,
    gyro_x,
    gyro_y,
    gyro_z,
    imu_time,
    lat,
    lon,
    alt,
    gnss_time,
):
    forward_velocity = calculateForwardVelocity(
        orientation, acc_x, acc_y, acc_z, imu_time
    )
    yaw_rates = calculateYawRate(orientation, gyro_x, gyro_y, gyro_z, imu_time)

    # convert gnss position to enu coordinate frame
    points_lla = np.array(list(zip(lon, lat, alt)))
    ref_lla = [lon[0], lat[0], alt[0]]
    traj_xyz = lla_to_enu(points_lla, ref_lla)
    pos_x = traj_xyz[:, 0]
    pos_y = traj_xyz[:, 1]

    ## Down sample IMU data to match gnss data
    forward_vel_downsampled = cubic_spline_interpolation(
        forward_velocity, imu_time, gnss_time
    )
    yaw_rate_downsampled = cubic_spline_interpolation(yaw_rates, imu_time, gnss_time)

    x = np.array([traj_xyz[0][0], traj_xyz[0][1], INITIAL_YAW])

    P = np.array(
        [
            [XY_OBS_NOISE_STD**2.0, 0.0, 0.0],
            [0.0, XY_OBS_NOISE_STD**2.0, 0.0],
            [0.0, 0.0, INITIAL_YAW_STD**2.0],
        ]
    )

    Q = np.array([[XY_OBS_NOISE_STD**2.0, 0.0], [0.0, XY_OBS_NOISE_STD**2.0]])

    R = np.array(
        [
            [FORWARD_VELOCITY_NOISE_STD**2.0, 0.0, 0.0],
            [0.0, FORWARD_VELOCITY_NOISE_STD**2.0, 0.0],
            [0.0, 0.0, YAW_RATE_NOISE_STD**2.0],
        ]
    )
    ## Feed into Filter
    kf = EKF(x, P, R, Q)
    est_x, est_y, est_heading = kf.run_filter(
        forward_vel_downsampled, yaw_rate_downsampled, pos_x, pos_y, gnss_time
    )

    ### Convert back to LLA
    est_xyz = np.stack([est_x, est_y, np.zeros_like(est_x)], axis=1)
    est_lla = enu_to_lla(est_xyz, ref_lla)

    return est_lla, est_heading


###################### Math for Kalman Filter ######################


def calculateForwardVelocity(orientation, acc_x, acc_y, acc_z, time):
    """
    Calculate the forward velocity of a vehicle given its orientation and accelerations over time.

    Parameters:
    orientation (numpy array): Array of orientation angles (in radians) over time [yaw, pitch, roll].
    acc_x (numpy array): Array of x-axis accelerations over time.
    acc_y (numpy array): Array of y-axis accelerations over time.
    acc_z (numpy array): Array of z-axis accelerations over time.
    time (numpy array): Array of time stamps.

    Returns:
    numpy array: Array of forward velocities over time.
    """
    # Calculate the time differences
    dt = np.diff(time)

    # Initialize velocity array
    velocity = np.zeros_like(time)

    yaw, pitch, roll = orientation
    # Iterate through each time step to compute velocity
    for i in range(1, len(time)):
        # Calculate the rotation matrix from the IMU frame to the vehicle frame
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cr = np.cos(roll)
        sr = np.sin(roll)

        # Rotation matrix (IMU to vehicle frame)
        R = np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ]
        )

        # Accelerometer readings in IMU frame
        acc_imu = np.array([acc_x[i], acc_y[i], acc_z[i]])

        # Transform accelerometer readings to the vehicle frame
        acc_vehicle = np.dot(R, acc_imu)

        # Forward acceleration is the x-component in the vehicle frame
        acc_forward = acc_vehicle[0]

        # Integrate acceleration to get velocity
        velocity[i] = velocity[i - 1] + acc_forward * dt[i - 1]

    # Deals with the first velocity not being calculated assign it the same alue as the first calcualted velocity value
    if len(velocity) > 1:
        velocity[0] = velocity[1]

    return velocity


def calculateYawRate(orientation, gyro_x, gyro_y, gyro_z, time):
    """
    Calculate the yaw rate of a vehicle given its orientation and gyroscope readings over time.

    Parameters:
    orientation (numpy array): Array of orientation angles (in radians) over time [yaw, pitch, roll].
    gyro_x (numpy array): Array of x-axis gyroscope readings over time.
    gyro_y (numpy array): Array of y-axis gyroscope readings over time.
    gyro_z (numpy array): Array of z-axis gyroscope readings over time.
    time (numpy array): Array of time stamps.

    Returns:
    numpy array: Array of yaw rates over time.
    """
    # Initialize yaw rate array
    yaw_rate = np.zeros_like(time)

    yaw, pitch, roll = orientation
    # Iterate through each time step to compute yaw rate
    for i in range(len(time)):
        # Calculate the rotation matrix from the IMU frame to the vehicle frame
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cr = np.cos(roll)
        sr = np.sin(roll)

        # Rotation matrix (IMU to vehicle frame)
        R = np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ]
        )

        # Gyroscope readings in IMU frame
        gyro_imu = np.array([gyro_x[i], gyro_y[i], gyro_z[i]])

        # Transform gyroscope readings to the vehicle frame
        gyro_vehicle = np.dot(R, gyro_imu)

        # Extract the yaw rate from the transformed gyroscope readings
        yaw_rate[i] = gyro_vehicle[2]  # The z-component in the vehicle frame

    # Deals with the first yaw rate not being calculated assign it the same alue as the first calcualted yaw rate value
    if len(yaw_rate) > 1:
        yaw_rate[0] = yaw_rate[1]

    return yaw_rate


def cubic_spline_interpolation(signal, signal_timestamps, desired_timestamps):
    """
    Perform cubic spline interpolation on a given signal to match it to desired timestamps.

    Parameters:
    signal (numpy array): N x M array where N is the number of samples and M is the number of dimensions of the signal.
    signal_timestamps (numpy array): Array of N timestamps corresponding to the signal samples.
    desired_timestamps (numpy array): Array of K desired timestamps to interpolate the signal to.

    Returns:
    numpy array: Interpolated signal at the desired timestamps.
    """
    # Ensure input is a numpy array
    signal = np.array(signal)
    signal_timestamps = np.array(signal_timestamps)
    desired_timestamps = np.array(desired_timestamps)

    # Initialize an array to hold the interpolated signal
    interpolated_signal = np.zeros((len(desired_timestamps), signal.shape[1]))

    # Perform cubic spline interpolation for each dimension of the signal
    for i in range(signal.shape[1]):
        # Create a cubic spline interpolation function for the i-th dimension
        cs = CubicSpline(signal_timestamps, signal[:, i])

        # Interpolate the signal to the desired timestamps
        interpolated_signal[:, i] = cs(desired_timestamps)

    return interpolated_signal


###################### Other ######################
def grab_most_recent_raw_data_session(
    rawData,
    processedIndex,
    furthest_index,
):
    """Requires rawData to have session data. Processes the raw data and returns the raw data
    with only the starting session and the new index

    Args:
        rawData (_type_): _description_
    """
    next_index = -1
    # Handle pulling singluar drive sessions here
    starting_session = rawData[0].session
    ending_session = rawData[-1].session
    if starting_session != ending_session:
        # iterate backwards through the data to find the start of the new session
        for i in range(len(rawData) - 1, -1, -1):
            if rawData[i].session == starting_session:
                # reduce rawData to only the new session
                rawData = rawData[: i + 1]
                print(f"Reduced data to {len(rawData)} data points")
                next_index = processedIndex + i + 1
                break
    else:
        next_index = furthest_index + 1

    return rawData, next_index
