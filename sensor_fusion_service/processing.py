from typing import List
import numpy as np
from scipy.interpolate import CubicSpline

from sensor_fusion.sensor_fusion_service.telemetry_math import (
    extract_smooth_imu_data,
    extract_gnss_data,
    calculate_stationary_status,
)
from sensor_fusion.sensor_fusion_service.conversions import (
    lists_to_dicts,
    lla_to_enu,
    enu_to_lla,
    convert_time_to_epoch,
)
from sensor_fusion.sensor_fusion_service.data_definitions import IMUData, GNSSData
from sensor_fusion.sensor_fusion_service.filter import ExtendedKalmanFilter as EKF


##### Constants for Kalman Filter #####
XY_OBS_NOISE_STD = 0.9
FORWARD_VELOCITY_NOISE_STD = 0.3
YAW_RATE_NOISE_STD = 0.01
INITIAL_YAW_STD = 0.01
INITIAL_YAW = 0.01
#######################################


####################### Main Processing Function #######################


def process_raw_data(
    gnss_data: List[GNSSData],
    imu_data: List[IMUData],
    state_values: dict,
    debug: bool = False,
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
        imu_time,
        temperature,
        imu_session,
        row_id,
        imu_freq,
        imu_converted_time,
    ) = extract_smooth_imu_data(imu_data)
    (
        lat,
        lon,
        alt,
        gnss_speed,
        gnss_heading,
        gnss_heading_accuracy,
        hdop,
        gdop,
        gnss_system_time,  ### time given in epoch format that should be used for calculations
        gnss_real_time,  ### time given by gnss model
        gnss_time_resolved,
        gnss_time,  ### Time that is in original format
        gnss_session,
        gnss_freq,
    ) = extract_gnss_data(gnss_data)

    ########### Process Sensor Data ###########
    stationary = calculate_stationary_status(
        acc_x,
        acc_y,
        acc_z,
        gyro_x,
        gyro_y,
        gyro_z,
        imu_freq,
        imu_converted_time,
    )

    fused_position, fused_heading, forward_vel, yaw_rates = calculate_fused_position(
        state_values,
        [0.0, 0.0, 0.0],  ### For now orientation is set to 0,0,0
        acc_x,
        acc_y,
        acc_z,
        gyro_x,
        gyro_y,
        gyro_z,
        imu_converted_time,
        lat,
        lon,
        alt,
        gnss_system_time,
    )

    fused_lons = fused_position[0, :]
    fused_lats = fused_position[1, :]

    ########### Package Processed Data ###########
    processed_imu_keys = [
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
    processed_imu_data = lists_to_dicts(
        processed_imu_keys,
        acc_x,
        acc_y,
        acc_z,
        gyro_x,
        gyro_y,
        gyro_z,
        stationary,
        imu_time,
        temperature,
        imu_session,
        row_id,
    )

    fused_keys = [
        "gnss_lat",
        "gnss_lon",
        "fused_lat",
        "fused_lon",
        "fused_heading",
        "forward_velocity",
        "yaw_rate",
        "time",
        "session",
    ]
    fused_position_data = lists_to_dicts(
        fused_keys,
        lat,
        lon,
        fused_lats,
        fused_lons,
        fused_heading,
        forward_vel,
        yaw_rates,
        gnss_system_time,
        gnss_session,
    )

    if debug:
        return (
            processed_imu_data,
            fused_position_data,
            acc_x,
            acc_y,
            acc_z,
            gyro_x,
            gyro_y,
            gyro_z,
            imu_converted_time,
        )

    return processed_imu_data, fused_position_data


###################### Kalman Filter Main Function #######################


def calculate_fused_position(
    state_values,
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
    forward_velocity = calculate_forward_velocity(
        orientation, state_values["current_velocity"], acc_x, acc_y, acc_z, imu_time
    )
    # Update the state values with the current velocity
    state_values["current_velocity"] = forward_velocity[-1]
    yaw_rates = calculate_yaw_rate(orientation, gyro_x, gyro_y, gyro_z)

    # convert gnss position to enu coordinate frame
    points_lla = np.array([lon, lat, alt])
    ref_lla = [lon[0], lat[0], alt[0]]
    traj_xyz = lla_to_enu(points_lla, ref_lla)
    pos_x = traj_xyz[0, :]
    pos_y = traj_xyz[1, :]

    ## Down sample IMU data to match gnss data
    # forward_vel_downsampled = cubic_spline_interpolation(
    #     forward_velocity, imu_time, gnss_time
    # )
    # yaw_rate_downsampled = cubic_spline_interpolation(yaw_rates, imu_time, gnss_time)

    forward_vel_downsampled = np.interp(gnss_time, imu_time, forward_velocity)
    yaw_rate_downsampled = np.interp(gnss_time, imu_time, yaw_rates)

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
    est_xyz = np.array([est_x, est_y, traj_xyz[2, :]])
    # est_xyz = np.array([pos_x, pos_y, traj_xyz[2, :]])
    est_lla = enu_to_lla(est_xyz, ref_lla)

    return est_lla, est_heading, forward_vel_downsampled, yaw_rate_downsampled


###################### Math for Kalman Filter ######################


def calculate_forward_velocity(orientation, initial_vel, acc_x, acc_y, acc_z, time):
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
    # Convert time from milliseconds to seconds
    time_sec = np.array(time) / 1000.0
    # Calculate the time differences
    dt = np.diff(time_sec)

    # Initialize velocity array
    velocity = np.zeros_like(time_sec)
    velocity[0] = initial_vel

    yaw, pitch, roll = orientation
    # Iterate through each time step to compute velocity
    for i in range(1, len(time_sec)):
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

        # TODO:HARDCODED LIMITS, hope to remove this in the future
        # Limit the velocity to 35 m/s
        if velocity[i] > 35.0:
            velocity[i] = 35.0

        # limit the velocity to 0 m/s
        if velocity[i] < 0.0:
            velocity[i] = 0.0

    return velocity


# def calculate_forward_velocity(orientation, initial_vel, acc_x, acc_y, acc_z, time):
#     """
#     Calculate the forward velocity of a vehicle given its x-axis accelerations over time.

#     Parameters:
#     acc_x (numpy array): Array of x-axis accelerations over time.
#     time (numpy array): Array of time stamps in milliseconds epoch.

#     Returns:
#     numpy array: Array of forward velocities over time.
#     """
#     # Convert time from milliseconds to seconds
#     time = np.array(time) / 1000.0

#     # Calculate the time differences in seconds
#     dt = np.diff(time)

#     # Initialize velocity array
#     velocity = np.zeros_like(time)
#     velocity[0] = initial_vel

#     # Iterate through each time step to compute velocity
#     for i in range(1, len(time)):
#         # Forward acceleration is the x-component
#         acc_forward = acc_x[i]

#         # Integrate acceleration to get velocity

#         velocity[i] = velocity[i - 1] + acc_forward * dt[i - 1]

#         # TODO:HARDCODED LIMITS, hope to remove this in the future
#         # Limit the velocity to 35 m/s
#         if velocity[i] > 35.0:
#             velocity[i] = 35.0

#         # limit the velocity to 0 m/s
#         if velocity[i] < 0.0:
#             velocity[i] = 0.0

#     return velocity


def calculate_yaw_rate(orientation, gyro_x, gyro_y, gyro_z):
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
    yaw_rate = np.zeros_like(gyro_x)

    yaw, pitch, roll = orientation
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
    # Iterate through each time step to compute yaw rate
    for i in range(len(gyro_x)):
        # Gyroscope readings in IMU frame
        gyro_imu = np.array([gyro_x[i], gyro_y[i], gyro_z[i]])
        # Transform gyroscope readings to the vehicle frame
        gyro_vehicle = np.dot(R, gyro_imu)

        # Extract the yaw rate from the transformed gyroscope readings
        yaw_rate[i] = gyro_vehicle[2]  # The z-component in the vehicle frame

    return yaw_rate


def cubic_spline_interpolation(signal, signal_timestamps, desired_timestamps):
    """
    Perform cubic spline interpolation on a given one-dimensional signal to match it to desired timestamps.

    Parameters:
    signal (numpy array): Array of N signal samples.
    signal_timestamps (numpy array): Array of N timestamps corresponding to the signal samples.
    desired_timestamps (numpy array): Array of K desired timestamps to interpolate the signal to.

    Returns:
    numpy array: Interpolated signal at the desired timestamps.
    """
    # Ensure input is a numpy array
    signal = np.array(signal)
    signal_timestamps = np.array(signal_timestamps)
    desired_timestamps = np.array(desired_timestamps)
    # Check if signal_timestamps is strictly increasing
    index = check_strictly_increasing(signal_timestamps)
    if index != -1:
        raise ValueError(
            f"`signal_timestamps` must be strictly increasing. Problem at index {index}: "
            f"{signal_timestamps[index-1]} >= {signal_timestamps[index]}"
        )

    # Create a cubic spline interpolation function for the signal
    cs = CubicSpline(signal_timestamps, signal)

    # Interpolate the signal to the desired timestamps
    interpolated_signal = cs(desired_timestamps)

    return interpolated_signal


def check_strictly_increasing(arr):
    """
    Check if an array is strictly increasing and return the index where it fails, if any.
    """
    for i in range(1, len(arr)):
        if arr[i] <= arr[i - 1]:
            return i
    return -1


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


def remove_duplicate_imu_data(raw_data):
    """Removes duplicate data points from the raw data based on the `time` attribute.

    Args:
        raw_data (list): List of IMUData objects.

    Returns:
        list: List of IMUData objects with duplicates removed.
    """
    new_data = []
    last_time = 0

    for data_point in raw_data:
        cur_time = convert_time_to_epoch(data_point.time)
        if cur_time != last_time:
            new_data.append(data_point)
            last_time = cur_time

    return new_data
