import os
import math
import numpy as np
from typing import List

env = os.getenv("HIVE_ENV")

if env == "local":
    from sensor_fusion.sensor_fusion_service.filter import butter_lowpass_filter
    from sensor_fusion.sensor_fusion_service.data_definitions import IMUData, GNSSData
    from sensor_fusion.sensor_fusion_service.conversions import convert_time_to_epoch
else:
    from filter import butter_lowpass_filter
    from data_definitions import IMUData, GNSSData
    from conversions import convert_time_to_epoch


# Constants for threshold-based window averaging
WINDOW_SIZE = 1000
THRESHOLD_ACCEL = 0.00005
THRESHOLD_GYRO = 0.00005

# Constants for IMU LOW PASS FILTER
IMU_LOW_PASS_CUTOFF_FREQ = 3
IMU_LOW_PASS_ORDER = 3
IMU_STATIONARY_LOW_PASS_CUTOFF_FREQ = 5


# for: /Users/rogerberman/Desktop/4.9.11_DB_data/testData/2024-07-17T15:34:38.000Z.db
# OFFSETS = {
#     "acc_x": 0.23540433574567557,
#     "acc_y": 0.0009506753661703987,
#     "acc_z": 0.017152832308572785,
#     "gyro_x": 0.05056635014217598,
#     "gyro_y": 0.010372298896629956,
#     "gyro_z": 0.0854693413851669,
# }

OFFSETS = {
    "acc_x": -0.04633832350834084,
    "acc_y": -0.024603480328423244,
    "acc_z": 0.02175024562121597,
    "gyro_x": 0.0016928365665450007,
    "gyro_y": -0.0028576082846062836,
    "gyro_z": 0.00037565459775269636,
}

SCALING = {
    "acc_x": 1.0,
    "acc_y": 1.0,
    "acc_z": 1.0,
    "gyro_x": 1.0,
    "gyro_y": 1.0,
    "gyro_z": 1.0,
}


def calculate_average_frequency(epoch_times_ms):
    """
    Calculates the average frequency of events given a list of epoch times in milliseconds starting from zero.
    Parameters:
    - epoch_times_ms: List of epoch times in milliseconds. The list should start at zero and represent successive events.
    Returns:
    - float: The average frequency of the events in Hz (events per second).
    """
    if len(epoch_times_ms) < 2:
        return 0

    periods_seconds = [
        (epoch_times_ms[i] - epoch_times_ms[i - 1]) / 1000.0
        for i in range(1, len(epoch_times_ms))
    ]
    average_period_seconds = sum(periods_seconds) / len(periods_seconds)
    average_frequency = 1 / average_period_seconds if average_period_seconds != 0 else 0

    return average_frequency


def extract_smooth_imu_data(imu_data: List[IMUData], offsets: dict = OFFSETS):
    """
    Extracts accelerometer, gyroscope data, and time differences from the given data.
    Parameters:
    - imu_data: List of IMUData instances containing 'ax', 'ay', 'az', 'gx', 'gy', 'gz', and 'time'.
    Returns:
    - Tuple containing lists for 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', and time differences.
    """
    acc_x, acc_y, acc_z = [], [], []
    gyro_x, gyro_y, gyro_z = [], [], []
    converted_time, time, temperature, session = [], [], [], []

    for point in imu_data:
        acc_x.append(point.acc_x)
        acc_y.append(point.acc_y)
        acc_z.append(point.acc_z)
        gyro_x.append(math.radians(point.gyro_x))
        gyro_y.append(math.radians(point.gyro_y))
        gyro_z.append(math.radians(point.gyro_z))
        time.append(point.time)
        converted_time.append(convert_time_to_epoch(point.time))
        temperature.append(point.temperature)
        session.append(point.session)

    freq = math.floor(calculate_average_frequency(converted_time))
    print(f"IMU data frequency: {freq} Hz")

    acc_x = butter_lowpass_filter(
        acc_x, freq, IMU_LOW_PASS_CUTOFF_FREQ, IMU_LOW_PASS_ORDER
    )
    acc_y = butter_lowpass_filter(
        acc_y, freq, IMU_LOW_PASS_CUTOFF_FREQ, IMU_LOW_PASS_ORDER
    )
    acc_z = butter_lowpass_filter(
        acc_z, freq, IMU_LOW_PASS_CUTOFF_FREQ, IMU_LOW_PASS_ORDER
    )
    gyro_x = butter_lowpass_filter(
        gyro_x, freq, IMU_LOW_PASS_CUTOFF_FREQ, IMU_LOW_PASS_ORDER
    )
    gyro_y = butter_lowpass_filter(
        gyro_y, freq, IMU_LOW_PASS_CUTOFF_FREQ, IMU_LOW_PASS_ORDER
    )
    gyro_z = butter_lowpass_filter(
        gyro_z, freq, IMU_LOW_PASS_CUTOFF_FREQ, IMU_LOW_PASS_ORDER
    )

    # Handle Offsets
    acc_x = np.subtract(acc_x, offsets["acc_x"])
    acc_y = np.subtract(acc_y, offsets["acc_y"])
    acc_z = np.subtract(acc_z, offsets["acc_z"])
    gyro_x = np.subtract(gyro_x, offsets["gyro_x"])
    gyro_y = np.subtract(gyro_y, offsets["gyro_y"])
    gyro_z = np.subtract(gyro_z, offsets["gyro_z"])

    # Handle Scaling
    acc_x = np.multiply(acc_x, SCALING["acc_x"])
    acc_y = np.multiply(acc_y, SCALING["acc_y"])
    acc_z = np.multiply(acc_z, SCALING["acc_z"])
    gyro_x = np.multiply(gyro_x, SCALING["gyro_x"])
    gyro_y = np.multiply(gyro_y, SCALING["gyro_y"])
    gyro_z = np.multiply(gyro_z, SCALING["gyro_z"])

    return (
        acc_x,
        acc_y,
        acc_z,
        gyro_x,
        gyro_y,
        gyro_z,
        time,
        temperature,
        session,
        freq,
        converted_time,
    )


def extract_gnss_data(data: List[GNSSData]):
    """
    Extracts GNSS data from the given data.
    Parameters:
    - data: List of GNSSData instances containing 'lat', 'lon', 'alt', 'speed', 'heading', 'heading_accuracy', 'hdop', 'gdop', and 'time'.
    Returns:
    - Tuple containing lists for 'lat', 'lon', 'alt', 'speed', 'heading', 'heading_accuracy', 'hdop', 'gdop', and time differences.
    """
    lat, lon, alt = [], [], []
    speed, heading, heading_accuracy = [], [], []
    hdop, gdop = [], []
    system_time = []
    gnss_real_time, time_resolved, time, session = [], [], [], []

    for point in data:
        lat.append(point.latitude)
        lon.append(point.longitude)
        alt.append(point.altitude)
        speed.append(point.speed)
        heading.append(point.heading)
        heading_accuracy.append(point.heading_accuracy)
        hdop.append(point.hdop)
        gdop.append(point.gdop)
        system_time.append(convert_time_to_epoch(point.system_time))
        gnss_real_time.append(convert_time_to_epoch(point.time))
        time_resolved.append(point.time_resolved)
        time.append(point.system_time)
        session.append(point.session)

    freq = math.floor(calculate_average_frequency(system_time))
    print(f"GNSS data frequency: {freq} Hz")

    return (
        lat,
        lon,
        alt,
        speed,
        heading,
        heading_accuracy,
        hdop,
        gdop,
        system_time,
        gnss_real_time,
        time_resolved,
        time,
        session,
        freq,
    )


def threshold_based_window_averaging(data, times, window_size_ms, threshold):
    """
    Applies threshold-based window averaging to the input data.

    Parameters:
        data (list or ndarray): Input data points.
        times (list or ndarray): Timestamps in epoch milliseconds corresponding to the data points.
        window_size_ms (int): Size of the time window in milliseconds.
        threshold (float): Threshold value for determining the output (0 or 1).

    Returns:
        list: Output list of 0s and 1s based on threshold-based window averaging.
    """
    output = []
    # Convert data and times to numpy arrays if they are not already
    data = np.array(data)
    times = np.array(times)
    # Find the total number of data points
    num_points = len(times)
    # Initialize start and end indices for the sliding window
    window_start = 0
    window_end = 0

    # Slide the window and compute output
    while window_end < num_points:
        # Determine the end time for the current window
        window_end_time = times[window_start] + window_size_ms

        # Find data points within the current window
        window_data_indices = np.where(
            (times >= times[window_start]) & (times < window_end_time)
        )[0]

        # Calculate the average value within the window
        window_average = np.mean(data[window_data_indices])

        # Determine the output for the current window based on the threshold
        window_output = 0 if window_average >= threshold else 1

        # Add the window output to the output list
        output.extend([window_output] * len(window_data_indices))

        # Slide the window
        window_start = window_data_indices[-1] + 1
        window_end = window_start

    return np.array(output)


def calculate_stationary_status(
    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, imu_freq, imu_time
):
    # get point to point diffs for all sensors
    acc_x_diff = np.diff(acc_x)
    padded_acc_x_diff = np.concatenate(([acc_x_diff[0]], acc_x_diff))
    acc_y_diff = np.diff(acc_y)
    padded_acc_y_diff = np.concatenate(([acc_y_diff[0]], acc_y_diff))
    acc_z_diff = np.diff(acc_z)
    padded_acc_z_diff = np.concatenate(([acc_z_diff[0]], acc_z_diff))
    gyro_x_diff = np.diff(gyro_x)
    padded_gyro_x_diff = np.concatenate(([gyro_x_diff[0]], gyro_x_diff))
    gyro_y_diff = np.diff(gyro_y)
    padded_gyro_y_diff = np.concatenate(([gyro_y_diff[0]], gyro_y_diff))
    gyro_z_diff = np.diff(gyro_z)
    padded_gyro_z_diff = np.concatenate(([gyro_z_diff[0]], gyro_z_diff))

    abs_padded_acc_x_diff = np.abs(padded_acc_x_diff)
    abs_padded_acc_y_diff = np.abs(padded_acc_y_diff)
    abs_padded_acc_z_diff = np.abs(padded_acc_z_diff)
    abs_padded_gyro_x_diff = np.abs(padded_gyro_x_diff)
    abs_padded_gyro_y_diff = np.abs(padded_gyro_y_diff)
    abs_padded_gyro_z_diff = np.abs(padded_gyro_z_diff)

    # Apply low-pass filter for each cutoff frequency
    acc_x_diff_lowpass = butter_lowpass_filter(
        abs_padded_acc_x_diff, imu_freq, IMU_STATIONARY_LOW_PASS_CUTOFF_FREQ
    )
    acc_y_diff_lowpass = butter_lowpass_filter(
        abs_padded_acc_y_diff, imu_freq, IMU_STATIONARY_LOW_PASS_CUTOFF_FREQ
    )
    acc_z_diff_lowpass = butter_lowpass_filter(
        abs_padded_acc_z_diff, imu_freq, IMU_STATIONARY_LOW_PASS_CUTOFF_FREQ
    )
    gyro_x_diff_lowpass = butter_lowpass_filter(
        abs_padded_gyro_x_diff, imu_freq, IMU_STATIONARY_LOW_PASS_CUTOFF_FREQ
    )
    gyro_y_diff_lowpass = butter_lowpass_filter(
        abs_padded_gyro_y_diff, imu_freq, IMU_STATIONARY_LOW_PASS_CUTOFF_FREQ
    )
    gyro_z_diff_lowpass = butter_lowpass_filter(
        abs_padded_gyro_z_diff, imu_freq, IMU_STATIONARY_LOW_PASS_CUTOFF_FREQ
    )

    # Gyro threshold +- 0.001, Accel threshold +- 0.001
    acc_x_stopped = threshold_based_window_averaging(
        acc_x_diff_lowpass, imu_time, WINDOW_SIZE, THRESHOLD_ACCEL
    )
    acc_y_stopped = threshold_based_window_averaging(
        acc_y_diff_lowpass, imu_time, WINDOW_SIZE, THRESHOLD_ACCEL
    )
    acc_z_stopped = threshold_based_window_averaging(
        acc_z_diff_lowpass, imu_time, WINDOW_SIZE, THRESHOLD_ACCEL
    )
    gyro_x_stopped = threshold_based_window_averaging(
        gyro_x_diff_lowpass, imu_time, WINDOW_SIZE, THRESHOLD_GYRO
    )
    gyro_y_stopped = threshold_based_window_averaging(
        gyro_y_diff_lowpass, imu_time, WINDOW_SIZE, THRESHOLD_GYRO
    )
    gyro_z_stopped = threshold_based_window_averaging(
        gyro_z_diff_lowpass, imu_time, WINDOW_SIZE, THRESHOLD_GYRO
    )

    # acc_res = acc_x_stopped & acc_y_stopped & acc_z_stopped
    acc_res = (
        (acc_x_stopped & acc_y_stopped)
        | (acc_y_stopped & acc_z_stopped)
        | (acc_x_stopped & acc_z_stopped)
    ).astype(int)
    gyro_res = (
        (gyro_x_stopped & gyro_y_stopped)
        | (gyro_y_stopped & gyro_z_stopped)
        | (gyro_x_stopped & gyro_z_stopped)
    ).astype(int)
    combined_gyro_accel = acc_res | gyro_res

    combined_gyro_accel = np.where(combined_gyro_accel == 0, 0.0, 1.0)

    return combined_gyro_accel
