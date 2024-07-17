import math
import numpy as np

from typing import List
from filter import butter_lowpass_filter
from sqliteInterface import IMUData, GNSSData
from conversions import convertTimeToEpoch


# from plottingCode import plot_signals_over_time, plot_sensor_data, plot_signal_over_time
# import matplotlib.pyplot as plt

WINDOW_SIZE = 1000
THRESHOLD_ACCEL = 0.00005
THRESHOLD_GYRO = 0.00005


def calculateAverageFrequency(epoch_times_ms):
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


def extractAndSmoothImuData(imu_data: List[IMUData]):
    """
    Extracts accelerometer, gyroscope data, and time differences from the given data.
    Parameters:
    - imu_data: List of IMUData instances containing 'ax', 'ay', 'az', 'gx', 'gy', 'gz', and 'time'.
    Returns:
    - Tuple containing lists for 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', and time differences.
    """
    acc_x, acc_y, acc_z = [], [], []
    gyro_x, gyro_y, gyro_z = [], [], []
    converted_time, time, temperature, session, row_id = [], [], [], [], []

    for point in imu_data:
        acc_x.append(point.ax)
        acc_y.append(point.ay)
        acc_z.append(point.az)
        gyro_x.append(math.radians(point.gx))
        gyro_y.append(math.radians(point.gy))
        gyro_z.append(math.radians(point.gz))
        time.append(point.system_time)
        converted_time.append(convertTimeToEpoch(point.system_time))
        temperature.append(point.temperature)
        session.append(point.session)
        row_id.append(point.row_id)

    freq = math.floor(calculateAverageFrequency(converted_time))
    print(f"IMU data frequency: {freq} Hz")

    acc_x = butter_lowpass_filter(acc_x, freq)
    acc_y = butter_lowpass_filter(acc_y, freq)
    acc_z = butter_lowpass_filter(acc_z, freq)
    gyro_x = butter_lowpass_filter(gyro_x, freq)
    gyro_y = butter_lowpass_filter(gyro_y, freq)
    gyro_z = butter_lowpass_filter(gyro_z, freq)

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
        row_id,
        freq,
        converted_time,
    )


def extractGNSSData(data: List[GNSSData]):
    """
    Extracts GNSS data from the given data.
    Parameters:
    - data: List of GNSSData instances containing 'lat', 'lon', 'alt', 'speed', 'heading', 'headingAccuracy', 'hdop', 'gdop', and 'time'.
    Returns:
    - Tuple containing lists for 'lat', 'lon', 'alt', 'speed', 'heading', 'headingAccuracy', 'hdop', 'gdop', and time differences.
    """
    lat, lon, alt = [], [], []
    speed, heading, headingAccuracy = [], [], []
    hdop, gdop = [], []
    system_time = []
    gnss_real_time, time_resolved, time, session = [], [], [], []

    for point in data:
        lat.append(point.lat)
        lon.append(point.lon)
        alt.append(point.alt)
        speed.append(point.speed)
        heading.append(point.heading)
        headingAccuracy.append(point.headingAccuracy)
        hdop.append(point.hdop)
        gdop.append(point.gdop)
        system_time.append(convertTimeToEpoch(point.system_time))
        gnss_real_time.append(convertTimeToEpoch(point.time))
        time_resolved.append(point.time_resolved)
        time.append(point.system_time)
        session.append(point.session)

    freq = math.floor(calculateAverageFrequency(system_time))
    print(f"GNSS data frequency: {freq} Hz")

    return (
        lat,
        lon,
        alt,
        speed,
        heading,
        headingAccuracy,
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


def calculateStationary(
    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, imu_freq, imu_time, debug=False
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

    # # try low pass filter
    cutoff_freq = 0.1

    # Apply low-pass filter for each cutoff frequency
    acc_x_diff_lowpass = butter_lowpass_filter(
        abs_padded_acc_x_diff, imu_freq, cutoff_freq
    )
    acc_y_diff_lowpass = butter_lowpass_filter(
        abs_padded_acc_y_diff, imu_freq, cutoff_freq
    )
    acc_z_diff_lowpass = butter_lowpass_filter(
        abs_padded_acc_z_diff, imu_freq, cutoff_freq
    )
    gyro_x_diff_lowpass = butter_lowpass_filter(
        abs_padded_gyro_x_diff, imu_freq, cutoff_freq
    )
    gyro_y_diff_lowpass = butter_lowpass_filter(
        abs_padded_gyro_y_diff, imu_freq, cutoff_freq
    )
    gyro_z_diff_lowpass = butter_lowpass_filter(
        abs_padded_gyro_z_diff, imu_freq, cutoff_freq
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
    # if debug:
    #     plot_sensor_data(
    #         imu_time,
    #         acc_x_diff_lowpass,
    #         acc_y_diff_lowpass,
    #         acc_z_diff_lowpass,
    #         "Accelerometer Filtered Diff",
    #     )
    #     plot_sensor_data(
    #         imu_time,
    #         gyro_x_diff_lowpass,
    #         gyro_y_diff_lowpass,
    #         gyro_z_diff_lowpass,
    #         "Gyroscope Filtered Diff",
    #     )
    #     plot_signal_over_time(imu_time, combined_gyro_accel, "Stationary")
    #     plt.show()
    return combined_gyro_accel
