import numpy as np
import math
from typing import List

from .sqliteinterface import (
    convertTimeToEpoch,
    IMUData, 
    MagData, 
    GNSSData
)

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
    
    periods_seconds = [(epoch_times_ms[i] - epoch_times_ms[i-1]) / 1000.0 for i in range(1, len(epoch_times_ms))]
    average_period_seconds = sum(periods_seconds) / len(periods_seconds)
    average_frequency = 1 / average_period_seconds if average_period_seconds != 0 else 0
    
    return average_frequency

def calculateRollingAverage(data, window_size):
    """
    Calculates the rolling average of a list of numbers using numpy,
    padding the beginning and end with the first and last values respectively to maintain input array length.

    Parameters:
    - data: List or numpy array of numbers.
    - window_size: Size of the moving window to calculate the average.

    Returns:
    - Numpy array of rolling average values, same length as input data.
    """
    if window_size <= 1:
        return np.array(data)

    # Determine the amount of padding
    pad_size = window_size // 2
    # For even window sizes, reduce the padding by 1 at the start
    start_pad_size = pad_size if window_size % 2 != 0 else pad_size - 1
    # Pad the beginning with the first element and the end with the last element
    padded_data = np.pad(data, (start_pad_size, pad_size), 'edge')
    # Calculate the rolling average using 'valid' mode
    rolling_avg = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')

    return rolling_avg

def calculateAngularChange(headings, times):
    """
    Calculates the angular change between consecutive headings for each timestep.
    Parameters:
    - headings: List of headings in degrees.
    - times: List of times. The length must match the headings list.
    Returns:
    - List of tuples containing (time difference, angular change) for each timestep.
    """
    angular_changes = []
    for i in range(1, len(headings)):
        # Calculate time difference
        time_diff = times[i] - times[i-1]
        # Calculate angular change, accounting for angle wrap-around
        angle_diff = (headings[i] - headings[i-1] + 180) % 360 - 180
        angular_changes.append(math.radians(angle_diff))
    
    return angular_changes


def calculateAttributesAverage(data_list):
    """
    Calculates the average value of each attribute for a list of objects, ignoring non-numeric data.

    This function iterates over a list of objects, computing the average value of
    each attribute that is not a method, does not begin with '__', and is numeric.
    It assumes that all objects in the list have the same set of attributes.

    Parameters:
    - data_list (list): A list of objects with common attributes to be averaged.

    Returns:
    - dict: A dictionary where keys are attribute names and values are their average
            values across all objects in data_list. Returns None if data_list is empty.

    Raises:
    - AttributeError: If objects in data_list do not have matching attribute sets or
                      if an attribute's value is not numerical and thus cannot be summed.

    Example usage:
    Assuming a class Person with attributes 'age' and 'height', and a list of Person
    instances, calling calculate_average(list_of_people) will return a dictionary
    with the average 'age' and 'height' of the people in the list.
    """
    if not data_list:
        return None
    
    if len(data_list) == 1:
        return {k: v for k, v in data_list[0].__dict__.items() if isinstance(v, (int, float))}
    
    sum_values = {}
    count_values = {}
    attribute_names = [attr for attr in dir(data_list[0]) if not attr.startswith("__") and not callable(getattr(data_list[0], attr))]

    # Initialize sum_values and count_values dictionaries
    for attr in attribute_names:
        sum_values[attr] = 0
        count_values[attr] = 0
    
    # Sum up values for each attribute, only if they are numeric
    for instance in data_list:
        for attr in attribute_names:
            value = getattr(instance, attr)
            if isinstance(value, (int, float)):  # Check if value is numeric
                sum_values[attr] += value
                count_values[attr] += 1
    
    # Calculate the average for each attribute, considering only attributes with non-zero counts
    avg_values = {attr: sum_values[attr] / count_values[attr] for attr in attribute_names if count_values[attr] > 0}
    
    return avg_values

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
    time = []


    for point in imu_data:
        acc_x.append(point.ax)
        acc_y.append(point.ay)
        acc_z.append(point.az)
        gyro_x.append(math.radians(point.gx))
        gyro_y.append(math.radians(point.gy))
        gyro_z.append(math.radians(point.gz))
        time.append(convertTimeToEpoch(point.time))

    time = repair_time(time)

    freq = math.floor(calculateAverageFrequency(time))
    print(f"IMU data frequency: {freq} Hz")
    freq_fourth = freq // 4

    import sys
    sys.path.insert(0, '/Users/rogerberman/sensor-fusion/testingScripts')  # Add the project root to the Python path
    from testingScripts.plottingCode import plot_signal_over_time
    import matplotlib.pyplot as plt
    plot_signal_over_time(list(range(len(time))), time, 'IMU time')
    # time_diffs = []
    # indexes = []
    # for i in range(1, len(time)):
    #     diff = time[i] - time[i-1]
    #     time_diffs.append(diff)
    #     indexes.append((i-1, i))

    # # Sort time differences with the greatest at the front
    # sorted_diffs_with_indexes = sorted(zip(time_diffs, indexes), reverse=True)
    # # Extract top 15 time differences and their corresponding indexes
    # top_15_diffs_with_indexes = sorted_diffs_with_indexes[:15]
    # # Print top 15 time differences and their corresponding indexes
    # print("Top 15 Time Differences:")
    # for diff, (idx1, idx2) in top_15_diffs_with_indexes:
    #     print(f"Time difference: {diff}, Indexes: {idx1}, {idx2}")
    # plt.show()

    acc_x = calculateRollingAverage(acc_x, freq_fourth)
    acc_y = calculateRollingAverage(acc_y, freq_fourth)
    acc_z = calculateRollingAverage(acc_z, freq_fourth)
    gyro_x = calculateRollingAverage(gyro_x, freq_fourth)
    gyro_y = calculateRollingAverage(gyro_y, freq_fourth)
    gyro_z = calculateRollingAverage(gyro_z, freq_fourth)

    return acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, time

def extractAndSmoothMagData(data: List[MagData]):
    """
    Extracts magnetometer data and time differences from the given data.
    Parameters:
    - data: List of MagnetometerData instances containing 'mx', 'my', 'mz', and 'time'.
    Returns:
    - Tuple containing lists for 'mag_x', 'mag_y', 'mag_z', and time differences.
    """
    mag_x, mag_y, mag_z = [], [], []
    time = []

    for point in data:
        mag_x.append(point.mx)
        mag_y.append(point.my)
        mag_z.append(point.mz)
        time.append(convertTimeToEpoch(point.time))

    time = repair_time(time)

    import sys
    sys.path.insert(0, '/Users/rogerberman/sensor-fusion/testingScripts')  # Add the project root to the Python path
    from testingScripts.plottingCode import plot_signal_over_time
    import matplotlib.pyplot as plt
    plot_signal_over_time(list(range(len(time))), time, 'Mag time')
    # time_diffs = []
    # indexes = []
    # for i in range(1, len(time)):
    #     diff = time[i] - time[i-1]
    #     time_diffs.append(diff)
    #     indexes.append((i-1, i))

    # # Sort time differences with the greatest at the front
    # sorted_diffs_with_indexes = sorted(zip(time_diffs, indexes), reverse=True)
    # # Extract top 15 time differences and their corresponding indexes
    # top_15_diffs_with_indexes = sorted_diffs_with_indexes[:15]
    # # Print top 15 time differences and their corresponding indexes
    # print("Top 15 Time Differences:")
    # for diff, (idx1, idx2) in top_15_diffs_with_indexes:
    #     print(f"Time difference: {diff}, Indexes: {idx1}, {idx2}")
    # plt.show()
    

    freq = math.floor(calculateAverageFrequency(time))
    print(f"Magnetometer data frequency: {freq} Hz")
    freq_fourth = freq // 4

    mag_x = calculateRollingAverage(mag_x, freq_fourth)
    mag_y = calculateRollingAverage(mag_y, freq_fourth)
    mag_z = calculateRollingAverage(mag_z, freq_fourth)

    return mag_x, mag_y, mag_z, time

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
    time = []

    for point in data:
        lat.append(point.lat)
        lon.append(point.lon)
        alt.append(point.alt)
        speed.append(point.speed)
        heading.append(point.heading)
        headingAccuracy.append(point.headingAccuracy)
        hdop.append(point.hdop)
        gdop.append(point.gdop)
        time.append(convertTimeToEpoch(point.time))

    freq = math.floor(calculateAverageFrequency(time))
    print(f"GNSS data frequency: {freq} Hz")

    return lat, lon, alt, speed, heading, headingAccuracy, hdop, gdop, time, freq


def repair_time(time):
    # Repair imu and mag time when gnss drops
    DAY_IN_MS = 1000 * 60 * 60 * 24
    repaired_time = []
    repaired_time.append(time[0])
    
    # Find pairs of indices where time needs to be repaired
    offset = 0
    first = False
    for i in range(1, len(time)):
        time_diff = time[i] - time[i - 1]

        # identify when gnss drops occur
        if time_diff <= -DAY_IN_MS:
            print(f"GNSS DROP: Time difference: {time_diff} at index {i}")
            offset = abs(time_diff)
            first = True
        elif time_diff >= DAY_IN_MS:
            offset = 0
        
        # apply offset due to gnss drops
        # first case apply average period to compensate for the drop(best? compensation for the first drop)
        if offset > 0 and first:
            time_diffs = [repaired_time[i] - repaired_time[i - 1] for i in range(1, len(repaired_time))]
            average_period = sum(time_diffs) / len(time_diffs)
            offset = offset + average_period
            first = False
            repaired_time.append(time[i] + offset)
        # second case apply offset to the rest of the time where drop occurs
        elif offset > 0 and not first:
            repaired_time.append(time[i] + offset)
        # no drop append as normal
        else:
            repaired_time.append(time[i])

    # Check if the length of the time list changed
    if len(time) != len(repaired_time):
        print(f"Error: Time list length changed from {len(time)} to {len(repaired_time)}")
    return repaired_time