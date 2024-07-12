import numpy as np
import math
from typing import List
from scipy.signal import butter, filtfilt

from .sqliteinterface import convertTimeToEpoch, IMUData, MagData, GNSSData
from ahrs.utils.wmm import WMM


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


def calculate_rates_and_counts(timestamps):
    rates = []
    for i in range(1, len(timestamps)):
        time_diff = timestamps[i] - timestamps[i - 1]
        period = time_diff / 1000.0  # Convert milliseconds to seconds
        rate = 1.0 / period if period != 0 else 0  # Avoid division by zero
        rounded_rate = math.floor(rate)  # Floor the rate to keep it at the ones place
        rates.append(rounded_rate)

    # Initialize an empty dictionary to store rate counts
    rate_counts = {}

    # Count occurrences of each rate
    for rate in rates:
        if rate in rate_counts:
            rate_counts[rate] += 1
        else:
            rate_counts[rate] = 1

    return rate_counts, rates


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
    padded_data = np.pad(data, (start_pad_size, pad_size), "edge")
    # Calculate the rolling average using 'valid' mode
    rolling_avg = np.convolve(
        padded_data, np.ones(window_size) / window_size, mode="valid"
    )

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
        time_diff = times[i] - times[i - 1]
        # Calculate angular change, accounting for angle wrap-around
        angle_diff = (headings[i] - headings[i - 1] + 180) % 360 - 180
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
        return {
            k: v
            for k, v in data_list[0].__dict__.items()
            if isinstance(v, (int, float))
        }

    sum_values = {}
    count_values = {}
    attribute_names = [
        attr
        for attr in dir(data_list[0])
        if not attr.startswith("__") and not callable(getattr(data_list[0], attr))
    ]

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
    avg_values = {
        attr: sum_values[attr] / count_values[attr]
        for attr in attribute_names
        if count_values[attr] > 0
    }

    return avg_values


def extractAndSmoothImuData(imu_data: List[IMUData], inital_gnss_time: str = None):
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

    if inital_gnss_time is None:
        initial_time = 0
    else:
        initial_time = convertTimeToEpoch(inital_gnss_time)

    for point in imu_data:
        cur_time = convertTimeToEpoch(point.system_time)
        if cur_time >= initial_time:
            acc_x.append(point.ax)
            acc_y.append(point.ay)
            acc_z.append(point.az)
            gyro_x.append(math.radians(point.gx))
            gyro_y.append(math.radians(point.gy))
            gyro_z.append(math.radians(point.gz))
            time.append(cur_time)

    freq = math.floor(calculateAverageFrequency(time))
    print(f"IMU data frequency: {freq} Hz")

    acc_x = butter_lowpass_filter(acc_x, freq)
    acc_y = butter_lowpass_filter(acc_y, freq)
    acc_z = butter_lowpass_filter(acc_z, freq)
    gyro_x = butter_lowpass_filter(gyro_x, freq)
    gyro_y = butter_lowpass_filter(gyro_y, freq)
    gyro_z = butter_lowpass_filter(gyro_z, freq)

    return acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, time, freq


def extractAndSmoothMagData(data: List[MagData], inital_gnss_time: str):
    """
    Extracts magnetometer data and time differences from the given data.
    Parameters:
    - data: List of MagnetometerData instances containing 'mx', 'my', 'mz', and 'time'.
    Returns:
    - Tuple containing lists for 'mag_x', 'mag_y', 'mag_z', and time differences.
    """
    mag_x, mag_y, mag_z = [], [], []
    time = []

    initial_time = convertTimeToEpoch(inital_gnss_time)

    for point in data:
        cur_time = convertTimeToEpoch(point.time)
        if cur_time >= initial_time:
            mag_x.append(point.mx)
            mag_y.append(point.my)
            mag_z.append(point.mz)
            time.append(cur_time)

    freq = math.floor(calculateAverageFrequency(time))
    print(f"Magnetometer data frequency: {freq} Hz")

    mag_x = butter_lowpass_filter(mag_x, freq)
    mag_y = butter_lowpass_filter(mag_y, freq)
    mag_z = butter_lowpass_filter(mag_z, freq)

    return mag_x, mag_y, mag_z, time, freq


def butter_lowpass_filter(data, fs, cutoff=1, order=2):
    nyq = 0.5 * fs  # Define Nyquist Frequency
    normal_cutoff = cutoff / nyq  # Normalize cutoff frequency
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)
    return y


def extractGNSSData(data: List[GNSSData], inital_gnss_time: str = None):
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
    gnss_real_time, time_resolved = [], []

    if inital_gnss_time is None:
        initial_time = 0
    else:
        initial_time = convertTimeToEpoch(inital_gnss_time)

    for point in data:
        cur_time = convertTimeToEpoch(point.system_time)
        if cur_time >= initial_time:
            lat.append(point.lat)
            lon.append(point.lon)
            alt.append(point.alt)
            speed.append(point.speed)
            heading.append(point.heading)
            headingAccuracy.append(point.headingAccuracy)
            hdop.append(point.hdop)
            gdop.append(point.gdop)
            system_time.append(cur_time)
            gnss_real_time.append(convertTimeToEpoch(point.time))
            time_resolved.append(point.time_resolved)

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
        freq,
    )


def calculate_mag_headings(mag_bundle, accel_bundle, position):
    lat, lon, alt = position
    wmm = WMM(latitude=lat, longitude=lon, height=alt)
    declination = wmm.magnetic_elements["D"]
    # print(f"Declination: {declination} degrees")

    m_xs = mag_bundle[:, 0]
    m_ys = mag_bundle[:, 1]
    m_zs = mag_bundle[:, 2]
    a_xs = accel_bundle[:, 0]
    a_ys = accel_bundle[:, 1]
    a_zs = accel_bundle[:, 2]

    headings = []

    # Loop through all measurements
    for m_x, m_y, m_z, a_x, a_y, a_z in zip(m_xs, m_ys, m_zs, a_xs, a_ys, a_zs):
        # Calculate pitch and roll from accelerometer data
        pitch = math.atan2(-a_x, math.sqrt(a_y**2 + a_z**2))
        roll = math.atan2(a_y, a_z)
        # print(f"Pitch: {math.degrees(pitch)}, Roll: {math.degrees(roll)}")

        # Compensate the magnetometer data
        m_x_prime = m_x * math.cos(roll) + m_z * math.sin(roll)
        m_y_prime = (
            m_x * math.sin(pitch) * math.sin(roll)
            + m_y * math.cos(pitch)
            - m_z * math.sin(pitch) * math.cos(roll)
        )

        # Calculate the magnetic heading
        magnetic_heading = math.atan2(m_x_prime, m_y_prime) * (180 / math.pi)

        # Adjust for magnetic declination to get true heading
        true_heading = magnetic_heading + declination
        # Normalize the heading to be within 0-360 degrees
        if true_heading >= 360:
            true_heading -= 360
        elif true_heading < 0:
            true_heading += 360

        # Append the calculated true heading to the list
        headings.append(true_heading)

    return headings


def calculate_imu_forward_velocity(
    acc_bundle, imu_time, gnss_speed, angle_offset_degrees
):
    # Constants
    GNSS_ONE_SECOND = (
        8  # Adjust this to your actual number of samples per second in GNSS data
    )
    ACCEL_GRAVITY = 9.81  # Acceleration due to gravity (m/s^2)
    ONE_SECOND = 1000  # Time units in milliseconds
    RADIANS = np.pi / 180  # Conversion factor from degrees to radians

    # Convert angle offset from degrees to radians
    angle_offset_radians = angle_offset_degrees * RADIANS

    # Grab initial velocity from first second of GNSS data
    gnss_speed = [data for data in gnss_speed[:GNSS_ONE_SECOND]]
    velocity_forward = sum(gnss_speed) / len(gnss_speed) if gnss_speed else 0

    # Calculate deltaTImu
    totalTime = imu_time[-1] - imu_time[0]
    deltaTImu = totalTime / len(imu_time) / ONE_SECOND  # Convert ms to seconds

    # Lists to store velocity for analysis
    velocity_history = []

    # Iterate through the IMU data
    for data in acc_bundle:
        acc_x = data[0]
        acc_y = data[1]

        # Adjust acceleration components based on the offset angle
        acc_forward = acc_x * np.cos(angle_offset_radians) + acc_y * np.sin(
            angle_offset_radians
        )

        # Integrate forward acceleration to obtain velocity
        velocity_forward += acc_forward * ACCEL_GRAVITY * deltaTImu
        velocity_history.append(velocity_forward)

    return velocity_history
