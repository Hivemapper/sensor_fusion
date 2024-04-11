import math
import csv
import os
import numpy as np
from datetime import datetime, timezone

from plottingCode import plot_signal_over_time, plot_signals_over_time, create_map
import fusion.sensorFusion as sensorFusion
from ellipsoid_fit import data_regularize, ellipsoid_fit
import ahrs

import matplotlib.pyplot as plt

# import fusion

# Function to convert each value to its appropriate data type
def convert(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value  # Return as string if neither int nor float

def importDatafromCSV(path: str):
    data = []
    with open(path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # Convert each value in the row to its appropriate data type
            typed_row = {key: convert(value) for key, value in row.items()}
            data.append(typed_row)
    return data


def convertTimeToEpoch(time_str):
    """
    Converts a time string in the format 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD HH:MM:SS.sss' to epoch milliseconds.
    Parameters:
    - time_str: A string representing the time, possibly with milliseconds ('YYYY-MM-DD HH:MM:SS[.sss]').
    Returns:
    - int: The epoch time in milliseconds.
    """
    # Determine if the time string includes milliseconds
    if '.' in time_str:
        format_str = '%Y-%m-%d %H:%M:%S.%f'
    else:
        format_str = '%Y-%m-%d %H:%M:%S'
    
    timestamp_dt = datetime.strptime(time_str, format_str)
    epoch_ms = int(timestamp_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    return epoch_ms

def convertEpochToTime(epoch_ms):
    """
    Converts an epoch time in milliseconds to a time string in the format 'YYYY-MM-DD HH:MM:SS.sss'.

    Parameters:
    - epoch_ms: The epoch time in milliseconds.

    Returns:
    - str: A string representing the time in the format 'YYYY-MM-DD HH:MM:SS.sss'.
    """
    # Convert milliseconds to seconds
    epoch_s = epoch_ms / 1000.0
    # Convert to datetime object
    datetime_obj = datetime.fromtimestamp(epoch_s, tz=timezone.utc)
    return datetime_obj.strftime('%Y-%m-%d %H:%M:%S.%f')

def extractIMUData(data):
    """
    Extracts accelerometer, gyroscope data, and time differences from the given data.
    Parameters:
    - data: List of dictionaries containing 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', and 'time'.
    Returns:
    - Tuple containing lists for 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', and time differences.
    """
    acc_x = []
    acc_y = []
    acc_z = []
    gyro_x = []
    gyro_y = []
    gyro_z = []
    time_data = []

    for point in data:
        acc_x.append(point['acc_x'])
        acc_y.append(point['acc_y'])
        acc_z.append(point['acc_z'])
        gyro_x.append(math.radians(point['gyro_x']))
        gyro_y.append(math.radians(point['gyro_y']))
        gyro_z.append(math.radians(point['gyro_z']))
        time_data.append(convertTimeToEpoch(point['time']))

    freq = math.floor(calculate_average_frequency(time_data))
    print(f"IMU data frequency: {freq} Hz")
    freq_half = freq // 4

    # Calculate rolling averages for each data type
    acc_x = calculate_rolling_average(acc_x, freq_half)
    acc_y = calculate_rolling_average(acc_y, freq_half)
    acc_z = calculate_rolling_average(acc_z, freq_half)
    gyro_x = calculate_rolling_average(gyro_x, freq_half)
    gyro_y = calculate_rolling_average(gyro_y, freq_half)
    gyro_z = calculate_rolling_average(gyro_z, freq_half)

    return acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, time_data


def extractMagData(data):
    """
    Extracts magnetometer data from the given data and computes the time differences.
    Parameters:
    - data: List of dictionaries containing 'mag_x', 'mag_y', 'mag_z', and 'system_time'.
    Returns:
    - Tuple containing lists for 'mag_x', 'mag_y', 'mag_z', and time differences.
    """
    mag_x = []
    mag_y = []
    mag_z = []
    time_data = []

    for point in data:
        mag_x.append(point['mag_x'])
        mag_y.append(point['mag_y'])
        mag_z.append(point['mag_z'])
        time_data.append(convertTimeToEpoch(point['system_time']))

    freq = math.floor(calculate_average_frequency(time_data))
    print(f"Magnetometer data frequency: {freq} Hz")
    freq_half = freq // 4

    # Calculate rolling averages for each magnetometer component
    mag_x = calculate_rolling_average(mag_x, freq_half)
    mag_y = calculate_rolling_average(mag_y, freq_half)
    mag_z = calculate_rolling_average(mag_z, freq_half)

    return mag_x, mag_y, mag_z, time_data

def extractGNSSData(data):
    """
    Extracts GNSS data from the given data and computes the time differences.
    Parameters:
    - data: List of dictionaries containing 'latitude', 'longitude', 'altitude', 'speed', 'heading', 'heading_accuracy', 'hdop', 'gdop', and 'system_time'.
    Returns:
    - Tuple containing lists for 'latitude', 'longitude', 'altitude', 'speed', 'heading', 'heading_accuracy', 'hdop', 'gdop', and time differences.
    """
    latitude = []
    longitude = []
    altitude = []
    speed = []
    heading = []
    heading_accuracy = []
    hdop = []
    gdop = []
    time_data = []

    for point in data:
        latitude.append(point['latitude'])
        longitude.append(point['longitude'])
        altitude.append(point['altitude'])
        speed.append(point['speed'])
        heading.append(point['heading'])
        heading_accuracy.append(point['heading_accuracy'])
        hdop.append(point['hdop'])
        gdop.append(point['gdop'])
        time_data.append(convertTimeToEpoch(point['system_time']))

    freq = math.floor(calculate_average_frequency(time_data))
    print(f"GNSS data frequency: {freq} Hz")

    return latitude, longitude, altitude, speed, heading, heading_accuracy, hdop, gdop, time_data

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
    
    periods_seconds = [(epoch_times_ms[i] - epoch_times_ms[i-1]) / 1000.0 for i in range(1, len(epoch_times_ms))]
    average_period_seconds = sum(periods_seconds) / len(periods_seconds)
    average_frequency = 1 / average_period_seconds if average_period_seconds != 0 else 0
    
    return average_frequency

def calculate_rolling_average(data, window_size):
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

def calculate_angular_change(headings, times):
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

def calibrate_mag(data):
    # Compute calibration center and transformation matrix
    data_regularized = data_regularize(data, divs=8)
    center, evecs, radii, v = ellipsoid_fit(data_regularized)

    a, b, c = radii
    r = np.abs(a * b * c) ** (1. / 3.)
    D = np.array([[r/a, 0., 0.], [0., r/b, 0.], [0., 0., r/c]])
    #http://www.cs.brandeis.edu/~cs155/Lecture_07_6.pdf
    #affine transformation from ellipsoid to sphere (translation excluded)
    TR = evecs.dot(D).dot(evecs.T)

    # Subtract the center offset from each data point in the dataset
    # and apply the calibration transformation
    calibrated_data = np.array([np.dot(TR, data_point - center) for data_point in data])

    return calibrated_data

def calculateHeading(accel_data, gyro_data, mag_data, gnss_initial_heading):
    # ensure same number of samples
    if not len(accel_data) == len(mag_data):
        raise ValueError("Both lists must equal size.")

    q = ahrs.Quaternion()
    # yaw, pitch, roll in radians
    q = q.from_angles(np.array([math.radians(gnss_initial_heading),0.0, 0.0]))

    # quats = sensorFusion.orientationFLAE(mag_data, accel_data)
    quats = sensorFusion.orientationEKF(mag_data, accel_data, gyro_data, q)
    euler_list = sensorFusion.convertToEuler(quats)
    heading = [euler[2] for euler in euler_list]
    pitch = [euler[1] for euler in euler_list]
    roll = [euler[0] for euler in euler_list]
    return heading, pitch, roll
    
if __name__ == "__main__":
    # Load data from a csv
    dir_path = '/Users/rogerberman/Desktop/YawFusionDrives'
    drive = 'drive3'
    gnss_path = os.path.join(dir_path, drive, f'{drive}_gnss.csv')
    imu_path = os.path.join(dir_path, drive, f'{drive}_imu.csv')
    mag_path = os.path.join(dir_path, drive, f'{drive}_mag.csv')
    print(f"Loading data from {gnss_path}")
    gnss_data = importDatafromCSV(gnss_path)
    imu_data = importDatafromCSV(imu_path)
    mag_data = importDatafromCSV(mag_path)
    print(f"Data loaded successfully!")

    print("Extracting data...")
    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, imu_time = extractIMUData(imu_data)
    mag_x, mag_y, mag_z, mag_time = extractMagData(mag_data)
    latitude, longitude, altitude, speed, heading, heading_accuracy, hdop, gdop, gnss_time = extractGNSSData(gnss_data)
    print("Data extracted successfully!")
    
    print("Downsampling data...")
    acc_x_down = np.interp(gnss_time, imu_time, acc_x)
    acc_y_down = np.interp(gnss_time, imu_time, acc_y)
    acc_z_down = np.interp(gnss_time, imu_time, acc_z)
    gyro_x_down = np.interp(gnss_time, imu_time, gyro_x)
    gyro_y_down = np.interp(gnss_time, imu_time, gyro_y)
    gyro_z_down = np.interp(gnss_time, imu_time, gyro_z)
    mag_x_down = np.interp(gnss_time, mag_time, mag_x)
    mag_y_down = np.interp(gnss_time, mag_time, mag_y)
    mag_z_down = np.interp(gnss_time, mag_time, mag_z)
    print("Data downsampled successfully!")

    print("Calculate additional OFFSETS")
    # find the indices where the speed is zero
    zero_speed_indices = [i for i, speed_val in enumerate(speed) if speed_val < 0.1]
    # Grab all indexes where the speed is zero for accel and gyro
    acc_x_down_zero_speed = [acc_x_down[i] for i in zero_speed_indices]
    acc_y_down_zero_speed = [acc_y_down[i] for i in zero_speed_indices]
    acc_z_down_zero_speed = [acc_z_down[i] for i in zero_speed_indices]
    gyro_x_down_zero_speed = [gyro_x_down[i] for i in zero_speed_indices]
    gyro_y_down_zero_speed = [gyro_y_down[i] for i in zero_speed_indices]
    gyro_z_down_zero_speed = [gyro_z_down[i] for i in zero_speed_indices]

    # Calculate the average of the zero speed values
    acc_x_down_zero_speed_avg = np.mean(acc_x_down_zero_speed)
    acc_y_down_zero_speed_avg = np.mean(acc_y_down_zero_speed)
    acc_z_down_zero_speed_avg = np.mean(acc_z_down_zero_speed) - 1  # handle the fact this needs to be 1 when at 0 velocity not 0
    gyro_x_down_zero_speed_avg = np.mean(gyro_x_down_zero_speed)
    gyro_y_down_zero_speed_avg = np.mean(gyro_y_down_zero_speed)
    gyro_z_down_zero_speed_avg = np.mean(gyro_z_down_zero_speed)

    # Apply the offsets to the data
    acc_x_down = [a - acc_x_down_zero_speed_avg for a in acc_x_down]
    acc_y_down = [a - acc_y_down_zero_speed_avg for a in acc_y_down]
    acc_z_down = [a - acc_z_down_zero_speed_avg for a in acc_z_down]
    gyro_x_down = [g - gyro_x_down_zero_speed_avg for g in gyro_x_down]
    gyro_y_down = [g - gyro_y_down_zero_speed_avg for g in gyro_y_down]
    gyro_z_down = [g - gyro_z_down_zero_speed_avg for g in gyro_z_down]

    # print all offsets in one line
    print(f"Accel offsets: {acc_x_down_zero_speed_avg}, {acc_y_down_zero_speed_avg}, {acc_z_down_zero_speed_avg}")
    print(f"Gyro offsets: {gyro_x_down_zero_speed_avg}, {gyro_y_down_zero_speed_avg}, {gyro_z_down_zero_speed_avg}")

    print("Offsets calculated successfully!")

    print("Calibrating magnetometer...")
    mag_bundle = np.array(list(zip(mag_x_down, mag_y_down, mag_z_down)))
    calibrated_mag_bundle = calibrate_mag(mag_bundle)
    print("Magnetometer calibrated successfully!")

    acc_bundle = np.array(list(zip(acc_x_down, acc_y_down, acc_z_down)))
    gyro_bundle = np.array(list(zip(gyro_x_down, gyro_y_down, gyro_z_down)))
    print("Calculating heading...")
    fused_heading, pitch, roll = calculateHeading(acc_bundle, gyro_bundle, calibrated_mag_bundle, heading[0])

    # Mag Straight Heading
    # mag_headings = [(math.atan2(y, x) * 180 / math.pi) % 360 for x, y in zip(mag_x_down, mag_y_down)]

    # used to translate the fused heading to the correct range
    fused_heading = [heading_val + 360 if heading_val < 0 else heading_val for heading_val in fused_heading]

    # used when the heading is off by 180 degrees
    fused_heading = [heading_val - 180 for heading_val in fused_heading]

    # Calculate the difference between the GNSS heading and the fused heading 
    heading_diff = []
    for i in range(len(heading_accuracy)):
        if heading_accuracy[i] < 3.0:
            heading_diff.append((heading[i] - fused_heading[i] + 180) % 360 - 180)

    heading_diff_mean = np.mean(heading_diff)
    print(f"Mean heading difference: {heading_diff_mean}")

    # plot_signal_over_time(gnss_time, heading_diff, 'Heading Diff')
    # gnss_angular_changes = calculate_angular_change(heading, gnss_time)

    # plot_path = os.path.join(dir_path, drive, f'EKF_plot_testing_{heading_diff}.png')
    # plot_signals_over_time(gnss_time, heading_diff, heading_accuracy, 'GNSS Heading', 'Heading Accuracy', None)
    plt.show()

    # print("Creating map...")
    # map_path = os.path.join(dir_path, drive, f'{drive}_EKF_map_testing.html')
    # create_map(latitude, longitude, fused_heading, map_path, 3)
    # print("Map created successfully!")