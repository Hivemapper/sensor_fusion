import math
import csv
import os
import numpy
from datetime import datetime, timezone

from plottingCode import plot_signal_over_time, plot_signals_over_time, create_map
import fusion.sensorFusion as sensorFusion
from ahrs.common.orientation import rpy2q, q2euler, q2rpy
import ahrs

from scipy.spatial.transform import Rotation as R

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
        return numpy.array(data)

    # Determine the amount of padding
    pad_size = window_size // 2
    # For even window sizes, reduce the padding by 1 at the start
    start_pad_size = pad_size if window_size % 2 != 0 else pad_size - 1
    # Pad the beginning with the first element and the end with the last element
    padded_data = numpy.pad(data, (start_pad_size, pad_size), 'edge')
    # Calculate the rolling average using 'valid' mode
    rolling_avg = numpy.convolve(padded_data, numpy.ones(window_size) / window_size, mode='valid')

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


def calculateHeading(accel_data, gyro_data, mag_data, gnss_initial_heading):
    # ensure same number of samples
    if not len(accel_data) == len(mag_data):
        raise ValueError("Both lists must equal size.")
    
    # print(gnss_initial_heading)

    # q = ahrs.Quaternion()
    # q = q.from_angles(numpy.array([math.radians(gnss_initial_heading),0.0, 0.0]))     # roll, pitch, yaw in radians
    # full_q = ahrs.Quaternion(q)
    # print(math.degrees(full_q.to_angles()[0]))

    # quats = sensorFusion.orientationFLAE(mag_data, accel_data)
    quats = sensorFusion.orientationEKF(mag_data, accel_data, gyro_data)
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
    acc_x_down = numpy.interp(gnss_time, imu_time, acc_x)
    acc_y_down = numpy.interp(gnss_time, imu_time, acc_y)
    acc_z_down = numpy.interp(gnss_time, imu_time, acc_z)
    gyro_x_down = numpy.interp(gnss_time, imu_time, gyro_x)
    gyro_y_down = numpy.interp(gnss_time, imu_time, gyro_y)
    gyro_z_down = numpy.interp(gnss_time, imu_time, gyro_z)
    mag_x_down = numpy.interp(gnss_time, mag_time, mag_x)
    mag_y_down = numpy.interp(gnss_time, mag_time, mag_y)
    mag_z_down = numpy.interp(gnss_time, mag_time, mag_z)
    print("Data downsampled successfully!")
    # OFFSET Removal 
    gyro_z_down = [g-0.025 for g in gyro_z_down]
    gyro_y_down = [g+0.015 for g in gyro_y_down]
    gyro_x_down = [g+0.038 for g in gyro_x_down]

    acc_x_down = [a-0.155 for a in acc_x_down]
    acc_y_down = [a-0.005 for a in acc_y_down]
    acc_z_down = [a-0.009 for a in acc_z_down]


    acc_bundle = numpy.array(list(zip(acc_x_down, acc_y_down, acc_z_down)))
    gyro_bundle = numpy.array(list(zip(gyro_x_down, gyro_y_down, gyro_z_down)))
    mag_bundle = numpy.array(list(zip(mag_x_down, mag_y_down, mag_z_down)))
    print("Calculating heading...")
    fused_heading, pitch, roll = calculateHeading(acc_bundle, gyro_bundle, mag_bundle, heading[0])

    # Adjust the fused heading to match the GNSS heading
    # heading_diff = heading[0] - fused_heading[0]
    # fused_heading = [heading_val + heading_diff for heading_val in fused_heading]

    plot_signal_over_time(gnss_time, roll, 'Pitch')
    # gnss_angular_changes = calculate_angular_change(heading, gnss_time)
    # print(len(gnss_angular_changes), len(gnss_time))
    # plot_signals_over_time(gnss_time[100:], heading[100:], fused_heading[100:], 'GNSS Heading', 'Fused Heading')

    print("Creating map...")
    map_path = os.path.join(dir_path, drive, 'map_testing.html')
    create_map(latitude, longitude, fused_heading, map_path, 3)
    print("Map created successfully!")