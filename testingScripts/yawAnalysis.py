import math
import csv
import os
import numpy as np
from datetime import datetime, timezone

from plottingCode import plot_signal_over_time, plot_signals_over_time, create_map
from fusion.sensorFusion import calculateHeading
from fusion.utils import calculateAverageFrequency, calculateRollingAverage
from fusion.ellipsoid_fit import calibrate_mag
import ahrs

import matplotlib.pyplot as plt

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

    freq = math.floor(calculateAverageFrequency(time_data))
    print(f"IMU data frequency: {freq} Hz")
    freq_half = freq // 4

    # Calculate rolling averages for each data type
    acc_x = calculateRollingAverage(acc_x, freq_half)
    acc_y = calculateRollingAverage(acc_y, freq_half)
    acc_z = calculateRollingAverage(acc_z, freq_half)
    gyro_x = calculateRollingAverage(gyro_x, freq_half)
    gyro_y = calculateRollingAverage(gyro_y, freq_half)
    gyro_z = calculateRollingAverage(gyro_z, freq_half)

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

    freq = math.floor(calculateAverageFrequency(time_data))
    print(f"Magnetometer data frequency: {freq} Hz")
    freq_half = freq // 4

    # Calculate rolling averages for each magnetometer component
    mag_x = calculateRollingAverage(mag_x, freq_half)
    mag_y = calculateRollingAverage(mag_y, freq_half)
    mag_z = calculateRollingAverage(mag_z, freq_half)

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

    freq = math.floor(calculateAverageFrequency(time_data))
    print(f"GNSS data frequency: {freq} Hz")

    return latitude, longitude, altitude, speed, heading, heading_accuracy, hdop, gdop, time_data
    
if __name__ == "__main__":
    # Load data from a csv
    dir_path = '/Users/rogerberman/Desktop/YawFusionDrives'
    drive = 'drive2'
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
    # print(f"GNSS Start Time: {gnss_time[0]}",
    #       f"\nGNSS End Time: {gnss_time[-1]}")
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

    # check last heading diff to make decision
    if abs(heading[-1] - fused_heading[-1]) > 100:
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
    plot_signals_over_time(gnss_time, heading, fused_heading, 'GNSS Heading', 'Fused Heading', None)
    plt.show()

    # print("Creating map...")
    # map_path = os.path.join(dir_path, drive, f'{drive}_EKF_map_testing.html')
    # create_map(latitude, longitude, fused_heading, map_path, 3)
    # print("Map created successfully!")