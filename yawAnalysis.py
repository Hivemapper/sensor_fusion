import folium
import math
import csv
import os
import matplotlib.pyplot as plt
import numpy
from datetime import datetime, timezone

# import fusion

def create_map(data, map_filename, plot_every=1):
    """
    Creates a map and plots every 'plot_every' points from the 'data'.
    
    Parameters:
    - data: List of dictionaries containing 'latitude', 'longitude', and 'heading'.
    - map_filename: Filename for the saved map.
    - plot_every: Interval at which points are plotted (1 = every point, 2 = every second point, etc.).
    """
    # Create a map object centered around the average location
    m = folium.Map(location=[sum(p['latitude'] for p in data) / len(data), 
                             sum(p['longitude'] for p in data) / len(data)], 
                   zoom_start=13)  # Adjust zoom level as needed

    # Add points and short lines to the map, plotting every 'plot_every' points
    for i, point in enumerate(data):
        if i % plot_every == 0:  # Plot only every 'plot_every' points
            # Add a small circle dot for the point
            folium.CircleMarker(
                location=[point['latitude'], point['longitude']],
                radius=3,  # small radius for the circle marker
                color='red',
                fill=True,
                fill_color='red'
            ).add_to(m)

            # Calculate end point for the short line
            line_length = 0.00003  # Adjust this for line length
            end_lat = point['latitude'] + line_length * math.cos(math.radians(point['heading']))
            end_lon = point['longitude'] + line_length * math.sin(math.radians(point['heading']))

            # Create a short line
            folium.PolyLine([(point['latitude'], point['longitude']), (end_lat, end_lon)], 
                            color='blue', weight=3, opacity=1).add_to(m)

    # Save the map to an HTML file
    m.save(map_filename)


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


def plot_heading_over_time(seconds, headings):
    """
    Plots heading values over time, where time is represented in seconds.

    Parameters:
    - seconds: A list of timestamps in seconds.
    - headings: A list of heading values corresponding to each timestamp.
    """
    # Ensure the lists are of the same length
    if len(seconds) != len(headings):
        print("Error: The lists of timestamps and headings must have the same length.")
        return
    
    plt.figure(figsize=(10, 6))  # Adjust figure size as desired
    plt.plot(seconds, headings, marker='o', linestyle='-', color='b')

    # Formatting the plot
    plt.title('Heading Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Heading')

    plt.grid(True)
    plt.show()

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
    Extracts accelerometer and gyroscope data from the given data, and zeros the time values.
    Parameters:
    - data: List of dictionaries containing 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', and 'time'.
    Returns:
    - Tuple[numpy.ndarray, numpy.ndarray, List]: 
      A tuple containing two numpy arrays of accelerometer and gyroscope data, and a list of time differences.
    """
    accel_data = []
    gyro_data = []
    time_data = []
    time_start = convertTimeToEpoch(data[0]['time'])

    for point in data:
        accel_data.append((point['acc_x'], point['acc_y'], point['acc_z']))
        gyro_data.append((point['gyro_x'], point['gyro_y'], point['gyro_z']))
        time_data.append(convertTimeToEpoch(point['time']) - time_start)

    # Convert lists of tuples to numpy arrays for accelerometer and gyroscope data
    accel_data = numpy.array(accel_data)
    gyro_data = numpy.array(gyro_data)
    return accel_data, gyro_data, time_data


def extractMagData(data):
    """
    Extracts magnetometer data from the given data, and zeros the time values.
    Parameters:
    - data: List of dictionaries containing 'mag_x', 'mag_y', 'mag_z', and 'system_time'.
    Returns:
    - Tuple[numpy.ndarray, List]: 
      A tuple containing a numpy array of magnetometer data, and a list of time differences.
    """
    mag_data = []
    time_data = []
    time_start = convertTimeToEpoch(data[0]['system_time'])

    for point in data:
        mag_data.append((point['mag_x'], point['mag_y'], point['mag_z']))
        time_data.append(convertTimeToEpoch(point['system_time']) - time_start)

    mag_data = numpy.array(mag_data)
    return mag_data, time_data

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


def calculateHeading(imu_data, mag_data):
    # extract accel xyz, gyro xyz, and mag xyz
    accel_xyz, gyro_xyz, imu_time = extractIMUData(imu_data)
    mag_xyz, mag_time = extractMagData(mag_data)
    print(f"Period of IMU data: {calculate_average_frequency(imu_time)} Hz")
    print(f"Period of magnetometer data: {calculate_average_frequency(mag_time)} Hz")
    # ensure same number of samples
    # if not len(accel_xyz) == len(mag_xyz):
    #     print(f'Length of accel_xyz: {len(accel_xyz)}, Length of mag_xyz: {len(mag_xyz)}')
    #     raise ValueError("Both lists must equal size.")
        # implement equalizer for list sizes
    
    # quats = fusion.orientationFLAE(mag_xyz, accel_xyz, gyro_xyz)
    # euler_list = fusion.convertToEuler(quats)
    # return [euler[2] for euler in euler_list]
    


if __name__ == "__main__":
    # Load data from a csv
    dir_path = '/Users/rogerberman/Desktop/YawFusionDrives'
    drive = 'drive3'
    gnss_path = os.path.join(dir_path, drive, f'{drive}_gnss.csv')
    imu_path = os.path.join(dir_path, drive, f'{drive}_imu.csv')
    mag_path = os.path.join(dir_path, drive, f'{drive}_mag.csv')
    map_path = os.path.join(dir_path, drive, 'map.html')
    print(f"Loading data from {gnss_path}")
    gnss_data = importDatafromCSV(gnss_path)
    imu_data = importDatafromCSV(imu_path)
    mag_data = importDatafromCSV(mag_path)
    print(f"Data loaded successfully!")
    calculateHeading(imu_data, mag_data)

    # print("Creating map...")
    # create_map(gnss_data, map_path, 3)
    # print("Map created successfully!")