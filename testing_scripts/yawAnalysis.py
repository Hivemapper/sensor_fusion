import math
import csv
import os
import numpy as np
from datetime import datetime, timezone
import sys

sys.path.insert(
    0, "/Users/rogerberman/sensor-fusion"
)  # Add the project root to the Python path

from offline_code.utils.plotting_code import (
    plot_signal_over_time,
    plot_signals_over_time,
    create_map,
)
from fusion_old import (
    SqliteInterface,
    extractAndSmoothImuData,
    extractAndSmoothMagData,
    extractGNSSData,
    calculateHeading,
    calibrate_mag,
    getCleanGNSSHeading,
    GNSS_LOW_SPEED_THRESHOLD,
    HEADING_DIFF_MAGNETOMETER_FLIP_THRESHOLD,
    GNSS_HEADING_ACCURACY_THRESHOLD,
    ASC,
)

import matplotlib.pyplot as plt


def make_headings_continuous(headings):
    if not headings:
        return []

    corrected_headings = [headings[0]]
    full_circle = 360

    for i in range(1, len(headings)):
        diff = headings[i] - headings[i - 1]

        # Correct the difference to be the minimal angle difference
        diff -= full_circle * round(diff / full_circle)

        # Adjust the current heading based on the normalized difference
        corrected_heading = corrected_headings[-1] + diff
        corrected_headings.append(corrected_heading)

    return corrected_headings


def is_chronological(epoch_times):
    # Check each item in the list after the first
    for i in range(1, len(epoch_times)):
        # If a previous item is greater than the current, list is not in order
        if epoch_times[i - 1] > epoch_times[i]:
            return False
    return True


if __name__ == "__main__":
    # Load data from a csv
    dir_path = "/Users/rogerberman/Desktop/YawFusionDrives"
    drive = "drive4"
    data_logger_path = os.path.join(dir_path, drive, "data-logger.v1.4.4.db")
    print(f"Loading data from {data_logger_path}")
    sql_db = SqliteInterface(data_logger_path)
    gnss_data = sql_db.queryGnss(0, 0, ASC)
    imu_data = sql_db.queryImu(0, 0, ASC)
    mag_data = sql_db.queryMagnetometer(0, 0, ASC)
    print(f"len(imu_data): {len(imu_data)}")
    print(f"len(mag_data): {len(mag_data)}")
    print(f"len(gnss_data): {len(gnss_data)}")

    print(f"Data loaded successfully!")

    print("Extracting data...")
    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, imu_time = extractAndSmoothImuData(
        imu_data
    )
    print(f"IMU Order: {is_chronological(imu_time)}")
    print(f"imu start time: {imu_time[0]}")
    print(f"imu end time: {imu_time[-1]}")
    mag_x, mag_y, mag_z, mag_time = extractAndSmoothMagData(mag_data)
    print(f"Mag Order: {is_chronological(mag_time)}")
    print(f"mag start time: {mag_time[0]}")
    print(f"mag end time: {mag_time[-1]}")
    (
        latitude,
        longitude,
        altitude,
        speed,
        heading,
        heading_accuracy,
        hdop,
        gdop,
        gnss_time,
        gnssFreq,
    ) = extractGNSSData(gnss_data)
    print(f"GNSS Order: {is_chronological(gnss_time)}")
    print(f"gnss start time: {gnss_time[0]}")
    print(f"gnss end time: {gnss_time[-1]}")
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
    zero_speed_indices = [
        i for i, speed_val in enumerate(speed) if speed_val < GNSS_LOW_SPEED_THRESHOLD
    ]
    # Grab all indexes where the speed is zero for accel and gyro
    acc_x_down_zero_speed, acc_y_down_zero_speed, acc_z_down_zero_speed = [], [], []
    gyro_x_down_zero_speed, gyro_y_down_zero_speed, gyro_z_down_zero_speed = [], [], []
    for i in zero_speed_indices:
        acc_x_down_zero_speed.append(acc_x_down[i])
        acc_y_down_zero_speed.append(acc_y_down[i])
        acc_z_down_zero_speed.append(acc_z_down[i])
        gyro_x_down_zero_speed.append(gyro_x_down[i])
        gyro_y_down_zero_speed.append(gyro_y_down[i])
        gyro_z_down_zero_speed.append(gyro_z_down[i])

    # Calculate the average of the zero speed values
    acc_x_down_zero_speed_avg = np.mean(acc_x_down_zero_speed)
    acc_y_down_zero_speed_avg = np.mean(acc_y_down_zero_speed)
    acc_z_down_zero_speed_avg = (
        np.mean(acc_z_down_zero_speed) - 1
    )  # handle the fact this needs to be 1 when at 0 velocity not 0
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
    print(
        f"Accel offsets: {acc_x_down_zero_speed_avg}, {acc_y_down_zero_speed_avg}, {acc_z_down_zero_speed_avg}"
    )
    print(
        f"Gyro offsets: {gyro_x_down_zero_speed_avg}, {gyro_y_down_zero_speed_avg}, {gyro_z_down_zero_speed_avg}"
    )

    print("Offsets calculated successfully!")

    print("Calibrating magnetometer...")
    mag_bundle = np.array(list(zip(mag_x_down, mag_y_down, mag_z_down)))
    calibrated_mag_bundle = calibrate_mag(mag_bundle)
    print("Magnetometer calibrated successfully!")

    acc_bundle = np.array(list(zip(acc_x_down, acc_y_down, acc_z_down)))
    gyro_bundle = np.array(list(zip(gyro_x_down, gyro_y_down, gyro_z_down)))
    print("Calculating heading...")
    print(f"data length: {len(acc_bundle)}")
    fused_heading, pitch, roll = calculateHeading(
        acc_bundle, gyro_bundle, calibrated_mag_bundle, heading[0], gnssFreq
    )

    # Mag Straight Heading
    mag_headings = [
        (math.atan2(y, x) * 180 / math.pi) % 360 for x, y in zip(mag_x_down, mag_y_down)
    ]

    # used to translate the fused heading to the correct range
    fused_heading = [
        heading_val + 360 if heading_val < 0 else heading_val
        for heading_val in fused_heading
    ]

    # check last heading diff to make decision to flip the heading
    if abs(fused_heading[-1] - heading[-1]) > HEADING_DIFF_MAGNETOMETER_FLIP_THRESHOLD:
        # handle wrap around and shift by 180 degrees
        fused_heading = [(heading_val - 180) % 360 for heading_val in fused_heading]

    cleanHeading, _ = getCleanGNSSHeading(sql_db, 0, 0)
    # Calculate the difference between the GNSS heading and the fused heading
    heading_diff = []
    time = []
    for i in range(len(heading_accuracy)):
        if heading_accuracy[i] < GNSS_HEADING_ACCURACY_THRESHOLD:
            heading_diff.append((fused_heading[i] - heading[i] + 180) % 360 - 180)
            time.append(gnss_time[i])

    heading_diff_mean = np.mean(heading_diff)
    print(f"Mean heading difference: {heading_diff_mean}")
    # print(len(heading_diff))
    step_size = 10
    number_of_points = []
    mean_diff = []
    for i in range(step_size, len(heading_diff), step_size):
        number_of_points.append(i)
        mean_diff.append(np.mean(heading_diff[:i]))

    plot_signal_over_time(number_of_points, mean_diff, "Heading Diff Mean")
    plot_signal_over_time(time, heading_diff, "Heading Diff")

    # plot_path = os.path.join(dir_path, drive, f'EKF_plot_testing_{heading_diff}.png')
    plot_signals_over_time(
        gnss_time,
        cleanHeading,
        fused_heading,
        "Clean GNSS Heading",
        "Fused Heading",
        None,
    )
    plt.show()

    # print("Creating map...")
    # map_path = os.path.join(dir_path, drive, f'{drive}_EKF_map_testing.html')
    # create_map(latitude, longitude, fused_heading, map_path, 3)
    # print("Map created successfully!")

    # x axis is number of data points
    # y axis is the mean heading difference
