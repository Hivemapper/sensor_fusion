import sys
from processDBs import validate_dbs, process_db_file_for_individual_drives
sys.path.insert(0, '/Users/rogerberman/hivemapper/sensor-fusion')  # Add the project root to the Python path
from fusion import (
    SqliteInterface,
    extractAndSmoothImuData,
    extractAndSmoothMagData,
    extractGNSSData,
    calculateHeading, 
    calculate_rates_and_counts,
    calculateRollingAverage,
    butter_lowpass_filter,
    calibrate_mag, 
    getCleanGNSSHeading,
    getDashcamToVehicleHeadingOffset,
    convertTimeToEpoch,
    convertEpochToTime,
    GNSS_LOW_SPEED_THRESHOLD,
    HEADING_DIFF_MAGNETOMETER_FLIP_THRESHOLD,
    GNSS_HEADING_ACCURACY_THRESHOLD,
    ASC,
)
from plottingCode import (
    plot_signal_over_time, 
    plot_signals_over_time, 
    create_map_with_highlighted_indexes, 
    plot_rate_counts,
    plot_sensor_data,
    plot_sensor_data_classified,
    plot_lat_lon_with_highlights
)
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.interpolate import CubicSpline

def align_sensor_with_reference(sensor_time, sensor_data, ref_time):
    """
    Aligns sensor data with a reference time using cubic spline interpolation.
    
    Parameters:
        sensor_time (ndarray): Timestamps of the sensor data.
        sensor_data (ndarray): Sensor data values.
        ref_time (ndarray): Reference timestamps to align with.
    
    Returns:
        ndarray: Aligned sensor data interpolated to match the reference timestamps.
    """
    # Create a cubic spline interpolation function for sensor data
    sensor_interp_func = CubicSpline(sensor_time, sensor_data)

    # Interpolate sensor data at the timestamps of the reference time
    aligned_sensor_data = sensor_interp_func(ref_time)

    return aligned_sensor_data

def threshold_based_window_averaging(data, times, window_size_ms, threshold):
    """
    Applies threshold-based window averaging to the input data.

    Parameters:
        data (list or ndarray): Input data points.
        times (list or ndarray): Timestamps in epoch milliseconds corresponding to the data points.
        window_size_ms (int): Size of the time window in milliseconds.
        threshold (float): Threshold value for determining the output (0 or 1).

    Returns:
        list: Output list of 0s and 0.25s based on threshold-based window averaging.
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
        window_data_indices = np.where((times >= times[window_start]) & (times < window_end_time))[0]

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

if __name__ == "__main__":
    # Load data and filter data
    dir_path = '/Users/rogerberman/dev_ground/test_data/imu_stationary_test_data-decoded-recovered'
    time_now = time.time()
    drives = validate_dbs(dir_path)
    print(f"Validation Done in {time.time()-time_now} seconds")


    for user in sorted(drives):
        for drive_info in sorted(drives[user]):
            db_file_path = drive_info[0]
            camera_type = drive_info[1]
            if camera_type != 'hdcs':
                continue
            usable_drives = process_db_file_for_individual_drives(db_file_path, camera_type)
            try:
                for session in usable_drives:
                    print(f"Processing Session: {session}")
                    drive_data = usable_drives[session]
                    gnss_data = drive_data['gnss_data']
                    imu_data = drive_data['imu_data']
                    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, imu_time, imu_freq = extractAndSmoothImuData(imu_data)
                    lats, lons, alts, speed, heading, headingAccuracy, hdop, gdop, gnss_system_time, gnss_real_time, time_resolved, gnss_freq = extractGNSSData(gnss_data)
                    print(f"GNSS Start Time: {gnss_system_time[0]}, IMU Start Time: {imu_time[0]}")
                    if imu_time[0] > gnss_system_time[0]:
                        print("IMU Data starts after GNSS Data***")
                        continue

                    # if imu_freq < 15:
                    #     print("IMU Frequency is too low***")
                    #     continue

                    # gnss_rates = calculate_rates_and_counts(gnss_system_time)
                    # imu_rates = calculate_rates_and_counts(imu_time)
                    # plot_rate_counts(gnss_rates, f'GNSS Freq Rates, Session:{session}')
                    # plot_rate_counts(imu_rates, f'IMU Freq Rates, Session:{session}')

                    # plot_signal_over_time(list(range(len(gnss_system_time))), gnss_system_time, 'GNSS Time')
                    # plot_signal_over_time(list(range(len(imu_time))), imu_time, 'IMU Time')

                    # Align sensor data with GNSS timestamps
                    acc_x_aligned = align_sensor_with_reference(imu_time, acc_x, gnss_system_time)
                    acc_y_aligned = align_sensor_with_reference(imu_time, acc_y, gnss_system_time)
                    acc_z_aligned = align_sensor_with_reference(imu_time, acc_z, gnss_system_time)
                    gyro_x_aligned = align_sensor_with_reference(imu_time, gyro_x, gnss_system_time)
                    gyro_y_aligned = align_sensor_with_reference(imu_time, gyro_y, gnss_system_time)
                    gyro_z_aligned = align_sensor_with_reference(imu_time, gyro_z, gnss_system_time)
                    # get point to point diffs for all sensors
                    acc_x_diff = np.diff(acc_x_aligned)
                    padded_acc_x_diff = np.concatenate(([acc_x_diff[0]], acc_x_diff))
                    acc_y_diff = np.diff(acc_y_aligned)
                    padded_acc_y_diff = np.concatenate(([acc_y_diff[0]], acc_y_diff))
                    acc_z_diff = np.diff(acc_z_aligned)
                    padded_acc_z_diff = np.concatenate(([acc_z_diff[0]], acc_z_diff))
                    gyro_x_diff = np.diff(gyro_x_aligned)
                    padded_gyro_x_diff = np.concatenate(([gyro_x_diff[0]], gyro_x_diff))
                    gyro_y_diff = np.diff(gyro_y_aligned)
                    padded_gyro_y_diff = np.concatenate(([gyro_y_diff[0]], gyro_y_diff))
                    gyro_z_diff = np.diff(gyro_z_aligned)
                    padded_gyro_z_diff = np.concatenate(([gyro_z_diff[0]], gyro_z_diff))

                    abs_padded_acc_x_diff = np.abs(padded_acc_x_diff)
                    abs_padded_acc_y_diff = np.abs(padded_acc_y_diff)
                    abs_padded_acc_z_diff = np.abs(padded_acc_z_diff)
                    abs_padded_gyro_x_diff = np.abs(padded_gyro_x_diff)
                    abs_padded_gyro_y_diff = np.abs(padded_gyro_y_diff)
                    abs_padded_gyro_z_diff = np.abs(padded_gyro_z_diff)

                    # # try low pass filter
                    cutoff_freq = 0.025

                    # Apply low-pass filter for each cutoff frequency
                    acc_x_diff_lowpass = butter_lowpass_filter(abs_padded_acc_x_diff, gnss_freq, cutoff_freq)
                    acc_y_diff_lowpass = butter_lowpass_filter(abs_padded_acc_y_diff, gnss_freq, cutoff_freq)
                    acc_z_diff_lowpass = butter_lowpass_filter(abs_padded_acc_z_diff, gnss_freq, cutoff_freq)
                    gyro_x_diff_lowpass = butter_lowpass_filter(abs_padded_gyro_x_diff, gnss_freq, cutoff_freq)
                    gyro_y_diff_lowpass = butter_lowpass_filter(abs_padded_gyro_y_diff, gnss_freq, cutoff_freq)
                    gyro_z_diff_lowpass = butter_lowpass_filter(abs_padded_gyro_z_diff, gnss_freq, cutoff_freq)


                    # Gyro threshold +- 0.001, Accel threshold +- 0.001
                    acc_x_stopped = threshold_based_window_averaging(acc_x_diff_lowpass, gnss_system_time, 1000, 0.002)
                    acc_y_stopped = threshold_based_window_averaging(acc_y_diff_lowpass, gnss_system_time, 1000, 0.002)
                    acc_z_stopped = threshold_based_window_averaging(acc_z_diff_lowpass, gnss_system_time, 1000, 0.002)
                    gyro_x_stopped = threshold_based_window_averaging(gyro_x_diff_lowpass, gnss_system_time, 1000, 0.001)
                    gyro_y_stopped = threshold_based_window_averaging(gyro_y_diff_lowpass, gnss_system_time, 1000, 0.001)
                    gyro_z_stopped = threshold_based_window_averaging(gyro_z_diff_lowpass, gnss_system_time, 1000, 0.001)

                    # acc_res = acc_x_stopped & acc_y_stopped & acc_z_stopped
                    acc_res = ((acc_x_stopped & acc_y_stopped) | (acc_y_stopped & acc_z_stopped) | (acc_x_stopped & acc_z_stopped)).astype(int)
                    # gyro_res = gyro_x_stopped & gyro_y_stopped & gyro_z_stopped
                    gyro_res = ((gyro_x_stopped & gyro_y_stopped) | (gyro_y_stopped & gyro_z_stopped) | (gyro_x_stopped & gyro_z_stopped)).astype(int)
                    # gyro_res = np.where(gyro_res == 0, 10, 0)
                    combined_gyro_accel = acc_res | gyro_res

                    for i in range(len(hdop)):
                        if hdop[i] < 3 and speed[i] > 2 and combined_gyro_accel[i] == 1:
                            combined_gyro_accel[i] = 0

                    combined_gyro_accel = np.where(combined_gyro_accel == 0, 10, 0)
                    # plot_signals_over_time(gnss_system_time, speed, acc_res, 'SPEED', 'ACCEL', f'Accel Stopped, Session:{session}')
                    plot_signals_over_time(gnss_system_time, speed, combined_gyro_accel, 'Speed', 'IMU Combined', f'IMU Stopped, Session:{session}')
                    # plot_signals_over_time(gnss_system_time, speed, hdop, 'Speed', 'HDOP', f'HDOP, Session:{session}')


                    # plot_sensor_data_classified(gnss_system_time, acc_x_diff_lowpass, acc_y_diff_lowpass, acc_z_diff_lowpass, acc_x_stopped, acc_y_stopped, acc_z_stopped, 'Accel', f'IMU Accelerometer Aligned, Session:{session}')
                    # plot_sensor_data_classified(gnss_system_time, gyro_x_diff_lowpass, gyro_y_diff_lowpass, gyro_z_diff_lowpass, gyro_x_stopped, gyro_y_stopped, gyro_z_stopped, 'Gyro', f'IMU Gyroscope Aligned, Session:{session}')


                    # low_speed_indexes = [index for index,val in enumerate(speed) if val < 2]
                    # poor_quality_indexes = [index for index,val in enumerate(hdop) if val > 5]

                    stationary_points = [index for index,val in enumerate(combined_gyro_accel) if val == 0]

                    # combined_indexes = list(set(low_speed_indexes).union(set(poor_quality_indexes)))
                    # plot_lat_lon_with_highlights(lats, lons, low_speed_indexes, f'Low Speed Points, Session:{session}')
                    create_map_with_highlighted_indexes(lats, lons, stationary_points, 'imu_stationary.html')
                    plt.show()

            except Exception as e:
                print(f"Error: {e}")