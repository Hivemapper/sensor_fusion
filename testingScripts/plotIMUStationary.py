import sys
import json
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


if __name__ == "__main__":
    # read in json data
    json_file_path = "stationary_detection_result.json"
    with open(json_file_path, "r") as json_file:
        all_data = json.load(json_file)
    # Deconstructing the dictionary
    stationary = all_data["stationary"]
    gnssTime = all_data["gnssTime"]
    gnssSpeed = all_data["gnssSpeed"]
    acc_x = all_data["acc_x"]
    acc_y = all_data["acc_y"]
    acc_z = all_data["acc_z"]
    gyro_x = all_data["gyro_x"]
    gyro_y = all_data["gyro_y"]
    gyro_z = all_data["gyro_z"]
    acc_x_diff_abs = all_data["acc_x_diff_abs"]
    acc_y_diff_abs = all_data["acc_y_diff_abs"]
    acc_z_diff_abs = all_data["acc_z_diff_abs"]
    gyro_x_diff_abs = all_data["gyro_x_diff_abs"]
    gyro_y_diff_abs = all_data["gyro_y_diff_abs"]
    gyro_z_diff_abs = all_data["gyro_z_diff_abs"]


    plot_sensor_data(gnssTime, acc_x, acc_y, acc_z, "Stationary Accel Data")
    plot_sensor_data(gnssTime, gyro_x, gyro_y, gyro_z, "Stationary Gyro Data")
    plot_sensor_data(gnssTime, acc_x_diff_abs, acc_y_diff_abs, acc_z_diff_abs, "Stationary Accel Diff Abs Data")
    plot_sensor_data(gnssTime, gyro_x_diff_abs, gyro_y_diff_abs, gyro_z_diff_abs, "Stationary Gyro Diff Abs Data")
    plot_signals_over_time(gnssTime, gnssSpeed, stationary, "Stationary Speed Data")
    plt.show()
