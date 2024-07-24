import sys
import os
from offline_code.utils.process_validate_dbs import (
    validate_db_file,
    process_db_file_for_individual_drives,
)

sys.path.insert(0, "/data/sensorExp")  # Add the project root to the Python path
from fusion_old import (
    extractAndSmoothImuData,
    extractAndSmoothMagData,
    extractGNSSData,
    calculate_rates_and_counts,
)
from offline_code.utils.plotting_code import (
    plot_signal_over_time,
    plot_signals_over_time,
    plot_rate_counts,
    plot_sensor_data,
    plot_sensor_data_classified,
    plot_lat_lon_with_highlights,
    plot_time,
    plot_periods_over_time,
)
import matplotlib.pyplot as plt
import time


if __name__ == "__main__":
    # check and load data
    dataBPath = "/Users/rogerberman/Desktop/data-logger-test-data-decoded-recovered"
    user = "hospitable-cardinal-shark"
    file = "2024-06-05T23:15:11.000Z.db"
    complete_path = os.path.join(dataBPath, user, file)
    time_now = time.time()
    validatedPath, camera_type = validate_db_file(complete_path)
    processedData = process_db_file_for_individual_drives(validatedPath, camera_type)
    print(f"Validation & Proccessing done in {time.time()-time_now} seconds")
    for session in processedData:
        print(f"Processing Session: {session}")
        drive_data = processedData[session]
        gnss_data = drive_data["gnss_data"]
        imu_data = drive_data["imu_data"]
        acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, imu_time, imu_freq = (
            extractAndSmoothImuData(imu_data)
        )
        (
            lats,
            lons,
            alts,
            speed,
            heading,
            headingAccuracy,
            hdop,
            gdop,
            gnss_system_time,
            gnss_real_time,
            time_resolved,
            gnss_freq,
        ) = extractGNSSData(gnss_data)
        print(f"GNSS Start Time: {gnss_system_time[0]}, IMU Start Time: {imu_time[0]}")

        gnss_rates, gnss_rates_list = calculate_rates_and_counts(gnss_system_time)
        imu_rates, imu_rates_list = calculate_rates_and_counts(imu_time)
        # print(len(imu_rates_list), len(imu_time))

        plot_periods_over_time(imu_time[1:], imu_rates_list, "IMU Periods")

        # plot_rate_counts(gnss_rates, f"GNSS Freq Rates, Session:{session}")
        # plot_rate_counts(imu_rates, f"IMU Freq Rates, Session:{session}")
        # plot_signal_over_time(imu_time, acc_x, "IMU Acc X")
        # plot_time(imu_time)
        plt.show()
