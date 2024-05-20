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
    create_map, 
    plot_rate_counts,
    plot_sensor_data,
)
import matplotlib.pyplot as plt
import time

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
            usable_drives = process_db_file_for_individual_drives(db_file_path, camera_type)
            try:
                for session in usable_drives:
                    drive_data = usable_drives[session]
                    gnss_data = drive_data['gnss_data']
                    imu_data = drive_data['imu_data']
                    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, imu_time, imu_freq = extractAndSmoothImuData(imu_data)
                    lats, lons, alts, speed, heading, headingAccuracy, hdop, gdop, gnss_system_time, gnss_real_time, time_resolved, gnss_freq = extractGNSSData(gnss_data)

                    # gnss_rates = calculate_rates_and_counts(gnss_system_time)
                    # imu_rates = calculate_rates_and_counts(imu_time)
                    # plot_rate_counts(gnss_rates, f'GNSS Freq Rates, Session:{session}')
                    # plot_rate_counts(imu_rates, f'IMU Freq Rates, Session:{session}')

                    plot_sensor_data(imu_time, acc_x, acc_y, acc_z, 'Accel', f'IMU Accelerometer, Session:{session}')
                    plot_sensor_data(imu_time, gyro_x, gyro_y, gyro_z, 'Gyro', f'IMU Gyroscope, Session:{session}')
                    plot_sensor_data(gnss_system_time, speed, headingAccuracy, hdop, 'GNSS', f'GNSS Data, Session:{session}')

                    # plot_signal_over_time(list(range(len(gnss_system_time))), gnss_system_time, 'GNSS Time')
                    # plot_signal_over_time(list(range(len(imu_time))), imu_time, 'IMU Time')
                    plt.show()

            except Exception as e:
                print(f"Error: {e}")