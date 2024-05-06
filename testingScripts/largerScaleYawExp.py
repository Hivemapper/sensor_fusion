import os
import sys
sys.path.insert(0, '/Users/rogerberman/sensor-fusion')  # Add the project root to the Python path
import numpy as np
from fusion import (
    SqliteInterface,
    extractAndSmoothImuData,
    extractAndSmoothMagData,
    extractGNSSData,
    calculateHeading, 
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
from plottingCode import plot_signal_over_time, plot_signals_over_time, create_map
import matplotlib.pyplot as plt


def process_db_and_visualize(dbpath: str):
    print(f"Loading data from {dbpath}")
    sql_db = SqliteInterface(dbpath)
    gnss_min_time, gnss_max_time = sql_db.get_min_max_system_time('gnss')
    imu_min_time, imu_max_time = sql_db.get_min_max_system_time('imu')
    mag_min_time, mag_max_time = sql_db.get_min_max_system_time('magnetometer')
    # convert all to epoch
    epoch_gnss_min_time = convertTimeToEpoch(gnss_min_time)
    epoch_gnss_max_time = convertTimeToEpoch(gnss_max_time)
    epoch_imu_min_time = convertTimeToEpoch(imu_min_time)
    epoch_imu_max_time = convertTimeToEpoch(imu_max_time)
    epoch_mag_min_time = convertTimeToEpoch(mag_min_time)
    epoch_mag_max_time = convertTimeToEpoch(mag_max_time)
    # get the max min times
    min_time = min(epoch_gnss_min_time, epoch_imu_min_time, epoch_mag_min_time)
    max_time = max(epoch_gnss_max_time, epoch_imu_max_time, epoch_mag_max_time)
    total_time = max_time - min_time 
    converted_total_time = convertEpochToTime(total_time)
    print(f"Total time: {converted_total_time}")

    current_time = max_time
    pastRange = current_time - min_time
    # print(f"Current time: {current_time}, Past range: {pastRange}")
    heading_diff_mean, fused_heading, clean_gnss_heading, gnss_time, heading_diff, number_of_points, mean_diff = getDashcamToVehicleHeadingOffset(
        sql_db, 
        current_time=current_time, 
        pastRange=pastRange, 
    )
    plot_signal_over_time(number_of_points, mean_diff, 'Clean Heading Diff Mean')
    plot_signal_over_time(gnss_time, heading_diff, 'Clean Heading Diff')
    plot_signals_over_time(gnss_time, clean_gnss_heading, fused_heading, 'Clean GNSS Heading', 'Fused Heading')
    plt.show()



if __name__ == "__main__":
    # Load data and filter data
    # dir_path = '/Users/rogerberman/dev_ground/CtpDbsDecoded'
    # drives = {}
    # for user in os.listdir(dir_path):
    #     drives[user] = []
    #     for drive in os.listdir(os.path.join(dir_path, user)):
    #         if '-shm' not in drive and '-wal' not in drive and '.db' in drive:
    #             file_path = os.path.join(dir_path, user, drive)
    #             if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
    #                 sql_db = SqliteInterface(file_path)
    #                 # check for existence of mag table
    #                 if sql_db.table_exists('magnetometer'):
    #                     # check to ensure data is written to mag table
    #                     if sql_db.get_min_max_system_time('magnetometer') != (None, None):
    #                         drives[user].append(file_path)


    # # for user in drives:
    #     for drive_path in drives[user]:
    #         try:
    #             process_db_and_visualize(drive_path)
    #         except Exception as e:
    #             print(f"Error: {e}")


    # get max min times for all tables
    # input this into yaw function, plot output

    # what am i giving it
    # current time
    # past range

    old_rive_dir = '/Users/rogerberman/dev_ground/YawFusionDrives'
    drive = 'drive3'
    data_logger_path = os.path.join(old_rive_dir, drive, 'data-logger.v1.4.4.db')
    process_db_and_visualize(data_logger_path)






