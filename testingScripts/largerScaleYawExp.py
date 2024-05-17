import os
import sys
from processDBs import validate_dbs, process_db_file_for_individual_drives
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


if __name__ == "__main__":
    # Load data and filter data
    dir_path = '/Users/rogerberman/dev_ground/CTP-decoded-05-14-2024-recovered'
    # dir_path = '/Users/rogerberman/dev_ground/CTP-decoded-05-14-2024'
    # dir_path = '/Users/rogerberman/dev_ground/CtpDbsDecoded'
    drives = validate_dbs(dir_path)
    print("Validation Done")


    for user in drives:
        for drive_path in drives[user]:
            usable_drives = process_db_file_for_individual_drives(drive_path)
            try:
                for session in usable_drives:
                    getDashcamToVehicleHeadingOffset(usable_drives[session], session)
            except Exception as e:
                print(f"Error: {e}")



    # malformed_count = 0
    # total_count = 0
    # for user in drives:
    #     for drive_path in drives[user]:
    #         try:
    #             total_count += 1
    #             # if 'exotic-purple-mapmaker/2024-05-14T13:31:35.000Z.db' in drive_path:
    #             process_db_and_visualize(drive_path)
    #         except Exception as e:
    #             print(f"Error: {e}")
    #             if 'malformed' in str(e):
    #                 malformed_count += 1
    #             exit(1)
    # print(f"Malformed count: {malformed_count}, total count: {total_count}")


    ### Old Drives

    # old_rive_dir = '/Users/rogerberman/dev_ground/YawFusionDrives'
    # drives = ['drive1', 'drive2', 'drive3', 'drive4', 'lafayette' ,'highway_sf']
    # for drive in drives:
    #     data_logger_path = os.path.join(old_rive_dir, drive, 'data-logger.v1.4.4.db')
    #     process_db_and_visualize(data_logger_path)






