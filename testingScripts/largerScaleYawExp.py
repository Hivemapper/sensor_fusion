import os
import sys
from processDBs import validate_dbs, process_db_file_for_individual_drives

sys.path.insert(
    0, "/Users/rogerberman/hivemapper/sensor-fusion"
)  # Add the project root to the Python path
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
from testingScripts.plottingCode import (
    plot_signal_over_time,
    plot_signals_over_time,
    plot_rate_counts,
    create_map_with_headings,
)
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Load data and filter data
    dir_path = "/Users/rogerberman/Desktop/data-logger-test-data-decoded-recovered"
    # dir_path = '/Users/rogerberman/dev_ground/50_Hz_data_trial-decoded'
    # dir_path = '/Users/rogerberman/dev_ground/CTP-decoded-05-14-2024-recovered'
    # dir_path = '/Users/rogerberman/dev_ground/CTP-decoded-05-14-2024'
    # dir_path = '/Users/rogerberman/dev_ground/CtpDbsDecoded'
    drives = validate_dbs(dir_path)
    print("Validation Done")

    for user in drives:
        for drive in drives[user]:
            db_path = drive[0]
            camera_type = drive[1]
            usable_drives = process_db_file_for_individual_drives(db_path, camera_type)
            try:
                for session in usable_drives:
                    getDashcamToVehicleHeadingOffset(usable_drives[session], session)
            except Exception as e:
                print(f"Error: {e}")

    ### Old Drives

    # old_rive_dir = '/Users/rogerberman/dev_ground/YawFusionDrives'
    # drives = ['drive1', 'drive2', 'drive3', 'drive4', 'lafayette' ,'highway_sf']
    # for drive in drives:
    #     data_logger_path = os.path.join(old_rive_dir, drive, 'data-logger.v1.4.4.db')
    #     process_db_and_visualize(data_logger_path)
