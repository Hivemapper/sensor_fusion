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


def validate_dbs(dir_path):
    drives = {}
    successful_count = 0
    failed_files = []

    for user in os.listdir(dir_path):
        user_path = os.path.join(dir_path, user)
        if os.path.isdir(user_path):
            drives[user] = []
            for drive in os.listdir(user_path):
                if '-shm' in drive or '-wal' in drive or '.db' not in drive:
                    continue

                file_path = os.path.join(user_path, drive)
                if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
                    failed_files.append((file_path, "File does not exist or is empty"))
                    continue

                try:
                    sql_db = SqliteInterface(file_path)
                    if not sql_db.table_exists('magnetometer'):
                        failed_files.append((file_path, "Magnetometer table does not exist"))
                        continue
                    
                    if sql_db.get_min_max_system_time('magnetometer') == (None, None):
                        failed_files.append((file_path, "No data in magnetometer table"))
                        continue

                    # Attempt to pull all data from the database to check for malformed data
                    sql_db = SqliteInterface(file_path)
                    gnss_data = sql_db.queryAllGnss()
                    imu_data = sql_db.queryAllImu()
                    mag_data = sql_db.queryAllMagnetometer()
                    if len(gnss_data) == 0 or len(imu_data) == 0 or len(mag_data) == 0:
                        failed_files.append((file_path, "No data in one of the tables"))
                        continue

                    # Check for sessions
                    gnss_sessions = set([d.session for d in gnss_data])
                    imu_sessions = set([d.session for d in imu_data])
                    mag_sessions = set([d.session for d in mag_data])
                    if len(gnss_sessions) == 0 or len(imu_sessions) == 0 or len(mag_sessions) == 0:
                        failed_files.append((file_path, "No sessions in one of the tables"))
                        continue
                    # guard against empty sessions
                    if len(gnss_sessions) == 1 and gnss_sessions[0] == '':
                        failed_files.append((file_path, "Empty session in GNSS table"))
                        continue
                    
                    drives[user].append(file_path)
                    successful_count += 1
                except Exception as e:
                    failed_files.append((file_path, str(e)))

    failed_count = len(failed_files)

    print(f"Successful files: {successful_count}")
    print(f"Failed files: {failed_count}")
    # for file, reason in failed_files:
    #     print(f"File: {file}, Reason: {reason}")

    return drives


def process_db_file_for_individual_drives(filename):
    print(f"********* Loading data from {filename}")
    sql_db = SqliteInterface(filename)
    gnss_data = sql_db.queryAllGnss()
    imu_data = sql_db.queryAllImu()
    mag_data = sql_db.queryAllMagnetometer()
    print(f"GNSS data: {len(gnss_data)}, IMU data: {len(imu_data)}, Mag data: {len(mag_data)}")
    # get unique session ids for all three
    gnss_sessions = set([d.session for d in gnss_data])
    imu_sessions = set([d.session for d in imu_data])
    mag_sessions = set([d.session for d in mag_data])
    # Filter out empty sessions
    if '' in gnss_sessions:
        gnss_sessions.remove('')
    if '' in imu_sessions:
        imu_sessions.remove('')
    if '' in mag_sessions:
        mag_sessions.remove('')
    # only look at data where session exists in all three
    common_sessions = gnss_sessions.intersection(imu_sessions).intersection(mag_sessions)
    print(f"Common sessions: {common_sessions}")
    print(f"GNSS sessions: {gnss_sessions}")
    print(f"IMU sessions: {imu_sessions}")
    print(f"Mag sessions: {mag_sessions}")
    # split out data into each individual common session
    for session in common_sessions:
        gnss_data_session = [d for d in gnss_data if d.session == session]
        imu_data_session = [d for d in imu_data if d.session == session]
        mag_data_session = [d for d in mag_data if d.session == session]
        print(f"Session: {session}, GNSS data: {len(gnss_data_session)}, IMU data: {len(imu_data_session)}, Mag data: {len(mag_data_session)}")
        # check if there is enough data to process
        DATA_MINIMUM = 100
        if len(gnss_data_session) < DATA_MINIMUM or len(imu_data_session) < DATA_MINIMUM or len(mag_data_session) < DATA_MINIMUM:
            print(f"Not enough data to process for session {session}")
            continue


