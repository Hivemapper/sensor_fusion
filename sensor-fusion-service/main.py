import time
import argparse
from datetime import datetime
from typing import List
# import matplotlib.pyplot as plt

from conversions import lists_to_dicts
from telemetryMath import extractAndSmoothImuData, extractGNSSData, calculateStationary
from sqliteInterface import (
    SqliteInterface,
    IMUData,
    TableName,
    DATA_LOGGER_PATH,
)
# from plottingCode import plot_signals_over_time, plot_sensor_data, plot_signal_over_time

IMU_SET_FREQUENCY = 100.0  # Hz
AMOUNT_OF_DATA_IN_SECONDS = 60.0  # seconds
MIN_DATA_POINTS = (
    IMU_SET_FREQUENCY * AMOUNT_OF_DATA_IN_SECONDS
)  # X seconds worth of data at Y Hz
LOOP_SLEEP_TIME = (MIN_DATA_POINTS / IMU_SET_FREQUENCY) / 8.0  # seconds
print(f"Loop Sleep Time: {LOOP_SLEEP_TIME}")


def processIMUData(data: List[IMUData], debug: bool = False):
    """
    This function processes the raw IMU data and returns the processed data.
    Parameters:
    - data: A list of dictionaries, where each dictionary represents a row of raw IMU data.
    Returns:
    - list: A list of dictionaries, where each dictionary represents a row of processed IMU data.
    """
    (
        acc_x,
        acc_y,
        acc_z,
        gyro_x,
        gyro_y,
        gyro_z,
        time,
        temperature,
        session,
        row_id,
        imu_freq,
        imu_converted_time,
    ) = extractAndSmoothImuData(data)
    stationary = calculateStationary(
        acc_x,
        acc_y,
        acc_z,
        gyro_x,
        gyro_y,
        gyro_z,
        imu_freq,
        imu_converted_time,
        debug,
    )

    keys = [
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "stationary",
        "time",
        "temperature",
        "session",
        "row_id",
    ]
    processedData = lists_to_dicts(
        keys,
        acc_x,
        acc_y,
        acc_z,
        gyro_x,
        gyro_y,
        gyro_z,
        stationary,
        time,
        temperature,
        session,
        row_id,
    )
    if debug:
        return (
            processedData,
            acc_x,
            acc_y,
            acc_z,
            gyro_x,
            gyro_y,
            gyro_z,
            imu_converted_time,
        )

    return processedData


def main(dbPath: str, debug: bool = False):
    # Setup
    db = SqliteInterface(dbPath)
    # Setup Error Logging Table
    if not db.check_table_exists(TableName.SENSOR_FUSION_ERROR_LOG_TABLE.value):
        db.create_service_log_table()
    # Remove the processed table if it exists to start fresh for debugging
    if debug:
        print("Debugging Mode")
        db.drop_table(TableName.IMU_PROCESSED_TABLE.value)

    # These indexes are used to track diff between raw and processed tables
    # They will track row ids from the raw table
    rawCurTableIndex = -1
    processedCurTableIndex = -1
    # Setup Processed Data Table
    if not db.check_table_exists(TableName.IMU_PROCESSED_TABLE.value):
        db.create_processed_imu_table()
        processedCurTableIndex = db.find_starting_row_id(TableName.IMU_RAW_TABLE.value)
    else:
        processedCurTableIndex = db.find_most_recent_row_id(
            TableName.IMU_PROCESSED_TABLE.value
        )

    # infinite loop to process data as it comes in
    while 1:
        # Check for need to purge DB every loop
        # TODO: Modify to not be every loop, this can be done much less frequently
        db.purge()

        ### Find where to start rwo index
        rawCurTableIndex = db.find_most_recent_row_id(TableName.IMU_RAW_TABLE.value)
        # Catch in case either index is not
        if rawCurTableIndex == None:
            time.sleep(LOOP_SLEEP_TIME)
            continue

        if processedCurTableIndex == None:
            processedCurTableIndex = db.find_most_recent_row_id(
                TableName.IMU_PROCESSED_TABLE.value
            )
            continue

        if debug:
            print("Raw Table Index: ", rawCurTableIndex)
            print("Processed Table Index: ", processedCurTableIndex)

        index_window_size = rawCurTableIndex - processedCurTableIndex

        if index_window_size >= MIN_DATA_POINTS:
            now = time.time()
            # Limit the number of data points to process at once
            furthest_index = processedCurTableIndex + MIN_DATA_POINTS

            try:
                rawData = db.get_raw_imu_by_row_range(
                    processedCurTableIndex, furthest_index
                )
                ## If there is no data to process, skip the processing step, increment the processedCurTableIndex, and continue
                if len(rawData) == 0:
                    print("No data to process")
                    processedCurTableIndex = furthest_index + 1
                    continue
            except Exception as e:
                db.service_log_msg("Retrieving IMU Data", str(e))
                continue

            if debug:
                print(f"Processing {len(rawData)} data points")

            next_index = -1
            # Handle pulling singluar drive sessions here
            starting_session = rawData[0].session
            ending_session = rawData[-1].session
            if starting_session != ending_session:
                # iterate backwards through the data to find the start of the new session
                for i in range(len(rawData) - 1, -1, -1):
                    if rawData[i].session == starting_session:
                        # reduce rawData to only the new session
                        rawData = rawData[: i + 1]
                        print(f"Reduced data to {len(rawData)} data points")
                        next_index = processedCurTableIndex + i + 1
                        break
            else:
                next_index = furthest_index + 1

            ### Processing Section
            if debug:
                processedData, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, imu_time = (
                    processIMUData(rawData, debug)
                )
            else:
                try:
                    processedData = processIMUData(rawData)
                except Exception as e:
                    db.service_log_msg("Processing IMU Data", str(e))
                    continue
            ### End Processing Section
            try:
                db.insert_processed_imu_data(processedData)
            except Exception as e:
                db.service_log_msg("Inserting Processed IMU Data", str(e))
                continue

            totalTime = time.time() - now
            if debug:
                print(
                    f"Processed {len(processedData)} data points in {totalTime} seconds"
                )
                print(
                    f"Inserted {len(processedData)} data points into the processed table"
                )
            ## Only set next index if all data was processed
            if next_index != -1:
                processedCurTableIndex = int(next_index)

            ## Print out data for evaluation
            # if debug:
            #     print("Gathering GNSS Data for Analysis")
            #     print("Starting Session: ", starting_session)
            #     gnss_start_index = db.get_nearest_row_id_to_time(
            #         "gnss", processedData[0]["time"], starting_session
            #     )
            #     gnss_end_index = db.get_nearest_row_id_to_time(
            #         "gnss", processedData[-1]["time"], starting_session
            #     )
            #     print(
            #         "GNSS Start Index: ",
            #         gnss_start_index,
            #         "GNSS End Index: ",
            #         gnss_end_index,
            #     )
            #     gnssData = db.get_gnss_by_row_range(gnss_start_index, gnss_end_index)
            #     (
            #         lat,
            #         lon,
            #         alt,
            #         speed,
            #         heading,
            #         headingAccuracy,
            #         hdop,
            #         gdop,
            #         gnss_system_time,
            #         gnss_real_time,
            #         time_resolved,
            #         gnss_freq,
            #     ) = extractGNSSData(gnssData)
            # plot_signal_over_time(gnss_system_time, speed, "Speed")
            # plot_sensor_data(
            #     imu_time,
            #     acc_x,
            #     acc_y,
            #     acc_z,
            #     "Accelerometer",
            # )
            # plot_sensor_data(
            #     imu_time,
            #     gyro_x,
            #     gyro_y,
            #     gyro_z,
            #     "Gyroscope",
            # )
            # plt.show()

        time.sleep(LOOP_SLEEP_TIME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optional arguments for the Python Data Layer, for testing purposes."
    )
    parser.add_argument(
        "-db", "--dbPath", type=str, help="Optional db file for local testing"
    )
    # Parse the arguments
    args = parser.parse_args()

    # Check if dbFile is provided, otherwise use a default value
    db_path = args.dbPath if args.dbPath is not None else DATA_LOGGER_PATH
    debug_mode = True if args.dbPath is not None else False

    print("Starting IMU processing")
    # This is for letting system set up everything before starting the main loop
    time.sleep(5)
    print("Starting Python Data Layer Processing ...")
    main(db_path, False)
