import time
import argparse
# import matplotlib.pyplot as plt

from processing import grab_most_recent_raw_data_session, process_raw_data
from sqliteInterface import (
    SqliteInterface,
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


def main(dbPath: str, debug: bool = False):
    ########### Setup service tables and grab starting indexes for raw data processing ###########
    db = SqliteInterface(dbPath)
    if not db.check_table_exists(TableName.SENSOR_FUSION_ERROR_LOG_TABLE.value):
        db.create_service_log_table()
    # Remove the processed table if it exists to start fresh for debugging
    if debug:
        print("Debugging Mode")
        db.drop_table(TableName.IMU_PROCESSED_TABLE.value)

    if not db.check_table_exists(TableName.FUSED_POSITION_TABLE.value):
        db.create_fused_position_table()

    # These indexes track what data has been processed
    rawIMUIndex = -1
    processedIMUIndex = -1
    # Setup Processed IMU Data Table
    if not db.check_table_exists(TableName.IMU_PROCESSED_TABLE.value):
        db.create_processed_imu_table()
        processedIMUIndex = db.find_starting_row_id(TableName.IMU_RAW_TABLE.value)
    else:
        processedIMUIndex = db.find_most_recent_row_id(
            TableName.IMU_PROCESSED_TABLE.value
        )

    ############################### Main Service Loop ###############################
    while 1:
        ########### Purge DB if required ###########
        # TODO: Modify to not be every loop, this can be done much less frequently
        db.purge()

        ########### Check for enough Data for Processing ###########
        ### Find where to start raw index
        rawIMUIndex = db.find_most_recent_row_id(TableName.IMU_RAW_TABLE.value)
        # Catch in case either index is None
        if rawIMUIndex == None:
            time.sleep(LOOP_SLEEP_TIME)
            continue

        if processedIMUIndex == None:
            processedIMUIndex = db.find_most_recent_row_id(
                TableName.IMU_PROCESSED_TABLE.value
            )
            continue

        if debug:
            print("Raw Table Index: ", rawIMUIndex)
            print("Processed Table Index: ", processedIMUIndex)

        index_window_size = rawIMUIndex - processedIMUIndex

        ########### Enough Data to Retrieve ###########
        if index_window_size >= MIN_DATA_POINTS:
            if debug:
                now = time.time()
            # Limit the number of data points to process at once
            furthestIMUIndex = processedIMUIndex + MIN_DATA_POINTS

            try:
                rawIMUData = db.get_raw_imu_by_row_range(
                    processedIMUIndex, furthestIMUIndex
                )
                ## If there is no data to process, skip the processing step, increment the processedIMUIndex, and continue
                if len(rawIMUData) == 0:
                    print("No data to process")
                    processedIMUIndex = furthestIMUIndex + 1
                    continue

                rawIMUData, next_index = grab_most_recent_raw_data_session(
                    rawIMUData,
                    processedIMUIndex,
                )
                # Using imu raw data, determine GNSS data to process
                imu_session = rawIMUData[0].session
                imu_chunk_start_time = rawIMUData[0].time
                imu_chunk_end_time = rawIMUData[-1].time
                gnss_start_index = db.get_nearest_row_id_to_time(
                    TableName.GNSS_TABLE.value, imu_chunk_start_time, imu_session
                )
                gnss_end_index = db.get_nearest_row_id_to_time(
                    TableName.GNSS_TABLE.value, imu_chunk_end_time, imu_session
                )
                gnssData = db.get_gnss_by_row_range(gnss_start_index, gnss_end_index)

            except Exception as e:
                db.service_log_msg("Retrieving IMU or GNSS Data", str(e))
                continue

            if debug:
                print(f"Processing {len(rawIMUData)} imu data points")

            ########### Section for processing Data ###########
            if debug:
                (
                    processedIMUData,
                    acc_x,
                    acc_y,
                    acc_z,
                    gyro_x,
                    gyro_y,
                    gyro_z,
                    imu_time,
                ) = process_raw_data(gnssData, rawIMUData, debug)
            else:
                try:
                    processedIMUData = process_raw_data(gnssData, rawIMUData)
                except Exception as e:
                    db.service_log_msg("Processing IMU Data", str(e))
                    continue
            ########### Section for inserting processed Data ###########
            try:
                db.insert_processed_imu_data(processedIMUData)
            except Exception as e:
                db.service_log_msg("Inserting Processed IMU Data", str(e))
                continue

            if debug:
                totalTime = time.time() - now
                print(
                    f"Processed {len(processedIMUData)} imu data points in {totalTime} seconds"
                )
                print(
                    f"Inserted {len(processedIMUData)} imu data points into the processed table"
                )
            ## Only set next index if all data was processed
            if next_index != -1:
                processedIMUIndex = int(next_index)

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

    print("Waiting for Sensor Fusion Service to start ...")
    # This is for letting system set up everything before starting the main loop
    time.sleep(5)
    print("Starting Sensor Fusion Service ...")
    main(db_path, False)
