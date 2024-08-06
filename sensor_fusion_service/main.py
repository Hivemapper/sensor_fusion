import os
import time
import argparse

env = os.getenv("HIVE_ENV")

if env == "local":
    from sensor_fusion.sensor_fusion_service.processing import (
        grab_most_recent_raw_data_session,
        process_raw_data,
        remove_duplicate_imu_data,
    )
    from sensor_fusion.sensor_fusion_service.sqlite_interface import (
        SqliteInterface,
        DATA_LOGGER_PATH,
    )
    from sensor_fusion.sensor_fusion_service.data_definitions import (
        ProcessedIMUData,
        FusedPositionData,
        IMUData,
        GNSSData,
        TableName,
        get_class_field_names,
    )
else:
    from processing import (
        grab_most_recent_raw_data_session,
        process_raw_data,
        remove_duplicate_imu_data,
    )
    from sqlite_interface import (
        SqliteInterface,
        DATA_LOGGER_PATH,
    )
    from data_definitions import (
        ProcessedIMUData,
        FusedPositionData,
        IMUData,
        GNSSData,
        TableName,
        get_class_field_names,
    )

IMU_SET_FREQUENCY = 100.0  # Hz
AMOUNT_OF_DATA_IN_SECONDS = 60.0  # seconds
MIN_DATA_POINTS = int(
    IMU_SET_FREQUENCY * AMOUNT_OF_DATA_IN_SECONDS
)  # X seconds worth of data at Y Hz
LOOP_SLEEP_TIME = (MIN_DATA_POINTS / IMU_SET_FREQUENCY) / 8.0  # seconds
print(f"Minimum Data Points: {MIN_DATA_POINTS}")
print(f"Loop Sleep Time: {LOOP_SLEEP_TIME}")


def table_and_initial_index_setup(db: SqliteInterface, debug: bool = False):
    # Remove the processed table if it exists to start fresh for debugging
    if debug:
        print("Debugging Mode")
        db.drop_table(TableName.IMU_PROCESSED_TABLE.value)
        db.drop_table(TableName.FUSED_POSITION_TABLE.value)
        db.drop_table(TableName.SENSOR_FUSION_LOG_TABLE.value)
        db.drop_table(TableName.GNSS_PROCESSED_TABLE.value)
    ########### Setup service tables, check columns, grab starting indexes for raw data processing ###########
    if not db.check_table_exists(TableName.ERROR_LOG_TABLE.value):
        print(f"Table {TableName.ERROR_LOG_TABLE.value} does not exist")
        db.create_error_logs_table()
        print("Table created")

    if not db.check_table_exists(TableName.SENSOR_FUSION_LOG_TABLE.value):
        print(f"Table {TableName.SENSOR_FUSION_LOG_TABLE.value} does not exist")
        db.create_service_log_table()
        print("Table created")

    if not db.check_table_exists(TableName.FUSED_POSITION_TABLE.value):
        print(f"Table {TableName.FUSED_POSITION_TABLE.value} does not exist")
        db.create_fused_position_table()
        print("Table created")
    else:
        # check if table has changed
        if not db.table_columns_match(
            TableName.FUSED_POSITION_TABLE.value,
            get_class_field_names(FusedPositionData),
        ):
            print("Fused Position Table columns do not match")
            db.drop_table(TableName.FUSED_POSITION_TABLE.value)
            db.create_fused_position_table()
            print("Table created")

    if not db.check_table_exists(TableName.GNSS_PROCESSED_TABLE.value):
        print(f"Table {TableName.GNSS_PROCESSED_TABLE.value} does not exist")
        db.create_processed_gnss_table()
        print("Table created")
    else:
        # check if table has changed
        if not db.table_columns_match(
            TableName.GNSS_PROCESSED_TABLE.value, get_class_field_names(GNSSData)
        ):
            print("GNSS Processed Table columns do not match")
            db.drop_table(TableName.GNSS_PROCESSED_TABLE.value)
            db.create_processed_gnss_table()
            print("Table created")

    # These indexes track what data has been processed
    processed_imu_index = -1
    # Setup Processed IMU Data Table
    if not db.check_table_exists(TableName.IMU_PROCESSED_TABLE.value):
        db.create_processed_imu_table()
        processed_imu_index = db.get_starting_row_id(TableName.IMU_RAW_TABLE.value)
        print("Does not exist, creating table")
        print("Starting Index: ", processed_imu_index)
    else:
        if not db.table_columns_match(
            TableName.IMU_PROCESSED_TABLE.value, get_class_field_names(ProcessedIMUData)
        ):
            print("Processed IMU Table columns do not match")
            db.drop_table(TableName.IMU_PROCESSED_TABLE.value)
            db.create_processed_imu_table()
            print("Table created")

        processed_imu_index = db.get_most_recent_row_id(
            TableName.IMU_PROCESSED_TABLE.value
        )
    return processed_imu_index


def main(db_path: str, debug: bool = False):
    db = SqliteInterface(db_path)
    # Setup tables and indexes for processing
    processed_imu_index = table_and_initial_index_setup(db, debug)
    raw_imu_index = -1
    ### Setup Variables needed between loops
    state_values = {
        "current_velocity": 0.0,
    }

    ############################### Main Service Loop ###############################
    if debug:
        loop_counter = 0

    loop_time = time.time()
    while 1:
        ########### Purge DB if required ###########
        # Check every minute if the db needs to be purged
        if loop_time + 60 < time.time():
            db.purge()
            loop_time = time.time()

        ########### Check for enough Data for Processing ###########
        ### Find where to start raw index
        raw_imu_index = db.get_most_recent_row_id(TableName.IMU_RAW_TABLE.value)
        # Catch for raw_imu table being empty for any reason, allow time for more values to be inserted
        if raw_imu_index == None or raw_imu_index == -1:
            time.sleep(LOOP_SLEEP_TIME)
            continue

        # Catch for imu_processed table being empty for any reason,
        # assume the first value in the raw table is the starting point
        if processed_imu_index == None:
            processed_imu_index = db.get_starting_row_id(TableName.IMU_RAW_TABLE.value)
            continue

        index_window_size = raw_imu_index - processed_imu_index

        ########### Enough Data to Retrieve ###########
        if index_window_size >= MIN_DATA_POINTS:
            if debug:
                now = time.time()
            # Limit the number of data points to process at once
            furthest_imu_index = processed_imu_index + MIN_DATA_POINTS

            try:
                raw_imu_data = db.get_data_by_row_range(
                    TableName.IMU_RAW_TABLE.value,
                    IMUData,
                    processed_imu_index,
                    furthest_imu_index,
                )
                ## If there is no data to process, skip the processing step, increment the processed_imu_index, and continue
                if len(raw_imu_data) == 0:
                    print("No data to process")
                    processed_imu_index = furthest_imu_index + 1
                    continue

                raw_imu_data, next_index = grab_most_recent_raw_data_session(
                    raw_imu_data,
                    processed_imu_index,
                    furthest_imu_index,
                )

                # IMU data is imperfect and can have duplicates in time, remove them
                raw_imu_data = remove_duplicate_imu_data(raw_imu_data)

                # Using imu raw data, determine GNSS data to process
                imu_session = raw_imu_data[0].session
                imu_chunk_start_time = raw_imu_data[0].time
                imu_chunk_end_time = raw_imu_data[-1].time

                gnss_start_index = db.get_nearest_row_id_to_time(
                    TableName.GNSS_RAW_TABLE.value, imu_chunk_start_time, imu_session
                )
                gnss_end_index = db.get_nearest_row_id_to_time(
                    TableName.GNSS_RAW_TABLE.value, imu_chunk_end_time, imu_session
                )
                gnss_data = db.get_data_by_row_range(
                    TableName.GNSS_RAW_TABLE.value,
                    GNSSData,
                    gnss_start_index,
                    gnss_end_index,
                )

            except Exception as e:
                db.service_log_msg("Retrieving IMU or GNSS Data", str(e))
                continue

            if debug:
                print(f"Processing {len(raw_imu_data)} imu data points")
                print(f"Processing {len(gnss_data)} gnss data points")

            ########### Section for processing Data ###########
            if debug:
                (
                    processed_imu_data,
                    fused_position_data,
                    acc_x,
                    acc_y,
                    acc_z,
                    gyro_x,
                    gyro_y,
                    gyro_z,
                    imu_time,
                ) = process_raw_data(gnss_data, raw_imu_data, state_values, debug)
            else:
                try:
                    processed_imu_data, fused_position_data = process_raw_data(
                        gnss_data, raw_imu_data, state_values
                    )
                except Exception as e:
                    db.service_log_msg("Processing IMU Data", str(e))
                    continue
            ########### Section for inserting processed Data ###########
            try:
                db.insert_data(TableName.IMU_PROCESSED_TABLE.value, processed_imu_data)
                db.insert_data(
                    TableName.FUSED_POSITION_TABLE.value, fused_position_data
                )
                db.insert_data(TableName.GNSS_PROCESSED_TABLE.value, gnss_data)
            except Exception as e:
                db.service_log_msg("Inserting Processed Data", str(e))
                continue

            if debug:
                totalTime = time.time() - now
                print(
                    f"Processed {len(processed_imu_data)} imu data points in {totalTime} seconds"
                )
                print(
                    f"Inserted {len(processed_imu_data)} imu data points into the processed table"
                )
            ## Only set next index if all data was processed
            if next_index != -1:
                processed_imu_index = int(next_index)

        if debug:
            print("Loop Counter: ", loop_counter)
            loop_counter += 1
        else:
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
    if debug_mode:
        print(f"Using {db_path} for local testing")
    main(db_path, debug_mode)
