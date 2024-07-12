import os
from multiprocessing import Process, Queue

from sensor_fusion.fusion import (
    SqliteInterface,
)

SESSION_DATA_MINIMUM = 500  # number of points


def validate_db_file(file_path):
    if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
        print(file_path, "File does not exist or is empty")
        return None, None

    try:
        # Assume hdcs and check for magnetometer table
        camera_type = "hdc"
        sql_db = SqliteInterface(file_path)
        # Check for HDCS
        if sql_db.table_exists("magnetometer"):
            mag_data = sql_db.queryAllMagnetometer()
            if len(mag_data) > 0:
                camera_type = "hdcs"

        # Attempt to pull all data from the database to check for malformed data
        gnss_data = sql_db.queryAllGnss()
        imu_data = sql_db.queryAllImu()
        if len(gnss_data) == 0 or len(imu_data) == 0:
            print(file_path, "No data in one of the tables")
            return None, None

        # Check for sessions
        gnss_sessions = set([d.session for d in gnss_data])
        imu_sessions = set([d.session for d in imu_data])
        if camera_type == "hdcs":
            mag_sessions = set([d.session for d in mag_data])
        if len(gnss_sessions) == 0 or len(imu_sessions) == 0:
            print(file_path, "No sessions in one of the tables")
            return None, None
        if camera_type == "hdcs" and len(mag_sessions) == 0:
            print(file_path, "No sessions in magnetometer table")
            return None, None
        # guard against empty sessions
        if len(gnss_sessions) == 1 and "" in gnss_sessions:
            print(file_path, "Empty session in GNSS table")
            return None, None

        return file_path, camera_type
    except Exception as e:
        print(file_path, str(e))
        return None, None


def validate_user_dir(user_path, drives_queue, failed_files_queue):
    drives = []
    failed_files = []

    for drive in os.listdir(user_path):
        if "-shm" in drive or "-wal" in drive or ".db" not in drive:
            continue

        file_path = os.path.join(user_path, drive)
        result, camera_type = validate_db_file(file_path)
        if result is not None:
            drives.append((result, camera_type))
        else:
            failed_files.append((file_path, "Validation failed"))

    # Put results into queues
    drives_queue.put(drives)
    failed_files_queue.put(failed_files)


def validate_dbs(dir_path):
    drives = {}
    successful_count = 0
    failed_files = []

    # Create queues for storing results
    drives_queue = Queue()
    failed_files_queue = Queue()

    # Create processes for each user directory
    processes = []
    for user in os.listdir(dir_path):
        user_path = os.path.join(dir_path, user)
        if os.path.isdir(user_path):
            process = Process(
                target=validate_user_dir,
                args=(user_path, drives_queue, failed_files_queue),
            )
            process.start()
            processes.append(process)

    # Wait for all processes to finish
    for process in processes:
        process.join()

    # Get results from queues
    while not drives_queue.empty():
        user_drives = drives_queue.get()
        user_failed_files = failed_files_queue.get()
        if user_drives:
            user = os.path.basename(user_drives[0][0])
            drives[user] = user_drives
            failed_files.extend(user_failed_files)
            successful_count += len(user_drives)

    failed_count = len(failed_files)

    print(f"Successful files: {successful_count}")
    print(f"Failed files: {failed_count}")
    for file, reason in failed_files:
        print(f"File: {file}, Reason: {reason}")

    return drives


# Not written in the most effiecient way, but in the most clear way
def process_db_file_for_individual_drives(filename, camera_type):
    print(f"--- Loading data from {filename} ---")
    print(f"Camera type: {camera_type}")
    sql_db = SqliteInterface(filename)
    fusion_service = False
    if camera_type == "hdcs":
        gnss_data = sql_db.queryAllGnss()
        imu_data = sql_db.queryAllImu()
        mag_data = sql_db.queryAllMagnetometer()
        if sql_db.table_exists("imu_processed"):
            fusion_service = True
            imu_processed_data = sql_db.queryAllProcessedImu()
            if len(imu_processed_data) == 0:
                print("No processed IMU data found")
        # get unique session ids for all three
        gnss_sessions = set([d.session for d in gnss_data])
        imu_sessions = set([d.session for d in imu_data])
        mag_sessions = set([d.session for d in mag_data])
        # Filter out empty sessions
        if "" in gnss_sessions:
            gnss_sessions.remove("")
        if "" in imu_sessions:
            imu_sessions.remove("")
        if "" in mag_sessions:
            mag_sessions.remove("")
        # only look at data where session exists in all three
        common_sessions = gnss_sessions.intersection(imu_sessions).intersection(
            mag_sessions
        )
        if len(common_sessions) == 0:
            print(f"No common sessions found for {filename}")
            return {}
        useable_sessions = {}
        # split out data into each individual common session
        for session in common_sessions:
            gnss_data_session = [d for d in gnss_data if d.session == session]
            imu_data_session = [d for d in imu_data if d.session == session]
            mag_data_session = [d for d in mag_data if d.session == session]
            if fusion_service:
                imu_processed_data_session = [
                    d for d in imu_processed_data if d.session == session
                ]
            # ensure enough data to be useful
            if (
                len(gnss_data_session) < SESSION_DATA_MINIMUM
                or len(imu_data_session) < SESSION_DATA_MINIMUM
                or len(mag_data_session) < SESSION_DATA_MINIMUM
            ):
                print(f"Not enough data to process for session {session}")
                continue

            if fusion_service:
                print(
                    f"  Session: {session}, gnss: {len(gnss_data_session)}, raw_imu: {len(imu_data_session)}, processed_imu: {len(imu_processed_data_session)}, mag: {len(mag_data_session)}"
                )
                useable_sessions[session] = {
                    "gnss_data": gnss_data_session,
                    "imu_data": imu_data_session,
                    "imu_processed_data": imu_processed_data_session,
                    "mag_data": mag_data_session,
                }
            else:
                print(
                    f"  Session: {session}, gnss: {len(gnss_data_session)}, raw_imu: {len(imu_data_session)}, mag: {len(mag_data_session)}"
                )
                useable_sessions[session] = {
                    "gnss_data": gnss_data_session,
                    "imu_data": imu_data_session,
                    "mag_data": mag_data_session,
                }
    # HDC route
    else:
        gnss_data = sql_db.queryAllGnss()
        imu_data = sql_db.queryAllImu()
        if sql_db.table_exists("imu_processed"):
            imu_processed_data = sql_db.queryAllProcessedImu()
            if len(imu_processed_data) == 0:
                print("No processed IMU data found")
        # get unique session ids for all three
        gnss_sessions = set([d.session for d in gnss_data])
        imu_sessions = set([d.session for d in imu_data])
        # Filter out empty sessions
        if "" in gnss_sessions:
            gnss_sessions.remove("")
        if "" in imu_sessions:
            imu_sessions.remove("")
        # only look at data where session exists in all three
        common_sessions = gnss_sessions.intersection(imu_sessions)
        if len(common_sessions) == 0:
            print(f"No common sessions found for {filename}")
            return {}
        useable_sessions = {}
        # split out data into each individual common session
        for session in common_sessions:
            gnss_data_session = [d for d in gnss_data if d.session == session]
            imu_data_session = [d for d in imu_data if d.session == session]
            if sql_db.table_exists("imu_processed"):
                imu_processed_data_session = [
                    d for d in imu_processed_data if d.session == session
                ]
            # ensure enough data to be useful
            if (
                len(gnss_data_session) < SESSION_DATA_MINIMUM
                or len(imu_data_session) < SESSION_DATA_MINIMUM
            ):
                print(f"Not enough data to process for session {session}")
                continue
            print(
                f"Session: {session}, gnss data: {len(gnss_data_session)}, IMU data: {len(imu_data_session)}"
            )
            useable_sessions[session] = {
                "gnss_data": gnss_data_session,
                "imu_data": imu_data_session,
                "imu_processed_data": imu_processed_data_session,
            }

    print(f"--- Finished loading data from {filename} ---")
    return useable_sessions


### Helper Functions
def aggregate_data(data_list):
    """
    Aggregates data from a list of IMUData or ProcessedIMUData objects into a dictionary.
    Args:
        data_list (list): A list of IMUData or ProcessedIMUData objects.
    Returns:
        dict: A dictionary with keys as attribute names and values as lists of attribute values.
    """
    if not data_list:
        return {}

    result = {}
    first_item = data_list[0]

    # Initialize dictionary keys with empty lists
    for key in first_item.__dict__.keys():
        result[key] = []

    # Populate the dictionary with values from each object
    for item in data_list:
        for key, value in item.__dict__.items():
            result[key].append(value)

    return result


def transform_list_of_dicts(list_of_dicts):
    """
    Transforms a list of dictionaries into a dictionary of lists, where each key from the dictionaries
    maps to a list of values from the original dictionaries.

    Parameters:
    list_of_dicts (list): A list of dictionaries with the same keys.

    Returns:
    dict: A dictionary where each key maps to a list of values from the original dictionaries.
    """
    if not list_of_dicts:
        return {}

    # Extract all keys from the dictionaries
    keys = list_of_dicts[0].keys()

    # Initialize a dictionary where each key maps to an empty list
    transformed_dict = {key: [] for key in keys}

    # Iterate through the list of dictionaries
    for dictionary in list_of_dicts:
        for key in dictionary:
            transformed_dict[key].append(dictionary[key])

    return transformed_dict
