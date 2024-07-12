import time
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(
    0, "/Users/rogerberman/hivemapper/sensor-fusion"
)  # Add the project root to the Python path
from fusion import SqliteInterface, aggregate_data, convertTimeToEpoch
from plottingCode import (
    plot_signal_over_time,
    plot_signals_over_time,
    create_map_with_highlighted_indexes,
    plot_rate_counts,
    plot_sensor_data,
    plot_sensor_data_classified,
    plot_lat_lon_with_highlights,
)
from processDBs import validate_db_file, process_db_file_for_individual_drives


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


def check_table_copy(raw_data, processed_data):
    for i in range(len(processed_data)):
        if (
            raw_data[i]["time"] != processed_data[i]["time"]
            and raw_data[i]["id"] != processed_data[i]["row_id"]
        ):
            print("Data does not match")
            return False
    return True


if __name__ == "__main__":
    # Load data and filter data
    # file_path = "/Users/rogerberman/Desktop/telemetryProcessing/testDBs/2024-06-14T19:04:56.000Z.db"
    file_path = "/Users/rogerberman/Desktop/telemetryProcessing/testDBs/2024-06-13T12:57:34.000Z.db"
    time_now = time.time()
    _, camera_type = validate_db_file(file_path)
    print(f"Validation Done in {time.time()-time_now} seconds")
    try:
        db_interface = SqliteInterface(file_path)
        raw_imu = db_interface.get_all_rows_as_dicts("imu")
        processed_imu = db_interface.get_all_rows_as_dicts("imu_processed")
        print(f"Copied Table correctly: {check_table_copy(raw_imu, processed_imu)}")

        useable_sessions = process_db_file_for_individual_drives(file_path, "hdc")
        for session in useable_sessions:
            raw_imu = aggregate_data(useable_sessions[session]["imu_data"])
            processed_imu = aggregate_data(
                useable_sessions[session]["imu_processed_data"]
            )
            gnss_data = aggregate_data(useable_sessions[session]["gnss_data"])
            print(f"Session: {session}")

            print(f"Raw IMU: {len(raw_imu['time'])}")
            print(f"Processed IMU: {len(processed_imu['time'])}")
            print(f"GNSS: {len(gnss_data['system_time'])}")
            # Convert time to epoch for all values
            for i in range(len(gnss_data["system_time"])):
                gnss_data["system_time"][i] = convertTimeToEpoch(
                    gnss_data["system_time"][i]
                )
            for i in range(len(raw_imu["time"])):
                raw_imu["time"][i] = convertTimeToEpoch(raw_imu["time"][i])

            for i in range(len(processed_imu["time"])):
                processed_imu["time"][i] = convertTimeToEpoch(processed_imu["time"][i])

            print("Downsampling data")
            # downsample stationary to compare with GNSS
            stationary_down = np.interp(
                gnss_data["system_time"],
                processed_imu["time"],
                processed_imu["stationary"],
            )

            plot_signals_over_time(
                gnss_data["system_time"],
                gnss_data["speed"],
                stationary_down,
                downsample_factor=1,
                signal1_label="Speed",
                signal2_label="Stationary",
            )

            plot_sensor_data(
                processed_imu["time"],
                processed_imu["ax"],
                processed_imu["ay"],
                processed_imu["az"],
                sensor_name="Processed ACCEL",
                title="Processed ACCEL Data",
                downsample_factor=10,
            )
            plot_sensor_data(
                processed_imu["time"],
                processed_imu["gx"],
                processed_imu["gy"],
                processed_imu["gz"],
                sensor_name="Processed GYRO",
                title="Processed GYRO Data",
                downsample_factor=10,
            )

            plt.show()

    except Exception as e:
        print(f"Error: {e}")
