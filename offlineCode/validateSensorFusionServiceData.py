import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from sensor_fusion.fusion import SqliteInterface, convertTimeToEpoch
from sensor_fusion.offlineCode.utils.plottingCode import (
    plot_signals_over_time,
    plot_sensor_data,
    plot_sensor_timestamps,
)
from sensor_fusion.offlineCode.utils.processDBs import (
    validate_db_file,
    process_db_file_for_individual_drives,
    aggregate_data,
)


def check_table_copy(raw_data, processed_data):
    for i in range(len(processed_data)):
        if (
            raw_data[i]["time"] != processed_data[i]["time"]
            and raw_data[i]["id"] != processed_data[i]["row_id"]
        ):
            print("Data does not match")
            return False
    return True


def main(file_path):
    # Check if the path exists
    if os.path.exists(file_path):
        print(f"The path {file_path} exists.")
        # You can add your processing code here
    else:
        print(f"The path {file_path} does not exist.")

    ### Main Loop
    time_now = time.time()
    _, camera_type = validate_db_file(file_path)
    print(f"Validation Done in {time.time()-time_now} seconds")
    try:
        db_interface = SqliteInterface(file_path)
        raw_imu = db_interface.get_all_rows_as_dicts("imu")
        processed_imu = db_interface.get_all_rows_as_dicts("imu_processed")
        print(f"Copied Table correctly: {check_table_copy(raw_imu, processed_imu)}")

        useable_sessions = process_db_file_for_individual_drives(file_path, camera_type)
        for session in useable_sessions:
            raw_imu = aggregate_data(useable_sessions[session]["imu_data"])
            processed_imu = aggregate_data(
                useable_sessions[session]["imu_processed_data"]
            )
            gnss_data = aggregate_data(useable_sessions[session]["gnss_data"])
            raw_imu_len = len(raw_imu)
            processed_imu_len = len(processed_imu)
            gnss_len = len(gnss_data)
            if raw_imu_len == 0 or processed_imu_len == 0 or gnss_len == 0:
                print(
                    f"Session: {session} -> Missing Data -- gnss: {gnss_len} raw_imu: {raw_imu_len} processed_imu: {processed_imu_len}"
                )
                continue

            print(
                f"Session: {session} -> gnss: {len(gnss_data['system_time'])} raw_imu: {len(raw_imu['time'])} processed_imu: {len(processed_imu['time'])}"
            )
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
            sensor_data = {
                "gnss": gnss_data["system_time"],
                "raw_imu": raw_imu["time"],
                "processed_imu": processed_imu["time"],
            }
            plot_sensor_timestamps(sensor_data)

            plt.show()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a local path.")
    parser.add_argument("path", type=str, help="The local path to process")
    args = parser.parse_args()
    file_path = args.path
    # Check if the path exists
    for user in os.listdir(file_path):
        if ".DS_Store" in user:
            continue
        # print(f"Processing {user}")
        user_path = os.path.join(file_path, user)
        for possible_dir in os.listdir(user_path):
            possible_dir_path = os.path.join(user_path, possible_dir)
            if possible_dir == "recovered" and os.path.isdir(possible_dir_path):
                # print(f"Processing {possible_dir_path}")
                for recovered_file in os.listdir(possible_dir_path):
                    db_file_path = os.path.join(possible_dir_path, recovered_file)
                    if recovered_file.endswith(".db"):
                        print(f"Processing {db_file_path}")
                        main(db_file_path)
