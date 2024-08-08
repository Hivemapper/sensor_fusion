import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from sensor_fusion.sensor_fusion_service.conversions import convert_time_to_epoch
from sensor_fusion.offline_code.utils.plotting_code import (
    plot_signals_over_time,
    plot_sensor_data,
    plot_sensor_data_classified,
)
from sensor_fusion.offline_code.utils.process_validate_dbs import (
    validate_db_file,
    process_db_file_for_individual_drives,
    transform_class_list_to_dict,
)
from sensor_fusion.offline_code.utils.utils import valid_dir, valid_file


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
        useable_sessions = process_db_file_for_individual_drives(file_path, camera_type)
        for session in useable_sessions:
            ##### Ingest Data #####
            raw_imu = transform_class_list_to_dict(
                useable_sessions[session]["imu_data"]
            )
            processed_imu = transform_class_list_to_dict(
                useable_sessions[session]["imu_processed_data"]
            )
            gnss_data = transform_class_list_to_dict(
                useable_sessions[session]["gnss_data"]
            )
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

            ### Convert time to epoch for all values
            for i in range(len(gnss_data["system_time"])):
                gnss_data["system_time"][i] = convert_time_to_epoch(
                    gnss_data["system_time"][i]
                )
            for i in range(len(raw_imu["time"])):
                raw_imu["time"][i] = convert_time_to_epoch(raw_imu["time"][i])

            for i in range(len(processed_imu["time"])):
                processed_imu["time"][i] = convert_time_to_epoch(
                    processed_imu["time"][i]
                )

            ##### Plottting #####
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

            # plot_sensor_data(
            #     processed_imu["time"],
            #     processed_imu["ax"],
            #     processed_imu["ay"],
            #     processed_imu["az"],
            #     sensor_name="Processed ACCEL",
            #     title="Processed ACCEL Data",
            #     downsample_factor=5,
            # )
            # plot_sensor_data(
            #     processed_imu["time"],
            #     processed_imu["gx"],
            #     processed_imu["gy"],
            #     processed_imu["gz"],
            #     sensor_name="Processed GYRO",
            #     title="Processed GYRO Data",
            #     downsample_factor=5,
            # )

            ### Downsample raw data
            acc_x_down = np.interp(
                gnss_data["system_time"], processed_imu["time"], processed_imu["acc_x"]
            )
            acc_y_down = np.interp(
                gnss_data["system_time"], processed_imu["time"], processed_imu["acc_y"]
            )
            acc_z_down = np.interp(
                gnss_data["system_time"], processed_imu["time"], processed_imu["acc_z"]
            )
            gyro_x_down = np.interp(
                gnss_data["system_time"], processed_imu["time"], processed_imu["gyro_x"]
            )
            gyro_y_down = np.interp(
                gnss_data["system_time"], processed_imu["time"], processed_imu["gyro_y"]
            )
            gyro_z_down = np.interp(
                gnss_data["system_time"], processed_imu["time"], processed_imu["gyro_z"]
            )

            stationary_down *= 0.5
            plot_sensor_data_classified(
                gnss_data["system_time"],
                acc_x_down,
                acc_y_down,
                acc_z_down,
                stationary_down,
                stationary_down,
                stationary_down,
                sensor_name="Processed ACCEL",
                title="Processed ACCEL Data",
            )
            plot_sensor_data_classified(
                gnss_data["system_time"],
                gyro_x_down,
                gyro_y_down,
                gyro_z_down,
                stationary_down,
                stationary_down,
                stationary_down,
                sensor_name="Processed GYRO",
                title="Processed GYRO Data",
            )

            plt.show()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a local path.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dirpath", type=valid_dir, help="The local directory path to process"
    )
    group.add_argument(
        "--filepath", type=valid_file, help="The local file path to process"
    )
    args = parser.parse_args()

    if args.dirpath:
        file_path = args.dirpath
        # Check if the path exists
        for user in os.listdir(file_path):
            if ".DS_Store" in user:
                continue
            user_path = os.path.join(file_path, user)
            for possible_dir in os.listdir(user_path):
                possible_dir_path = os.path.join(user_path, possible_dir)
                if possible_dir == "output" and os.path.isdir(possible_dir_path):
                    for recovered_file in os.listdir(possible_dir_path):
                        db_file_path = os.path.join(possible_dir_path, recovered_file)
                        if recovered_file.endswith(".db"):
                            print(f"Processing {db_file_path}")
                            main(db_file_path)
    else:
        file_path = args.filepath
        print(f"Processing {file_path}")
        main(file_path)
