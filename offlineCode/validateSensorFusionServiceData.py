import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from sensor_fusion.fusion import SqliteInterface, convertTimeToEpoch
from sensor_fusion.offlineCode.utils.plottingCode import (
    plot_signal_over_time,
    plot_signals_over_time,
    plot_sensor_data,
    plot_sensor_timestamps,
    create_map,
)
from sensor_fusion.offlineCode.utils.processDBs import (
    validate_db_file,
    process_db_file_for_individual_drives,
    aggregate_data,
)
from sensor_fusion.sensor_fusion_service.processing import cubic_spline_interpolation
from sensor_fusion.offlineCode.utils.utils import (
    valid_dir,
    valid_file,
    remove_duplicate_data,
)
from get_imu_offset import get_imu_offsets


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
        useable_sessions = process_db_file_for_individual_drives(file_path, camera_type)
        for session in useable_sessions:
            raw_imu = aggregate_data(useable_sessions[session]["imu_data"])
            processed_imu = aggregate_data(
                useable_sessions[session]["imu_processed_data"]
            )
            gnss_data = aggregate_data(useable_sessions[session]["gnss_data"])
            fused_data = aggregate_data(useable_sessions[session]["fused_data"])
            ## Filter out duplicates
            fused_data = remove_duplicate_data(fused_data, "time")
            raw_imu_len = len(raw_imu)
            processed_imu_len = len(processed_imu)
            gnss_len = len(gnss_data)
            fused_len = len(fused_data)
            if (
                raw_imu_len == 0
                or processed_imu_len == 0
                or gnss_len == 0
                or fused_len == 0
            ):
                print(
                    f"Session: {session} -> Missing Data -- gnss: {gnss_len} raw_imu: {raw_imu_len} processed_imu: {processed_imu_len} fused: {fused_len}"
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
            forward_vel = np.interp(
                gnss_data["system_time"],
                fused_data["time"],
                fused_data["forward_velocity"],
            )

            plot_signals_over_time(
                gnss_data["system_time"],
                gnss_data["speed"],
                forward_vel,
                downsample_factor=1,
                signal1_label="Speed",
                signal2_label="Processed Speed",
            )

            # Match fused heading to gnss heading
            fused_heading = np.interp(
                gnss_data["system_time"],
                fused_data["time"],
                fused_data["fused_heading"],
            )
            fused_heading = np.rad2deg(fused_heading)
            fused_heading = [
                heading_val + 360 if heading_val < 0 else heading_val
                for heading_val in fused_heading
            ]

            plot_signals_over_time(
                gnss_data["system_time"],
                gnss_data["heading"],
                fused_heading,
                downsample_factor=10,
                signal1_label="Heading",
                signal2_label="Processed Heading",
            )

            # plot yaw rates
            yaw_rate_deg = fused_data["yaw_rate"]
            plot_signal_over_time(
                fused_data["time"], yaw_rate_deg, signal_label="Yaw Rate"
            )

            plot_sensor_data(
                processed_imu["time"],
                processed_imu["ax"],
                processed_imu["ay"],
                processed_imu["az"],
                sensor_name="Processed ACCEL",
                title="Processed ACCEL Data",
                downsample_factor=5,
            )
            plot_sensor_data(
                processed_imu["time"],
                processed_imu["gx"],
                processed_imu["gy"],
                processed_imu["gz"],
                sensor_name="Processed GYRO",
                title="Processed GYRO Data",
                downsample_factor=1,
            )
            sensor_data = {
                "gnss": gnss_data["system_time"],
                "raw_imu": raw_imu["time"],
                "processed_imu": processed_imu["time"],
            }
            plot_sensor_timestamps(sensor_data)

            get_imu_offsets(
                processed_imu["ax"],
                processed_imu["ay"],
                processed_imu["az"],
                processed_imu["gx"],
                processed_imu["gy"],
                processed_imu["gz"],
                processed_imu["time"],
                gnss_data["system_time"],
                gnss_data["speed"],
            )

            create_map(
                fused_data["gnss_lat"],
                fused_data["gnss_lon"],
            )

            create_map(
                fused_data["fused_lat"],
                fused_data["fused_lon"],
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
