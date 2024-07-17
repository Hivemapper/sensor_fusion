from typing import List

from telemetryMath import extractAndSmoothImuData, extractGNSSData, calculateStationary
from conversions import lists_to_dicts
from sqliteInterface import IMUData, GNSSData


def grab_most_recent_raw_data_session(
    rawData,
    processedIndex,
    furthest_index,
):
    """Requires rawData to have session data. Processes the raw data and returns the raw data
    with only the starting session and the new index

    Args:
        rawData (_type_): _description_
    """
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
                next_index = processedIndex + i + 1
                break
    else:
        next_index = furthest_index + 1

    return rawData, next_index


def processRawData(
    gnssData: List[GNSSData], imuData: List[IMUData], debug: bool = False
):
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
    ) = extractAndSmoothImuData(imuData)
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
