import sys
import json
from processDBs import validate_db_file, process_db_file_for_individual_drives
sys.path.insert(0, '/Users/rogerberman/hivemapper/sensor-fusion')  # Add the project root to the Python path
from fusion import (
    extractAndSmoothImuData,
    extractGNSSData,
)


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python script_name.py file_path")
    #     sys.exit(1)
    
    # db_path = sys.argv[1]
    db_path = "/Users/rogerberman/dev_ground/test_data/imu_stationary_test_data-decoded-recovered/spectacular-grass-window/2024-05-16T12:32:02.000Z.db"
    validated_file_path, camera_type = validate_db_file(db_path)
    usable_drives = process_db_file_for_individual_drives(validated_file_path, camera_type)

    all_data = {}
    for session in usable_drives:
        drive_data = usable_drives[session]
        gnss_data = drive_data['gnss_data']
        imu_data = drive_data['imu_data']
        acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, imu_time, imu_freq = extractAndSmoothImuData(imu_data)
        lats, lons, alts, speed, heading, headingAccuracy, hdop, gdop, gnss_system_time, gnss_real_time, time_resolved, gnss_freq = extractGNSSData(gnss_data)
        # Create a dictionary to store sensor data for the session
        session_dict = {
            'acc_x': list(acc_x),
            'acc_y': list(acc_y),
            'acc_z': list(acc_z),
            'gyro_x': list(gyro_x),
            'gyro_y': list(gyro_y),
            'gyro_z': list(gyro_z),
            'imu_time': list(imu_time),
            'lats': list(lats),
            'lons': list(lons),
            'alts': list(alts),
            'speed': list(speed),
            'heading': list(heading),
            'headingAccuracy': list(headingAccuracy),
            'hdop': list(hdop),
            'gdop': list(gdop),
            'gnss_system_time': list(gnss_system_time),
            'gnss_real_time': list(gnss_real_time),
            'time_resolved': list(time_resolved),
        }
        
        # Store the session data in the overarching dictionary
        all_data[session] = session_dict

    # Define the path to the JSON file
    json_file_path = "spectacular-grass-window_session_data.json"

    # Write the session_data dictionary to a JSON file
    with open(json_file_path, "w") as json_file:
        json.dump(all_data, json_file)

    print("Session data has been written to", json_file_path)
