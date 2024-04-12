import numpy as np
from geopy.distance import geodesic

from .sensorFusion import calculateHeading, equalize_list_lengths, convertToEuler, averageEulerAngles, orientationFLAE
from .sqliteinterface import getImuData, getMagnetometerData, getGnssData, ASC
from .utils import calculateAttributesAverage, extractAndSmoothImuData, extractAndSmoothMagData, extractGNSSData
from .ellipsoid_fit import calibrate_mag


TEN_MINUTES = 1000 * 60 * 10 # in millisecond epoch time
QUARTER_SECOND = 250 # in millisecond epoch time
HALF_SECOND = 500 # in millisecond epoch time
ONE_SECOND = 1000 # in millisecond epoch time
THIRTY_SECONDS = 1000 * 30 # in millisecond epoch time

ACCEL_Z_UPSIDE_DOWN_THRESHOLD = -0.1
GNSS_LOW_SPEED_THRESHOLD = 0.1
HEADING_DIFF_MAGNETOMETER_FLIP_THRESHOLD = 100 #in degreed
GNSS_HEADING_ACCURACY_THRESHOLD = 3.0
GNSS_DISTANCE_THRESHOLD = 1.5 # meters

def getEulerAngle(desiredTime: int):
    """ 
    Returns the average Euler angles (roll, pitch, yaw) in degrees for the given epoch millisecond times.
    Args:
        desiredTime (int): The epoch millisecond time to query for sensor data.
    Returns:
        Tuple[float, float, float]: The average Euler angles (roll, pitch, yaw) in degrees.
    """
    # get data from the database
    imu_data = getImuData(desiredTime)
    mag_data = getMagnetometerData(desiredTime)

    # Ensure same number of samples
    imu_data, mag_data = equalize_list_lengths(imu_data, mag_data)

    # extract the data from the objects
    accel_list = []
    gyro_list = []
    for data in imu_data:
        accel_list.append(data.getAccel())
        gyro_list.append(data.getGyro())

    accel = np.array(accel_list)
    gyro = np.array(gyro_list)    
    mag = np.array([data.getMag() for data in mag_data])

    # get the orientation
    quats = orientationFLAE(mag, accel, gyro)
    euler_list = convertToEuler(quats)
    avg_euler = averageEulerAngles(euler_list)
    # Modification for FLAE
    avg_euler = [avg_euler[0], avg_euler[1]*-1, avg_euler[2]]
    return avg_euler

def isUpsideDown(time: int = None):
    """ 
    Returns True if the device is upside down, False otherwise.
    Returns:
        bool: True if the device is upside down, False otherwise.
    """
    # if no time given get data for ~now
    if time is None:
        time = int(time.time()*ONE_SECOND) - HALF_SECOND

    # get data from the database
    imu_data = getImuData(time)
    imu_ave = calculateAttributesAverage(imu_data)
    # check if the device is upside down
    return imu_ave['az'] < ACCEL_Z_UPSIDE_DOWN_THRESHOLD

def getGNSSHeading(time: int = None):
    """
    Returns the GNSS heading in degrees.
    Returns:
        float: The GNSS heading in degrees.
    """
    # if no time given get data for ~now
    if time is None:
        time = int(time.time()*ONE_SECOND) - HALF_SECOND

    # get data from the database
    gnss_data = getGnssData(time)
    current_heading = gnss_data[0].heading
    current_heading_accuracy = gnss_data[0].headingAccuracy
    current_position = (gnss_data[0].latitude, gnss_data[0].longitude)
    if current_heading_accuracy < GNSS_HEADING_ACCURACY_THRESHOLD:
        return current_heading
    else:
        older_gnss_data = getGnssData(time-QUARTER_SECOND,THIRTY_SECONDS)
        for data in older_gnss_data:
            old_position = (data.latitude, data.longitude)
            distance = geodesic(current_position, old_position).meters
            if (data.headingAccuracy < GNSS_HEADING_ACCURACY_THRESHOLD and
                distance < GNSS_DISTANCE_THRESHOLD):
                return data.heading
    return None

def getDashcamToVehicleHeadingOffset(time: int = None, pastRange: int= None):
    """
    Returns the yaw offset between the dashcam and vehicle in degrees.
    Returns:
        float: The yaw offset between the dashcam and vehicle in degrees.
    """
    # if no time given get data for ~now
    if time is None:
        time = int(time.time()*ONE_SECOND) - HALF_SECOND

    # if no pastRange given get data for 7 minutes
    if pastRange is None:
        pastRange = TEN_MINUTES

    # get data from the database
    imu_data = getImuData(time, pastRange, ASC)
    mag_data = getMagnetometerData(time, pastRange, ASC)
    gnss_data = getGnssData(time, pastRange, ASC)

    # Extract the data from the objects
    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, imu_time = extractAndSmoothImuData(imu_data)
    mag_x, mag_y, mag_z, mag_time = extractAndSmoothMagData(mag_data)
    _, _, _, speed, heading, headingAccuracy, hdop, gdop, gnss_time = extractGNSSData(gnss_data)

    # downsample the data to match GNSS frequency
    acc_x_down = np.interp(gnss_time, imu_time, acc_x)
    acc_y_down = np.interp(gnss_time, imu_time, acc_y)
    acc_z_down = np.interp(gnss_time, imu_time, acc_z)
    gyro_x_down = np.interp(gnss_time, imu_time, gyro_x)
    gyro_y_down = np.interp(gnss_time, imu_time, gyro_y)
    gyro_z_down = np.interp(gnss_time, imu_time, gyro_z)
    mag_x_down = np.interp(gnss_time, mag_time, mag_x)
    mag_y_down = np.interp(gnss_time, mag_time, mag_y)
    mag_z_down = np.interp(gnss_time, mag_time, mag_z)

    # Calculate bias for accel and gyro
    zero_speed_indices = [i for i, speed_val in enumerate(speed) if speed_val < GNSS_LOW_SPEED_THRESHOLD]

    acc_x_down_zero_speed, acc_y_down_zero_speed, acc_z_down_zero_speed = [], [], []
    gyro_x_down_zero_speed, gyro_y_down_zero_speed, gyro_z_down_zero_speed = [], [], []
    for i in zero_speed_indices:
        acc_x_down_zero_speed.append(acc_x_down[i])
        acc_y_down_zero_speed.append(acc_y_down[i])
        acc_z_down_zero_speed.append(acc_z_down[i])
        gyro_x_down_zero_speed.append(gyro_x_down[i])
        gyro_y_down_zero_speed.append(gyro_y_down[i])
        gyro_z_down_zero_speed.append(gyro_z_down[i])

    # Calculate the average of the zero speed values
    acc_x_down_zero_speed_avg = np.mean(acc_x_down_zero_speed)
    acc_y_down_zero_speed_avg = np.mean(acc_y_down_zero_speed)
    acc_z_down_zero_speed_avg = np.mean(acc_z_down_zero_speed) - 1  # handle the fact this needs to be 1 when at 0 velocity not 0
    gyro_x_down_zero_speed_avg = np.mean(gyro_x_down_zero_speed)
    gyro_y_down_zero_speed_avg = np.mean(gyro_y_down_zero_speed)
    gyro_z_down_zero_speed_avg = np.mean(gyro_z_down_zero_speed)

    # Apply the bias to the data
    acc_x_down = [a - acc_x_down_zero_speed_avg for a in acc_x_down]
    acc_y_down = [a - acc_y_down_zero_speed_avg for a in acc_y_down]
    acc_z_down = [a - acc_z_down_zero_speed_avg for a in acc_z_down]
    gyro_x_down = [g - gyro_x_down_zero_speed_avg for g in gyro_x_down]
    gyro_y_down = [g - gyro_y_down_zero_speed_avg for g in gyro_y_down]
    gyro_z_down = [g - gyro_z_down_zero_speed_avg for g in gyro_z_down]

    print(f"Accel offsets: {acc_x_down_zero_speed_avg}, {acc_y_down_zero_speed_avg}, {acc_z_down_zero_speed_avg}")
    print(f"Gyro offsets: {gyro_x_down_zero_speed_avg}, {gyro_y_down_zero_speed_avg}, {gyro_z_down_zero_speed_avg}")

    # Calibrate Mag
    mag_bundle = np.array(list(zip(mag_x_down, mag_y_down, mag_z_down)))
    calibrated_mag_bundle = calibrate_mag(mag_bundle)

    acc_bundle = np.array(list(zip(acc_x_down, acc_y_down, acc_z_down)))
    gyro_bundle = np.array(list(zip(gyro_x_down, gyro_y_down, gyro_z_down)))

    fused_heading, _, _ = calculateHeading(acc_bundle, gyro_bundle, calibrated_mag_bundle, heading[0])

    # used to translate the fused heading from -180:180 to the correct range 0:360
    fused_heading = [heading_val + 360 if heading_val < 0 else heading_val for heading_val in fused_heading]

    # used when the heading is off by 180 degrees
    # check last heading diff to make decision
    if abs(heading[-1] - fused_heading[-1]) > HEADING_DIFF_MAGNETOMETER_FLIP_THRESHOLD:
        fused_heading = [heading_val - 180 for heading_val in fused_heading]

    heading_diff = []
    for i in range(len(headingAccuracy)):
        if headingAccuracy[i] < GNSS_HEADING_ACCURACY_THRESHOLD:
            heading_diff.append((heading[i] - fused_heading[i] + 180) % 360 - 180)

    heading_diff_mean = np.mean(heading_diff)
    print(f"Mean heading difference: {heading_diff_mean}")

    return heading_diff_mean

    
    