import numpy
from .sensorFusion import convertToEuler, averageEulerAngles, equalize_list_lengths, orientationFLAE
from .sqliteinterface import getImuData, getMagnetometerData, calculate_average


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

    accel = numpy.array(accel_list)
    gyro = numpy.array(gyro_list)    
    mag = numpy.array([data.getMag() for data in mag_data])

    # get the orientation
    quats = orientationFLAE(mag, accel, gyro)
    euler_list = convertToEuler(quats)
    avg_euler = averageEulerAngles(euler_list)
    # Modification for FLAE
    avg_euler = (avg_euler[0], avg_euler[1]*-1, avg_euler[2])
    return avg_euler


def isUpsideDown(time: int = None):
    """ 
    Returns True if the device is upside down, False otherwise.
    Returns:
        bool: True if the device is upside down, False otherwise.
    """
    # if no time given get data for ~now
    if time is None:
        time = int(time.time()*1000) - 500

    # get data from the database
    imu_data = getImuData(time)
    imu_ave = calculate_average(imu_data)
    # check if the device is upside down
    return imu_ave['az'] < -0.1

    
    