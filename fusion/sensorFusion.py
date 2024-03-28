from .sqliteinterface import getGnssData, getImuData, getMagnetometerData
import numpy
import ahrs


########### TOP LEVEL SENSOR FUSION ORIENTATION ALGO ############

def getEulerAngle(desiredTime: int):
    """ 
    Returns the average Euler angles (roll, pitch, yaw) in degrees for the given epoch timestamp.
    Args:
        desiredTime (int): The epoch timestamp to query for sensor data.
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
        accel_list.append(data.getAccelerometer())
        gyro_list.append(data.getGyroscope())

    accel = numpy.array(accel_list)
    gyro = numpy.array(gyro_list)    
    mag = numpy.array(data.getMagnetometer() for data in mag_data)

    # get the orientation
    quats = orientationFLAE(mag, accel, gyro)
    euler_list = convertToEuler(quats)
    avg_euler = averageEulerAngles(euler_list)
    return avg_euler


########### HELPER FUNCTIONS ############

def convertToEuler(values):
    """
    Converts quaternions to Euler angles in degrees.
    Args:
        values (list): List of quaternions as numpy arrays.
    Returns:
        list: Euler angles (roll, pitch, yaw) in degrees for each quaternion.
    """
    quaternion_list = [ahrs.Quaternion(q) for q in values]
    euler_deg_list = []
    for quaternion in quaternion_list:
        euler_rad = quaternion.to_angles()
        euler_deg = numpy.degrees(euler_rad)
        euler_deg_list.append(euler_deg)
    return euler_deg_list

# returns the average of a list of euler angles
def averageEulerAngles(euler_deg_list):
    """
    Averages a list of Euler angles in degrees.
    Args:
        euler_deg_list (list): A list of Euler angles in degrees.
    Returns:
        Tuple[float, float, float]: The average Euler angles in degrees.
    """
    return tuple(numpy.mean(euler_deg_list, axis=0))

def equalize_list_lengths(list1, list2):
    """ Equalizes the lengths of two lists by removing elements from the end of the longer list,
    provided both lists have elements. If either list is empty, no changes are made.
    Args:
        list1 (list): The first list.
        list2 (list): The second list.
    Returns:
        Tuple[List, List]: A tuple containing the potentially modified lists with equal lengths.
    """
    # Check if either list is empty
    if not list1 or not list2:
        print("One of the lists of sensor data is empty")
        return list1, list2 
    
    len1, len2 = len(list1), len(list2)
    
    # If list1 is longer, trim elements from its end
    if len1 > len2:
        list1 = list1[:len2]
    # If list2 is longer, trim elements from its end
    elif len2 > len1:
        list2 = list2[:len1]
    
    return list1, list2


########### FUSION OPTIONS ############
SF_MAGNETIC_REF = [22371.8, 5180, 41715.1]
SF_MAGNETIC_DIP = 61.16779
SENSOR_FREQ = 38.0

def orientationAQUA(mag, accel, gyro):
    orientation = ahrs.filters.AQUA(gyr=gyro, acc=accel, mag=mag, frequency=38.0, adaptive=True)
    return orientation.Q

def orientationComplementary(mag, accel, gyro):
    orientation = ahrs.filters.Complementary(
        gyr=gyro, 
        acc=accel, 
        mag=mag, 
        frequency=SENSOR_FREQ, 
        gain=0.5,
        representation='quaternion'
    )
    return orientation.Q

def orientationEKF(mag, accel, gyro):
    orientation = ahrs.filters.EKF(
        gyr=gyro, 
        acc=accel, 
        mag=mag, 
        frequency= SENSOR_FREQ, 
        magnetic_ref = SF_MAGNETIC_REF,
        # g, a, m
        # noises=[0.1, 0.5, 0.5]
    )
    return orientation.Q

def orientationFourati(mag, accel, gyro):
    orientation = ahrs.filters.Fourati(
        gyr=gyro, 
        acc=accel, 
        mag=mag, 
        frequency= SENSOR_FREQ, 
        magnetic_dip= SF_MAGNETIC_DIP,
        gain=0.01,
    )
    return orientation.Q

def orientationMadgwick(mag, accel, gyro):
    orientation = ahrs.filters.Madgwick(
        gyr=gyro, 
        acc=accel, 
        mag=mag, 
        frequency= SENSOR_FREQ,
        # gain=0.01,
    )
    return orientation.Q

def orientationMahony(mag, accel, gyro):
    orientation = ahrs.filters.Mahony(
        gyr=gyro, 
        acc=accel, 
        mag=mag, 
        frequency= SENSOR_FREQ,
        k_i=0.2,
    )
    return orientation.Q

def orientationROLEQ(mag, accel, gyro):
    orientation = ahrs.filters.ROLEQ(
        gyr=gyro, 
        acc=accel, 
        mag=mag, 
        frequency= SENSOR_FREQ, 
    )
    return orientation.Q

def orientationSAAM(mag, accel, gyro):
    orientation = ahrs.filters.SAAM(
        acc=accel, 
        mag=mag, 
    )
    return orientation.Q

def orientationDavenport(mag, accel, gyro):
    orientation = ahrs.filters.Davenport( 
        acc=accel, 
        mag=mag,
        # weights=[0.5, 0.5], 
        magnetic_dip= SF_MAGNETIC_DIP,
    )
    return orientation.Q

def orientationFAMC(mag, accel, gyro):
    orientation = ahrs.filters.FAMC(
        acc=accel, 
        mag=mag, 
    )
    return orientation.Q

def orientationFLAE(mag, accel, gyro):
    orientation = ahrs.filters.FLAE(
        acc=accel, 
        mag=mag,
        magnetic_dip= SF_MAGNETIC_DIP,
        weights=[0.9, 0.1], 
    )
    return orientation.Q

def orientationFQA(mag, accel, gyro):
    orientation = ahrs.filters.FQA(
        acc=accel, 
        mag=mag,
        # mag_ref= SF_MAGNETIC_REF, 
    )
    return orientation.Q

def orientationOLEQ(mag, accel, gyro):
    orientation = ahrs.filters.OLEQ(
        acc=accel, 
        mag=mag,
        # magnetic_ref= SF_MAGNETIC_REF, 
    )
    return orientation.Q

def orientationQUEST(mag, accel, gyro):
    orientation = ahrs.filters.QUEST(
        acc=accel, 
        mag=mag,
        magnetic_dip= SF_MAGNETIC_DIP, 
    )
    return orientation.Q

def oreintationTilt(mag, accel, gyro):
    orientation = ahrs.filters.Tilt(
        acc=accel, 
        mag=mag, 
    )
    return orientation.Q