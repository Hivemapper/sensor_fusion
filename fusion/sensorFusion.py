import numpy
import ahrs


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
        raise ValueError("Both lists must have elements.")
    
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
SENSOR_FREQ = 7.0

########### FUSION ALGORITHM FUNCTIONS ############

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

def orientationEKF(mag, accel, gyro, q0):
    orientation = ahrs.filters.EKF(
        gyr=gyro, 
        acc=accel, 
        mag=mag, 
        frequency= SENSOR_FREQ, 
        magnetic_ref = SF_MAGNETIC_REF,
        # magnetic_ref= SF_MAGNETIC_DIP,
        # g, a, m
        noises=[0.001, 0.175, 0.9],
        # noises=[0.0005, 0.21, 2.9],
        # noises=[0.01, 0.85, 5],
        # frame='NED',
        q0=q0,
    )
    return orientation.Q

def orientationFourati(mag, accel, gyro):
    orientation = ahrs.filters.Fourati(
        gyr=gyro, 
        acc=accel, 
        mag=mag, 
        frequency= SENSOR_FREQ, 
        magnetic_dip= SF_MAGNETIC_DIP,
        gain=0.5,
    )
    return orientation.Q

def orientationMadgwick(mag, accel, gyro):
    orientation = ahrs.filters.Madgwick(
        gyr=gyro, 
        acc=accel, 
        mag=mag, 
        frequency= SENSOR_FREQ,
        gain_imu=0.05,
        gain_mag=0.0,
    )
    return orientation.Q

def orientationMahony(mag, accel, gyro):
    orientation = ahrs.filters.Mahony(
        gyr=gyro, 
        acc=accel, 
        mag=mag, 
        frequency= SENSOR_FREQ,
        k_P = 0.001,
        k_I=0.2,
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

def orientationSAAM(mag, accel):
    orientation = ahrs.filters.SAAM(
        acc=accel, 
        mag=mag, 
    )
    return orientation.Q

def orientationDavenport(mag, accel):
    orientation = ahrs.filters.Davenport( 
        acc=accel, 
        mag=mag,
        # weights=[0.5, 0.5], 
        magnetic_dip= SF_MAGNETIC_DIP,
    )
    return orientation.Q

def orientationFAMC(mag, accel):
    orientation = ahrs.filters.FAMC(
        acc=accel, 
        mag=mag, 
    )
    return orientation.Q

def orientationFLAE(mag, accel):
    orientation = ahrs.filters.FLAE(
        acc=accel, 
        mag=mag,
        magnetic_dip= SF_MAGNETIC_DIP,
        weights=[0.2, 0.8], 
    )
    return orientation.Q

def orientationFQA(mag, accel):
    orientation = ahrs.filters.FQA(
        acc=accel, 
        mag=mag,
        # mag_ref= SF_MAGNETIC_REF, 
    )
    return orientation.Q

def orientationOLEQ(mag, accel):
    orientation = ahrs.filters.OLEQ(
        acc=accel, 
        mag=mag,
        # magnetic_ref= SF_MAGNETIC_REF, 
    )
    return orientation.Q

def orientationQUEST(mag, accel):
    orientation = ahrs.filters.QUEST(
        acc=accel, 
        mag=mag,
        magnetic_dip= SF_MAGNETIC_DIP, 
    )
    return orientation.Q

def oreintationTilt(mag, accel):
    orientation = ahrs.filters.Tilt(
        acc=accel, 
        mag=mag, 
    )
    return orientation.Q