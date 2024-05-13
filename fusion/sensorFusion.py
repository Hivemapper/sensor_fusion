import numpy as np
import ahrs
import math
from ahrs.utils.wmm import WMM
from .filter import estimate, quaternion_to_euler_degrees, quaternion_from_euler

########### FUSION OPTIONS ############
SF_MAGNETIC_REF = [22371.8, 5180, 41715.1]
SF_MAGNETIC_DIP = 61.16779
SENSOR_FREQ = 7.0

############ MAIN FUNCTIONS ############
def calculateHeading(accel_data, gyro_data, mag_data, time, gnss_initial_heading, initialPosition, gnssFreq=SENSOR_FREQ, P=None):
    # ensure same number of samples
    if not len(accel_data) == len(mag_data):
        raise ValueError("Both lists must equal size.")

    q0 = ahrs.Quaternion()
    # yaw, pitch, roll in radians
    q0 = q0.from_angles(np.array([math.radians(gnss_initial_heading),0.0, 0.0]))
    quats = orientationEKF(mag=mag_data, accel=accel_data, gyro=gyro_data, time=time, q0=q0, initialPosition=initialPosition, freq=gnssFreq, P=P)
    # quats = orientationAQUA(mag=mag_data, accel=accel_data, gyro=gyro_data, time=time, q0=q0, initialPosition=initialPosition, freq=gnssFreq)
    euler_list = convertToEuler(quats)
    yaw, pitch, roll = [], [], []
    for euler in euler_list:
        roll.append(euler[0])
        pitch.append(euler[1])
        yaw.append(euler[2])

    # initialOritentation = quaternion_from_euler(0, 0, math.radians(gnss_initial_heading))
    # initialOritentation = quaternion_from_euler(0, 0, 0)
    # # print(f"Initial Orientation: {initialOritentation}")
    # quats = orientationEKF2(accel=accel_data, gyro=gyro_data, time=time, q0=initialOritentation)
    # roll, pitch, yaw = quaternion_to_euler_degrees(quats)   
    return yaw, pitch, roll

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
        euler_deg = np.degrees(euler_rad)
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
    return tuple(np.mean(euler_deg_list, axis=0))

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

########### FUSION ALGORITHM FUNCTIONS ############

def orientationAQUA(mag, accel, gyro, time, q0, initialPosition, freq):
    orientation = ahrs.filters.AQUA(
        gyr=gyro, 
        acc=accel, 
        mag=mag, 
        frequency=freq, 
        adaptive=False,
        q0=q0,
    )
    return orientation.Q

def orientationComplementary(mag, accel, gyro, time, q0, initialPosition, freq):
    orientation = ahrs.filters.Complementary(
        gyr=gyro, 
        acc=accel, 
        mag=mag, 
        frequency=freq, 
        gain=0.75,
        q0=q0,
        representation='quaternion'
    )
    return orientation.Q

def orientationEKF(mag, accel, gyro, time, q0, initialPosition, freq, P=None):
    lat,lon,alt = initialPosition
    wmm = WMM(latitude=lat, longitude=lon, height=alt)
    # print(f"Initial Position: {initialPosition}, magnetic ref: {wmm.X, wmm.Y, wmm.Z}")

    orientation = ahrs.filters.EKF(
        gyr=gyro, 
        acc=accel, 
        # mag=mag, 
        frequency= freq, 
        magnetic_ref = np.array([wmm.X, wmm.Y, wmm.Z]),
        # g, a, m
        noises=[0.085, 0.5, 0.5], # Roger's values
        # noises=[0.001, 0.165, 0.2], # Alexi's values
        # noises=[0.0085, 0.8, 0.9],
        # frame='NED',
        q0=q0,
        P=P
    )
    # print(f"Error covariance: {orientation.P}")
    return orientation.Q

def orientationEKF2( accel, gyro, time, q0):
    estimated_q, estimated_b, b_err = estimate(time, gyro.T, accel.T, q0)
    return estimated_q


# def orientationEKF(mag, accel, gyro, time, q0, initialPosition, freq):
#     lat,lon,alt = initialPosition
#     wmm = WMM(latitude=lat, longitude=lon, height=alt)
#     print(f"Initial Position: {initialPosition}, magnetic ref: {wmm.X, wmm.Y, wmm.Z}")

#     ekf_filter = ahrs.filters.EKF(
#         magnetic_ref=np.array([wmm.X, wmm.Y, wmm.Z]),
#         noises=[0.2, 0.5, 0.5],
#         q0=q0,
#     )
#     num_samples = len(accel)
#     Q = np.zeros((num_samples, 4))
#     Q[0] = q0
#     Q[0] /= np.linalg.norm(Q[0])
#     print(f"Q0: {Q[0]}")
#     for t in range(1, num_samples):
#         diff_time = (time[t] - time[t-1])/1000.0
#         # print(f"diff_time: {type(diff_time)}")
#         Q[t] = ekf_filter.update(q=Q[t-1], gyr=gyro[t], acc=accel[t], mag=mag[t], dt=diff_time)
#     return Q

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