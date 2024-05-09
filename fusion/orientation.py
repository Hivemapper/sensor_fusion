import numpy as np
import time

from .sensorFusion import(
    calculateHeading, 
    equalize_list_lengths, 
    convertToEuler, 
    averageEulerAngles, 
    orientationFLAE
) 
from .sqliteinterface import SqliteInterface, ASC, convertTimeToEpoch
from .utils import (
    calculateAttributesAverage, 
    extractAndSmoothImuData, 
    extractAndSmoothMagData, 
    extractGNSSData,
    calculate_mag_headings,
    calculate_imu_forward_velocity,
)
from .ellipsoid_fit import calibrate_mag

TEN_MINUTES = 1000 * 60 * 10 # in millisecond epoch time
QUARTER_SECOND = 250 # in millisecond epoch time
HALF_SECOND = 500 # in millisecond epoch time
ONE_SECOND = 1000 # in millisecond epoch time
THIRTY_SECONDS = 1000 * 30 # in millisecond epoch time

ACCEL_Z_UPSIDE_DOWN_THRESHOLD = -0.1
GNSS_LOW_SPEED_THRESHOLD = 0.1
HEADING_DIFF_MAGNETOMETER_FLIP_THRESHOLD = 140 #in degreed
GNSS_HEADING_ACCURACY_THRESHOLD = 3.0

def getEulerAngle(db_interface: SqliteInterface, desiredTime: int):
    """ 
    Returns the average Euler angles (roll, pitch, yaw) in degrees for the given epoch millisecond times.
    Args:
        desiredTime (int): The epoch millisecond time to query for sensor data.
    Returns:
        Tuple[float, float, float]: The average Euler angles (roll, pitch, yaw) in degrees.
    """
    # get data from the database
    imu_data = db_interface.queryImu(desiredTime)
    mag_data = db_interface.queryMagnetometer(desiredTime)

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

def isUpsideDown(db_interface: SqliteInterface, current_time: int = None):
    """ 
    Returns True if the device is upside down, False otherwise.
    Returns:
        bool: True if the device is upside down, False otherwise.
    """
    # if no time given get data for ~now
    if current_time is None:
        current_time = int(time.time()*ONE_SECOND) - HALF_SECOND

    # get data from the database
    imu_data = db_interface.queryImu(current_time)
    imu_ave = calculateAttributesAverage(imu_data)
    # check if the device is upside down
    return imu_ave['az'] < ACCEL_Z_UPSIDE_DOWN_THRESHOLD

def getCleanGNSSHeading(db_interface: SqliteInterface, current_time: int = None, pastRange: int = None):
    """
    Returns the GNSS heading in degrees.
    Returns:
        float: The GNSS heading in degrees.
    """
    # if no time given get data for ~now
    if current_time is None:
        current_time = int(time.time()*ONE_SECOND) - HALF_SECOND

    # get data from the database
    gnss_data = db_interface.queryGnss(current_time, pastRange, ASC)
    _, _, _, _, heading, headingAccuracy, _, _, gnss_time, _ = extractGNSSData(gnss_data)
    # setup for heading correction
    dataLength = len(heading)
    forward_loop = [None]*dataLength
    backward_loop = [None]*dataLength
    last_forward_good = None
    last_backward_good = None
     # iterate through data forward and backward to get the best heading
    for i in range(dataLength):
        forward_index = i
        backward_index = dataLength - 1 - i
        # Handle forward direction
        if headingAccuracy[forward_index] < GNSS_HEADING_ACCURACY_THRESHOLD:
            last_forward_good = heading[forward_index]
        forward_loop[forward_index] = last_forward_good
        # Handle backward direction
        if headingAccuracy[backward_index] < GNSS_HEADING_ACCURACY_THRESHOLD:
            last_backward_good = heading[backward_index]
        backward_loop[backward_index] = last_backward_good
    # correct the forward loop with the backward loop
    for i in range(dataLength):
        if forward_loop[i] == None:
            if backward_loop[i] != None:
                forward_loop[i] = backward_loop[i]
            else:
                forward_loop[i] = heading[i]

    return forward_loop, gnss_time

def getDashcamToVehicleHeadingOffset(db_interface: SqliteInterface, current_time: int = None, pastRange: int= None):
    """
    Returns the yaw offset between the dashcam and vehicle in degrees.
    Returns:
        float: The yaw offset between the dashcam and vehicle in degrees.
    """
    # if no time given get data for ~now
    if current_time is None:
        current_time = int(time.time()*ONE_SECOND) - HALF_SECOND

    # if no pastRange given get data for 7 minutes
    if pastRange is None:
        pastRange = TEN_MINUTES

    # get data from the database
    imu_data = db_interface.queryImuUsingRowID(current_time, pastRange, ASC)
    mag_data = db_interface.queryMagnetometerUsingRowID(current_time, pastRange, ASC)
    gnss_data = db_interface.queryGnss(current_time, pastRange, ASC)
    # print(f"len(imu_data): {len(imu_data)}")
    # print(f"len(mag_data): {len(mag_data)}")
    # print(f"len(gnss_data): {len(gnss_data)}")

    # Extract the data from the objects
    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, imu_time, imu_freq = extractAndSmoothImuData(imu_data, gnss_data[0].time)
    mag_x, mag_y, mag_z, mag_time, mag_freq = extractAndSmoothMagData(mag_data, gnss_data[0].time)
    lats, lons, alts, speed, heading, headingAccuracy, hdop, gdop, gnss_time, gnssFreq = extractGNSSData(gnss_data)
    clean_gnss_heading, _ = getCleanGNSSHeading(db_interface, current_time, pastRange)
    print(f" GNSS initial time: {gnss_time[0]}, IMU initial time: {imu_time[0]}, Mag initial time: {mag_time[0]}")

    print("Data extracted successfully!")

    # downsample the data to match GNSS frequency
    acc_x_down = np.interp(gnss_time, imu_time, acc_x)
    acc_y_down = np.interp(gnss_time, imu_time, acc_y)
    acc_z_down = np.interp(gnss_time, imu_time, acc_z)
    gyro_x_down = np.interp(gnss_time, imu_time, gyro_x)
    gyro_y_down = np.interp(gnss_time, imu_time, gyro_y)
    gyro_z_down = np.interp(gnss_time, imu_time, gyro_z)
    # mag_x_down = np.interp(gnss_time, mag_time, mag_x)
    # mag_y_down = np.interp(gnss_time, mag_time, mag_y)
    # mag_z_down = np.interp(gnss_time, mag_time, mag_z)

    print("Data downsampled successfully!")

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


    # ignore gnss and sample to match highest frequency between imu and mag
    acc_x_input = []
    acc_y_input = []
    acc_z_input = []
    gyro_x_input = []
    gyro_y_input = []
    gyro_z_input = []
    mag_x_input = []
    mag_y_input = []
    mag_z_input = []
    time_input = []
    freq_input = 0
    if imu_freq > mag_freq:
        acc_x_input = acc_x
        acc_y_input = acc_y
        acc_z_input = acc_z
        gyro_x_input = gyro_x
        gyro_y_input = gyro_y
        gyro_z_input = gyro_z
        mag_x_input = np.interp(imu_time, mag_time, mag_x)
        mag_y_input = np.interp(imu_time, mag_time, mag_y)
        mag_z_input = np.interp(imu_time, mag_time, mag_z)
        time_input = imu_time
        freq_input = imu_freq
    else:
        acc_x_input = np.interp(mag_time, imu_time, acc_x)
        acc_y_input = np.interp(mag_time, imu_time, acc_y)
        acc_z_input = np.interp(mag_time, imu_time, acc_z)
        gyro_x_input = np.interp(mag_time, imu_time, gyro_x)
        gyro_y_input = np.interp(mag_time, imu_time, gyro_y)
        gyro_z_input = np.interp(mag_time, imu_time, gyro_z)
        mag_x_input = mag_x
        mag_y_input = mag_y
        mag_z_input = mag_z
        time_input = mag_time
        freq_input = mag_freq   

    # Apply the bias to the data
    acc_x_input = [a - acc_x_down_zero_speed_avg for a in acc_x_input]
    acc_y_input = [a - acc_y_down_zero_speed_avg for a in acc_y_input]
    acc_z_input = [a - acc_z_down_zero_speed_avg for a in acc_z_input]
    gyro_x_input = [g - gyro_x_down_zero_speed_avg for g in gyro_x_input]
    gyro_y_input = [g - gyro_y_down_zero_speed_avg for g in gyro_y_input]
    gyro_z_input = [g - gyro_z_down_zero_speed_avg for g in gyro_z_input]

    import sys
    sys.path.insert(0, '/Users/rogerberman/sensor-fusion/testingScripts')  # Add the project root to the Python path
    from testingScripts.plottingCode import plot_signal_over_time
    import matplotlib.pyplot as plt
    plot_signal_over_time(time_input, acc_x_input, 'acc x time')
    plot_signal_over_time(time_input, acc_y_input, 'acc y time')
    plot_signal_over_time(time_input, gyro_z_input, 'gyro z time')

    print(f"Accel offsets: {acc_x_down_zero_speed_avg}, {acc_y_down_zero_speed_avg}, {acc_z_down_zero_speed_avg}")
    print(f"Gyro offsets: {gyro_x_down_zero_speed_avg}, {gyro_y_down_zero_speed_avg}, {gyro_z_down_zero_speed_avg}")

    # Calibrate Mag
    mag_bundle = np.array(list(zip(mag_x_input, mag_y_input, mag_z_input)))
    calibrated_mag_bundle = calibrate_mag(mag_bundle)


    acc_bundle = np.array(list(zip(acc_x_input, acc_y_input, acc_z_input)))
    gyro_bundle = np.array(list(zip(gyro_x_input, gyro_y_input, gyro_z_input)))

    # print(f"heading: {heading[0]}, clean_gnss_heading: {clean_gnss_heading[0]}")

    initialPosition = [lats[0], lons[0], alts[0]]
    for i in range(len(lats)):
        if headingAccuracy[i] < GNSS_HEADING_ACCURACY_THRESHOLD:
            initialPosition = [lats[i], lons[i], alts[i]]
            break
    # initialHeading = clean_gnss_heading[0] - 360 if clean_gnss_heading[0] > 180 else clean_gnss_heading[0]
    initialHeading = clean_gnss_heading[0]
    # P = [
    #     [ 1.03503015e-05, -5.03595339e-06, -5.79910035e-06, -2.25042690e-07],
    #     [-5.03595339e-06,  3.74206730e-05,  4.71497232e-05,  9.90866868e-07],
    #     [-5.79910035e-06,  4.71497232e-05,  6.00017114e-05,  2.19874434e-06],
    #     [-2.25042690e-07,  9.90866868e-07,  2.19874434e-06,  9.67468524e-06],
    # ]
    P = None
    fused_heading, _, _ = calculateHeading(acc_bundle, gyro_bundle, calibrated_mag_bundle, time_input, initialHeading, initialPosition, freq_input, P)
    mag_calc_heading = calculate_mag_headings(calibrated_mag_bundle, acc_bundle, initialPosition)
    imu_vel = calculate_imu_forward_velocity(acc_bundle, time_input, speed, 0)

    # used to translate the fused heading from -180:180 to the correct range 0:360
    fused_heading = [heading_val + 360 if heading_val < 0 else heading_val for heading_val in fused_heading]

    print(f"Initial Heading: {initialHeading}, Fused Heading: {fused_heading[0]}")

    fused_heading = np.interp(gnss_time, time_input, fused_heading)
    mag_calc_heading = np.interp(gnss_time, time_input, mag_calc_heading)
    imu_vel = np.interp(gnss_time, time_input, imu_vel)

    # used when the heading is off by 180 degrees
    # check last heading diff to make decision
    # if abs(fused_heading[-1] - heading[-1]) > HEADING_DIFF_MAGNETOMETER_FLIP_THRESHOLD:
    #         # handle wrap around and shift by 180 degrees
    #         fused_heading = [(heading_val - 180) % 360 for heading_val in fused_heading]

    # heading_diff = []
    # time_diff = []
    # for i in range(len(headingAccuracy)):
    #     if headingAccuracy[i] < GNSS_HEADING_ACCURACY_THRESHOLD:
    #         heading_diff.append((fused_heading[i] - heading[i] + 180) % 360 - 180)
    #         time_diff.append(gnss_time[i])

    clean_heading_diff = []
    for i in range(len(clean_gnss_heading)):
        clean_heading_diff.append((fused_heading[i] - clean_gnss_heading[i] + 180) % 360 - 180)


    step_size = 10
    number_of_points = []
    mean_diff = []
    for i in range(step_size,len(clean_heading_diff), step_size):
        number_of_points.append(i)
        mean_diff.append(np.mean(clean_heading_diff[:i]))

    heading_diff_mean = np.mean(clean_heading_diff)
    print(f"Mean heading difference: {heading_diff_mean}")


    import sys
    sys.path.insert(0, '/Users/rogerberman/sensor-fusion/testingScripts')  # Add the project root to the Python path
    from testingScripts.plottingCode import plot_signal_over_time, plot_signals_over_time
    import matplotlib.pyplot as plt
    plot_signal_over_time(number_of_points, mean_diff, 'Clean Heading Diff Mean')
    plot_signal_over_time(gnss_time, clean_heading_diff, 'Clean Heading Diff')
    print(f"Initial Clean GNSS Heading: {clean_gnss_heading[0]}, Fused Heading: {fused_heading[0]}")

    plot_signals_over_time(gnss_time, clean_gnss_heading, fused_heading, 'Clean GNSS Heading', 'Fused Heading')
    plot_signals_over_time(gnss_time, clean_gnss_heading, mag_calc_heading, 'Clean GNSS Heading', 'Mag Calc Heading')
    plot_signals_over_time(gnss_time, speed, imu_vel, 'GNSS Speed', 'IMU Forward Velocity')


    # for angle in range(18, 10, -2):
    #     imu_vel = calculate_imu_forward_velocity(acc_bundle, time_input, speed, angle)
    #     imu_vel = np.interp(gnss_time, time_input, imu_vel)
    #     plot_signals_over_time(gnss_time, speed, imu_vel, 'GNSS Speed', f'IMU Forward Velocity: {angle} degrees')


    plt.show()

    return heading_diff_mean

    
    