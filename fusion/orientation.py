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
    # _, _, _, _, heading, headingAccuracy, _, _, gnss_system_time, _, _, _ = extractGNSSData(gnss_data)
    _, _, _, _, heading, headingAccuracy, _, _, gnss_system_time, _, = extractGNSSData(gnss_data)
    # setup for heading correction
    clean_gnss_heading = getBestHeading(heading, headingAccuracy)

    return clean_gnss_heading, gnss_system_time

def getBestHeading(gnssHeadingData, gnssHeadingAccuracyData):
    dataLength = len(gnssHeadingData)
    forward_loop = [None]*dataLength
    backward_loop = [None]*dataLength
    last_forward_good = None
    last_backward_good = None
     # iterate through data forward and backward to get the best heading
    for i in range(dataLength):
        forward_index = i
        backward_index = dataLength - 1 - i
        # Handle forward direction
        if gnssHeadingAccuracyData[forward_index] < GNSS_HEADING_ACCURACY_THRESHOLD:
            last_forward_good = gnssHeadingData[forward_index]
        forward_loop[forward_index] = last_forward_good
        # Handle backward direction
        if gnssHeadingAccuracyData[backward_index] < GNSS_HEADING_ACCURACY_THRESHOLD:
            last_backward_good = gnssHeadingData[backward_index]
        backward_loop[backward_index] = last_backward_good
    # correct the forward loop with the backward loop
    for i in range(dataLength):
        if forward_loop[i] == None:
            if backward_loop[i] != None:
                forward_loop[i] = backward_loop[i]
            else:
                forward_loop[i] = gnssHeadingData[i]
    return forward_loop
    

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
    print(f"IMU data length: {len(imu_data)}, Mag data length: {len(mag_data)}, GNSS data length: {len(gnss_data)}")

    # Trim beginning until vehicle is at rest
    starting_index = 0
    STOPPED_POINTS = 25
    counter = 0
    for i in range(len(gnss_data)):
        if gnss_data[i].speed < GNSS_LOW_SPEED_THRESHOLD:
            counter += 1
            if counter > STOPPED_POINTS:
                starting_index = i - STOPPED_POINTS
                break
        else:
            counter = 0
        
    print(f"Starting index: {starting_index}")
    # Extract the data from the objects
    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, imu_time, imu_freq = extractAndSmoothImuData(imu_data, gnss_data[starting_index].system_time)
    mag_x, mag_y, mag_z, mag_time, mag_freq = extractAndSmoothMagData(mag_data, gnss_data[starting_index].system_time)
    lats, lons, alts, speed, heading, headingAccuracy, hdop, gdop, gnss_system_time, gnssFreq = extractGNSSData(gnss_data, gnss_data[starting_index].system_time)
    clean_gnss_heading = getBestHeading(heading, headingAccuracy)
    # print(f"clean heading length: {len(clean_gnss_heading)}, heading length: {len(heading)}")
    # print(f" GNSS initial time: {gnss_time[0]}, IMU initial time: {imu_time[0]}, Mag initial time: {mag_time[0]}")




    # downsample the data to match GNSS frequency
    acc_x_down = np.interp(gnss_system_time, imu_time, acc_x)
    acc_y_down = np.interp(gnss_system_time, imu_time, acc_y)
    acc_z_down = np.interp(gnss_system_time, imu_time, acc_z)
    gyro_x_down = np.interp(gnss_system_time, imu_time, gyro_x)
    gyro_y_down = np.interp(gnss_system_time, imu_time, gyro_y)
    gyro_z_down = np.interp(gnss_system_time, imu_time, gyro_z)
    # mag_x_down = np.interp(gnss_time, mag_time, mag_x)
    # mag_y_down = np.interp(gnss_time, mag_time, mag_y)
    # mag_z_down = np.interp(gnss_time, mag_time, mag_z)


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

    # import sys
    # sys.path.insert(0, '/Users/rogerberman/sensor-fusion/testingScripts')  # Add the project root to the Python path
    # from testingScripts.plottingCode import plot_signal_over_time, plot_signals_over_time
    # plot_signal_over_time(time_input, acc_x_input, 'Accel X')
    # plot_signal_over_time(time_input, acc_y_input, 'Accel Y')
    # plot_signal_over_time(time_input, acc_z_input, 'Accel Z')
    # plot_signal_over_time(time_input, gyro_x_input, 'Gyro X')
    # plot_signal_over_time(time_input, gyro_y_input, 'Gyro Y')
    # plot_signal_over_time(time_input, gyro_z_input, 'Gyro Z')

    # print(f"Accel offsets: {acc_x_down_zero_speed_avg}, {acc_y_down_zero_speed_avg}, {acc_z_down_zero_speed_avg}")
    # print(f"Gyro offsets: {gyro_x_down_zero_speed_avg}, {gyro_y_down_zero_speed_avg}, {gyro_z_down_zero_speed_avg}")

    # Calibrate Mag
    mag_bundle = np.array(list(zip(mag_x_input, mag_y_input, mag_z_input)))
    calibrated_mag_bundle = calibrate_mag(mag_bundle)


    acc_bundle = np.array(list(zip(acc_x_input, acc_y_input, acc_z_input)))
    gyro_bundle = np.array(list(zip(gyro_x_input, gyro_y_input, gyro_z_input)))


    initialPosition = [lats[0], lons[0], alts[0]]
    for i in range(len(lats)):
        if headingAccuracy[i] < GNSS_HEADING_ACCURACY_THRESHOLD:
            initialPosition = [lats[i], lons[i], alts[i]]
            break

    initialHeading = clean_gnss_heading[0]
    P = None
    fused_heading, fused_pitch, fused_roll = calculateHeading(acc_bundle, gyro_bundle, calibrated_mag_bundle, time_input, initialHeading, initialPosition, freq_input, P)
    mag_calc_heading = calculate_mag_headings(calibrated_mag_bundle, acc_bundle, initialPosition)
    imu_vel = calculate_imu_forward_velocity(acc_bundle, time_input, speed, 0)

    # used to translate the fused heading from -180:180 to the correct range 0:360
    fused_heading = [heading_val + 360 if heading_val < 0 else heading_val for heading_val in fused_heading]
    fused_pitch = [pitch_val + 360 if pitch_val < 0 else pitch_val for pitch_val in fused_pitch]
    fused_roll = [roll_val + 360 if roll_val < 0 else roll_val for roll_val in fused_roll]

    fused_heading = np.interp(gnss_system_time, time_input, fused_heading)
    fused_pitch = np.interp(gnss_system_time, time_input, fused_pitch)
    fused_roll = np.interp(gnss_system_time, time_input, fused_roll)
    mag_calc_heading = np.interp(gnss_system_time, time_input, mag_calc_heading)
    imu_vel = np.interp(gnss_system_time, time_input, imu_vel)

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
    print(f"Mean Pitch: {np.mean(fused_pitch)}, Mean Roll: {np.mean(fused_roll)}")


    import sys
    sys.path.insert(0, '/Users/rogerberman/sensor-fusion/testingScripts')  # Add the project root to the Python path
    from testingScripts.plottingCode import plot_signal_over_time, plot_signals_over_time
    import matplotlib.pyplot as plt
    # plot_signal_over_time(number_of_points, mean_diff, 'Clean Heading Diff Mean')
    # plot_signal_over_time(gnss_system_time, clean_heading_diff, 'Clean Heading Diff')
    # print(f"Initial Clean GNSS Heading: {clean_gnss_heading[0]}, Fused Heading: {fused_heading[0]}")


    plot_signals_over_time(gnss_system_time, clean_gnss_heading, mag_calc_heading, 'Clean GNSS Heading', 'Mag Calc Heading')
    # plot_signals_over_time(gnss_system_time, speed, imu_vel, 'GNSS Speed', 'IMU Forward Velocity')
    # plot_signals_over_time(gnss_system_time, clean_gnss_heading, fused_heading, 'Clean GNSS Heading', 'Fused Heading')
    # plot_signal_over_time(gnss_system_time, fused_pitch, 'Pitch')
    # plot_signal_over_time(gnss_system_time, fused_roll, 'Roll')

    # for angle in range(18, 10, -2):
    #     imu_vel = calculate_imu_forward_velocity(acc_bundle, time_input, speed, angle)
    #     imu_vel = np.interp(gnss_system_time, time_input, imu_vel)
    #     plot_signals_over_time(gnss_system_time, speed, imu_vel, 'GNSS Speed', f'IMU Forward Velocity: {angle} degrees')


    plt.show()

    return heading_diff_mean

    
    