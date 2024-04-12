import fusion
import time


################ Main Functions ################
def printEulerAngle():
    while True:
        now = int(time.time()*1000) - 500
        roll, pitch, yaw = fusion.getEulerAngle(now)
        print(f"Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")
        time.sleep(0.1)

def printAccelData():
    while True:
        now = int(time.time()*1000) - 500
        imu = fusion.getImuData(now)
        imu_now = fusion.calculate_average(imu)
        print(f"accel_x: {imu_now['ax']}, accel_y:{imu_now['ay']}, accel_z:{imu_now['az']}")
        time.sleep(0.1)

def printGyroData():
    while True:
        now = int(time.time()*1000) - 500
        imu = fusion.getImuData(now)
        imu_now = fusion.calculate_average(imu)
        print(f"gyro_x: {imu_now['gx']}, gyro_y:{imu_now['gy']}, gyro_z:{imu_now['gz']}")
        time.sleep(0.1)

def printMagData():
    while True:
        now = int(time.time()*1000) - 500
        mag = fusion.getMagnetometerData(now)
        mag_now = fusion.calculate_average(mag)
        print(f"mag_x: {mag_now['mx']}, mag_y:{mag_now['my']}, mag_z:{mag_now['mz']}")
        time.sleep(0.1)

def printGNSSData():
    while True:
        now = int(time.time()*1000) - 500
        gnss = fusion.getGnssData(now)
        if len(gnss) == 0:
            # print("No GNSS data available")
            continue
        gnss_now = fusion.calculate_average(gnss)
        print(f"speed:{gnss_now['speed']}, heading:{gnss_now['heading']}, headingAccuracy:{gnss_now['headingAccuracy']}")
        time.sleep(0.1)

def printIsUpsideDown():
    while True:
        now = int(time.time()*1000) - 500
        print(f"Device is upside down: {fusion.isUpsideDown(now)}")
        time.sleep(0.1)

def printDashcamToVehicleHeadingOffset():
    fusion.getDashcamToVehicleHeadingOffset()

################# Whats on #################

# Record Data using 
if __name__ == "__main__":
    ##### UNCOMMMENT THE FUNCTION YOU WISH TO USE #####
    # printEulerAngle()
    # printAccelData()
    # printGyroData()
    # printMagData()
    # printGNSSData()
    # printIsUpsideDown()
    printDashcamToVehicleHeadingOffset()
