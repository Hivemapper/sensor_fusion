import fusion
import time
import os
import sys
import json


################ Main Functions ################

def dataRecording(filename: str):
    filePath = os.path.join('/data/sensorExp', filename+'.json')
    imuFilePath = os.path.join('/data/sensorExp', filename+'_imu.json')
    magFilePath = os.path.join('/data/sensorExp', filename+'_mag.json')
    while True:
        now = int(time.time()*1000) - 500
        euler, imu, mag = fusion.getEulerAngle(now)
        roll = euler[0]
        pitch = -1*euler[1]
        yaw = euler[2]
        eulerDict = {'roll': roll, 'pitch': pitch, 'yaw': yaw, 'time': now}
        print(f"Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")
        imu_now = fusion.calculate_average(imu)
        imu_now['time'] = now
        mag_now = fusion.calculate_average(mag)
        mag_now['time'] = now
        record_data(filePath, eulerDict)
        record_data(imuFilePath, imu_now)
        record_data(magFilePath, mag_now)
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

def printIsUpsideDown():
    while True:
        now = int(time.time()*1000) - 500
        print(f"Device is upside down: {fusion.isUpsideDown(now)}")
        time.sleep(0.1)

################ Helper Functions ################
def record_data(filename, data_dict):
    try:
        # Open the file in append mode
        with open(filename, 'a') as file:
            # Write the JSON data to the file
            json.dump(data_dict, file)
            file.write('\n')  # Add a newline character to separate each JSON object
        # print("Data recorded successfully.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

################# Whats on #################

# Record Data using 
if __name__ == "__main__":
    ##### Run the data recording function #####
    # if len(sys.argv) != 2:
    #     print("Usage: python script_name.py filename")
    #     sys.exit(1)
    # dataRecording(sys.argv[1])

    ##### UNCOMMMENT THE FUNCTION YOU WISH TO USE #####
    # printAccelData()
    # printGyroData()
    # printMagData()
    printIsUpsideDown()
