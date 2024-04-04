import sqlite3
from datetime import datetime

DATA_LOGGER_PATH = '/data/recording/data-logger.v1.4.4.db'

class IMUData():
    def __init__(self, ax, ay, az, gx, gy, gz):
        self.ax = ax
        self.ay = ay
        self.az = az
        self.gx = gx
        self.gy = gy
        self.gz = gz

    def getAccel(self):
        return [self.ax, self.ay, self.az]

    def getGyro(self):
        return [self.gx, self.gy, self.gz]

class MagnetometerData():
    def __init__(self, mx, my, mz):
        self.mx = mx
        self.my = my
        self.mz = mz
    
    def getMag(self):
        return [self.mx, self.my, self.mz]

class SqliteInterface:
    def __init__(self) -> None:
        self.connection = sqlite3.connect(DATA_LOGGER_PATH)
        self.cursor = self.connection.cursor()

    def queryImu(self, desiredTime: int, pastRange: int = 250):
        """ 
        Queries the IMU table for accelerometer and gyroscope data for a given epoch timestamp.
        Args:
            desiredTime (int): The epoch timestamp to query for sensor data.
            pastRange (int, optional): The number of seconds before desiredTime to query for. Defaults to 250(quarter of a second). 
        Returns:
            list: A list of IMUData objects containing the accelerometer and gyroscope data.
        """
        query = f'''
                    SELECT acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z 
                    FROM imu 
                    WHERE time > \'{datetime.fromtimestamp((desiredTime - pastRange)/1000.0)}\'
                    AND time <= \'{datetime.fromtimestamp(desiredTime/1000.0)}\'
                    ORDER BY time DESC
                '''
        rows = self.cursor.execute(query).fetchall()
        results = [IMUData(row[0], row[1], row[2], row[3], row[4], row[5]) for row in rows]
        return results
    
    # desiredTime is a epoch timestamp
    # pastRange is the number of seconds before desiredTime to query for defaults tp 250 which is a quarter of a second
    def queryMagnetometer(self, desiredTime: int, pastRange: int = 250):
        """
        Queries the magnetometer table for magnetometer data for a given epoch timestamp.
        Args:
            desiredTime (int): The epoch timestamp to query for sensor data.
            pastRange (int, optional): The number of seconds before desiredTime to query for. Defaults to 250(quarter of a second).
        Returns:
            list: A list of MagnetometerData objects containing the magnetometer data.
        """
        query = f'''
                    SELECT mag_x, mag_y, mag_z 
                    FROM magnetometer
                    WHERE system_time > \'{datetime.fromtimestamp((desiredTime - pastRange)/1000.0)}\'
                    AND system_time <= \'{datetime.fromtimestamp(desiredTime/1000.0)}\'
                    ORDER BY system_time DESC
                '''
        rows = self.cursor.execute(query).fetchall()
        results = [MagnetometerData(row[0], row[1], row[2]) for row in rows]
        return results
    
    
    # TODO: Implement if needed, this is a skeleton for the gnss query
    def queryGnss(self, since, until):
        query = f'''
                    SELECT *
                    FROM gnss 
                    WHERE system_time > '{since}'
                '''
        if until is not None:
            query += f' AND system_time < \'{until}\''

        results = self.cursor.execute(query).fetchall()
        return results

sqliteInterface = SqliteInterface()

def getImuData(desiredTime: int):
    return sqliteInterface.queryImu(desiredTime)

def getMagnetometerData(desiredTime: int):
    return sqliteInterface.queryMagnetometer(desiredTime)

#TODO: Fully implement this function
def getGnssData(since, until=None):
    return sqliteInterface.queryGnss(since, until)


################ Helper Functions ################

def calculate_average(data_list):
    if not data_list:
        return None
    
    num_instances = len(data_list)
    sum_values = {}
    attribute_names = [attr for attr in dir(data_list[0]) if not attr.startswith("__") and not callable(getattr(data_list[0], attr))]
    
    # Initialize sum_values dictionary
    for attr in attribute_names:
        sum_values[attr] = 0
    
    # Sum up values for each attribute
    for instance in data_list:
        for attr in attribute_names:
            sum_values[attr] += getattr(instance, attr)
    
    # Calculate the average for each attribute
    avg_values = {attr: total / num_instances for attr, total in sum_values.items()}
    
    return avg_values