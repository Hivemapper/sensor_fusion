import sqlite3
from datetime import datetime

DATA_LOGGER_PATH = '/data/recording/data-logger.v1.4.4.db'
DESC = 'DESC'
ASC = 'ASC'

class IMUData():
    def __init__(self, ax, ay, az, gx, gy, gz, time):
        self.ax = ax
        self.ay = ay
        self.az = az
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.time = time

    def getAccel(self):
        return [self.ax, self.ay, self.az]

    def getGyro(self):
        return [self.gx, self.gy, self.gz]

class MagData():
    def __init__(self, mx, my, mz, time):
        self.mx = mx
        self.my = my
        self.mz = mz
        self.time = time
    
    def getMag(self):
        return [self.mx, self.my, self.mz]
    
class GNSSData():
    def __init__(self, lat, lon, alt, speed, heading, headingAccuracy, hdop, gdop, time):
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.speed = speed
        self.heading = heading
        self.headingAccuracy = headingAccuracy
        self.hdop = hdop
        self.gdop = gdop
        self.time = time

    def getLatLon(self):
        return [self.lat, self.lon]

    def getAltitude(self):
        return self.alt

    def getSpeed(self):
        return self.speed

    def getHeading(self):
        return self.heading
    
    def getHeadingAccuracy(self):
        return self.headingAccuracy

    def getHdop(self):
        return self.hdop
    
    def getGdop(self):
        return self.gdop
    


class SqliteInterface:
    def __init__(self) -> None:
        self.connection = sqlite3.connect(DATA_LOGGER_PATH)
        self.cursor = self.connection.cursor()

    def queryImu(self, desiredTime: int, pastRange: int, order: str):
        """ 
        Queries the IMU table for accelerometer and gyroscope data for a given epoch timestamp.
        Args:
            desiredTime (int): The epoch timestamp to query for sensor data.
            pastRange (int, optional): The number of seconds before desiredTime to query for. Defaults to 250(quarter of a second). 
        Returns:
            list: A list of IMUData objects containing the accelerometer and gyroscope data.
        """
        query = f'''
                    SELECT acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, time
                    FROM imu 
                    WHERE time > \'{datetime.fromtimestamp((desiredTime - pastRange)/1000.0)}\'
                    AND time <= \'{datetime.fromtimestamp(desiredTime/1000.0)}\'
                    ORDER BY time {order}
                '''
        rows = self.cursor.execute(query).fetchall()
        results = [IMUData(row[0], row[1], row[2], row[3], row[4], row[5], row[6]) for row in rows]
        return results
    
    # desiredTime is a epoch timestamp
    # pastRange is the number of seconds before desiredTime to query for defaults tp 250 which is a quarter of a second
    def queryMagnetometer(self, desiredTime: int, pastRange: int, order: str):
        """
        Queries the magnetometer table for magnetometer data for a given epoch timestamp.
        Args:
            desiredTime (int): The epoch timestamp to query for sensor data.
            pastRange (int, optional): The number of seconds before desiredTime to query for. Defaults to 250(quarter of a second).
        Returns:
            list: A list of MagnetometerData objects containing the magnetometer data.
        """
        query = f'''
                    SELECT mag_x, mag_y, mag_z, system_time
                    FROM magnetometer
                    WHERE system_time > \'{datetime.fromtimestamp((desiredTime - pastRange)/1000.0)}\'
                    AND system_time <= \'{datetime.fromtimestamp(desiredTime/1000.0)}\'
                    ORDER BY system_time {order}
                '''
        rows = self.cursor.execute(query).fetchall()
        results = [MagData(row[0], row[1], row[2], row[3]) for row in rows]
        return results
    
    def queryGnss(self, desiredTime: int, pastRange: int, order: str):
        query = f'''
                    SELECT latitude, longitude, altitude, speed, heading, heading_accuracy, hdop, gdop, system_time
                    FROM gnss 
                    WHERE system_time > \'{datetime.fromtimestamp((desiredTime - pastRange)/1000.0)}\'
                    AND system_time <= \'{datetime.fromtimestamp(desiredTime/1000.0)}\'
                    ORDER BY system_time {order}
                '''
        results = self.cursor.execute(query).fetchall()
        results = [GNSSData(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]) for row in results]
        return results

# sqliteInterface = SqliteInterface()

# def getImuData(desiredTime: int, pastRange: int = 250, order: str = DESC):
#     return sqliteInterface.queryImu(desiredTime, pastRange, order)

# def getMagnetometerData(desiredTime: int, pastRange: int = 250, order: str = DESC):
#     return sqliteInterface.queryMagnetometer(desiredTime, pastRange, order)

# def getGnssData(desiredTime: int, pastRange: int = 250, order: str = DESC):
#     return sqliteInterface.queryGnss(desiredTime, pastRange, order)
