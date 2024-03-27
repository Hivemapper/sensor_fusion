import sqlite3

DATA_LOGGER_PATH = './data-logger.v1.4.4-gnss.db'

class SqliteInterface:
    def __init__(self) -> None:
        self.connection = sqlite3.connect(DATA_LOGGER_PATH)
        self.cursor = self.connection.cursor()

    def queryImu(self, since, until):
        query = f'''
                    SELECT time, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z 
                    FROM imu 
                    WHERE time > '{since}'
                '''
        if until is not None:
            query += f' AND time < \'{until}\''

        results = self.cursor.execute(query).fetchall()
        return results
    
    def queryMagnetometer(self, since, until):
        query = f'''
                    SELECT system_time, mag_x, mag_y, mag_z 
                    FROM magnetometer
                    WHERE system_time > '{since}'
                '''
        if until is not None:
            query += f' AND system_time < \'{until}\''

        results = self.cursor.execute(query).fetchall()
        return results
    
    
    
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

def getImuData(since, until=None):
    return sqliteInterface.queryImu(since, until)

def getMagnetometerData(since, until=None):
    return sqliteInterface.queryMagnetometer(since, until)

def getGnssData(since, until=None):
    return sqliteInterface.queryGnss(since, until)
