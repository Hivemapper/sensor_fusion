import sqlite3
from datetime import datetime, timezone

# For SQLite interface
DATA_LOGGER_PATH = "/data/recording/data-logger.v1.4.4.db"
DESC = "DESC"
ASC = "ASC"


### Time functions placed here to avoid circular imports insted of utils.py
def convertTimeToEpoch(time_str):
    """
    Converts a time string in the format 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD HH:MM:SS.sss' to epoch milliseconds.
    Parameters:
    - time_str: A string representing the time, possibly with milliseconds ('YYYY-MM-DD HH:MM:SS[.sss]').
    Returns:
    - int: The epoch time in milliseconds.
    """
    # Determine if the time string includes milliseconds
    if "." in time_str:
        format_str = "%Y-%m-%d %H:%M:%S.%f"
    else:
        format_str = "%Y-%m-%d %H:%M:%S"

    timestamp_dt = datetime.strptime(time_str, format_str)
    epoch_ms = int(timestamp_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    return epoch_ms


def convertEpochToTime(epoch_ms):
    """
    Converts an epoch time in milliseconds to a time string in the format 'YYYY-MM-DD HH:MM:SS.sss'.

    Parameters:
    - epoch_ms: The epoch time in milliseconds.

    Returns:
    - str: A string representing the time in the format 'YYYY-MM-DD HH:MM:SS.sss'.
    """
    # Convert milliseconds to seconds
    epoch_s = epoch_ms / 1000.0
    # Convert to datetime object
    datetime_obj = datetime.fromtimestamp(epoch_s, tz=timezone.utc)
    return datetime_obj.strftime("%Y-%m-%d %H:%M:%S.%f")


class IMUData:
    def __init__(self, ax, ay, az, gx, gy, gz, time, temperature, session):
        self.ax = ax
        self.ay = ay
        self.az = az
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.time = time
        self.temperature = temperature
        self.session = session


class ProcessedIMUData:
    def __init__(
        self,
        ax,
        ay,
        az,
        gx,
        gy,
        gz,
        time,
        temperature,
        session,
        stationary,
        row_id,
    ):
        self.ax = ax
        self.ay = ay
        self.az = az
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.time = time
        self.temperature = temperature
        self.session = session
        self.stationary = stationary
        self.row_id = row_id


class MagData:
    def __init__(self, mx, my, mz, time, session):
        self.mx = mx
        self.my = my
        self.mz = mz
        self.time = time
        self.session = session


class GNSSData:
    def __init__(
        self,
        lat,
        lon,
        alt,
        speed,
        heading,
        headingAccuracy,
        hdop,
        gdop,
        system_time,
        time,
        time_resolved,
        session,
    ):
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.speed = speed
        self.heading = heading
        self.headingAccuracy = headingAccuracy
        self.hdop = hdop
        self.gdop = gdop
        self.system_time = system_time
        self.time = time
        self.time_resolved = time_resolved
        self.session = session


class SqliteInterface:
    def __init__(self, data_logger_path: str = DATA_LOGGER_PATH) -> None:
        self.connection = sqlite3.connect(data_logger_path)
        self.cursor = self.connection.cursor()

    def queryAllImu(self, order: str = "ASC"):
        """
        Queries the whole IMU table for accelerometer and gyroscope data and sorts by row ID.
        Columns queried acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, time
        Args:
            order (str, optional): The order of retrieval, either 'ASC' or 'DESC'. Defaults to 'DESC'.
        Returns:
            list: A list of IMUData objects containing the accelerometer and gyroscope data.
        """
        query = f"""
                    SELECT acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, time, temperature, session
                    FROM imu 
                    ORDER BY id {order}
                """
        rows = self.cursor.execute(query).fetchall()
        results = [
            IMUData(
                row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]
            )
            for row in rows
        ]
        return results

    def queryAllProcessedImu(self, order: str = "ASC"):
        """
        Queries the whole IMU table for accelerometer and gyroscope data and sorts by row ID.
        Columns queried acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, time
        Args:
            order (str, optional): The order of retrieval, either 'ASC' or 'DESC'. Defaults to 'DESC'.
        Returns:
            list: A list of ProcessedIMUData objects containing the accelerometer and gyroscope data.
        """
        query = f"""
                    SELECT acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, time, temperature, session, stationary, id
                    FROM imu_processed 
                    ORDER BY id {order}
                """
        rows = self.cursor.execute(query).fetchall()
        results = [
            ProcessedIMUData(
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                row[6],
                row[7],
                row[8],
                row[9],
                row[10],
            )
            for row in rows
        ]
        return results

    def queryAllMagnetometer(self, order: str = "ASC"):
        """
        Queries the magnetometer table for magnetometer data and sorts by row ID.
        Columns queried mag_x, mag_y, mag_z, system_time
        Args:
            order (str, optional): The order of retrieval, either 'ASC' or 'DESC'. Defaults to 'DESC'.
        Returns:
            list: A list of MagnetometerData objects containing the magnetometer data.
        """
        query = f"""
                    SELECT mag_x, mag_y, mag_z, system_time, session
                    FROM magnetometer
                    ORDER BY id {order}
                """
        rows = self.cursor.execute(query).fetchall()
        results = [MagData(row[0], row[1], row[2], row[3], row[4]) for row in rows]
        return results

    def queryAllGnss(self, order: str = "ASC"):
        """
        Queries the GNSS table for all GNSS data and sorts by row ID.
        Columns queried latitude, longitude, altitude, speed, heading, heading_accuracy, hdop, gdop, system_time, time, time_resolved, session
        Args:
            order (str, optional): The order of retrieval, either 'ASC' or 'DESC'. Defaults to 'DESC'.
        Returns:
            list: A list of GNSSData objects containing the GNSS data.
        """
        query = f"""
                    SELECT latitude, longitude, altitude, speed, heading, heading_accuracy, hdop, gdop, system_time, time, time_resolved, session
                    FROM gnss 
                    ORDER BY id {order}
                """
        results = self.cursor.execute(query).fetchall()
        results = [
            GNSSData(
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                row[6],
                row[7],
                row[8],
                row[9],
                row[10],
                row[11],
            )
            for row in results
        ]
        return results

    def table_exists(self, tableName):
        """
        Checks if the given table exists in the database.
        Args:
            table_name (str): The name of the table to check.
        Returns:
            bool: True if the table exists, False otherwise.
        """
        # Query to check if the table exists in SQLite's internal schema
        query = (
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{tableName}'"
        )
        result = self.cursor.execute(query).fetchone()
        return result is not None

    def get_min_row_id(self, tableName):
        """
        Queries the specified table to retrieve the minimum row id.
        Args:
            tableName (str): The name of the table to query.
        Returns:
            int: The minimum row id of the table.
        """
        # Query to retrieve the minimum row id
        query = f"SELECT MIN(id) FROM {tableName}"
        min_row_id = self.cursor.execute(query).fetchone()
        return min_row_id[0] if min_row_id else None

    def get_min_max_system_time(self, tableName):
        """
        Queries the specified table to retrieve the smallest and largest values of the system_time column.
        Args:
            tableName (str): The name of the table to query.
        Returns:
            Tuple[datetime, datetime]: A tuple containing the smallest and largest system_time values.
        """
        timeVariable = "system_time"
        if tableName == "imu":
            timeVariable = "time"

        # Get the minimum row id
        min_row_id = self.get_min_row_id(tableName)

        # Query to retrieve the smallest system_time value corresponding to the minimum row id
        min_query = f"SELECT {timeVariable} FROM {tableName} WHERE id = {min_row_id}"
        min_result = self.cursor.execute(min_query).fetchone()
        min_system_time = min_result[0] if min_result else None

        # Query to retrieve the largest system_time value
        max_query = f"SELECT MAX({timeVariable}) FROM {tableName}"
        max_result = self.cursor.execute(max_query).fetchone()
        max_system_time = max_result[0] if max_result else None

        return min_system_time, max_system_time

    def get_all_rows_as_dicts(self, table_name):
        """
        Retrieves all rows from a given table in the database and returns them as a list of dictionaries.

        Parameters:
        table_name (str): The name of the table to retrieve the rows from.

        Returns:
        list: A list of dictionaries where each dictionary represents a row in the table with column names as keys.
        """
        try:
            # SQL command to select all rows from the specified table
            query = f"SELECT * FROM {table_name};"

            # Execute the SQL command
            self.cursor.execute(query)

            # Fetch all rows from the executed query
            rows = self.cursor.fetchall()

            # Get the column names from the cursor description
            column_names = [description[0] for description in self.cursor.description]

            # Convert rows to a list of dictionaries
            result = [dict(zip(column_names, row)) for row in rows]

            return result

        except sqlite3.Error as e:
            # Handle any SQLite errors
            print(
                f"An error occurred while retrieving rows from table {table_name}: {e}"
            )
            return []

    ##### Functions below are a bit outdated as of now but can be used for reference #####

    # def get_nearest_row_id_to_time(self, tableName, desiredTime):
    #     """
    #     Queries the specified table to retrieve the row id of the nearest row to the given time.
    #     Args:
    #         tableName (str): The name of the table to query.
    #         desiredTime (int): The desired time in milliseconds since the epoch.
    #     Returns:
    #         int: The row id of the nearest row to the given time.
    #     """
    #     # Assign correct identifier for time column
    #     timeVariable = 'system_time'
    #     if tableName == 'imu':
    #         timeVariable = 'time'

    #     # Convert desiredTime to datetime object, corrected variable name to 'desired'
    #     desired = convertEpochToTime(desiredTime)

    #     # Query to retrieve the row id of the nearest row to the given time
    #     query = f"""
    #             SELECT id FROM {tableName}
    #             ORDER BY ABS(strftime('%s', {timeVariable}) - strftime('%s', '{desired}'))
    #             LIMIT 1
    #             """
    #     nearest_row_id = self.cursor.execute(query).fetchone()
    #     return nearest_row_id[0] if nearest_row_id else None

    # def get_rows_between_ids(self, tableName, startRowId, endRowId, order='ASC'):
    #     """
    #     Queries the specified table to retrieve all rows between the given start and end row IDs, including those two IDs.
    #     Args:
    #         tableName (str): The name of the table to query.
    #         startRowId (int): The starting row ID.
    #         endRowId (int): The ending row ID.
    #         order (str): The order of retrieval, either 'ASC' or 'DESC'. Defaults to 'ASC'.
    #     Returns:
    #         list: A list of rows between the start and end row IDs, including those two IDs.
    #     """
    #     # desired columns for each table
    #     columns = '*'
    #     if tableName == 'imu':
    #         columns = 'acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, time'
    #     elif tableName == 'magnetometer':
    #         columns = 'mag_x, mag_y, mag_z, system_time'
    #     elif tableName == 'gnss':
    #         columns = 'latitude, longitude, altitude, speed, heading, heading_accuracy, hdop, gdop, system_time'

    #     query = f"""
    #             SELECT {columns} FROM {tableName}
    #             WHERE id <= {endRowId} AND id >= {startRowId}
    #             ORDER BY id {order}
    #         """

    #     # Execute the query and fetch the results
    #     rows = self.cursor.execute(query).fetchall()
    #     return rows

    # def get_max_row_id(self, tableName):
    #     """
    #     Queries the specified table to retrieve the maximum row id.
    #     Args:
    #         tableName (str): The name of the table to query.
    #     Returns:
    #         int: The maximum row id of the table.
    #     """
    #     # Query to retrieve the maximum row id
    #     query = f"SELECT MAX(id) FROM {tableName}"
    #     max_row_id = self.cursor.execute(query).fetchone()
    #     return max_row_id[0] if max_row_id else None

    # def queryImu(self, desiredTime: int, pastRange: int = 250, order: str = DESC):
    #     """
    #     Queries the IMU table for accelerometer and gyroscope data for a given epoch timestamp.
    #     IMPORTANT: This DOES NOT account for GNSS dropping causing the time to be off in this table at points
    #     Args:
    #         desiredTime (int): The epoch timestamp to query for sensor data.
    #         pastRange (int, optional): The number of seconds before desiredTime to query for. Defaults to 250(quarter of a second).
    #         order (str, optional): The order of retrieval, either 'ASC' or 'DESC'. Defaults to 'DESC'.
    #     Returns:
    #         list: A list of IMUData objects containing the accelerometer and gyroscope data.
    #     """
    #     min_time = convertEpochToTime(desiredTime - pastRange)
    #     max_time = convertEpochToTime(desiredTime)
    #     query = f'''
    #                 SELECT acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, time
    #                 FROM imu
    #                 WHERE time > \'{min_time}\'
    #                 AND time <= \'{max_time}\'
    #                 ORDER BY time {order}
    #             '''
    #     rows = self.cursor.execute(query).fetchall()
    #     results = [IMUData(row[0], row[1], row[2], row[3], row[4], row[5], row[6]) for row in rows]
    #     return results

    # def queryMagnetometer(self, desiredTime: int, pastRange: int = 250, order: str = DESC):
    #     """
    #     Queries the magnetometer table for magnetometer data for a given epoch timestamp.
    #     IMPORTANT: This DOES NOT account for GNSS dropping causing the time to be off in this table at points
    #     Args:
    #         desiredTime (int): The epoch timestamp to query for sensor data.
    #         pastRange (int, optional): The number of seconds before desiredTime to query for. Defaults to 250(quarter of a second).
    #         order (str, optional): The order of retrieval, either 'ASC' or 'DESC'. Defaults to 'DESC'.
    #     Returns:
    #         list: A list of MagnetometerData objects containing the magnetometer data.
    #     """
    #     min_time = convertEpochToTime(desiredTime - pastRange)
    #     max_time = convertEpochToTime(desiredTime)
    #     query = f'''
    #                 SELECT mag_x, mag_y, mag_z, system_time
    #                 FROM magnetometer
    #                 WHERE system_time > \'{min_time}\'
    #                 AND system_time <= \'{max_time}\'
    #                 ORDER BY system_time {order}
    #             '''
    #     rows = self.cursor.execute(query).fetchall()
    #     results = [MagData(row[0], row[1], row[2], row[3]) for row in rows]
    #     return results

    # def queryGnss(self, desiredTime: int, pastRange: int = 250, order: str = DESC):
    #     """
    #     Queries the GNSS table for GNSS data for a given epoch timestamp.
    #     IMPORTANT: GNSS dropping causes gaps in this data but otherwise the time is accurate
    #     Args:
    #         desiredTime (int): The epoch timestamp to query for sensor data.
    #         pastRange (int, optional): The number of seconds before desiredTime to query for. Defaults to 250(quarter of a second).
    #         order (str, optional): The order of retrieval, either 'ASC' or 'DESC'. Defaults to 'DESC'.
    #     Returns:
    #         list: A list of GNSSData objects containing the GNSS data.
    #     """
    #     min_time = convertEpochToTime(desiredTime - pastRange)
    #     max_time = convertEpochToTime(desiredTime)
    #     query = f'''
    #                 SELECT latitude, longitude, altitude, speed, heading, heading_accuracy, hdop, gdop, system_time
    #                 FROM gnss
    #                 WHERE system_time > \'{min_time}\'
    #                 AND system_time <= \'{max_time}\'
    #                 ORDER BY system_time {order}
    #             '''
    #     results = self.cursor.execute(query).fetchall()
    #     results = [GNSSData(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]) for row in results]
    #     return results
