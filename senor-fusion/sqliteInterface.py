import sqlite3
from conversions import convertTimeToEpoch, convertEpochToTime
import time
from functools import wraps

# For SQLite interface
DATA_LOGGER_PATH = "/data/recording/data-logger.v1.4.5.db"
DESC = "DESC"
ASC = "ASC"


class IMUData:
    def __init__(
        self, ax, ay, az, gx, gy, gz, system_time, temperature, session, row_id
    ):
        self.ax = ax
        self.ay = ay
        self.az = az
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.system_time = system_time
        self.temperature = temperature
        self.session = session
        self.row_id = row_id

    def to_dict(self):
        return {
            "acc_x": self.ax,
            "acc_y": self.ay,
            "acc_z": self.az,
            "gyro_x": self.gx,
            "gyro_y": self.gy,
            "gyro_z": self.gz,
            "time": self.system_time,
            "temperature": self.temperature,
            "session": self.session,
            "row_id": self.row_id,
        }


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


# Function decorator to retry a function a specified number of times with a delay between each attempt
def retry(max_attempts: int = 3, delay: float = 0.2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise e
                    time.sleep(delay)

        return wrapper

    return decorator


class SqliteInterface:
    def __init__(self, data_logger_path: str = DATA_LOGGER_PATH) -> None:
        self.connection = sqlite3.connect(data_logger_path)
        self.cursor = self.connection.cursor()

    @retry()
    def create_processed_imu_table(self):
        """
        Creates the processed tables in the database if they do not already exist.
        """
        try:
            # SQL command to create the imu table if it doesn't already exist
            create_imu_table_sql = """
            CREATE TABLE IF NOT EXISTS imu_processed (
                        id INTEGER NOT NULL PRIMARY KEY,
                        row_id INTEGER NOT NULL,
                        time TIMESTAMP NOT NULL,
                        acc_x REAL NOT NULL,
                        acc_y REAL NOT NULL,
                        acc_z REAL NOT NULL,
                        gyro_x REAL NOT NULL,
                        gyro_y REAL NOT NULL,
                        gyro_z REAL NOT NULL,
                        stationary REAL NOT NULL,
                        temperature REAL,
                        session TEXT NOT NULL DEFAULT ''
            );
            """
            # Execute the SQL command to create the imu table
            self.cursor.execute(create_imu_table_sql)

            # Commit the changes to the database
            self.connection.commit()

        except sqlite3.Error as e:
            # Handle any SQLite errors
            print(f"An error occurred while creating the imu table: {e}")
            self.connection.rollback()

    @retry()
    def create_error_log_table(self):
        """
        Creates the error log table in the database if it does not already exist.
        """
        try:
            # SQL command to create the error log table if it doesn't already exist
            create_error_log_table_sql = """
            CREATE TABLE IF NOT EXISTS python_layer_error_logs (
                        id INTEGER NOT NULL PRIMARY KEY,
                        system_time TIMESTAMP NOT NULL,
                        error_type TEXT NOT NULL,
                        error_message TEXT NOT NULL
            );
            """
            # Execute the SQL command to create the error log table
            self.cursor.execute(create_error_log_table_sql)

            # Commit the changes to the database
            self.connection.commit()

        except sqlite3.Error as e:
            # Handle any SQLite errors
            print(f"An error occurred while creating the error log table: {e}")
            self.connection.rollback()

    @retry()
    def find_starting_row_id(self, table_name):
        """
        Finds the starting row ID for a given table in the database.

        Parameters:
        table_name (str): The name of the table to find the starting row ID.

        Returns:
        int: The starting row ID, or None if the table is empty or an error occurs.
        """
        try:
            # SQL command to find the minimum row ID in the specified table
            query = f"SELECT MIN(id) FROM {table_name};"

            # Execute the SQL command
            self.cursor.execute(query)

            # Fetch the result
            result = self.cursor.fetchone()

            # Return the starting row ID if it exists, otherwise return None
            starting_row_id = result[0] if result and result[0] is not None else None

            return starting_row_id

        except sqlite3.Error as e:
            # Handle any SQLite errors
            print(
                f"An error occurred while finding the starting row ID for table {table_name}: {e}"
            )
            return None

    @retry()
    def find_most_recent_row_id(self, table_name):
        """
        Finds the most recent (maximum) row ID for a given table in the database.

        Parameters:
        table_name (str): The name of the table to find the most recent row ID.

        Returns:
        int: The most recent row ID, or None if the table is empty or an error occurs.
        """
        try:
            # SQL command to find the maximum row ID in the specified table
            query = f"SELECT MAX(id) FROM {table_name};"

            # Execute the SQL command
            self.cursor.execute(query)

            # Fetch the result
            result = self.cursor.fetchone()

            # Return the most recent row ID if it exists, otherwise return None
            most_recent_row_id = result[0] if result and result[0] is not None else None

            return most_recent_row_id

        except sqlite3.Error as e:
            # Handle any SQLite errors
            print(
                f"An error occurred while finding the most recent row ID for table {table_name}: {e}"
            )
            return None

    @retry()
    def get_raw_imu_by_row_range(self, start_id: int, end_id: int, order: str = "ASC"):
        """
        Queries the IMU table for accelerometer and gyroscope data for rows within a specified ID range
        and sorts by row ID.
        Columns queried: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, time, temperature, session.

        Args:
            start_id (int): The starting row ID for the query.
            end_id (int): The ending row ID for the query.
            order (str, optional): The order of retrieval, either 'ASC' or 'DESC'. Defaults to 'ASC'.

        Returns:
            list: A list of IMUData objects containing the accelerometer and gyroscope data.
        """
        try:
            query = f"""
                        SELECT acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, time, temperature, session, id
                        FROM imu
                        WHERE id BETWEEN ? AND ?
                        ORDER BY id {order}
                    """
            rows = self.cursor.execute(query, (start_id, end_id)).fetchall()
            results = [
                IMUData(
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
                )
                for row in rows
            ]
            return results

        except sqlite3.Error as e:
            print(
                f"An error occurred while querying the IMU table for rows {start_id} to {end_id}: {e}"
            )
            return []

    @retry()
    def get_gnss_by_row_range(self, start_id: int, end_id: int, order: str = "ASC"):
        """
        Queries the GNSS table for GNSS data for rows within a specified ID range
        and sorts by row ID.
        Columns queried: latitude, longitude, altitude, speed, heading, heading_accuracy, hdop, gdop, system_time, time, time_resolved, session.

        Args:
            start_id (int): The starting row ID for the query.
            end_id (int): The ending row ID for the query.
            order (str, optional): The order of retrieval, either 'ASC' or 'DESC'. Defaults to 'ASC'.

        Returns:
            list: A list of GNSSData objects containing the GNSS data.
        """
        try:
            query = f"""
                        SELECT latitude, longitude, altitude, speed, heading, heading_accuracy, hdop, gdop, system_time, time, time_resolved, session, id
                        FROM gnss
                        WHERE id BETWEEN ? AND ?
                        ORDER BY id {order}
                    """
            rows = self.cursor.execute(query, (start_id, end_id)).fetchall()
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
                for row in rows
            ]
            return results

        except sqlite3.Error as e:
            print(
                f"An error occurred while querying the GNSS table for rows {start_id} to {end_id}: {e}"
            )
            return []

    @retry()
    def insert_processed_imu_data(self, processed_data):
        """
        Inserts one or multiple rows of IMU data into the imu table.

        Args:
            data (list): A list of dictionaries, each containing the data for a row.

        Returns:
            bool: True if the insert was successful, False otherwise.
        """
        try:
            insert_query = """
                INSERT INTO imu_processed (row_id, time, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, stationary, temperature, session)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            # Prepare data for insertion
            data_to_insert = [
                (
                    entry["row_id"],
                    entry["time"],
                    entry["acc_x"],
                    entry["acc_y"],
                    entry["acc_z"],
                    entry["gyro_x"],
                    entry["gyro_y"],
                    entry["gyro_z"],
                    entry["stationary"],
                    entry["temperature"],
                    entry["session"],
                )
                for entry in processed_data
            ]
            self.cursor.executemany(insert_query, data_to_insert)
            self.connection.commit()
            return True
        except sqlite3.Error as e:
            print(
                f"An error occurred while inserting data into the processed_data table: {e}"
            )
            self.connection.rollback()
            return False

    @retry()
    def check_table_exists(self, table_name):
        """
        Checks if a given table exists in the database.

        Parameters:
        table_name (str): The name of the table to check for existence.

        Returns:
        bool: True if the table exists, False otherwise.
        """
        try:
            query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?;"
            self.cursor.execute(query, (table_name,))
            result = self.cursor.fetchone()
            return result is not None
        except sqlite3.Error as e:
            print(f"An error occurred while checking if table {table_name} exists: {e}")
            return False

    @retry()
    def drop_table(self, table_name):
        """
        Drops the specified table from the database if it exists.
        """
        try:
            # SQL command to drop the table if it exists
            drop_table_sql = f"DROP TABLE IF EXISTS {table_name};"

            # Execute the SQL command to drop the table
            self.cursor.execute(drop_table_sql)

            # Commit the changes to the database
            self.connection.commit()
            print(f"Table '{table_name}' dropped successfully.")

        except sqlite3.Error as e:
            # Handle any SQLite errors
            print(f"An error occurred while dropping the table '{table_name}': {e}")
            self.connection.rollback()

    @retry()
    def get_nearest_row_id_to_time(
        self, tableName: str, desiredTime: str, session: str = None
    ):
        """
        Queries the specified table to retrieve the row id of the nearest row to the given time,
        optionally within a specified session.
        Args:
            tableName (str): The name of the table to query.
            desiredTime (str): The desired time in the correct format.
            session (str, optional): The session identifier. Defaults to None.
        Returns:
            int: The row id of the nearest row to the given time.
        """
        timeVariable = "system_time" if tableName != "imu" else "time"

        if session:
            query = f"""
                    SELECT id FROM {tableName}
                    WHERE session = ?
                    ORDER BY ABS(strftime('%s', {timeVariable}) - strftime('%s', ?))
                    LIMIT 1
                    """
            params = (session, desiredTime)
        else:
            query = f"""
                    SELECT id FROM {tableName}
                    ORDER BY ABS(strftime('%s', {timeVariable}) - strftime('%s', ?))
                    LIMIT 1
                    """
            params = (desiredTime,)

        nearest_row_id = self.cursor.execute(query, params).fetchone()
        return nearest_row_id[0] if nearest_row_id else None

    @retry()
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

    @retry()
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

    @retry()
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

    @retry()
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

    @retry()
    def log_error(self, error_type: str, error: str):
        """
        Writes an error message to the error table in the database.
        Args:
            service (str): The name of the service that encountered the error.
            error (str): The error message.
        """
        try:
            query = "INSERT INTO python_layer_error_logs (system_time, error_type, error_message) VALUES (?, ?, ?);"
            self.cursor.execute(query, (time.time(), error_type, error))
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"An error occurred while writing to the error table: {e}")
            self.connection.rollback()

    ##### Functions below are a bit outdated as of now but can be used for reference #####

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
