import sqlite3
import os
import math
import time
from functools import wraps

env = os.getenv("HIVE_ENV")

if env == "local":
    from sensor_fusion.sensor_fusion_service.data_definitions import (
        IMUData,
        ProcessedIMUData,
        GNSSData,
        MagData,
        FusedPositionData,
        PURGE_GROUP,
    )
else:
    from data_definitions import (
        IMUData,
        ProcessedIMUData,
        GNSSData,
        MagData,
        FusedPositionData,
        PURGE_GROUP,
    )

# For SQLite interface
DATA_LOGGER_PATH = "/data/recording/data-logger.v1.4.5.db"
DESC = "DESC"
ASC = "ASC"

# Purging Constants
DB_SIZE_LIMIT = 1024 * 1024 * 200  # 200 MB


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
        self.data_logger_path = data_logger_path

    def get_db_size(self) -> int:
        """
        Returns the size of the SQLite database file in bytes.
        """
        try:
            return os.path.getsize(self.data_logger_path)
        except Exception as e:
            print(f"An error occurred: {e}")
            return -1

    ################# Table Creation Functions #################
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
    def create_service_log_table(self):
        """
        Creates the error log table in the database if it does not already exist.
        """
        try:
            # SQL command to create the error log table if it doesn't already exist
            create_error_log_table_sql = """
            CREATE TABLE IF NOT EXISTS sensor_fusion_logs (
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

    def create_fused_position_table(self):
        """
        Creates the fused position table in the database if it does not already exist.
        """
        try:
            # SQL command to create the fused position table if it doesn't already exist
            create_fused_position_table_sql = """
            CREATE TABLE IF NOT EXISTS fused_position (
                        id INTEGER NOT NULL PRIMARY KEY,
                        time TIMESTAMP NOT NULL,
                        gnss_lat REAL NOT NULL,
                        gnss_lon REAL NOT NULL,
                        fused_lat REAL NOT NULL,
                        fused_lon REAL NOT NULL,
                        fused_heading REAL NOT NULL,
                        forward_velocity REAL NOT NULL,
                        yaw_rate REAL NOT NULL,
                        session TEXT NOT NULL DEFAULT ''
            );
            """
            # Execute the SQL command to create the fused position table
            self.cursor.execute(create_fused_position_table_sql)

            # Commit the changes to the database
            self.connection.commit()

        except sqlite3.Error as e:
            # Handle any SQLite errors
            print(f"An error occurred while creating the fused position table: {e}")
            self.connection.rollback()

    ################# Table Write Functions #################
    @retry()
    def service_log_msg(self, msg: str, error: str):
        """
        Writes a message to the service log table in the database.
        Args:
            service (str): The name of the service that encountered the error.
            error (str): The error message.
        """
        try:
            query = "INSERT INTO sensor_fusion_logs (system_time, error_type, error_message) VALUES (?, ?, ?);"
            self.cursor.execute(query, (time.time(), msg, error))
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"An error occurred while writing to the error table: {e}")
            self.connection.rollback()

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
    def insert_fused_position_data(self, fused_position_data):
        """
        Inserts one or multiple rows of fused position data into the fused position table.

        Args:
            data (list): A list of dictionaries, each containing the data for a row.

        Returns:
            bool: True if the insert was successful, False otherwise.
        """
        try:
            insert_query = """
                INSERT INTO fused_position (time, gnss_lat, gnss_lon, fused_lat, fused_lon, fused_heading, forward_velocity, yaw_rate, session)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            # Prepare data for insertion
            data_to_insert = [
                (
                    entry["time"],
                    entry["gnss_lat"],
                    entry["gnss_lon"],
                    entry["fused_lat"],
                    entry["fused_lon"],
                    entry["fused_heading"],
                    entry["forward_velocity"],
                    entry["yaw_rate"],
                    entry["session"],
                )
                for entry in fused_position_data
            ]
            self.cursor.executemany(insert_query, data_to_insert)
            self.connection.commit()
            return True
        except sqlite3.Error as e:
            print(
                f"An error occurred while inserting data into the fused position table: {e}"
            )
            self.connection.rollback()
            return False

    ################# Table Read Functions #################
    @retry()
    def get_starting_row_id(self, table_name):
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
    def get_most_recent_row_id(self, table_name):
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
    def get_row_count(self, table_name: str) -> int:
        """
        Gets the number of rows in the specified table.

        Parameters:
            table_name (str): The name of the table to count rows from.

        Returns:
            int: The number of rows in the table.
        """
        try:
            # Ensure the table name is a valid identifier (this is a simple check; you may want more thorough validation)
            if not table_name.isidentifier():
                raise ValueError("Invalid table name")

            # SQL command to count the rows in the table
            count_rows_sql = f"SELECT COUNT(*) FROM {table_name}"

            # Execute the SQL command to count the rows
            self.cursor.execute(count_rows_sql)

            # Fetch the result and return the count
            row_count = self.cursor.fetchone()[0]
            return row_count

        except sqlite3.Error as e:
            # Handle any SQLite errors
            print(f"An error occurred while counting the rows: {e}")
            return -1

    @retry()
    def count_rows_by_value(self, table_name: str, column_name: str, value) -> int:
        """
        Counts the number of rows in the specified table where the specified column contains the given value.

        Parameters:
            table_name (str): The name of the table to count rows from.
            column_name (str): The name of the column to check the value.
            value: The value to match for counting rows.

        Returns:
            int: The number of rows in the table that satisfy the condition.
        """
        try:
            # Ensure the table and column names are valid identifiers (simple check; you may want more thorough validation)
            if not table_name.isidentifier() or not column_name.isidentifier():
                raise ValueError("Invalid table or column name")

            # SQL command to count the rows in the table where the column contains the specified value
            count_rows_sql = (
                f"SELECT COUNT(*) FROM {table_name} WHERE {column_name} = ?"
            )

            # Execute the SQL command to count the rows
            self.cursor.execute(count_rows_sql, (value,))

            # Fetch the result and return the count
            row_count = self.cursor.fetchone()[0]
            return row_count

        except (sqlite3.Error, ValueError) as e:
            # Handle any SQLite or validation errors
            print(f"An error occurred while counting the rows: {e}")
            return -1

    @retry()
    def get_unique_values_ordered(
        self, table_name: str, column_name: str, order_by: str
    ) -> list:
        """
        Gets all unique values in a column and orders them by the specified column.

        :param table_name: Name of the table.
        :param column_name: Name of the column to get unique values from.
        :param order_by: Name of the column to order by.
        :return: List of unique values ordered by the specified column.
        """
        try:
            # Validate the table and column names to avoid SQL injection
            if (
                not table_name.isidentifier()
                or not column_name.isidentifier()
                or not order_by.isidentifier()
            ):
                raise ValueError("Invalid table or column name")

            # SQL command to select unique values from the specified column and order by the order_by column
            get_unique_values_ordered_sql = f"""
            SELECT DISTINCT {column_name}
            FROM {table_name}
            ORDER BY {order_by} ASC;
            """
            # Execute the SQL command
            self.cursor.execute(get_unique_values_ordered_sql)
            results = self.cursor.fetchall()

            # Extract the unique values from the results
            unique_values = [row[0] for row in results]

            return unique_values

        except (sqlite3.Error, ValueError) as e:
            # Handle any SQLite or validation errors
            print(f"An error occurred while retrieving unique values: {e}")
            return []

    ####  All Data Query Functions ####
    def queryAllImu(self, order: str = "ASC"):
        """
        Queries the whole IMU table for accelerometer and gyroscope data and sorts by row ID.
        Columns queried acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, time
        Args:
            order (str, optional): The order of retrieval, either 'ASC' or 'DESC'. Defaults to 'DESC'.
        Returns:
            list: A list of IMUData objects containing the accelerometer and gyroscope data.
        """
        try:
            query = """
                        SELECT acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, time, temperature, session, id
                        FROM imu 
                        ORDER BY id {}
                    """.format(order)
            rows = self.cursor.execute(query).fetchall()
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
            print(f"SQLite error: {e}")
            return []

    def queryAllProcessedImu(self, order: str = "ASC"):
        """
        Queries the whole IMU table for accelerometer and gyroscope data and sorts by row ID.
        Columns queried acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, time
        Args:
            order (str, optional): The order of retrieval, either 'ASC' or 'DESC'. Defaults to 'DESC'.
        Returns:
            list: A list of ProcessedIMUData objects containing the accelerometer and gyroscope data.
        """
        try:
            query = """
                        SELECT acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, time, temperature, session, stationary, id
                        FROM imu_processed 
                        ORDER BY id {}
                    """.format(order)
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
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return []

    def queryAllMagnetometer(self, order: str = "ASC"):
        """
        Queries the magnetometer table for magnetometer data and sorts by row ID.
        Columns queried mag_x, mag_y, mag_z, system_time
        Args:
            order (str, optional): The order of retrieval, either 'ASC' or 'DESC'. Defaults to 'DESC'.
        Returns:
            list: A list of MagnetometerData objects containing the magnetometer data.
        """
        try:
            query = """
                        SELECT mag_x, mag_y, mag_z, system_time, session
                        FROM magnetometer
                        ORDER BY id {}
                    """.format(order)
            rows = self.cursor.execute(query).fetchall()
            results = [MagData(row[0], row[1], row[2], row[3], row[4]) for row in rows]
            return results
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return []

    def queryAllGnss(self, order: str = "ASC"):
        """
        Queries the GNSS table for all GNSS data and sorts by row ID.
        Columns queried latitude, longitude, altitude, speed, heading, heading_accuracy, hdop, gdop, system_time, time, time_resolved, session
        Args:
            order (str, optional): The order of retrieval, either 'ASC' or 'DESC'. Defaults to 'DESC'.
        Returns:
            list: A list of GNSSData objects containing the GNSS data.
        """
        try:
            query = """
                        SELECT latitude, longitude, altitude, speed, heading, heading_accuracy, hdop, gdop, system_time, time, time_resolved, session
                        FROM gnss 
                        ORDER BY id {}
                    """.format(order)
            rows = self.cursor.execute(query).fetchall()
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
            print(f"SQLite error: {e}")
            return []

    def queryAllFusedPosition(self, order: str = "ASC"):
        """
        Queries the whole fused_position table and sorts by row ID.
        Columns queried: id, time, gnss_lat, gnss_lon, fused_lat, fused_lon, fused_heading, forward_velocity, yaw_rate, session.
        Args:
            order (str, optional): The order of retrieval, either 'ASC' or 'DESC'. Defaults to 'ASC'.
        Returns:
            list: A list of FusedPositionData objects containing the fused position data.
        """
        try:
            query = """
                        SELECT id, time, gnss_lat, gnss_lon, fused_lat, fused_lon, fused_heading, forward_velocity, yaw_rate, session
                        FROM fused_position 
                        ORDER BY id {}
                    """.format(order)
            rows = self.cursor.execute(query).fetchall()
            results = [
                FusedPositionData(
                    row[0],  # id
                    row[1],  # time
                    row[2],  # gnss_lat
                    row[3],  # gnss_lon
                    row[4],  # fused_lat
                    row[5],  # fused_lon
                    row[6],  # fused_heading
                    row[7],  # forward_velocity
                    row[8],  # yaw_rate
                    row[9],  # session
                )
                for row in rows
            ]
            return results
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return []

    ################# Table Level Functions #################
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

    ################# Data Purging Functions #################
    @retry()
    def purge_oldest_rows(self, table_name: str, num_rows: int):
        """
        Purges the oldest num_rows rows from the specified table.

        Parameters:
            table_name (str): The name of the table to purge rows from.
            num_rows (int): The number of oldest rows to delete from the table.
        """
        try:
            # Validate the table and column names to avoid SQL injection
            if not table_name.isidentifier():
                raise ValueError("Invalid table or column name")

            # SQL command to delete the oldest num_rows rows
            delete_rows_sql = f"""
            DELETE FROM {table_name}
            WHERE rowid IN (
                SELECT rowid
                FROM {table_name}
                ORDER BY rowid ASC
                LIMIT ?
            );
            """
            # Execute the SQL command to delete the rows with parameterized query
            self.cursor.execute(delete_rows_sql, (num_rows,))

            # Commit the changes to the database
            self.connection.commit()
            print(
                f"Successfully deleted the oldest {num_rows} rows from the {table_name} table."
            )

        except sqlite3.Error as e:
            # Handle any SQLite errors
            print(f"An error occurred while deleting the rows: {e}")
            self.connection.rollback()

    @retry()
    def purge_rows_by_value(self, table_name: str, column_name: str, value) -> None:
        """
        Purges all rows where the specified column contains the given value.

        :param table_name: Name of the table.
        :param column_name: Name of the column to check the value.
        :param value: The value to match for purging rows.
        """
        try:
            # Validate the table and column names to avoid SQL injection
            if not table_name.isidentifier() or not column_name.isidentifier():
                raise ValueError("Invalid table or column name")

            # SQL command to delete rows where the column contains the specified value
            purge_rows_sql = f"""
            DELETE FROM {table_name}
            WHERE {column_name} = ?;
            """
            # Execute the SQL command to delete the rows
            self.cursor.execute(purge_rows_sql, (value,))

            # Commit the changes to the database
            self.connection.commit()
            print(
                f"Successfully deleted rows from the {table_name} table where {column_name} = {value}."
            )

        except (sqlite3.Error, ValueError) as e:
            # Handle any SQLite or validation errors
            print(f"An error occurred while deleting the rows: {e}")
            self.connection.rollback()

    @retry()
    def purge(self):
        # Check if DB is over the limit return if it is not
        try:
            if self.get_db_size() < DB_SIZE_LIMIT:
                return True

            print("Purging the database to reduce the file size.")

            # sessions here are oldest to newest, important to keep this in mind
            sessions_count = {}
            table_row_counts = {}
            for table in PURGE_GROUP:
                cur_table_sessions = self.get_unique_values_ordered(
                    table, "session", "id"
                )
                table_row_counts[table] = self.get_row_count(table)
                for session in cur_table_sessions:
                    if session not in sessions_count:
                        sessions_count[session] = 1
                    else:
                        sessions_count[session] += 1

            # If there is only one session, then we can purge the oldest rows
            if len(sessions_count) == 1:
                for table in PURGE_GROUP:
                    self.purge_oldest_rows(
                        table, math.ceil(table_row_counts[table] / 2.0)
                    )
            else:
                # Check for session consistency remove sessions that are not in all tables
                # This uses python 3.7+ behavior of keeping insertion order in dictionaries
                unique_sessions = list(sessions_count.keys())
                if len(unique_sessions) > 1:
                    oldest_session = unique_sessions[0]
                    for table in PURGE_GROUP:
                        self.purge_rows_by_value(
                            table,
                            "session",
                            oldest_session,
                        )
            self.vacuum()
            return True
        except Exception as e:
            print(f"An error occurred while purging the database: {e}")
            self.service_log_msg("Purging DB", str(e))
            return False

    def vacuum(self):
        """
        Reclaims the unused space in the database file and reduces its size.
        """
        try:
            start_time = time.time()
            self.cursor.execute("VACUUM")
            self.connection.commit()
            end_time = time.time()
            print(
                f"Successfully reclaimed space and reduced the database file size. Time taken: {end_time - start_time:.2f} seconds."
            )

        except sqlite3.Error as e:
            print(f"An error occurred while vacuuming the database: {e}")
            self.connection.rollback()
