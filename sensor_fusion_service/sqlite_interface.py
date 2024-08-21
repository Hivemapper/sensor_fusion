import sqlite3
import os
import math
import time
from functools import wraps
from typing import List, Union

env = os.getenv("HIVE_ENV")

if env == "local":
    from sensor_fusion.sensor_fusion_service.data_definitions import (
        IMUData,
        ProcessedIMUData,
        GNSSData,
        MagData,
        FrameKMData,
        FusedPositionData,
        PURGE_GROUP,
    )
else:
    from data_definitions import (
        IMUData,
        ProcessedIMUData,
        GNSSData,
        MagData,
        FrameKMData,
        FusedPositionData,
        PURGE_GROUP,
    )

# For SQLite interface
DATA_LOGGER_PATH = "/data/recording/data-logger.v1.4.5.db"
DESC = "DESC"
ASC = "ASC"

# Purging Constants
DB_SIZE_LIMIT = 1024 * 1024 * 100  # 100 MB


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

    @retry()
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
    def create_processed_gnss_table(self):
        """
        Creates the gnss_processed table in the database if it does not already exist
        and creates an index on the system_time column.
        """
        try:
            create_gnss_table_sql = """
            CREATE TABLE IF NOT EXISTS gnss_processed (
                        id INTEGER NOT NULL PRIMARY KEY,
                        system_time TIMESTAMP NOT NULL,
                        time TIMESTAMP NOT NULL,
                        fix TEXT NOT NULL,
                        ttff INTEGER NOT NULL,
                        latitude REAL NOT NULL,
                        longitude REAL NOT NULL,
                        altitude REAL NOT NULL,
                        speed REAL NOT NULL,
                        heading REAL NOT NULL,
                        satellites_seen INTEGER NOT NULL,
                        satellites_used INTEGER NOT NULL,
                        eph INTEGER NOT NULL,
                        horizontal_accuracy REAL NOT NULL,
                        vertical_accuracy REAL NOT NULL,
                        heading_accuracy REAL NOT NULL,
                        speed_accuracy REAL NOT NULL,
                        hdop REAL NOT NULL,
                        vdop REAL NOT NULL,
                        xdop REAL NOT NULL,
                        ydop REAL NOT NULL,
                        tdop REAL NOT NULL,
                        pdop REAL NOT NULL,
                        gdop REAL NOT NULL,
                        rf_jamming_state TEXT NOT NULL,
                        rf_ant_status TEXT NOT NULL,
                        rf_ant_power TEXT NOT NULL,
                        rf_post_status INTEGER NOT NULL,
                        rf_noise_per_ms INTEGER NOT NULL,
                        rf_agc_cnt INTEGER NOT NULL,
                        rf_jam_ind INTEGER NOT NULL,
                        rf_ofs_i INTEGER NOT NULL,
                        rf_mag_i INTEGER NOT NULL,
                        rf_ofs_q INTEGER NOT NULL,
                        gga TEXT NOT NULL,
                        rxm_measx TEXT NOT NULL,
                        actual_system_time TIMESTAMP NOT NULL DEFAULT '0000-00-00 00:00:00',
                        unfiltered_latitude REAL NOT NULL DEFAULT 0,
                        unfiltered_longitude REAL NOT NULL DEFAULT 0,
                        time_resolved INTEGER NOT NULL DEFAULT 0,
                        session TEXT NOT NULL DEFAULT ''
            );
            """
            # Execute the SQL command to create the gnss_processed table
            self.cursor.execute(create_gnss_table_sql)

            # SQL command to create an index on the system_time column
            create_time_index_sql = """
            CREATE INDEX IF NOT EXISTS gnss_time_idx ON gnss_processed(system_time);
            """
            # Execute the SQL command to create the index
            self.cursor.execute(create_time_index_sql)

            # Commit the changes to the database
            self.connection.commit()

        except sqlite3.Error as e:
            # Handle any SQLite errors
            print(
                f"An error occurred while creating the gnss_processed table or index: {e}"
            )
            self.connection.rollback()

    @retry()
    def create_processed_imu_table(self):
        """
        Creates the processed IMU table in the database if it does not already exist,
        and creates an index on the time column.
        """
        try:
            # SQL command to create the imu_processed table if it doesn't already exist
            create_imu_table_sql = """
            CREATE TABLE IF NOT EXISTS imu_processed (
                        id INTEGER NOT NULL PRIMARY KEY,
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
            # Execute the SQL command to create the imu_processed table
            self.cursor.execute(create_imu_table_sql)

            # SQL command to create an index on the time column
            create_time_index_sql = """
            CREATE INDEX IF NOT EXISTS imu_time_idx ON imu_processed(time);
            """
            # Execute the SQL command to create the index
            self.cursor.execute(create_time_index_sql)

            # Commit the changes to the database
            self.connection.commit()

        except sqlite3.Error as e:
            # Handle any SQLite errors
            print(
                f"An error occurred while creating the imu_processed table or index: {e}"
            )
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

    @retry()
    def create_error_logs_table(self):
        """
        Creates the error_logs table in the database if it does not already exist,
        and creates an index on the system_time column.
        """
        try:
            # SQL command to create the error_logs table if it doesn't already exist
            create_error_logs_table_sql = """
            CREATE TABLE IF NOT EXISTS error_logs (
                        id INTEGER NOT NULL PRIMARY KEY,
                        system_time TIMESTAMP NOT NULL,
                        service_name TEXT NOT NULL,
                        message TEXT NOT NULL
            );
            """
            # Execute the SQL command to create the error_logs table
            self.cursor.execute(create_error_logs_table_sql)

            # SQL command to create an index on the system_time column
            create_time_index_sql = """
            CREATE INDEX IF NOT EXISTS error_time_idx ON error_logs(system_time);
            """
            # Execute the SQL command to create the index
            self.cursor.execute(create_time_index_sql)

            # Commit the changes to the database
            self.connection.commit()

        except sqlite3.Error as e:
            # Handle any SQLite errors
            print(
                f"An error occurred while creating the error_logs table or index: {e}"
            )
            self.connection.rollback()

    @retry()
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
    def insert_data(
        self,
        table_name,
        data_list: List[
            Union[GNSSData, IMUData, ProcessedIMUData, MagData, FusedPositionData]
        ],
    ):
        """
        Inserts a list of objects into the specified table.

        Args:
            table_name (str): The name of the table to insert data into.
            data_list (list): A list of objects to be inserted into the table.

        Returns:
            bool: True if the insert was successful, False otherwise.
        """
        if not data_list:
            print("No data to insert.")
            return False

        try:
            # Get columns from the first object in the list
            columns = vars(data_list[0]).keys()
            placeholders = ", ".join("?" * len(columns))
            column_names = ", ".join(columns)

            insert_data_sql = f"""
            INSERT INTO {table_name} ({column_names}) 
            VALUES ({placeholders})
            """

            # Convert each object to a tuple of values
            values_list = [tuple(vars(data).values()) for data in data_list]

            # Execute the insert statement for each object
            self.cursor.executemany(insert_data_sql, values_list)

            # Commit the changes to the database
            self.connection.commit()
            return True

        except sqlite3.Error as e:
            # Handle any SQLite errors
            print(f"An error occurred while inserting data into {table_name}: {e}")
            self.connection.rollback()
            return False

    ################# Table Read Functions #################
    @retry()
    def table_columns_match(self, table_name, column_names):
        """
        Checks whether the table columns match the provided column names.

        Parameters:
        table_name (str): The name of the table to check.
        column_names (list): A list of column names to match against the table columns.

        Returns:
        bool: True if the table columns match the provided column names, False otherwise.
        """
        try:
            # SQL command to get the table schema
            query = f"PRAGMA table_info({table_name});"

            # Execute the SQL command
            self.cursor.execute(query)

            # Fetch the result
            result = self.cursor.fetchall()

            # Extract the column names from the result, ignoring 'id' column
            table_columns = [row[1] for row in result if row[1] != "id"]

            # Check if the provided column names match the table columns
            return set(table_columns) == set(column_names)

        except sqlite3.Error as e:
            # Handle any SQLite errors
            print(
                f"An error occurred while checking the columns for table {table_name}: {e}"
            )
            return False

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

    @retry()
    def get_data_by_row_range(
        self,
        table_name: str,
        data_class: Union[
            GNSSData, IMUData, ProcessedIMUData, MagData, FusedPositionData, FrameKMData
        ],
        start_id: int,
        end_id: int,
        order: str = "ASC",
    ) -> List[
        Union[
            GNSSData, IMUData, ProcessedIMUData, MagData, FusedPositionData, FrameKMData
        ]
    ]:
        """
        Queries the specified table for data within a specified ID range, sorts by row ID, and returns the data as a list of objects of the specified class.
        The columns queried depend on the data_class specified.

        Args:
            table_name (str): The name of the table to query.
            data_class: The class type to instantiate for each row.
            start_id (int): The starting row ID for the query.
            end_id (int): The ending row ID for the query.
            order (str, optional): The order of retrieval, either 'ASC' or 'DESC'. Defaults to 'ASC'.

        Returns:
            list: A list of objects of the specified class type.
        """
        try:
            # SQL command to select rows within a specified ID range from the table
            query = f"""
                        SELECT * FROM {table_name}
                        WHERE id BETWEEN ? AND ?
                        ORDER BY id {order}
                    """

            # Execute the SQL command
            rows = self.cursor.execute(query, (start_id, end_id)).fetchall()

            # Get the column names from the cursor description
            column_names = [desc[0] for desc in self.cursor.description]

            # Convert each row to an instance of the data_class
            data_list = [
                data_class(
                    **{
                        col: val
                        for col, val in zip(column_names, row)
                        if col not in ["id", "row_id"]
                    }
                )
                for row in rows
            ]

            return data_list

        except sqlite3.Error as e:
            # Handle any SQLite errors
            print(
                f"An error occurred while querying the {table_name} table for rows {start_id} to {end_id}: {e}"
            )
            return []

    @retry()
    def read_all_table_data(
        self,
        table_name: str,
        data_class: Union[
            GNSSData, IMUData, ProcessedIMUData, MagData, FusedPositionData, FrameKMData
        ],
    ) -> List[
        Union[
            GNSSData, IMUData, ProcessedIMUData, MagData, FusedPositionData, FrameKMData
        ]
    ]:
        """
        Reads all rows from the specified table and returns them as a list of objects of the specified class.

        Args:
            table_name (str): The name of the table to read data from.
            data_class: The class type to instantiate for each row.

        Returns:
            list: A list of objects of the specified class type.
        """
        # print(f"Reading data from {table_name} table...")
        try:
            # SQL command to select all rows from the table
            if data_class == FrameKMData:
                select_data_sql = f"SELECT * FROM {table_name} ORDER BY fkm_id ASC;"
            else:
                select_data_sql = f"SELECT * FROM {table_name} ORDER BY id ASC;"

            # Execute the SQL command
            self.cursor.execute(select_data_sql)
            rows = self.cursor.fetchall()
            # print(f"Read {len(rows)} rows from {table_name} table.")

            # Get the column names from the cursor description
            column_names = [desc[0] for desc in self.cursor.description]
            # print(f"Column names: {column_names}")

            # Convert each row to an instance of the data_class, excluding the 'id' column
            data_list = [
                data_class(
                    **{
                        col: val
                        for col, val in zip(column_names, row)
                        if col not in ["id", "row_id"]
                    }
                )
                for row in rows
            ]

            return data_list

        except sqlite3.Error as e:
            # Handle any SQLite errors
            print(f"An error occurred while reading data from {table_name}: {e}")
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

    @retry()
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
