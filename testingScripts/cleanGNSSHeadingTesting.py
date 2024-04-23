import os
import sys
sys.path.insert(0, '/Users/rogerberman/sensor-fusion')  # Add the project root to the Python path
from fusion import (
    SqliteInterface,
    extractAndSmoothImuData,
    extractAndSmoothMagData,
    extractGNSSData,
    calculateHeading, 
    calculateAverageFrequency, 
    calculateRollingAverage, 
    calibrate_mag, 
    convertTimeToEpoch,
    getCleanGNSSHeading,
    GNSS_LOW_SPEED_THRESHOLD,
    HEADING_DIFF_MAGNETOMETER_FLIP_THRESHOLD,
    GNSS_HEADING_ACCURACY_THRESHOLD,
    GNSS_DISTANCE_THRESHOLD,
    ASC,
)
from plottingCode import plot_signal_over_time, plot_signals_over_time
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("Testing GNSS Heading Cleanup")
    dir_path = '/Users/rogerberman/Desktop/YawFusionDrives'
    drive = 'drive4'
    data_logger_path = os.path.join(dir_path, drive, 'data-logger.v1.4.4.db')
    print(f"Loading data from {data_logger_path}")
    sql_db = SqliteInterface(data_logger_path)
    heading, cleanHeading, time = getCleanGNSSHeading(sql_db, 1713487816514 + 1000)   

    plot_signals_over_time(time, heading, cleanHeading, 'Heading', 'Forward Loop', None)
    plt.show()
    