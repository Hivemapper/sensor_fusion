from enum import Enum


### Tables in the Database
class TableName(Enum):
    GNSS_TABLE = "gnss"
    GNSS_AUTH_TABLE = "gnss_auth"
    IMU_RAW_TABLE = "imu"
    IMU_PROCESSED_TABLE = "imu_processed"
    MAG_TABLE = "magnetometer"
    SENSOR_FUSION_LOG_TABLE = "sensor_fusion_logs"
    FUSED_POSITION_TABLE = "fused_position"


### Tables in Purge Group
PURGE_GROUP = [
    TableName.GNSS_TABLE.value,
    TableName.GNSS_AUTH_TABLE.value,
    TableName.IMU_RAW_TABLE.value,
    TableName.IMU_PROCESSED_TABLE.value,
    TableName.MAG_TABLE.value,
    TableName.FUSED_POSITION_TABLE.value,
]


############## Data Classes ##############
class IMUData:
    def __init__(self, ax, ay, az, gx, gy, gz, time, temperature, session, row_id):
        self.ax = ax
        self.ay = ay
        self.az = az
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.time = time
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
            "time": self.time,
            "temperature": self.temperature,
            "session": self.session,
            "row_id": self.row_id,
        }


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
        heading_accuracy,
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
        self.heading_accuracy = heading_accuracy
        self.hdop = hdop
        self.gdop = gdop
        self.system_time = system_time
        self.time = time
        self.time_resolved = time_resolved
        self.session = session


class FusedPositionData:
    def __init__(
        self,
        id,
        time,
        gnss_lat,
        gnss_lon,
        fused_lat,
        fused_lon,
        fused_heading,
        forward_velocity,
        yaw_rate,
        session,
    ):
        self.id = id
        self.time = time
        self.gnss_lat = gnss_lat
        self.gnss_lon = gnss_lon
        self.fused_lat = fused_lat
        self.fused_lon = fused_lon
        self.fused_heading = fused_heading
        self.forward_velocity = forward_velocity
        self.yaw_rate = yaw_rate
        self.session = session
