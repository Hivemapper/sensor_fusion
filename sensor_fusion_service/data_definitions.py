from enum import Enum


### Tables in the Database
class TableName(Enum):
    GNSS_TABLE = "gnss"
    GNSS_PROCESSED_TABLE = "gnss_processed"
    GNSS_AUTH_TABLE = "gnss_auth"
    IMU_RAW_TABLE = "imu"
    IMU_PROCESSED_TABLE = "imu_processed"
    MAG_TABLE = "magnetometer"
    SENSOR_FUSION_LOG_TABLE = "sensor_fusion_logs"
    FUSED_POSITION_TABLE = "fused_position"


### Tables in Purge Group
PURGE_GROUP = [
    TableName.GNSS_TABLE.value,
    TableName.GNSS_PROCESSED_TABLE.value,
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
        system_time,
        time,
        fix,
        ttff,
        latitude,
        longitude,
        altitude,
        speed,
        heading,
        satellites_seen,
        satellites_used,
        eph,
        horizontal_accuracy,
        vertical_accuracy,
        heading_accuracy,
        speed_accuracy,
        hdop,
        vdop,
        xdop,
        ydop,
        tdop,
        pdop,
        gdop,
        rf_jamming_state,
        rf_ant_status,
        rf_ant_power,
        rf_post_status,
        rf_noise_per_ms,
        rf_agc_cnt,
        rf_jam_ind,
        rf_ofs_i,
        rf_mag_i,
        rf_ofs_q,
        gga,
        rxm_measx,
        actual_system_time,
        unfiltered_latitude,
        unfiltered_longitude,
        time_resolved,
        session,
    ):
        self.system_time = system_time
        self.time = time
        self.fix = fix
        self.ttff = ttff
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.speed = speed
        self.heading = heading
        self.satellites_seen = satellites_seen
        self.satellites_used = satellites_used
        self.eph = eph
        self.horizontal_accuracy = horizontal_accuracy
        self.vertical_accuracy = vertical_accuracy
        self.heading_accuracy = heading_accuracy
        self.speed_accuracy = speed_accuracy
        self.hdop = hdop
        self.vdop = vdop
        self.xdop = xdop
        self.ydop = ydop
        self.tdop = tdop
        self.pdop = pdop
        self.gdop = gdop
        self.rf_jamming_state = rf_jamming_state
        self.rf_ant_status = rf_ant_status
        self.rf_ant_power = rf_ant_power
        self.rf_post_status = rf_post_status
        self.rf_noise_per_ms = rf_noise_per_ms
        self.rf_agc_cnt = rf_agc_cnt
        self.rf_jam_ind = rf_jam_ind
        self.rf_ofs_i = rf_ofs_i
        self.rf_mag_i = rf_mag_i
        self.rf_ofs_q = rf_ofs_q
        self.gga = gga
        self.rxm_measx = rxm_measx
        self.actual_system_time = actual_system_time
        self.unfiltered_latitude = unfiltered_latitude
        self.unfiltered_longitude = unfiltered_longitude
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
