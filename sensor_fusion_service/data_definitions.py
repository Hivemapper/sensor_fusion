from enum import Enum


### Tables in the Database
class TableName(Enum):
    GNSS_RAW_TABLE = "gnss"
    GNSS_PROCESSED_TABLE = "gnss_processed"
    GNSS_AUTH_TABLE = "gnss_auth"
    IMU_RAW_TABLE = "imu"
    IMU_PROCESSED_TABLE = "imu_processed"
    MAG_TABLE = "magnetometer"
    FUSED_POSITION_TABLE = "fused_position"
    PACKED_FRAMEKM_TABLE = "packed_framekms"
    SENSOR_FUSION_LOG_TABLE = "sensor_fusion_logs"
    ERROR_LOG_TABLE = "error_logs"


### Tables in Purge Group
PURGE_GROUP = [
    TableName.GNSS_RAW_TABLE.value,
    TableName.GNSS_PROCESSED_TABLE.value,
    TableName.GNSS_AUTH_TABLE.value,
    TableName.IMU_RAW_TABLE.value,
    TableName.IMU_PROCESSED_TABLE.value,
    TableName.MAG_TABLE.value,
    TableName.FUSED_POSITION_TABLE.value,
]


############## Data Classes ##############
class IMUData:
    def __init__(
        self,
        time,
        acc_x,
        acc_y,
        acc_z,
        gyro_x,
        gyro_y,
        gyro_z,
        temperature,
        session,
    ):
        self.time = time
        self.acc_x = acc_x
        self.acc_y = acc_y
        self.acc_z = acc_z
        self.gyro_x = gyro_x
        self.gyro_y = gyro_y
        self.gyro_z = gyro_z
        self.temperature = temperature
        self.session = session


class ProcessedIMUData:
    def __init__(
        self,
        time,
        acc_x,
        acc_y,
        acc_z,
        gyro_x,
        gyro_y,
        gyro_z,
        stationary,
        temperature,
        session,
    ):
        self.time = time
        self.acc_x = acc_x
        self.acc_y = acc_y
        self.acc_z = acc_z
        self.gyro_x = gyro_x
        self.gyro_y = gyro_y
        self.gyro_z = gyro_z
        self.stationary = stationary
        self.temperature = temperature
        self.session = session


class MagData:
    def __init__(self, system_time, mag_x, mag_y, mag_z, session):
        self.mag_x = mag_x
        self.mag_y = mag_y
        self.mag_z = mag_z
        self.system_time = system_time
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
        self.time = time
        self.gnss_lat = gnss_lat
        self.gnss_lon = gnss_lon
        self.fused_lat = fused_lat
        self.fused_lon = fused_lon
        self.fused_heading = fused_heading
        self.forward_velocity = forward_velocity
        self.yaw_rate = yaw_rate
        self.session = session


class FrameKMData:
    def __init__(
        self,
        fkm_id,
        image_name,
        image_path,
        acc_x,
        acc_y,
        acc_z,
        gyro_x,
        gyro_y,
        gyro_z,
        xdop,
        ydop,
        tdop,
        vdop,
        pdop,
        gdop,
        hdop,
        eph,
        latitude,
        longitude,
        altitude,
        speed,
        time,
        frame_idx,
        system_time,
        satellites_used,
        dilution,
        created_at,
        ml_model_hash,
        ml_detections,
        ml_read_time,
        ml_write_time,
        ml_inference_time,
        ml_blur_time,
        ml_downscale_time,
        ml_upscale_time,
        ml_mask_time,
        ml_composite_time,
        ml_load_time,
        ml_transpose_time,
        ml_letterbox_time,
        ml_processed_at,
        ml_grid,
        postponed,
        orientation,
        dx,
        stationary,
        error,
    ):
        self.fkm_id = fkm_id
        self.image_name = image_name
        self.image_path = image_path
        self.acc_x = acc_x
        self.acc_y = acc_y
        self.acc_z = acc_z
        self.gyro_x = gyro_x
        self.gyro_y = gyro_y
        self.gyro_z = gyro_z
        self.xdop = xdop
        self.ydop = ydop
        self.tdop = tdop
        self.vdop = vdop
        self.pdop = pdop
        self.gdop = gdop
        self.hdop = hdop
        self.eph = eph
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.speed = speed
        self.time = time
        self.frame_idx = frame_idx
        self.system_time = system_time
        self.satellites_used = satellites_used
        self.dilution = dilution
        self.created_at = created_at
        self.ml_model_hash = ml_model_hash
        self.ml_detections = ml_detections
        self.ml_read_time = ml_read_time
        self.ml_write_time = ml_write_time
        self.ml_inference_time = ml_inference_time
        self.ml_blur_time = ml_blur_time
        self.ml_downscale_time = ml_downscale_time
        self.ml_upscale_time = ml_upscale_time
        self.ml_mask_time = ml_mask_time
        self.ml_composite_time = ml_composite_time
        self.ml_load_time = ml_load_time
        self.ml_transpose_time = ml_transpose_time
        self.ml_letterbox_time = ml_letterbox_time
        self.ml_processed_at = ml_processed_at
        self.ml_grid = ml_grid
        self.postponed = postponed
        self.orientation = orientation
        self.dx = dx
        self.stationary = stationary
        self.error = error


def get_class_field_names(cls):
    """
    Returns a list of field names from the __init__ method of the given class.

    Parameters:
    cls (type): The class to extract field names from.

    Returns:
    list: A list of field names.
    """
    return [field for field in cls.__init__.__code__.co_varnames if field != "self"]


def convert_columns_to_class_instances(data_dict, data_class):
    """
    Converts a dictionary of columns to a list of instances of the specified class.

    Args:
        data_dict (dict): A dictionary where each key is a column name and each value is a list of column values.
        data_class (type): The class type to convert the dictionary into.

    Returns:
        list: A list of instances of the specified class.
    """
    # Get the number of rows by checking the length of any column (assuming all columns have the same length)
    num_rows = len(next(iter(data_dict.values())))

    # Create the list of class instances
    instances = [
        data_class(**{key: data_dict[key][i] for key in data_dict})
        for i in range(num_rows)
    ]

    return instances
