import numpy as np

from sensor_fusion.sensor_fusion_service.processing import cubic_spline_interpolation

GNSS_LOW_SPEED_THRESHOLD = 0.1


def get_imu_offsets(
    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, imu_time, gnss_system_time, speed
):
    # downsample the data to match GNSS frequency
    # acc_x_down = cubic_spline_interpolation(acc_x, imu_time, gnss_system_time)
    # acc_y_down = cubic_spline_interpolation(acc_y, imu_time, gnss_system_time)
    # acc_z_down = cubic_spline_interpolation(acc_z, imu_time, gnss_system_time)
    # gyro_x_down = cubic_spline_interpolation(gyro_x, imu_time, gnss_system_time)
    # gyro_y_down = cubic_spline_interpolation(gyro_y, imu_time, gnss_system_time)
    # gyro_z_down = cubic_spline_interpolation(gyro_z, imu_time, gnss_system_time)

    acc_x_down = np.interp(gnss_system_time, imu_time, acc_x)
    acc_y_down = np.interp(gnss_system_time, imu_time, acc_y)
    acc_z_down = np.interp(gnss_system_time, imu_time, acc_z)
    gyro_x_down = np.interp(gnss_system_time, imu_time, gyro_x)
    gyro_y_down = np.interp(gnss_system_time, imu_time, gyro_y)
    gyro_z_down = np.interp(gnss_system_time, imu_time, gyro_z)

    # Calculate bias for accel and gyro
    zero_speed_indices = [
        i for i, speed_val in enumerate(speed) if speed_val < GNSS_LOW_SPEED_THRESHOLD
    ]

    if len(zero_speed_indices) < 20:
        print(f"No enough zero speed data found: {len(zero_speed_indices)}")
        return

    acc_x_down_zero_speed, acc_y_down_zero_speed, acc_z_down_zero_speed = [], [], []
    gyro_x_down_zero_speed, gyro_y_down_zero_speed, gyro_z_down_zero_speed = [], [], []
    for i in zero_speed_indices:
        acc_x_down_zero_speed.append(acc_x_down[i])
        acc_y_down_zero_speed.append(acc_y_down[i])
        acc_z_down_zero_speed.append(acc_z_down[i])
        gyro_x_down_zero_speed.append(gyro_x_down[i])
        gyro_y_down_zero_speed.append(gyro_y_down[i])
        gyro_z_down_zero_speed.append(gyro_z_down[i])

    # Calculate the average of the zero speed values
    acc_x_down_zero_speed_avg = np.mean(acc_x_down_zero_speed)
    acc_y_down_zero_speed_avg = np.mean(acc_y_down_zero_speed)
    acc_z_down_zero_speed_avg = (
        np.mean(acc_z_down_zero_speed) - 1
    )  # handle the fact this needs to be 1 when at 0 velocity not 0
    gyro_x_down_zero_speed_avg = np.mean(gyro_x_down_zero_speed)
    gyro_y_down_zero_speed_avg = np.mean(gyro_y_down_zero_speed)
    gyro_z_down_zero_speed_avg = np.mean(gyro_z_down_zero_speed)

    print(
        f"Accel offsets: {acc_x_down_zero_speed_avg}, {acc_y_down_zero_speed_avg}, {acc_z_down_zero_speed_avg}"
    )
    print(
        f"Gyro offsets: {gyro_x_down_zero_speed_avg}, {gyro_y_down_zero_speed_avg}, {gyro_z_down_zero_speed_avg}"
    )

    return
