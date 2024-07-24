from datetime import datetime, timezone
import numpy as np

# references:
# [1] https://www.enri.go.jp/~fks442/K_MUSEN/1st/1st060428rev2.pdf

#################### Coordinate Conversions Functions ####################
# Code adapted from:
# https://github.com/motokimura/kalman_filter_with_kitti/blob/master/src/utils/geo_transforms.py

# constant parameters defined in [1]
_a = 6378137.0
_f = 1.0 / 298.257223563
_b = (1.0 - _f) * _a
_e = np.sqrt(_a**2.0 - _b**2.0) / _a
_e_prime = np.sqrt(_a**2.0 - _b**2.0) / _b


def Rx(theta):
    """rotation matrix around x-axis"""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def Ry(theta):
    """rotation matrix around y-axis"""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def Rz(theta):
    """rotation matrix around z-axis"""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def lla_to_ecef(points_lla):
    """transform N x [longitude(deg), latitude(deg), altitude(m)] coords into
    N x [x, y, z] coords measured in Earth-Centered-Earth-Fixed frame.
    """
    lon = np.radians(points_lla[0])  # [N,]
    lat = np.radians(points_lla[1])  # [N,]
    alt = points_lla[2]  # [N,]

    N = _a / np.sqrt(1.0 - (_e * np.sin(lat)) ** 2.0)  # [N,]
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1.0 - _e**2.0) + alt) * np.sin(lat)

    points_ecef = np.stack([x, y, z], axis=0)  # [3, N]
    return points_ecef


def ecef_to_enu(points_ecef, ref_lla):
    """transform N x [x, y, z] coords measured in Earth-Centered-Earth-Fixed frame into
    N x [x, y, z] coords measured in a local East-North-Up frame.
    """
    lon = np.radians(ref_lla[0])
    lat = np.radians(ref_lla[1])
    alt = ref_lla[2]

    ref_ecef = lla_to_ecef(ref_lla)  # [3,]

    relative = points_ecef - ref_ecef[:, np.newaxis]  # [3, N]

    R = Rz(np.pi / 2.0) @ Ry(np.pi / 2.0 - lat) @ Rz(lon)  # [3, 3]
    points_enu = R @ relative  # [3, N]
    return points_enu


def lla_to_enu(points_lla, ref_lla):
    """transform N x [longitude(deg), latitude(deg), altitude(m)] coords into
    N x [x, y, z] coords measured in a local East-North-Up frame.
    """
    points_ecef = lla_to_ecef(points_lla)
    points_enu = ecef_to_enu(points_ecef, ref_lla)
    return points_enu


def enu_to_ecef(points_enu, ref_lla):
    """transform N x [x, y, z] coords measured in a local East-North-Up frame into
    N x [x, y, z] coords measured in Earth-Centered-Earth-Fixed frame.
    """
    # inverse transformation of `ecef_to_enu`

    lon = np.radians(ref_lla[0])
    lat = np.radians(ref_lla[1])
    alt = ref_lla[2]

    ref_ecef = lla_to_ecef(ref_lla)  # [3,]

    R = Rz(np.pi / 2.0) @ Ry(np.pi / 2.0 - lat) @ Rz(lon)  # [3, 3]
    R = R.T  # inverse rotation
    relative = R @ points_enu  # [3, N]

    points_ecef = ref_ecef[:, np.newaxis] + relative  # [3, N]
    return points_ecef


def ecef_to_lla(points_ecef):
    """transform N x [x, y, z] coords measured in Earth-Centered-Earth-Fixed frame into
    N x [longitude(deg), latitude(deg), altitude(m)] coords.
    """
    # approximate inverse transformation of `lla_to_ecef`

    x = points_ecef[0]  # [N,]
    y = points_ecef[1]  # [N,]
    z = points_ecef[2]  # [N,]

    p = np.sqrt(x**2.0 + y**2.0)  # [N,]
    theta = np.arctan(z * _a / (p * _b))  # [N,]

    lon = np.arctan(y / x)  # [N,]
    lat = np.arctan(
        (z + (_e_prime**2.0) * _b * (np.sin(theta) ** 3.0))
        / (p - (_e**2.0) * _a * (np.cos(theta)) ** 3.0)
    )  # [N,]
    N = _a / np.sqrt(1.0 - (_e * np.sin(lat)) ** 2.0)  # [N,]
    alt = p / np.cos(lat) - N  # [N,]

    lon = np.degrees(lon)
    lat = np.degrees(lat)

    points_lla = np.stack([lon, lat, alt], axis=0)  # [3, N]
    return points_lla


def enu_to_lla(points_enu, ref_lla):
    """transform N x [x, y, z] coords measured in a local East-North-Up frame into
    N x [longitude(deg), latitude(deg), altitude(m)] coords.
    """
    points_ecef = enu_to_ecef(points_enu, ref_lla)
    points_lla = ecef_to_lla(points_ecef)
    return points_lla


#################### Time Conversions Functions ####################


def convert_time_to_epoch(time_str):
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


def convert_epoch_to_time(epoch_ms):
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


#################### Data Conversions Functions ####################


def lists_to_dicts(keys, *lists):
    # Find the length of the shortest list to avoid index errors
    min_length = min(len(lst) for lst in lists)

    # Ensure the number of keys matches the number of input lists
    if len(keys) != len(lists):
        raise ValueError("Number of keys must match the number of input lists")

    # Create a list of dictionaries using the specified keys and elements at each index from the input lists
    result = [
        dict(zip(keys, values)) for values in zip(*[lst[:min_length] for lst in lists])
    ]

    return result
