from datetime import datetime, timezone


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
