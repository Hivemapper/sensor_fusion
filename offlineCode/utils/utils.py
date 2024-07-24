import os
import argparse


def remove_duplicate_data(aggregated_data, filter_column):
    """Removes duplicate data points from the aggregated data based on the specified filter column.

    Args:
        aggregated_data (dict): Dictionary with keys as attribute names and values as lists of attribute values.
        filter_column (str): The column name to filter on for duplicates.

    Returns:
        dict: Dictionary with duplicates removed based on the filter column.
    """
    if filter_column not in aggregated_data:
        raise ValueError(f"Column {filter_column} not found in the aggregated data")

    new_data = {key: [] for key in aggregated_data.keys()}
    seen_values = set()
    column_values = aggregated_data[filter_column]

    for i, value in enumerate(column_values):
        if value not in seen_values:
            seen_values.add(value)
            for key in aggregated_data.keys():
                new_data[key].append(aggregated_data[key][i])

    return new_data


def valid_dir(path):
    """Check if the provided path is a valid directory."""
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid directory.")


def valid_file(path):
    """Check if the provided path is a valid file."""
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid file.")
