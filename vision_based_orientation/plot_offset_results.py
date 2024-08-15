import matplotlib.pyplot as plt
import json
import os
import argparse
from collections import defaultdict
import numpy as np


def load_and_group_data(json_file):
    """Load and group data from the JSON file by km identifier for regular and undistorted entries."""
    with open(json_file, "r") as f:
        data = json.load(f)

    grouped_data_regular = defaultdict(list)
    grouped_data_undistorted = defaultdict(list)

    for entry in data:
        km_identifier = extract_km_identifier(entry["directory"])
        file_name = os.path.basename(entry["directory"])

        if "_undistorted" in file_name:
            grouped_data_undistorted[km_identifier].append(entry["device_offset"])
        else:
            grouped_data_regular[km_identifier].append(entry["device_offset"])

    return grouped_data_regular, grouped_data_undistorted


def process_entries_for_plotting(grouped_data):
    """Process entries to calculate min, max, average, and diff of offsets."""
    processed_data = {}

    for km_identifier, offsets in grouped_data.items():
        min_offset = min(offsets)
        max_offset = max(offsets)
        avg_offset = np.mean(offsets)
        diff_offset = max_offset - min_offset

        processed_data[km_identifier] = {
            "min": min_offset,
            "max": max_offset,
            "avg": avg_offset,
            "diff": diff_offset,
        }

    return processed_data


def plot_km_group(processed_data, title):
    """Plot the min, max, average, and diff for each km identifier as scatter plots."""
    km_identifiers = list(processed_data.keys())
    x_values = range(len(km_identifiers))

    min_offsets = [processed_data[km]["min"] for km in km_identifiers]
    max_offsets = [processed_data[km]["max"] for km in km_identifiers]
    avg_offsets = [processed_data[km]["avg"] for km in km_identifiers]
    diff_offsets = [processed_data[km]["diff"] for km in km_identifiers]

    plt.figure(figsize=(12, 8))

    # Scatter plot for each metric
    plt.scatter(x_values, min_offsets, marker="o", label="Min Offset", color="blue")
    plt.scatter(x_values, max_offsets, marker="o", label="Max Offset", color="red")
    plt.scatter(
        x_values, avg_offsets, marker="o", label="Average Offset", color="green"
    )
    # plt.scatter(
    #     x_values,
    #     diff_offsets,
    #     marker="o",
    #     label="Diff Offset (Max-Min)",
    #     color="orange",
    # )

    # Adding labels and title
    plt.xlabel("KM Identifier")
    plt.ylabel("Offset Values")
    plt.title(f"Device Offset Metrics for {title}")
    plt.xticks(ticks=x_values, labels=km_identifiers, rotation=45, ha="right")

    # Add legend
    plt.legend()

    # Turn on the major and minor grids
    plt.grid(True, which="major", linestyle="-", linewidth="0.5")
    plt.minorticks_on()
    plt.grid(True, which="minor", linestyle=":", linewidth="0.5")

    # Show the plot
    plt.tight_layout()


def main(json_file):
    # Load and group data
    grouped_data_regular, grouped_data_undistorted = load_and_group_data(json_file)

    # Process data for plotting
    processed_data_regular = process_entries_for_plotting(grouped_data_regular)
    processed_data_undistorted = process_entries_for_plotting(grouped_data_undistorted)

    # Plot the processed data
    plot_km_group(processed_data_regular, "Regular Entries")
    plot_km_group(processed_data_undistorted, "Undistorted Entries")
    plt.show()


def extract_km_identifier(directory):
    """Extracts the km_ identifier from the directory path."""
    parts = directory.split("/")
    for part in parts:
        if part.startswith("km_"):
            return part
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot device offsets from JSON data.")
    parser.add_argument(
        "--json_file", type=str, help="Path to the JSON file containing the data."
    )
    args = parser.parse_args()

    # Call the main function with the provided JSON file
    main(args.json_file)
