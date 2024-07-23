import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

import folium
import math
import webbrowser
import tempfile
import os


def plot_signal_over_time(
    seconds,
    signal_values,
    downsample_factor=1,
    signal_label="Signal",
    highlight_indices=[],
    save_path=None,
):
    """
    Plots signal values over time, where time is represented in seconds.
    Parameters:
    - seconds: A list of timestamps in seconds.
    - signal_values: A list of signal values corresponding to each timestamp.
    - signal_label: A string label for the signal being plotted (e.g., 'Temperature', 'Speed').
    - highlight_indices: A list of indices to highlight on the plot.
    """
    # Ensure the lists are of the same length
    seconds = seconds[::downsample_factor]
    signal_values = signal_values[::downsample_factor]
    if len(seconds) != len(signal_values):
        print(
            "Error: The lists of timestamps and signal values must have the same length."
        )
        return

    plt.figure(figsize=(10, 6))  # Adjust figure size as desired
    plt.plot(seconds, signal_values, linestyle="-", color="b", label=signal_label)

    # Highlight the specified indices
    if highlight_indices:
        # Extract the highlighted seconds and values using list comprehension
        highlighted_seconds = [seconds[i] for i in highlight_indices]
        highlighted_values = [signal_values[i] for i in highlight_indices]
        plt.scatter(highlighted_seconds, highlighted_values, color="r", s=2, zorder=5)

    # Formatting the plot
    plt.title(f"{signal_label} Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel(signal_label)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)  # Save the figure to the file path provided
        plt.close()  # Close the plot figure to prevent it from displaying in the notebook/output


def plot_time(time_series, save_path=None):
    """
    Plots time series data over count of indexes of the time series data.
    Parameters:
    - time_series: A list of timestamps in seconds.
    """
    # Ensure the lists are of the same length
    if len(time_series) == 0:
        print("Error: The list of timestamps must have at least one value.")
        return

    plt.figure(figsize=(10, 6))  # Adjust figure size as desired
    plt.plot(
        range(len(time_series)), time_series, linestyle="-", color="b", label="Time"
    )
    # Formatting the plot
    plt.title("Time Over Count of Indexes")
    plt.xlabel("Index")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)  # Save the figure to the file path provided
        plt.close()  # Close the plot figure to prevent it from displaying in the notebook/output


def plot_periods_over_time(time, periods, period_label="Period", save_path=None):
    """
    Plots period values over time, where time is represented in seconds.
    Parameters:
    - seconds: A list of timestamps in seconds.
    - periods: A list of period values corresponding to each timestamp.
    - period_label: A string label for the period being plotted (e.g., 'Period').
    """
    # Ensure the lists are of the same length
    if len(time) != len(periods):
        print(
            "Error: The lists of timestamps and period values must have the same length."
        )
        return
    plt.figure(figsize=(10, 6))  # Adjust figure size as desired
    plt.scatter(
        time,
        periods,
        color="b",
        s=0.5,
        label="Period",
        zorder=10,
    )

    # Formatting the plot
    plt.title("Time Over Count of Indexes")
    plt.xlabel("Index")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)  # Save the figure to the file path provided
        plt.close()  # Close the plot figure to prevent it from displaying in the notebook/output


def plot_signals_over_time(
    seconds,
    signal1_values,
    signal2_values,
    downsample_factor=1,
    signal1_label="Signal 1",
    signal2_label="Signal 2",
    title=None,
    save_path=None,
):
    """
    Plots two signal values over time, where time is represented in seconds, on the same plot for comparison.
    Optionally saves the plot to a file if a filepath is provided.

    Parameters:
    - seconds: A list of timestamps in seconds.
    - signal1_values: A list of first set of signal values corresponding to each timestamp.
    - signal2_values: A list of second set of signal values corresponding to each timestamp.
    - downsample_factor: An integer factor by which to downsample the data.
    - signal1_label: A string label for the first signal being plotted (e.g., 'Temperature').
    - signal2_label: A string label for the second signal being plotted (e.g., 'Humidity').
    - title: Optional. A string representing the title of the plot.
    - save_path: Optional. A string representing the file path where the plot will be saved. If None, the plot will be displayed.
    """
    # Ensure downsample_factor is an integer
    if not isinstance(downsample_factor, int):
        raise ValueError("downsample_factor must be an integer")

    # Downsample the data
    seconds = np.array(seconds)[::downsample_factor]
    signal1_values = np.array(signal1_values)[::downsample_factor]
    signal2_values = np.array(signal2_values)[::downsample_factor]

    # Ensure the lists are of the same length
    if len(seconds) != len(signal1_values) or len(seconds) != len(signal2_values):
        print(
            "Error: The lists of timestamps and signal values must have the same length."
        )
        return

    plt.figure(figsize=(12, 6))  # Adjust figure size as desired

    # Plot the signals
    plt.plot(seconds, signal1_values, linestyle="-", color="b", label=signal1_label)
    plt.plot(seconds, signal2_values, linestyle="-", color="r", label=signal2_label)

    # Formatting the plot
    if title:
        plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)  # Save the figure to the file path provided
        plt.close()  # Close the plot figure to prevent it from displaying in the notebook/output


def plot_sensor_data(
    time_series,
    x_series,
    y_series,
    z_series,
    sensor_name,
    title="Sensor Data",
    downsample_factor=1,
):
    # Downsample the data
    time_series = time_series[::downsample_factor]
    x_series = x_series[::downsample_factor]
    y_series = y_series[::downsample_factor]
    z_series = z_series[::downsample_factor]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

    # Plot x-axis data
    ax1.plot(time_series, x_series, color="r")
    ax1.set_xlabel("Time")
    ax1.set_ylabel(f"{sensor_name} X")
    ax1.set_title(f"{sensor_name} X-Axis")
    ax1.minorticks_on()
    # ax1.grid(which="both", linestyle="-", linewidth="0.5")
    ax1.grid(which="minor", linestyle=":", linewidth="0.5", color="gray")
    ax1.xaxis.set_minor_locator(MaxNLocator(nbins=10))
    ax1.yaxis.set_minor_locator(MaxNLocator(nbins=10))

    # Plot y-axis data
    ax2.plot(time_series, y_series, color="g")
    ax2.set_xlabel("Time")
    ax2.set_ylabel(f"{sensor_name} Y")
    ax2.set_title(f"{sensor_name} Y-Axis")
    ax2.minorticks_on()
    # ax2.grid(which="both", linestyle="-", linewidth="0.5")
    ax2.grid(which="minor", linestyle=":", linewidth="0.5", color="gray")
    ax2.xaxis.set_minor_locator(MaxNLocator(nbins=10))
    ax2.yaxis.set_minor_locator(MaxNLocator(nbins=10))

    # Plot z-axis data
    ax3.plot(time_series, z_series, color="b")
    ax3.set_xlabel("Time")
    ax3.set_ylabel(f"{sensor_name} Z")
    ax3.set_title(f"{sensor_name} Z-Axis")
    ax3.minorticks_on()
    # ax3.grid(which="both", linestyle="-", linewidth="0.5")
    ax3.grid(which="minor", linestyle=":", linewidth="0.5", color="gray")
    ax3.xaxis.set_minor_locator(MaxNLocator(nbins=10))
    ax3.yaxis.set_minor_locator(MaxNLocator(nbins=10))

    # Set overall title
    plt.suptitle(title)

    # Adjust layout to prevent overlap
    plt.tight_layout(
        rect=[0, 0.03, 1, 0.95]
    )  # Adjust the rect parameter to make room for suptitle


def plot_sensor_data_classified(
    time_series,
    x_series,
    y_series,
    z_series,
    x_zeros_and_ones,
    y_zeros_and_ones,
    z_zeros_and_ones,
    sensor_name,
    title="Sensor Data",
):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

    # Plot x-axis data
    ax1.plot(time_series, x_series, color="r")
    ax1.scatter(
        time_series,
        x_zeros_and_ones,
        color="black",
        s=0.5,
        label="Threshold Output",
        zorder=10,
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel(f"{sensor_name} X")
    ax1.set_title(f"{sensor_name} X-Axis")
    ax1.grid(True)
    ax1.legend()

    # Plot y-axis data
    ax2.plot(time_series, y_series, color="g")
    ax2.scatter(
        time_series,
        y_zeros_and_ones,
        color="black",
        s=0.5,
        label="Threshold Output",
        zorder=10,
    )
    ax2.set_xlabel("Time")
    ax2.set_ylabel(f"{sensor_name} Y")
    ax2.set_title(f"{sensor_name} Y-Axis")
    ax2.grid(True)
    ax2.legend()

    # Plot z-axis data
    ax3.plot(time_series, z_series, color="b")
    ax3.scatter(
        time_series,
        z_zeros_and_ones,
        color="black",
        s=0.5,
        label="Threshold Output",
        zorder=10,
    )
    ax3.set_xlabel("Time")
    ax3.set_ylabel(f"{sensor_name} Z")
    ax3.set_title(f"{sensor_name} Z-Axis")
    ax3.grid(True)
    ax3.legend()

    # Set overall title
    plt.suptitle(title)

    # Adjust layout to prevent overlap
    plt.tight_layout()


def plot_lat_lon_with_highlights(
    latitudes, longitudes, highlight_indices, title="Latitude and Longitude Points"
):
    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(longitudes, latitudes, c="blue", label="Points")

    # Highlight specific points
    highlighted_latitudes = [latitudes[i] for i in highlight_indices]
    highlighted_longitudes = [longitudes[i] for i in highlight_indices]
    plt.scatter(
        highlighted_longitudes,
        highlighted_latitudes,
        c="red",
        label="Highlighted Points",
    )

    # Add title and labels
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()

    # Show the plot
    plt.grid(True)


def plot_rate_counts(rate_counts, title, save_path=None):
    rates = list(rate_counts.keys())
    counts = list(rate_counts.values())

    # Create a bar graph
    plt.figure(figsize=(10, 6))
    bars = plt.bar(rates, counts, color="blue")

    # Add titles and labels
    plt.title(title)
    plt.xlabel("Rate (Hz)")
    plt.ylabel("Count")

    # Set x-axis ticks to be evenly distributed
    # min_rate = min(rates)
    # max_rate = max(rates)
    # plt.xticks(range(min_rate, max_rate + 1))

    # Add grid
    plt.grid(True)

    # Ensure bars are in front of grid
    for bar in bars:
        bar.set_zorder(2)

    # Add count over each bar
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )
    if save_path:
        plt.savefig(save_path)  # Save the figure to the file path provided
        plt.close()


def plot_sensor_timestamps(sensor_data):
    """
    Plots sensor timestamps to visualize gaps in measurements.

    Args:
    sensor_data (dict): Dictionary where keys are sensor names and values are lists of timestamps in epoch milliseconds.
    """
    # Convert epoch milliseconds to seconds for plotting
    sensor_data_in_seconds = {
        sensor: [ts / 1000 for ts in timestamps]
        for sensor, timestamps in sensor_data.items()
    }

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot each sensor's timestamps
    num_sensors = len(sensor_data_in_seconds)
    spacing = 0.01  # Adjust this value to bring lines closer together
    for i, (sensor, timestamps) in enumerate(sensor_data_in_seconds.items()):
        plt.scatter(timestamps, [i * spacing] * len(timestamps), label=sensor, s=20)

    # Customize the plot
    plt.xlabel("Time (seconds since epoch)")
    plt.ylabel("Sensors")
    plt.yticks([i * spacing for i in range(num_sensors)], sensor_data.keys())
    plt.ylim(-spacing, num_sensors * spacing)  # Adjust plot limits to ensure visibility
    plt.title("Sensor Timestamps")
    plt.legend(loc="upper right")
    plt.grid(True)


### Plotting Code for Maps ###
def create_map_with_headings(
    latitude, longitude, heading, map_filename="temp", plot_every=1
):
    """
    Creates a map and plots every 'plot_every' points from the 'data'.

    Parameters:
    - data: List of dictionaries containing 'latitude', 'longitude', and 'heading'.
    - map_filename: Filename for the saved map.
    - plot_every: Interval at which points are plotted (1 = every point, 2 = every second point, etc.).
    """
    keys = ["latitude", "longitude", "heading"]
    data = [
        {keys[0]: val1, keys[1]: val2, keys[2]: val3}
        for val1, val2, val3 in zip(latitude, longitude, heading)
    ]

    # Create a map object centered around the average location
    m = folium.Map(
        location=[
            sum(p["latitude"] for p in data) / len(data),
            sum(p["longitude"] for p in data) / len(data),
        ],
        zoom_start=13,
    )  # Adjust zoom level as needed

    # Add points and short lines to the map, plotting every 'plot_every' points
    for i, point in enumerate(data):
        if i % plot_every == 0:  # Plot only every 'plot_every' points
            # Add a small circle dot for the point
            folium.CircleMarker(
                location=[point["latitude"], point["longitude"]],
                radius=3,  # small radius for the circle marker
                color="red",
                fill=True,
                fill_color="red",
            ).add_to(m)

            # Calculate end point for the short line
            line_length = 0.00003  # Adjust this for line length
            end_lat = point["latitude"] + line_length * math.cos(
                math.radians(point["heading"])
            )
            end_lon = point["longitude"] + line_length * math.sin(
                math.radians(point["heading"])
            )

            # Create a short line
            folium.PolyLine(
                [(point["latitude"], point["longitude"]), (end_lat, end_lon)],
                color="blue",
                weight=3,
                opacity=1,
            ).add_to(m)

    # Save the map to a temporary HTML file
    _, html_filename = tempfile.mkstemp(suffix=".html")
    m.save(html_filename)
    if map_filename != "temp":
        m.save(map_filename)

    # Open the HTML file in the default web browser
    webbrowser.open("file://" + os.path.realpath(html_filename))

    return html_filename


def create_map(
    latitude, longitude, highlight_indices=[], map_filename="temp", plot_every=1
):
    """
    Creates a map and plots every 'plot_every' points from the 'latitude' and 'longitude' lists.
    Highlight points specified by 'highlight_indices'.

    Parameters:
    - latitude: List of latitude values.
    - longitude: List of longitude values.
    - highlight_indices: List of indices to highlight.
    - map_filename: Filename for the saved map.
    - plot_every: Interval at which points are plotted (1 = every point, 2 = every second point, etc.).
    """
    # Create a map object centered around the average location
    m = folium.Map(
        location=[sum(latitude) / len(latitude), sum(longitude) / len(longitude)],
        zoom_start=13,
    )  # Adjust zoom level as needed

    # Add points and short lines to the map, plotting every 'plot_every' points
    for i, (lat, lon) in enumerate(zip(latitude, longitude)):
        if i % plot_every == 0:  # Plot only every 'plot_every' points
            # Add a small circle dot for the point
            color = "red" if i in highlight_indices else "blue"
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,  # small radius for the circle marker
                color=color,
                fill=True,
                fill_color=color,
            ).add_to(m)

    # Save the map to a temporary HTML file
    _, html_filename = tempfile.mkstemp(suffix=".html")
    m.save(html_filename)
    if map_filename != "temp":
        m.save(map_filename)

    # Open the HTML file in the default web browser
    webbrowser.open("file://" + os.path.realpath(html_filename))

    return html_filename
