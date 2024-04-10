import matplotlib.pyplot as plt
import folium
import math

def plot_signal_over_time(seconds, signal_values, signal_label='Signal'):
    """
    Plots signal values over time, where time is represented in seconds.
    Parameters:
    - seconds: A list of timestamps in seconds.
    - signal_values: A list of signal values corresponding to each timestamp.
    - signal_label: A string label for the signal being plotted (e.g., 'Temperature', 'Speed').
    """
    # Ensure the lists are of the same length
    if len(seconds) != len(signal_values):
        print("Error: The lists of timestamps and signal values must have the same length.")
        return
    
    plt.figure(figsize=(10, 6))  # Adjust figure size as desired
    plt.plot(seconds, signal_values, linestyle='-', color='b')

    # Formatting the plot
    plt.title(f'{signal_label} Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel(signal_label)

    plt.grid(True)
    plt.show()

def plot_signals_over_time(seconds, signal1_values, signal2_values, signal1_label='Signal 1', signal2_label='Signal 2', save_path=None):
    """
    Plots two signal values over time, where time is represented in seconds, on the same plot for comparison.
    Optionally saves the plot to a file if a filepath is provided.
    
    Parameters:
    - seconds: A list of timestamps in seconds.
    - signal1_values: A list of first set of signal values corresponding to each timestamp.
    - signal2_values: A list of second set of signal values corresponding to each timestamp.
    - signal1_label: A string label for the first signal being plotted (e.g., 'Temperature').
    - signal2_label: A string label for the second signal being plotted (e.g., 'Humidity').
    - save_path: Optional. A string representing the file path where the plot will be saved. If None, the plot will be displayed.
    """
    # Ensure the lists are of the same length
    if len(seconds) != len(signal1_values) or len(seconds) != len(signal2_values):
        print("Error: The lists of timestamps and signal values must have the same length.")
        return
    
    plt.figure(figsize=(12, 6))  # Adjust figure size as desired
    
    # Plot the signals
    plt.plot(seconds, signal1_values, linestyle='-', color='b', label=signal1_label)
    plt.plot(seconds, signal2_values, linestyle='-', color='r', label=signal2_label)
    # Formatting the plot
    plt.title(f'{signal1_label} and {signal2_label} Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Value')
    plt.legend() 
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)  # Save the figure to the file path provided
        plt.close()  # Close the plot figure to prevent it from displaying in the notebook/output
    else:
        plt.show()  # Display the plot if no file path is provided


def create_map(latitude, longitude, heading, map_filename, plot_every=1):
    """
    Creates a map and plots every 'plot_every' points from the 'data'.
    
    Parameters:
    - data: List of dictionaries containing 'latitude', 'longitude', and 'heading'.
    - map_filename: Filename for the saved map.
    - plot_every: Interval at which points are plotted (1 = every point, 2 = every second point, etc.).
    """
    keys = ['latitude', 'longitude', 'heading']
    data = [{keys[0]: val1, keys[1]: val2, keys[2]: val3} for val1, val2, val3 in zip(latitude, longitude, heading)]

    # Create a map object centered around the average location
    m = folium.Map(location=[sum(p['latitude'] for p in data) / len(data), 
                             sum(p['longitude'] for p in data) / len(data)], 
                   zoom_start=13)  # Adjust zoom level as needed

    # Add points and short lines to the map, plotting every 'plot_every' points
    for i, point in enumerate(data):
        if i % plot_every == 0:  # Plot only every 'plot_every' points
            # Add a small circle dot for the point
            folium.CircleMarker(
                location=[point['latitude'], point['longitude']],
                radius=3,  # small radius for the circle marker
                color='red',
                fill=True,
                fill_color='red'
            ).add_to(m)

            # Calculate end point for the short line
            line_length = 0.00003  # Adjust this for line length
            end_lat = point['latitude'] + line_length * math.cos(math.radians(point['heading']))
            end_lon = point['longitude'] + line_length * math.sin(math.radians(point['heading']))

            # Create a short line
            folium.PolyLine([(point['latitude'], point['longitude']), (end_lat, end_lon)], 
                            color='blue', weight=3, opacity=1).add_to(m)

    # Save the map to an HTML file
    m.save(map_filename)
