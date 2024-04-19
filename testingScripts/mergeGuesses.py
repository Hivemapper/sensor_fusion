import folium
import json
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import viridis
import numpy as np


def plot_points_on_map(data):
    """
    Plot points on a map using Folium. Each point has a latitude, longitude, and label.
    Points with the same label are plotted in the same named color.

    Args:
        data (list of dicts): Each dictionary should have 'lat', 'lon', and 'label' keys.

    Returns:
        folium.Map: A Folium map object with plotted points.
    """
    # If more colors are needed look here
    # named_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
    #                 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
    #                 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
    #                 'gray', 'black', 'lightgray']
    color_map = {'stop-sign': 'red', 'speed-sign': 'blue', 'turn-rules-sign': 'green', 'camera': 'gray'}

    # Create a base map
    map_center = [data[0]['lat'], data[0]['lon']] if data else [0, 0]
    map = folium.Map(location=map_center, zoom_start=12)

    # Add points to the map
    for item in data:
        if item['label'] == 'camera':
            folium.CircleMarker(
                location=[item['lat'], item['lon']],
                radius=5,
                popup=f"{item['label']}_{item['frameId']}",
                color=color_map[item['label']],
                fill=True,
                fill_color=color_map[item['label']],
                fill_opacity=0.7
            ).add_to(map)
        else:
            folium.CircleMarker(
                location=[item['lat'], item['lon']],
                radius=5,
                popup=item['label'],
                color=color_map[item['label']],
                fill=True,
                fill_color=color_map[item['label']],
                fill_opacity=0.7
            ).add_to(map)

    map.save('map.html')

def make_confidence_heatmap(width, height, cellSize, maxHeatValue, minHeatValue):
        # Create an empty heatmap array
        heatmap = np.zeros((height, width))
        # Calculate the center
        center_x, center_y = width // 2, height // 2
        # Calculate the maximum distance from the center to a corner (for normalization)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        # Iterate over each cell in the heatmap
        for i in range(0, width, cellSize):
            for j in range(0, height, cellSize):
                # Calculate distance from the center
                dist = np.sqrt((center_x - i)**2 + (center_y - j)**2)
                # Normalize the distance to get a value between minHeatValue and maxHeatValue
                normalized_dist = dist / max_dist
                value = (1 - normalized_dist) * (maxHeatValue - minHeatValue) + minHeatValue
                # Set the value for all pixels within this cell
                heatmap[j:j+cellSize, i:i+cellSize] = value
        return heatmap

def get_confidence_heatmap_average(heatmap, bbox):
    """
    Get the average heat value of a bounding box from a heatmap.

    Parameters:
    - heatmap (np.array): The heatmap array to sample from.
    - bbox (list): The bounding box coordinates [x1, y1, x2, y2].

    Returns:
    - float: The average heat value of the bounding box.
    """
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > heatmap.shape[1] or y2 > heatmap.shape[0]:
        raise ValueError('Bounding box coordinates out of bounds')
    if x1 > x2 or y1 > y2:
        raise ValueError('Bounding box coordinates are invalid')

    return np.mean(heatmap[y1:y2, x1:x2])
    
def visualize_confidence_heatmap(heatmap, cell_size):
    """
    Visualizes a given heatmap with a grid overlay.

    Parameters:
    - heatmap (np.array): The heatmap array to visualize.
    - cell_size (int): The size of each cell in pixels.
    """
    height, width = heatmap.shape
    fig, ax = plt.subplots()
    
    # Set plot limits and grid properties
    ax.set_xlim([0, width])
    ax.set_ylim([0, height])
    ax.set_xticks(np.arange(0, width + 1, cell_size))
    ax.set_yticks(np.arange(0, height + 1, cell_size))
    ax.grid(which='both', color='gray', linestyle='-', linewidth=1)
    
    # Hide tick labels and remove ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    
    # Invert Y axis to match the image coordinate system
    ax.invert_yaxis()
    
    # Normalize values for color mapping
    norm = Normalize(vmin=np.min(heatmap), vmax=np.max(heatmap))
    cmap = viridis

    # Plot the heatmap with colored rectangles
    for i in range(0, width, cell_size):
        for j in range(0, height, cell_size):
            # Get the average value for this cell from the heatmap
            value = np.mean(heatmap[j:j + cell_size, i:i + cell_size])
            color = cmap(norm(value))
            rect = Rectangle((i, j), cell_size, cell_size, color=color, edgecolor=None)
            ax.add_patch(rect)
            # Add text annotation at the center of the rectangle
            ax.text(i + cell_size / 2, j + cell_size / 2, f'{value:.2f}',
                    color='white' if value < (np.max(heatmap) + np.min(heatmap)) / 2 else 'black',  # Contrast color for visibility
                    ha='center', va='center', fontsize=8)
    plt.show()



if __name__ == "__main__": 
    # read the json file
    with open('/Users/rogerberman/sensor-fusion/testingScripts/features.json') as f:
        data = json.load(f)

    filtered_data = []
    for detection in data:
        if detection['eph'] < 10:
            filtered_data.append(detection)
    print(f"Number of detections: {len(filtered_data)}")
    print(filtered_data[0])

    locations = []
    for detection in filtered_data:
        cam_dict = {}
        cam_dict['lat'] = detection['cam_lat']
        cam_dict['lon'] = detection['cam_lon']
        cam_dict['label'] = "camera"
        cam_dict['frameId'] = detection['frameId']
        locations.append(cam_dict)
        sign_dict = {}
        sign_dict['lat'] = detection['sign_lat']
        sign_dict['lon'] = detection['sign_lon']
        sign_dict['label'] = detection['label']
        sign_dict['frameId'] = detection['frameId']
        sign_dict['distance'] = detection['distance']
        locations.append(sign_dict)

    # for location in locations:
    #     print(location)
    # plot_points_on_map(locations)

    width, height = 480, 640
    cell_size = 40
    max_heat_value = 1
    min_heat_value = 0.5
    heatmap = make_confidence_heatmap(width, height, cell_size, max_heat_value, min_heat_value)
    # print(heatmap)
    visualize_confidence_heatmap(heatmap, cell_size)
    for detection in filtered_data:
        print(get_confidence_heatmap_average(heatmap, detection['box']))
    




    # given I have a heat map of the image
    # i want to create a function where i can input a bounding box and get the average heat value of the bounding box
    
    # bbox input is top left corner and bottom right corner
    # bbox = [x1, y1, x2, y2]
