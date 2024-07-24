import sys
sys.path.insert(0, '/Users/rogerberman/sensor-fusion')  # Add the project root to the Python path

import folium
import json
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import viridis
from PIL import Image
import numpy as np
from pyproj import Proj, Transformer, Geod
from detections import merge_guesses

# Define projections
wgs84 = Proj(proj='latlong', datum='WGS84')
cartesian = Proj(proj='geocent', datum='WGS84')

# Create a transformer object from WGS84 to Cartesian (geocentric)
forward_transformer = Transformer.from_proj(wgs84, cartesian, always_xy=True)
reverse_transformer = Transformer.from_proj(cartesian, wgs84, always_xy=True)


def plot_points_on_map(data, map_path='map.html'):
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
    color_map = {'stop-sign': 'red', 'speed-sign': 'blue', 'turn-rules-sign': 'green', 'camera': 'gray', 'ave-sign': 'purple'}

    # Create a base map
    map_center = [data[0]['lat'], data[0]['lon']] if data else [0, 0]
    map = folium.Map(location=map_center, zoom_start=12)

    # Add points to the map
    for item in data:
        if item['label'] == 'camera':
            folium.CircleMarker(
                location=[item['lat'], item['lon']],
                radius=5,
                popup=f"{item['label']}_{item['frame_id']}",
                color=color_map[item['label']],
                fill=True,
                fill_color=color_map[item['label']],
                fill_opacity=0.7
            ).add_to(map)
        elif 'ave' in item['label']:
            folium.CircleMarker(
                location=[item['lat'], item['lon']],
                radius=5,
                popup=f"{item['label']}",
                color=color_map['ave-sign'],
                fill=True,
                fill_color=color_map['ave-sign'],
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

    map.save(map_path)

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

def overlay_heatmap_on_image(image_path, heatmap):
    # Load the image
    img = Image.open(image_path)
    img = img.convert('RGBA')  # Ensure image is in RGBA format for transparency handling

    # Print the shape of the original image
    img_array = np.array(img)  # Convert to numpy array to access the shape
    print("Image shape:", img_array.shape)

    # Create an RGBA image for the heatmap
    norm = Normalize(vmin=np.min(heatmap), vmax=np.max(heatmap))
    cmap = viridis  # You can choose any other colormap that suits your preference
    heatmap_rgba = cmap(norm(heatmap))  # This will give us a RGBA heatmap
    heatmap_rgba = (heatmap_rgba * 255).astype(np.uint8)  # Scale to 0-255

    # Print the shape of the heatmap
    print("Heatmap shape:", heatmap_rgba.shape[:2])  # Only print the spatial dimensions

    # Create a PIL image from the heatmap array
    heatmap_img = Image.fromarray(heatmap_rgba, 'RGBA')
    
    # Blend the heatmap and the image
    blended_img = Image.blend(img, heatmap_img, alpha=0.7)  # alpha controls the transparency

    # Display the blended image
    plt.imshow(blended_img)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()


def meters_to_latlon(distance, lat, lon, heading):
    # Convert heading to radians
    bearing = math.radians(heading)

    # Convert the initial latitude and longitude to cartesian coordinates (meters)
    x, y = forward_transformer.transform(lon, lat)
    
    # Compute new coordinates based on distance and bearing
    delta_x = distance * math.sin(bearing)
    delta_y = distance * math.cos(bearing)
    new_x = x + delta_x
    new_y = y + delta_y
    
    # Convert back to latitude and longitude
    new_lon, new_lat = reverse_transformer.transform(new_x, new_y)

    return new_lat, new_lon

def calculate_distance(point1, point2):
    # Define the ellipsoid for WGS 84 (which is the default)
    geod = Geod(ellps='WGS84')
    lat1, lon1 = point1
    lat2, lon2 = point2
    # Calculate the geodesic distance between the two points
    _, _, distance = geod.inv(lon1, lat1, lon2, lat2)
    
    return distance

def k_means(coordinates, num_clusters=1, distance_func=calculate_distance, max_iters=100):
    # Initialize centroids by randomly selecting points from the dataset
    centroids = coordinates[np.random.choice(len(coordinates), num_clusters, replace=False)]

    for _ in range(max_iters):
        # Assign each point to the closest centroid
        clusters = {i: [] for i in range(num_clusters)}
        for x in coordinates:
            # Compute distance to each centroid
            closest_centroid = np.argmin([distance_func(x, centroid) for centroid in centroids])
            clusters[closest_centroid].append(x)

        # Update centroids to be the mean of points in each cluster
        new_centroids = np.array([np.mean(clusters[i], axis=0) for i in range(num_clusters)])

        # Check if centroids have changed
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, clusters

def average_coordinates(detections):

    num_detections = len(detections)
    # Prepare detections
    for d in detections:
        # Convert the coordinates to Cartesian
        longitude = d['sign_lon']
        latitude = d['sign_lat']
        altitude = 0.0
        x, y, z = forward_transformer.transform(longitude, latitude, altitude)
        d['x'] = x
        d['y'] = y
        d['z'] = z

    x_coords = []
    y_coords = []
    z_coords = []
    for d in detections:
        x_coords.append(d['x'])
        y_coords.append(d['y'])
        z_coords.append(d['z'])

    sum_x = sum(x_coords)
    sum_y = sum(y_coords)
    sum_z = sum(z_coords)
    
    # Calculating the weighted average coordinates
    average_x = sum_x / num_detections
    average_y = sum_y / num_detections
    average_z = sum_z / num_detections

    # Convert the average coordinates back to geographic
    average_lon, average_lat, average_alt = reverse_transformer.transform(average_x, average_y, average_z)

    return  average_lat, average_lon, average_alt

def find_detection_groups(detections, print_groups=False):
    groups_found = {}
    for detection in detections:
        if detection['label'] not in groups_found:
            groups_found[detection['label']] = [[detection]]
            # print(groups_found)
        else:
            latest_group = groups_found[detection['label']][-1]
            latest_group_len = len(latest_group)
            # print(latest_group)
            if detection['frame_id'] - latest_group[-1]['frame_id'] < 5 - latest_group_len:
                groups_found[detection['label']][-1].append(detection)
            else:
                groups_found[detection['label']].append([detection])

    if print_groups:
        for group in groups_found:
            print(f"Label: {group}")
            for i in range(len(groups_found[group])):
                print(f"Group {i+1}")
                for detection in groups_found[group][i]:
                    print(f"Frame: {detection['frame_id']}, Distance: {detection['distance']}, Time: {detection['timestamp']}")

    return groups_found


if __name__ == "__main__": 
    # read the json file
    with open('/Users/rogerberman/sensor-fusion/testData/features_minnesota_st.json') as f:
        data = json.load(f)

    # pre-process the data
    filtered_data = []
    for detection in data:
        if detection['eph'] < 10:
            filtered_data.append(detection)
    print(f"Number of detections: {len(filtered_data)}")

    # original detections in filtered_data:
    locations = []
    for detection in filtered_data:
        cam_dict = {}
        cam_dict['lat'] = detection['cam_lat']
        cam_dict['lon'] = detection['cam_lon']
        cam_dict['label'] = "camera"
        cam_dict['frame_id'] = detection['frame_id']
        locations.append(cam_dict)
        sign_dict = {}
        sign_dict['lat'] = detection['sign_lat']
        sign_dict['lon'] = detection['sign_lon']
        sign_dict['label'] = detection['label']
        sign_dict['frame_id'] = detection['frame_id']
        sign_dict['distance'] = detection['distance']
        locations.append(sign_dict)

    # for location in locations:
    #     print(location)
    plot_points_on_map(locations)

    # width, height = 640, 480
    # cell_size = 10
    # max_heat_value = 1
    # min_heat_value = 0.5
    # heatmap = make_confidence_heatmap(width, height, cell_size, max_heat_value, min_heat_value)


    ave_locations = merge_guesses(filtered_data)
    for location in ave_locations:
        location['label'] = 'ave-'+location['label']
    print(ave_locations)


    combined_locations = locations + ave_locations
    plot_points_on_map(combined_locations, '/Users/rogerberman/sensor-fusion/testData/combined_map.html')


            



    


    # Cases:
    # 1. Single detection
    #   - Take the detection as is
    #   - Potentially to apply distance correction based on confidence heatmap (Distance based)
    # 2. Two detections
    #   - Take the average of the two detections
    #   - Potentially to apply distance weight based on confidence heatmap (Distance based)
    # 3. Three or more detections
    #   - Find cluster by distance
    #   - Weight detections based on distance

    




    # Applied confidence heatmap to the detections (Experimental)
    

    # plot_points_on_map(mod_locations, 'mod_map.html')








# print(heatmap)
    # visualize_confidence_heatmap(heatmap, cell_size)
    # frame_id = 1689
    # image_path = f"/Users/rogerberman/Downloads/frames_with_depth/{frame_id}_0.jpg"
    # overlay_heatmap_on_image(image_path, heatmap)