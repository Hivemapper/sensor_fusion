import folium
import json
import math
from pyproj import Proj, Transformer

# Define projections
wgs84 = Proj(proj='latlong', datum='WGS84')
cartesian = Proj(proj='geocent', datum='WGS84')

# Create a transformer object from WGS84 to Cartesian (geocentric)
forward_transformer = Transformer.from_proj(wgs84, cartesian, always_xy=True)
reverse_transformer = Transformer.from_proj(cartesian, wgs84, always_xy=True)


def plot_points_on_map(data):
    """
    Plot points on a map using Folium. Each point has a latitude, longitude, and label.
    Points with the same label are plotted in the same named color.

    Args:
        data (list of dicts): Each dictionary should have 'lat', 'lon', and 'label' keys.

    Returns:
        folium.Map: A Folium map object with plotted points.
    """
    # Define a list of named colors
    named_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
                    'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
                    'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
                    'gray', 'black', 'lightgray']

    # Create a base map
    map_center = [data[0]['lat'], data[0]['lon']] if data else [0, 0]
    map = folium.Map(location=map_center, zoom_start=12)

    # Assign a unique color to each label
    labels = {item['label'] for item in data}  # Set of unique labels
    if len(labels) > len(named_colors):
        raise ValueError("Not enough colors for the number of labels")
    color_map = dict(zip(labels, named_colors))  # Map each label to a named color
    print(color_map)

    # Add points to the map
    for item in data:
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



if __name__ == "__main__": 
    # read the json file
    with open('/Users/rogerberman/sensor-fusion/testingScripts/features.json') as f:
        data = json.load(f)

    labels = []
    for item in data:
        if item['label'] not in labels:
            labels.append(item['label'])

    print(labels)
    print(data[0])

    locations = []
    for detection in data:
        cam_dict = {}
        cam_dict['lat'] = detection['cam_lat']
        cam_dict['lon'] = detection['cam_lon']
        cam_dict['label'] = "camera"
        locations.append(cam_dict)
        sign_dict = {}
        x, y, z = forward_transformer.transform(cam_dict['lon'], cam_dict['lat'], 0)
        sign_x = x + detection['distance'] * math.sin(math.radians(detection['heading']))
        sign_y = y + detection['distance'] * math.cos(math.radians(detection['heading']))
        sign_dict['lon'], sign_dict['lat'], _ = reverse_transformer.transform(sign_x, sign_y, z)
        sign_dict['label'] = detection['label']
        locations.append(sign_dict)

    for location in locations:
        print(location)
    plot_points_on_map(locations)