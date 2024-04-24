import numpy as np
from pyproj import Proj, Transformer, Geod

# Define projections
wgs84 = Proj(proj='latlong', datum='WGS84')
cartesian = Proj(proj='geocent', datum='WGS84')

# Create a transformer object from WGS84 to Cartesian (geocentric)
forward_transformer = Transformer.from_proj(wgs84, cartesian, always_xy=True)
reverse_transformer = Transformer.from_proj(cartesian, wgs84, always_xy=True)


################### Top Level Functions ###################
def merge_guesses(detection_data):
    groups_found = find_detection_groups(detection_data)

    # Find average coordinates for each group
    ave_locations = []
    for label in groups_found:
        for detections in groups_found[label]:
            ave_detection = {}
            if len(detections) == 0:
                continue
            elif len(detections) == 1:
                ave_detection['lat'] = detections[0]['sign_lat']
                ave_detection['lon'] = detections[0]['sign_lon']
                ave_detection['label'] = detections[0]['label']
                ave_locations.append(ave_detection)
            elif len(detections) == 2:
                lat, lon, alt = average_coordinates(detections)
                ave_detection['lat'] = lat
                ave_detection['lon'] = lon
                ave_detection['label'] = detections[0]['label']
                ave_locations.append(ave_detection)
            else:
                coordinates = []
                for detection in detections:
                    coordinates.append([detection['sign_lat'], detection['sign_lon']])
                centroids, _ = k_means(np.array(coordinates), num_clusters=1)
                ave_detection['lat'] = centroids[0][0]
                ave_detection['lon'] = centroids[0][1]
                ave_detection['label'] = detections[0]['label']
                ave_locations.append(ave_detection)
    return ave_locations

################### Helper Functions ###################
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