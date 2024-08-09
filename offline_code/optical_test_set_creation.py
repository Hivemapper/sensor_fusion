import os
import math
import shutil
from PIL import Image
from PIL.ExifTags import TAGS


def get_exif_data(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()
    if exif_data is not None:
        exif = {TAGS.get(tag): value for tag, value in exif_data.items() if tag in TAGS}
        return exif
    else:
        return {}


def read_images_from_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg"):
                image_path = os.path.join(root, file)
                print(f"Processing image: {image_path}")
                exif_data = get_exif_data(image_path)
                print(f"EXIF data for {file}:")
                for key, value in exif_data.items():
                    print(f"{key}: {value}")
                print("\n")


def calculate_bearing(pointA, pointB):
    """
    Calculate the bearing between two points.
    """
    lat1 = math.radians(pointA[0])
    lon1 = math.radians(pointA[1])
    lat2 = math.radians(pointB[0])
    lon2 = math.radians(pointB[1])

    dLon = lon2 - lon1

    x = math.sin(dLon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (
        math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    )

    initial_bearing = math.atan2(x, y)

    # Convert radians to degrees
    initial_bearing = math.degrees(initial_bearing)

    # Normalize the bearing to be within the range [0, 360)
    bearing = (initial_bearing + 360) % 360

    return bearing


def calculate_bearings(coordinates):
    """
    Calculate bearings between consecutive points in a list of coordinates.
    """
    bearings = []
    for i in range(len(coordinates) - 1):
        bearing = calculate_bearing(coordinates[i], coordinates[i + 1])
        bearings.append(bearing)
    return bearings


def check_straight_line_segments(coordinates, threshold=3):
    """
    Check for continuous straight-ish portions of a GNSS coordinate path.
    Returns a list of index ranges where the standard deviation of bearings is below a specified threshold.
    """
    if len(coordinates) < 3:
        return [
            (0, len(coordinates) - 1)
        ]  # Not enough points to determine, assume the entire path is straight-ish

    straight_segments = []
    bearings = calculate_bearings(coordinates)

    start_index = 0

    while start_index < len(bearings):
        for i in range(
            start_index + 2, len(bearings) + 1
        ):  # start with a minimum of 2 bearings
            segment_bearings = bearings[start_index:i]

            # Calculate the standard deviation of the segment bearings
            mean_bearing = sum(segment_bearings) / len(segment_bearings)
            variance = sum((b - mean_bearing) ** 2 for b in segment_bearings) / len(
                segment_bearings
            )
            std_dev = math.sqrt(variance)

            if std_dev > threshold:
                if (
                    i - start_index > 2
                ):  # Only consider segments with more than 2 points
                    straight_segments.append((start_index, i - 1))
                start_index = i - 1
                break
        else:
            if len(bearings) - start_index > 1:
                straight_segments.append((start_index, len(bearings)))
            break

        start_index += 1

    return straight_segments


def save_straight_segments_images(image_names, output_dir, straight_segments):
    """
    Save images corresponding to the straight segments to the output directory and remove them from the original location.

    Parameters:
    - image_names: List of image filenames (full paths).
    - output_dir: Directory where the images should be saved.
    - straight_segments: List of tuples with start and end indexes of straight segments.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create a new empty output directory

    for segment_num, (start_idx, end_idx) in enumerate(straight_segments):
        num_frames = end_idx - start_idx + 1

        if num_frames >= 10:
            segment_dir = os.path.join(output_dir, f"segment_{segment_num+1}")
            os.makedirs(segment_dir)

            for idx in range(start_idx, end_idx + 1):
                image_name = image_names[idx]
                shutil.move(image_name, segment_dir)  # Move the file instead of copying
                print(f"Moved {image_name} to {segment_dir}")
        else:
            for idx in range(start_idx, end_idx + 1):
                image_name = image_names[idx]
                os.remove(
                    image_name
                )  # Delete the file if the segment has fewer than 10 frames
                print(f"Deleted {image_name}")

    # Check if output_dir is empty
    if not any(os.scandir(output_dir)):
        shutil.rmtree(output_dir)
        print(f"Deleted empty output directory: {output_dir}")


if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing JPEG images: ")
    read_images_from_folder(folder_path)
