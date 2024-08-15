import os
import argparse
import json

from optical_test_set_creation import (
    check_straight_line_segments,
    save_straight_segments_images,
)


def extract_images_from_file(input_file, output_dir):
    with open(input_file, "rb") as f:
        file_content = f.read().hex()

    image = []
    prev = ""
    cursor = 0
    image_names = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    started = False

    for i in range(0, len(file_content), 2):
        hex_value = "0x" + file_content[i] + file_content[i + 1]
        if started:
            image.append(hex_value)
        if hex_value == "0xd8" and prev == "0xff":
            if not started:
                image.append("0xff")
                image.append("0xd8")
            started = True
        elif hex_value == "0xd9" and prev == "0xff":
            cursor += 1
            image_name = f"{cursor}.jpg"
            image_path = os.path.join(output_dir, image_name)
            with open(image_path, "wb") as img_file:
                img_file.write(bytes(int(h, 16) for h in image))
            image_names.append(image_path)
            image = []
            started = False  # Reset the started flag after completing an image
        prev = hex_value

    print(f"Found {cursor} images from {input_file}")
    return image_names


def map_images_to_frames(image_names, json_data):
    frames = json_data.get("frames", [])
    if len(image_names) != len(frames):
        raise ValueError("Number of images and frames do not match")

    image_to_frame_mapping = {
        image_names[index]: frames[index] for index in range(len(image_names))
    }
    return image_to_frame_mapping


def read_json_file(input_file):
    json_file = input_file + ".json"
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
            # print(json.dumps(data, indent=4))
            return data
    else:
        print(f"No JSON file found with the name {json_file}")
        return None


def remove_non_directory_files(directory):
    """
    Remove every file from the given directory that is not a directory.

    Parameters:
    - directory: Path to the directory from which files should be removed.
    """
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
            print(f"Deleted file: {item_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract images from a hex-encoded file"
    )
    parser.add_argument(
        "--input_dir", type=str, help="Path to the hex-encoded input file"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory to save the extracted images"
    )

    args = parser.parse_args()

    for file in os.listdir(args.input_dir):
        cur_frame_path = os.path.join(args.input_dir, file)
        frame_output_dir = os.path.join(args.output_dir, file)
        image_names = extract_images_from_file(cur_frame_path, args.output_dir)
        json_dir_path, _ = os.path.split(args.input_dir)
        json_file_path = os.path.join(json_dir_path, "metadata", file)
        json_data = read_json_file(json_file_path)

        if json_data:
            image_to_frame_mapping = map_images_to_frames(image_names, json_data)
            # print("Image to Frame Mapping:")
            coordinates = []
            for image, frame in image_to_frame_mapping.items():
                coordinates.append((frame["lat"], frame["lon"]))
            # print("Coordinates:")
            # print(coordinates)
            print("Is straight line:", check_straight_line_segments(coordinates))
            save_straight_segments_images(
                image_names, frame_output_dir, check_straight_line_segments(coordinates)
            )
            remove_non_directory_files(args.output_dir)
