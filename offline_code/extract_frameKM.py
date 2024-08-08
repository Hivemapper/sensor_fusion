import os
import argparse
import json


def extract_images_from_file(input_file, output_dir):
    with open(input_file, "rb") as f:
        file_content = f.read().hex()

    image = []
    prev = ""
    cursor = 0
    image_names = []

    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
    else:
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
            print("start of image detected")
        elif hex_value == "0xd9" and prev == "0xff":
            print("end of image detected")
            cursor += 1
            image_name = f"{cursor}.jpg"
            with open(os.path.join(output_dir, image_name), "wb") as img_file:
                img_file.write(bytes(int(h, 16) for h in image))
            image_names.append(image_name)
            image = []
            started = False  # Reset the started flag after completing an image
        prev = hex_value

    print(f"found {cursor} images")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract images from a hex-encoded file"
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the hex-encoded input file"
    )
    parser.add_argument(
        "output_dir", type=str, help="Directory to save the extracted images"
    )

    args = parser.parse_args()

    image_names = extract_images_from_file(args.input_file, args.output_dir)
    json_data = read_json_file(args.input_file)

    if json_data:
        image_to_frame_mapping = map_images_to_frames(image_names, json_data)
        print("Image to Frame Mapping:")
        print(json.dumps(image_to_frame_mapping, indent=4))
