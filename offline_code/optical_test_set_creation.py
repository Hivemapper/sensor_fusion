import os
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


if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing JPEG images: ")
    read_images_from_folder(folder_path)
