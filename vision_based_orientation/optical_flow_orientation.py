import cv2
import numpy as np
import os
import time
import argparse
import json

from sensor_fusion.vision_based_orientation.optical_flow_helpers import (
    calculate_farneback_optical_flow,
    find_intersections_within_bounds,
    count_intersections_in_grid,
    get_top_x_sections,
    get_horizontal_range_of_top_sections,
    process_mag_and_angle_for_lines_optimized,
    create_output_images,
    undistort_image,
    HORIZONTAL_FOV_DEGREE_PER_PIXEL,
    LINES_STEP_SIZE,
    GRID_SIZE,
    NUMBER_OF_TOP_SECTIONS,
)


def calculate_optical_flow(
    directory, results_directory=None, parent_dir_name=None, debug=False
):
    startTime = time.time()
    # Get optical flow map
    average_flow, last_image_path, img_count = calculate_farneback_optical_flow(
        directory
    )
    optical_flow_time_taken = time.time() - startTime
    if debug:
        print(f" Optical flow calculation time: {optical_flow_time_taken} seconds")
    # Grab magnitudes and angles from flow map
    h = average_flow.shape[0]
    w = average_flow.shape[1]
    pre_mag_time = time.time()
    magnitude, angle = cv2.cartToPolar(average_flow[..., 0], average_flow[..., 1])
    mag_and_angle_time_taken = time.time() - pre_mag_time
    if debug:
        print(
            f" Magnitude and angle calculation time: {mag_and_angle_time_taken} seconds"
        )
    line_time = time.time()
    lines = process_mag_and_angle_for_lines_optimized(
        magnitude, angle, h, w, step=LINES_STEP_SIZE
    )
    line_time_taken = time.time() - line_time
    if debug:
        print(f" Line processing time: {line_time_taken} seconds")
    # Find intersection points within bounds
    intersection_find_time = time.time()
    intersection_points = find_intersections_within_bounds(lines, w, h)
    intersection_find_time_taken = time.time() - intersection_find_time
    if debug:
        print(f" Intersection finding time: {intersection_find_time_taken} seconds")
    ### Math and drawing done for intersection point densities
    grid_count_time = time.time()
    grid = count_intersections_in_grid(intersection_points, w, h, GRID_SIZE)
    grid_count_time_taken = time.time() - grid_count_time
    print(f" Grid count time: {grid_count_time_taken} seconds")
    ##### Math and overlay for top grid densities
    top_sections_time = time.time()
    top_sections = get_top_x_sections(grid, NUMBER_OF_TOP_SECTIONS)
    top_sections_time_taken = time.time() - top_sections_time
    if debug:
        print(f" Top sections calculation time: {top_sections_time_taken} seconds")
    x_min, x_max, x_weighted_ave = get_horizontal_range_of_top_sections(
        top_sections, GRID_SIZE
    )
    # mid_x = (x_min + x_max) // 2
    device_offset = ((w // 2) - x_weighted_ave) * HORIZONTAL_FOV_DEGREE_PER_PIXEL
    total_time_taken = time.time() - startTime
    if debug:
        print(f" Total time taken: {total_time_taken} seconds")
        print(f" Device offset: {device_offset} degrees")

    create_output_images(
        last_image_path,
        top_sections,
        grid,
        x_weighted_ave,
        parent_dir_name,
        results_directory,
        img_count,
        h,
        w,
    )

    stats = {
        "directory": directory,
        "Optical flow calculation time": optical_flow_time_taken,
        "Magnitude and angle calculation time": mag_and_angle_time_taken,
        "Line processing time": line_time_taken,
        "Intersection finding time": intersection_find_time_taken,
        "Grid count time": grid_count_time_taken,
        "Top sections calculation time": top_sections_time_taken,
        "Total time taken": total_time_taken,
        "device_offset": device_offset,
    }

    return stats


def write_list_of_dicts_to_json(list_of_dicts, output_file):
    """
    Writes a list of dictionaries to a JSON file.

    Parameters:
    list_of_dicts (list): A list of dictionaries to write to the JSON file.
    output_file (str): The path to the output JSON file.

    Returns:
    bool: True if the file was written successfully, False otherwise.
    """
    try:
        with open(output_file, "w") as json_file:
            json.dump(list_of_dicts, json_file, indent=4)
        return True
    except (IOError, TypeError) as e:
        print(f"Error writing to {output_file}: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optical flow calculation for vision based orientation"
    )
    parser.add_argument(
        "-undistort", action="store_true", help="Undistort images in the directory"
    )
    parser.add_argument(
        "-input_dir",
        type=str,
        help="Directory containing images to calculate optical flow for",
    )
    parser.add_argument(
        "-res_name",
        type=str,
        help="Directory to save optical flow results to",
    )
    args = parser.parse_args()

    # Game plan:
    # 1. Iterate over all directories in the base directory
    # 2. For each directory, undistort all images in the directory (replace or create new dir for them)
    # 3. Calculate optical flow for all images in the directory (don't limit to number of frames)
    # - experiment with down sizing images + cropping for faster proccessing of pure optcal flow calculation
    # - figure out how to easily scale up and down the image size to see performance differences
    # 4. Save the results to a new directory in each segment repo

    if args.undistort:
        for dir in os.listdir(args.input_dir):
            if ".DS_Store" in dir or ".json" in dir:
                continue
            sub_dir = os.path.join(args.input_dir, dir)
            for segment in os.listdir(sub_dir):
                if ".DS_Store" in segment or "_undistorted" in segment:
                    continue
                segment_dir = os.path.join(sub_dir, segment)
                sgement_undistorted_dir = os.path.join(
                    sub_dir, segment + "_undistorted"
                )
                for image in os.listdir(segment_dir):
                    if image.endswith(".jpg"):
                        try:
                            undistort_image(
                                os.path.join(segment_dir, image),
                                os.path.join(sgement_undistorted_dir, image),
                                True,
                            )
                        except Exception as e:
                            print(f"Error with {image}: {e}")

    results = []
    for dir in os.listdir(args.input_dir):
        if ".DS_Store" in dir:
            continue
        sub_dir = os.path.join(args.input_dir, dir)
        for segment in os.listdir(sub_dir):
            if ".DS_Store" in segment or "_results" in segment:
                continue
            segment_dir = os.path.join(sub_dir, segment)
            results_dir = os.path.join(sub_dir, segment + "_results")
            stats = calculate_optical_flow(segment_dir, results_dir, segment, True)
            results.append(stats)

    res_output_path = os.path.join(
        args.input_dir, f"{args.res_name}_optical_flow_stats.json"
    )
    print(f"Writing results to {res_output_path}")
    write_list_of_dicts_to_json(results, res_output_path)
