import cv2
import numpy as np
import os
import time

from sensor_fusion.vision_based_orientation.optical_flow_helpers import (
    calculate_farneback_optical_flow,
    find_intersections_within_bounds,
    count_intersections_in_grid,
    get_top_x_sections,
    get_horizontal_range_of_top_sections,
    process_mag_and_angle_for_lines_optimized,
    create_output_images,
    undistort_via_exif,
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
    if debug:
        print(f" Optical flow calculation time: {time.time() - startTime} seconds")
    # Grab magnitudes and angles from flow map
    h = average_flow.shape[0]
    w = average_flow.shape[1]
    pre_mag_time = time.time()
    magnitude, angle = cv2.cartToPolar(average_flow[..., 0], average_flow[..., 1])
    if debug:
        print(
            f" Magnitude and angle calculation time: {time.time() - pre_mag_time} seconds"
        )
    line_time = time.time()
    lines = process_mag_and_angle_for_lines_optimized(
        magnitude, angle, h, w, step=LINES_STEP_SIZE
    )
    if debug:
        print(f" Line processing time: {time.time() - line_time} seconds")
    # Find intersection points within bounds
    intersection_find_time = time.time()
    intersection_points = find_intersections_within_bounds(lines, w, h)
    print(f" Intersection finding time: {time.time() - intersection_find_time} seconds")
    ### Math and drawing done for intersection point densities
    grid_count_time = time.time()
    grid = count_intersections_in_grid(intersection_points, w, h, GRID_SIZE)
    print(f" Grid count time: {time.time() - grid_count_time} seconds")
    ##### Math and overlay for top grid densities
    top_sections_time = time.time()
    top_sections = get_top_x_sections(grid, NUMBER_OF_TOP_SECTIONS)
    x_min, x_max, x_weighted_ave = get_horizontal_range_of_top_sections(
        top_sections, GRID_SIZE
    )
    # mid_x = (x_min + x_max) // 2
    device_offset = ((w // 2) - x_weighted_ave) * HORIZONTAL_FOV_DEGREE_PER_PIXEL
    print(f" Top sections calculation time: {time.time() - top_sections_time} seconds")
    # print(f" Post flow processing time: {time.time() - line_time} seconds")
    print(f" Total time taken: {time.time() - startTime} seconds")
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
    return device_offset


if __name__ == "__main__":
    base_directory = "/Users/rogerberman/Desktop/frameKMImages/frontMount"
    results_base_directory = os.path.join(base_directory, "results")
    dirs = [
        d
        for d in os.listdir(base_directory)
        if os.path.isdir(os.path.join(base_directory, d))
    ]

    # Undistort images
    for dir_name in dirs:
        if "results" in dir_name:
            continue
        full_path = os.path.join(base_directory, dir_name)
        new_path = os.path.join(base_directory, dir_name + "_undistorted")
        # create new directory
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        for image in os.listdir(full_path):
            if image.endswith(".jpg"):
                try:
                    undistort_via_exif(
                        os.path.join(full_path, image),
                        os.path.join(new_path, image),
                        True,
                    )
                except Exception as e:
                    print(f"Error with {image}: {e}")

    # Loop over each subdirectory and calculate/save optical flow
    # for dir_name in dirs:
    #     full_path = os.path.join(base_directory, dir_name)
    #     if "results" in dir_name:
    #         continue
    #     results_directory = os.path.join(results_base_directory, dir_name)
    #     calculate_optical_flow(full_path, results_directory, dir_name)
