import cv2
import numpy as np
import os
import time

from opticalFlowHelpers import (
    calculate_franeback_optical_flow,
    find_intersections_within_bounds,
    count_intersections_in_grid,
    get_top_x_sections,
    get_horizontal_range_of_top_sections,
    process_mag_and_angle_for_lines,
    draw_vertical_center_line,
    draw_horizontal_center_line,
    draw_overlay_flow_map,
    HORIZONTAL_FOV_DEGREE_PER_PIXEL,
)

LINES_STEP_SIZE = 12
GRID_SIZE = 16
NUMBER_OF_TOP_SECTIONS = 5


def calculate_optical_flow(directory, results_directory, parent_dir_name):
    startTime = time.time()
    # Get optical flow map
    average_flow, last_image_path, img_count = calculate_franeback_optical_flow(
        directory
    )
    print(f" Optical flow calculation time: {time.time() - startTime} seconds")
    # Grab magnitudes and angles from flow map
    h = average_flow.shape[0]
    w = average_flow.shape[1]
    pre_mag_time = time.time()
    magnitude, angle = cv2.cartToPolar(average_flow[..., 0], average_flow[..., 1])
    print(
        f" Magnitude and angle calculation time: {time.time() - pre_mag_time} seconds"
    )
    line_time = time.time()
    lines = process_mag_and_angle_for_lines(
        magnitude, angle, h, w, step=LINES_STEP_SIZE
    )
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

    ################# Image Plotting Section #################
    # Read a sample image for overlay
    sample_image = cv2.imread(last_image_path)
    intersection_img = np.zeros((h, w, 3), dtype=np.uint8)
    overlay_top_sections = np.zeros_like(sample_image)
    for point in intersection_points:
        cv2.line(
            intersection_img,
            (int(point[0]), int(point[1])),
            (int(point[0]), int(point[1])),
            (255, 0, 0),
            1,
        )

    # Highlight the top sections on the overlay image
    for y_idx, x_idx, _ in top_sections:
        cv2.rectangle(
            overlay_top_sections,
            (x_idx * GRID_SIZE, y_idx * GRID_SIZE),
            ((x_idx + 1) * GRID_SIZE, (y_idx + 1) * GRID_SIZE),
            (0, 255, 0),
            2,
        )
    # draw line at mid_x
    cv2.line(
        overlay_top_sections, (x_weighted_ave, 0), (x_weighted_ave, h), (0, 0, 255), 2
    )

    # Create a heatmap from the grid
    heatmap = np.uint8(255 * grid / np.max(grid))
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_NEAREST)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    ### Draw Veritical and Horizontal Center Lines
    intersection_img_resized = draw_vertical_center_line(intersection_img)
    sample_image_resized = draw_vertical_center_line(sample_image)
    intersection_img_resized = draw_horizontal_center_line(intersection_img_resized)
    sample_image_resized = draw_horizontal_center_line(sample_image_resized)

    ### Overlays
    flow_map_overlay = draw_overlay_flow_map(average_flow, sample_image_resized)
    # angle_map = draw_angle_map(angle, sample_image_resized)
    overlay_intersection_img = cv2.addWeighted(
        sample_image_resized, 0.5, intersection_img_resized, 0.5, 0
    )
    overlay_density_img = cv2.addWeighted(
        sample_image_resized, 0.5, colored_heatmap, 0.5, 0
    )
    overlay_top_sections_img = cv2.addWeighted(
        sample_image_resized, 0.7, overlay_top_sections, 0.3, 0
    )

    # Save the images to the results directory with the parent directory name in the filenames
    os.makedirs(results_directory, exist_ok=True)
    cv2.imwrite(
        os.path.join(
            results_directory,
            f"{parent_dir_name}_Farneback_intersection_{img_count}.jpg",
        ),
        intersection_img,
    )
    cv2.imwrite(
        os.path.join(
            results_directory,
            f"{parent_dir_name}_Farneback_overlay_intersection{img_count}.jpg",
        ),
        overlay_intersection_img,
    )
    cv2.imwrite(
        os.path.join(
            results_directory,
            f"{parent_dir_name}_Farneback_intersection_density_{img_count}.jpg",
        ),
        overlay_density_img,
    )
    cv2.imwrite(
        os.path.join(
            results_directory,
            f"{parent_dir_name}_Farneback_top_sections_{img_count}.jpg",
        ),
        overlay_top_sections_img,
    )
    cv2.imwrite(
        os.path.join(
            results_directory,
            f"{parent_dir_name}_Farneback_flow_map_overlay_{img_count}.jpg",
        ),
        flow_map_overlay,
    )
    # cv2.imwrite(
    #     os.path.join(results_directory, f"{parent_dir_name}_Farneback_angle_map.jpg"),
    #     angle_map,
    # )


if __name__ == "__main__":
    base_directory = "//Users/rogerberman/Desktop/frameKMImages/frontMount"
    results_base_directory = os.path.join(base_directory, "results")
    dirs = [
        d
        for d in os.listdir(base_directory)
        if os.path.isdir(os.path.join(base_directory, d))
    ]

    # Loop over each subdirectory and calculate/save optical flow
    for dir_name in dirs:
        full_path = os.path.join(base_directory, dir_name)
        if "results" in dir_name:
            continue
        results_directory = os.path.join(results_base_directory, dir_name)
        calculate_optical_flow(full_path, results_directory, dir_name)
