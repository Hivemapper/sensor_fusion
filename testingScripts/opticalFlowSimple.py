import cv2
import numpy as np
import os

from opticalFlowHelpers import (
    calculate_boundary_points,
    line_from_slope_and_point,
    find_intersections_within_bounds,
    count_intersections_in_grid,
    get_top_x_sections,
    get_horizontal_range_of_top_sections,
    draw_vertical_center_line,
    draw_horizontal_center_line,
    HORIZONTAL_FOV_DEGREE_PER_PIXEL,
)


def calculate_farneback_optical_flow(directory, results_directory, parent_dir_name):
    print(f"FrameKM being evaluated: {directory}")

    # Get a sorted list of image filenames
    image_files = sorted(
        [f for f in os.listdir(directory) if f.endswith(".jpg")],
        key=lambda x: int(x.split(".")[0]),
    )

    # Initialize the optical flow accumulator
    accumulated_flow = None
    count = 0

    # Iterate through the image files to calculate optical flow
    for i in range(1, len(image_files)):
        prev_img = cv2.imread(os.path.join(directory, image_files[i - 1]))
        curr_img = cv2.imread(os.path.join(directory, image_files[i]))

        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

        # Calculate dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Accumulate the flow vectors
        if accumulated_flow is None:
            accumulated_flow = np.zeros_like(flow)

        accumulated_flow += flow
        count += 1

    # Calculate the average optical flow
    average_flow = accumulated_flow / count

    # Calculate magnitudes and angles
    magnitude, angle = cv2.cartToPolar(average_flow[..., 0], average_flow[..., 1])

    # Ignore magnitudes within X pixels of any border
    border = 10
    for y in range(magnitude.shape[0]):  # Iterate over rows (top to bottom)
        for x in range(magnitude.shape[1]):  # Iterate over columns (left to right)
            if (
                x < border
                or x >= magnitude.shape[1] - border
                or y < border
                or y >= magnitude.shape[0] - border
                or magnitude[y, x] < 2.0
            ):
                magnitude[y, x] = 0  # Set magnitudes within the border to 0

    # Ignore angles where the magnitude is 0
    for y in range(angle.shape[0]):  # Iterate over rows (top to bottom)
        for x in range(angle.shape[1]):  # Iterate over columns (left to right)
            if magnitude[y, x] == 0:
                angle[y, x] = 0

    # Recreate the flow vectors from magnitude and angle
    fx = magnitude * np.cos(angle)
    fy = magnitude * np.sin(angle)

    # Store lines for intersection calculations
    lines = []
    # Create an image to display the flow
    h, w = average_flow.shape[:2]
    flow_img = np.zeros((h, w, 3), dtype=np.uint8)
    line_img = np.zeros((h, w, 3), dtype=np.uint8)
    intersection_img = np.zeros((h, w, 3), dtype=np.uint8)
    line_count = 0
    step = 16
    for y in range(0, h, step):
        for x in range(0, w, step):
            if x >= border and x < w - border and y >= border and y < h - border:
                start_point = (x, y)
                end_point = (int(x + fx[y, x]), int(y + fy[y, x]))
                cv2.line(flow_img, start_point, end_point, (0, 255, 0), 1)
                cv2.circle(flow_img, start_point, 1, (0, 255, 0), -1)
                if magnitude[y, x] > 0 and end_point[0] != start_point[0]:
                    # print(f"Start: {start_point}, End: {end_point}")
                    slope = (end_point[1] - start_point[1]) / (
                        end_point[0] - start_point[0]
                    )
                    lines.append(line_from_slope_and_point(slope, start_point))
                    if slope != 0:
                        new_end = calculate_boundary_points(start_point, slope, w, h)
                        if len(new_end) == 2:
                            # cv2.circle(line_img, start_point, 3, (255, 0, 0), -1)
                            # cv2.circle(line_img, new_end[0], 3, (0, 0, 255), -1)
                            # cv2.circle(line_img, new_end[1], 3, (0, 0, 255), -1)
                            cv2.line(line_img, start_point, new_end[0], (0, 255, 0), 1)
                            cv2.line(line_img, start_point, new_end[1], (0, 255, 0), 1)
                            line_count += 1

    print(f"Line count: {line_count}")
    lines = np.array(lines)
    intersections = find_intersections_within_bounds(lines, w, h)
    for point in intersections:
        cv2.line(
            intersection_img,
            (int(point[0]), int(point[1])),
            (int(point[0]), int(point[1])),
            (255, 0, 0),
            1,
        )
    ### Math and drawing done for intersection point densities
    grid_size = 16
    print(f"Grid Calculation with grid size: {grid_size}")
    grid = count_intersections_in_grid(intersections, grid_size, w, h)
    # Create a heatmap from the grid
    heatmap = np.uint8(255 * grid / np.max(grid))
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_NEAREST)
    # Apply a color map to the heatmap
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    ##### Math and overlay for top grid densities
    top_x = 5
    top_sections = get_top_x_sections(grid, top_x)
    print(f"Top {top_x} sections with the highest number of intersections:")
    for y_idx, x_idx, value in top_sections:
        print(f"Section at ({x_idx}, {y_idx}) with {value} intersections")
    x_min, x_max = get_horizontal_range_of_top_sections(top_sections, grid_size)
    mid_x = (x_min + x_max) // 2

    # Read a sample image for overlay
    sample_image = cv2.imread(os.path.join(directory, image_files[-1]))
    # Create an overlay image for the top sections
    overlay_top_sections = np.zeros_like(sample_image)
    # Highlight the top sections on the overlay image
    for y_idx, x_idx, _ in top_sections:
        cv2.rectangle(
            overlay_top_sections,
            (x_idx * grid_size, y_idx * grid_size),
            ((x_idx + 1) * grid_size, (y_idx + 1) * grid_size),
            (0, 255, 0),
            2,
        )
    # draw line at mid_x
    cv2.line(overlay_top_sections, (mid_x, 0), (mid_x, h), (0, 0, 255), 2)
    device_offset = ((w // 2) - mid_x) * HORIZONTAL_FOV_DEGREE_PER_PIXEL
    print(f"Device offset: {device_offset} degrees")
    # Draw vertical center line on images
    flow_img_resized = draw_vertical_center_line(flow_img)
    line_img_resized = draw_vertical_center_line(line_img)
    intersection_img_resized = draw_vertical_center_line(intersection_img)
    sample_image_resized = draw_vertical_center_line(sample_image)

    # Draw horizontal center line on images
    flow_img_resized = draw_horizontal_center_line(flow_img_resized)
    line_img_resized = draw_horizontal_center_line(line_img_resized)
    intersection_img_resized = draw_horizontal_center_line(intersection_img_resized)
    sample_image_resized = draw_horizontal_center_line(sample_image_resized)

    # Overlay the optical flow on the sample image
    overlay = cv2.addWeighted(sample_image_resized, 0.5, flow_img_resized, 0.5, 0)
    overlay_line = cv2.addWeighted(sample_image_resized, 0.7, line_img_resized, 0.3, 0)
    overlay_intersection_img = cv2.addWeighted(
        sample_image_resized, 0.5, intersection_img_resized, 0.5, 0
    )
    # Overlay the heatmap on the original image
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
            results_directory, f"{parent_dir_name}_Farneback_flow_img_resized.jpg"
        ),
        flow_img_resized,
    )
    cv2.imwrite(
        os.path.join(
            results_directory, f"{parent_dir_name}_Farneback_sample_image_resized.jpg"
        ),
        sample_image_resized,
    )
    cv2.imwrite(
        os.path.join(results_directory, f"{parent_dir_name}_Farneback_overlay.jpg"),
        overlay,
    )
    cv2.imwrite(
        os.path.join(
            results_directory, f"{parent_dir_name}_Farneback_line_img_resized.jpg"
        ),
        line_img_resized,
    )
    cv2.imwrite(
        os.path.join(
            results_directory, f"{parent_dir_name}_Farneback_overlay_line.jpg"
        ),
        overlay_line,
    )
    cv2.imwrite(
        os.path.join(
            results_directory, f"{parent_dir_name}_Farneback_intersection.jpg"
        ),
        intersection_img,
    )
    cv2.imwrite(
        os.path.join(
            results_directory, f"{parent_dir_name}_Farneback_overlay_intersection.jpg"
        ),
        overlay_intersection_img,
    )
    cv2.imwrite(
        os.path.join(
            results_directory, f"{parent_dir_name}_Farneback_intersection_density.jpg"
        ),
        overlay_density_img,
    )
    cv2.imwrite(
        os.path.join(
            results_directory, f"{parent_dir_name}_Farneback_top_sections.jpg"
        ),
        overlay_top_sections_img,
    )


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
    # calculate_lucas_kanade_optical_flow(full_path, results_directory, dir_name)
    calculate_farneback_optical_flow(full_path, results_directory, dir_name)
