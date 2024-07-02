import cv2
import numpy as np
import os
import math

HORIZONTAL_FOV = 142  # degrees
HORIZONTAL_FOV_DEGREE_PER_PIXEL = (
    HORIZONTAL_FOV / 1024
)  # original 1520 pixels but we trim to 1024 pixels
VERTICAL_FOV = (
    103 / 1520 * 1024
)  # original 1520 pixels but we trim to 1024 pixels, in degrees


def calculate_boundary_points(start_point, slope, image_width, image_height):
    x0, y0 = start_point
    boundary_points = []

    # Calculate intersection with the left boundary (x = 0)
    if slope != 0:
        y_left = y0 - x0 * slope
        if 0 <= y_left < image_height:
            boundary_points.append((0, int(y_left)))

    # Calculate intersection with the right boundary (x = image_width - 1)
    y_right = y0 + (image_width - 1 - x0) * slope
    if 0 <= y_right < image_height:
        boundary_points.append((image_width - 1, int(y_right)))

    # Calculate intersection with the top boundary (y = 0)
    if slope != 0:
        x_top = x0 - y0 / slope
        if 0 <= x_top < image_width:
            boundary_points.append((int(x_top), 0))

    # Calculate intersection with the bottom boundary (y = image_height - 1)
    x_bottom = x0 + (image_height - 1 - y0) / slope
    if 0 <= x_bottom < image_width:
        boundary_points.append((int(x_bottom), image_height - 1))

    # Return all valid boundary points
    # print(f"Boundary points: {boundary_points}")
    return boundary_points


def line_from_slope_and_point(slope, point):
    x, y = point
    intercept = y - slope * x
    return slope, intercept


def find_intersections_within_bounds(lines, width, height):
    intersections = []
    num_lines = len(lines)
    for i in range(num_lines):
        m1, c1 = lines[i]
        for j in range(i + 1, num_lines):
            m2, c2 = lines[j]
            if m1 != m2:
                x = (c2 - c1) / (m1 - m2)
                y = m1 * x + c1
                x_rounded = round(x)
                y_rounded = round(y)
                if 0 <= x_rounded < width and 0 <= y_rounded < height:
                    intersections.append((x_rounded, y_rounded))
    return np.array(intersections)


def count_intersections_in_grid(intersections, grid_size, width, height):
    grid_height = height // grid_size
    grid_width = width // grid_size
    grid = np.zeros((grid_height, grid_width), dtype=int)
    grid_indices = (intersections // grid_size).astype(int)

    for x_idx, y_idx in grid_indices:
        if 0 <= x_idx < grid_width and 0 <= y_idx < grid_height:
            grid[y_idx, x_idx] += 1

    return grid


def get_top_x_sections(grid, top_x):
    # Flatten the 2D grid into a 1D array
    flat_grid = grid.flatten()

    # Get indices of the top X values in the flattened array, sorted in descending order
    top_indices = np.argsort(flat_grid)[-top_x:][::-1]

    # Get the top values from the flattened grid
    top_values = flat_grid[top_indices]

    # Convert the 1D indices back to 2D coordinates
    top_coords = np.unravel_index(top_indices, grid.shape)

    # Combine the coordinates and values into a list of tuples (row, col, value)
    top_sections = list(zip(top_coords[0], top_coords[1], top_values))

    return top_sections


# Calculate the horizontal range of the top sections
def get_horizontal_range_of_top_sections(top_sections, grid_size):
    # Extract x-coordinates (column indices) of the top sections
    x_coords = [x_idx * grid_size for _, x_idx, _ in top_sections]

    # Find the minimum and maximum x-coordinates
    min_x = min(x_coords)
    max_x = max(x_coords)

    return min_x, max_x


def draw_dotted_line(image, start_point, end_point, color, thickness, gap):
    x1, y1 = start_point
    x2, y2 = end_point
    is_vertical = x1 == x2
    if is_vertical:
        for y in range(y1, y2, gap * 2):
            cv2.line(image, (x1, y), (x2, y + gap), color, thickness)


def draw_vertical_center_line(image, thickness=1, gap=10):
    h, w = image.shape[:2]
    center_x = w // 2
    # get 20 degree lines from center
    HORIZONTAL_FOV_DEGREE_PER_PIXEL = 0.1  # example value
    pixel_count_20 = math.ceil(20 / HORIZONTAL_FOV_DEGREE_PER_PIXEL)

    # Draw dotted off-center lines
    draw_dotted_line(
        image,
        (center_x - pixel_count_20, 0),
        (center_x - pixel_count_20, h),
        (255, 255, 255),
        thickness,
        gap,
    )
    draw_dotted_line(
        image,
        (center_x + pixel_count_20, 0),
        (center_x + pixel_count_20, h),
        (255, 255, 255),
        thickness,
        gap,
    )

    # Draw solid center line
    cv2.line(image, (center_x, 0), (center_x, h), (255, 255, 255), thickness)

    return image


def draw_horizontal_center_line(image, thickness=1):
    h, w = image.shape[:2]
    center_y = h // 2
    cv2.line(
        image, (0, center_y), (w, center_y), (255, 255, 255), thickness
    )  # White line with specified thickness
    return image


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
