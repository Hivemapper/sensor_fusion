import numpy as np
import cv2
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

########### Constants ############
HORIZONTAL_FOV = 142  # degrees
HORIZONTAL_FOV_DEGREE_PER_PIXEL = (
    HORIZONTAL_FOV / 1024
)  # original 1520 pixels but we trim to 1024 pixels
VERTICAL_FOV = (
    103 / 1520 * 1024
)  # original 1520 pixels but we trim to 1024 pixels, in degrees

########### Tuneable Values ############
MAGNITUDE_THRESHOLD = 2.5
MAGNITUDE_BORDER_TRIMMING = 10


########### Math Functions ############


def calculate_flow(pair):
    prev_gray, curr_gray = pair
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow


def calculate_franeback_optical_flow(directory):
    print(f"FrameKM being evaluated: {directory}")
    image_files = sorted(
        [f for f in os.listdir(directory) if f.endswith(".jpg")],
        key=lambda x: int(x.split(".")[0]),
    )
    img_count = len(image_files)
    print(f" Image files length: {img_count}")
    last_image_path = os.path.join(directory, image_files[-1])

    accumulated_flow = None
    count = 0

    if img_count < 8:
        step = 1
    else:
        step = 2

    pairs = [
        (
            cv2.cvtColor(
                cv2.imread(os.path.join(directory, image_files[i - step])),
                cv2.COLOR_BGR2GRAY,
            ),
            cv2.cvtColor(
                cv2.imread(os.path.join(directory, image_files[i])), cv2.COLOR_BGR2GRAY
            ),
        )
        for i in range(step, len(image_files), step)
    ]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(calculate_flow, pair) for pair in pairs]
        for future in as_completed(futures):
            flow = future.result()
            if accumulated_flow is None:
                accumulated_flow = np.zeros_like(flow)
            accumulated_flow += flow
            count += 1

    average_flow = accumulated_flow / count
    return average_flow, last_image_path


# def calculate_franeback_optical_flow(directory):
#     print(f"FrameKM being evaluated: {directory}")
#     # Get a sorted list of image filenames
#     image_files = sorted(
#         [f for f in os.listdir(directory) if f.endswith(".jpg")],
#         key=lambda x: int(x.split(".")[0]),
#     )
#     img_count = len(image_files)
#     print(f" Image files length: {img_count}")
#     last_image_path = os.path.join(directory, image_files[-1])

#     # Initialize the optical flow accumulator
#     accumulated_flow = None
#     count = 0

#     # Calculate step size depending on the number of images
#     if img_count < 8:
#         step = 1
#     elif img_count < 16:
#         step = 2
#     else:
#         step = 3

#     # Iterate through the image files to calculate optical flow
#     for i in range(1, len(image_files), step):
#         prev_img = cv2.imread(os.path.join(directory, image_files[i - 1]))
#         curr_img = cv2.imread(os.path.join(directory, image_files[i]))

#         prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
#         curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

#         # Calculate dense optical flow using Farneback method
#         flow = cv2.calcOpticalFlowFarneback(
#             prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
#         )

#         # Accumulate the flow vectors
#         if accumulated_flow is None:
#             accumulated_flow = np.zeros_like(flow)

#         accumulated_flow += flow
#         count += 1

#     # Calculate the average optical flow
#     average_flow = accumulated_flow / count
#     return average_flow, last_image_path


def process_mag_and_angle_for_lines(magnitude, angle, h, w, step=16):
    # Store lines for intersection calculations
    lines = []
    for y in range(magnitude.shape[0]):  # Iterate over rows (top to bottom)
        for x in range(magnitude.shape[1]):  # Iterate over columns (left to right)
            if (
                x < MAGNITUDE_BORDER_TRIMMING
                or x >= magnitude.shape[1] - MAGNITUDE_BORDER_TRIMMING
                or y < MAGNITUDE_BORDER_TRIMMING
                or y >= magnitude.shape[0] - MAGNITUDE_BORDER_TRIMMING
                or magnitude[y, x] < MAGNITUDE_THRESHOLD
            ):
                magnitude[y, x] = 0  # Set magnitudes within the border to 0

    # Ignore angles where the magnitude is 0
    # for y in range(angle.shape[0]):  # Iterate over rows (top to bottom)
    #     for x in range(angle.shape[1]):  # Iterate over columns (left to right)
    #         if magnitude[y, x] == 0:
    #             angle[y, x] = 0

    # Recreate the flow vectors from magnitude and angle
    fx = magnitude * np.cos(angle)
    fy = magnitude * np.sin(angle)

    for y in range(0, h, step):
        for x in range(0, w, step):
            start_point = (x, y)
            end_point = (int(x + fx[y, x]), int(y + fy[y, x]))
            if magnitude[y, x] > 0 and end_point[0] != start_point[0]:
                slope = (end_point[1] - start_point[1]) / (
                    end_point[0] - start_point[0]
                )
                lines.append(line_from_slope_and_point(slope, start_point))
    print(f" Number of lines: {len(lines)}")
    return np.array(lines)


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


def count_intersections_in_grid(intersections, width, height, grid_size=16):
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


########### Drawing Functions ############
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
