import numpy as np
import cv2
import math
import os
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial.distance import cdist

########### Constants ############
HORIZONTAL_FOV = 142  # degrees
HORIZONTAL_FOV_DEGREE_PER_PIXEL = (
    HORIZONTAL_FOV / 1024
)  # original 1520 pixels but we trim to 1024 pixels
VERTICAL_FOV = (
    103 / 1520 * 1024
)  # original 1520 pixels but we trim to 1024 pixels, in degrees
HORIZONTAL_20_DEGREES = math.ceil(20 / HORIZONTAL_FOV_DEGREE_PER_PIXEL)
HORIZONTAL_30_DEGREES = math.ceil(30 / HORIZONTAL_FOV_DEGREE_PER_PIXEL)

########### Tuneable Values ############
MAGNITUDE_THRESHOLD = 2.5
MAGNITUDE_BORDER_TRIMMING = 10


########### Math Functions ############


# def calculate_flow(pair):
#     try:
#         prev_gray, curr_gray = pair
#         flow = cv2.calcOpticalFlowFarneback(
#             prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
#         )
#         return flow
#     except Exception as e:
#         logger.error(f"Error in calculate_flow: {e}")
#         raise


# def process_batch(batch_pairs):
#     try:
#         accumulated_flow = None
#         count = 0
#         for pair in batch_pairs:
#             flow = calculate_flow(pair)
#             if accumulated_flow is None:
#                 accumulated_flow = np.zeros_like(flow)
#             accumulated_flow += flow
#             count += 1
#         return accumulated_flow, count
#     except Exception as e:
#         logger.error(f"Error in process_batch: {e}")
#         raise


# def calculate_farneback_optical_flow(directory, step=1, max_processes=5, batch_size=1):
#     logger.info(f"FrameKM being evaluated: {directory}")
#     image_files = sorted(
#         [f for f in os.listdir(directory) if f.endswith(".jpg")],
#         key=lambda x: int(x.split(".")[0]),
#     )
#     img_count = len(image_files)
#     if img_count > 20:
#         img_count = 20
#     logger.info(f" Image files length: {img_count}")
#     last_image_path = os.path.join(directory, image_files[img_count - 1])

#     pairs = [
#         (
#             cv2.cvtColor(
#                 cv2.imread(os.path.join(directory, image_files[i])), cv2.COLOR_BGR2GRAY
#             ),
#             cv2.cvtColor(
#                 cv2.imread(os.path.join(directory, image_files[i + step])),
#                 cv2.COLOR_BGR2GRAY,
#             ),
#         )
#         for i in range(0, img_count - step, step)
#     ]

#     total_accumulated_flow = None
#     total_count = 0

#     with ProcessPoolExecutor(max_workers=max_processes) as executor:
#         futures = []
#         for i in range(0, len(pairs), batch_size):
#             batch_pairs = pairs[i : i + batch_size]
#             futures.append(executor.submit(process_batch, batch_pairs))

#         for future in as_completed(futures):
#             try:
#                 accumulated_flow, count = future.result()
#                 if total_accumulated_flow is None:
#                     total_accumulated_flow = np.zeros_like(accumulated_flow)
#                 total_accumulated_flow += accumulated_flow
#                 total_count += count
#             except Exception as e:
#                 logger.error(f"Error processing batch: {e}")

#     if total_count > 0:
#         average_flow = total_accumulated_flow / total_count
#     else:
#         average_flow = None

#     return average_flow, last_image_path, img_count


def calculate_flow(pair):
    prev_gray, curr_gray = pair
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,  # Default is 0.5
        levels=4,  # Default is 3, fewer levels to reduce computation
        winsize=20,  # Increased window size
        iterations=1,  # Default is 3, fewer iterations
        poly_n=5,  # Default is 5 or 7, smaller neighborhood
        poly_sigma=1.1,  # Default is 1.2, slightly lower sigma
        flags=0,  # Default flags
    )
    return flow


def calculate_farneback_optical_flow(directory, step=1, max_threads=3):
    print(f"FrameKM being evaluated: {directory}")
    image_files = sorted(
        [f for f in os.listdir(directory) if f.endswith(".jpg")],
        key=lambda x: int(x.split(".")[0]),
    )
    img_count = len(image_files)
    if img_count > 15:
        img_count = 15
    print(f" Image files length: {img_count}")
    last_image_path = os.path.join(directory, image_files[img_count - 1])

    accumulated_flow = None
    count = 0

    pairs = [
        (
            cv2.cvtColor(
                cv2.imread(os.path.join(directory, image_files[i])), cv2.COLOR_BGR2GRAY
            ),
            cv2.cvtColor(
                cv2.imread(os.path.join(directory, image_files[i + step])),
                cv2.COLOR_BGR2GRAY,
            ),
        )
        for i in range(0, img_count - step, step)
    ]

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(calculate_flow, pair) for pair in pairs]
        for future in as_completed(futures):
            flow = future.result()
            if accumulated_flow is None:
                accumulated_flow = np.zeros_like(flow)
            accumulated_flow += flow
            count += 1

    average_flow = accumulated_flow / count
    return average_flow, last_image_path, img_count


# def calculate_farneback_optical_flow(directory, step=1):
#     print(f"FrameKM being evaluated: {directory}")
#     # Get a sorted list of image filenames
#     image_files = sorted(
#         [f for f in os.listdir(directory) if f.endswith(".jpg")],
#         key=lambda x: int(x.split(".")[0]),
#     )
#     img_count = len(image_files)
#     if img_count > 20:
#         img_count = 20
#     print(f" Image files length: {img_count}")
#     last_image_path = os.path.join(directory, image_files[img_count - 1])

#     # Initialize the optical flow accumulator
#     accumulated_flow = None
#     count = 0

#     # Iterate through the image files to calculate optical flow
#     for i in range(1, img_count, step):
#         print(f"Processing images {i} and {i - 1}")
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
#     return average_flow, last_image_path, img_count


# def process_mag_and_angle_for_lines(magnitude, angle, h, w, step=16):
#     # Store lines for intersection calculations
#     lines = []
#     for y in range(magnitude.shape[0]):  # Iterate over rows (top to bottom)
#         for x in range(magnitude.shape[1]):  # Iterate over columns (left to right)
#             if (
#                 x < MAGNITUDE_BORDER_TRIMMING
#                 or x >= magnitude.shape[1] - MAGNITUDE_BORDER_TRIMMING
#                 or y < MAGNITUDE_BORDER_TRIMMING
#                 or y >= magnitude.shape[0] - MAGNITUDE_BORDER_TRIMMING
#                 or magnitude[y, x] < MAGNITUDE_THRESHOLD
#             ):
#                 magnitude[y, x] = 0  # Set magnitudes within the border to 0
#                 angle[y, x] = 0  # Set angles where the magnitude is 0 to 0

#     # Recreate the flow vectors from magnitude and angle
#     fx = magnitude * np.cos(angle)
#     fy = magnitude * np.sin(angle)

#     for y in range(0, h, step):
#         for x in range(0, w, step):
#             start_point = (x, y)
#             end_point = (int(x + fx[y, x]), int(y + fy[y, x]))
#             if magnitude[y, x] > 0 and end_point[0] != start_point[0]:
#                 slope = (end_point[1] - start_point[1]) / (
#                     end_point[0] - start_point[0]
#                 )
#                 # Add filter conditions to ignore noisy lines
#                 lines.append(line_from_slope_and_point(slope, start_point))
#     print(f" Number of lines: {len(lines)}")
#     return np.array(lines)


# def process_mag_and_angle_for_lines(magnitude, angle, h, w, step=1):
#     # Create a mask to zero out magnitudes and angles within the border or below the threshold
#     mask = (
#         (magnitude < MAGNITUDE_THRESHOLD)
#         | (np.arange(magnitude.shape[0])[:, None] < MAGNITUDE_BORDER_TRIMMING)
#         | (
#             np.arange(magnitude.shape[0])[:, None]
#             >= magnitude.shape[0] - MAGNITUDE_BORDER_TRIMMING
#         )
#         | (np.arange(magnitude.shape[1]) < MAGNITUDE_BORDER_TRIMMING)
#         | (
#             np.arange(magnitude.shape[1])
#             >= magnitude.shape[1] - MAGNITUDE_BORDER_TRIMMING
#         )
#     )

#     magnitude[mask] = 0
#     angle[mask] = 0

#     # Recreate the flow vectors from magnitude and angle
#     fx = magnitude * np.cos(angle)
#     fy = magnitude * np.sin(angle)

#     # Generate grid of start points
#     y_grid, x_grid = np.meshgrid(
#         np.arange(0, h, step), np.arange(0, w, step), indexing="ij"
#     )

#     # Calculate end points
#     end_x = x_grid + fx[::step, ::step]
#     end_y = y_grid + fy[::step, ::step]

#     # Filter out points with zero magnitude and where end point is the same as start point
#     valid_mask = (magnitude[::step, ::step] > 0) & (
#         (end_x != x_grid) | (end_y != y_grid)
#     )

#     start_points = np.vstack([x_grid[valid_mask], y_grid[valid_mask]]).T
#     end_points = np.vstack([end_x[valid_mask], end_y[valid_mask]]).T

#     # Calculate slopes and create lines
#     slopes = (end_points[:, 1] - start_points[:, 1]) / (
#         end_points[:, 0] - start_points[:, 0]
#     )
#     lines = [
#         line_from_slope_and_point(slope, start_point)
#         for slope, start_point in zip(slopes, start_points)
#     ]

#     print(f" Number of lines: {len(lines)}")
#     return np.array(lines)


# Optimized function with memory error handling
def process_mag_and_angle_for_lines_optimized(magnitude, angle, h, w, step=1):
    try:
        # Create border masks
        row_mask = (np.arange(magnitude.shape[0]) < MAGNITUDE_BORDER_TRIMMING) | (
            np.arange(magnitude.shape[0])
            >= magnitude.shape[0] - MAGNITUDE_BORDER_TRIMMING
        )
        col_mask = (np.arange(magnitude.shape[1]) < MAGNITUDE_BORDER_TRIMMING) | (
            np.arange(magnitude.shape[1])
            >= magnitude.shape[1] - MAGNITUDE_BORDER_TRIMMING
        )

        # Apply the border mask to magnitude and angle
        magnitude[row_mask, :] = 0
        magnitude[:, col_mask] = 0
        angle[row_mask, :] = 0
        angle[:, col_mask] = 0

        # Apply magnitude threshold
        magnitude[magnitude < MAGNITUDE_THRESHOLD] = 0
        angle[magnitude < MAGNITUDE_THRESHOLD] = 0

        # Recreate the flow vectors from magnitude and angle
        fx = magnitude * np.cos(angle)
        fy = magnitude * np.sin(angle)

        # Generate grid of start points
        y_grid, x_grid = np.meshgrid(
            np.arange(0, h, step), np.arange(0, w, step), indexing="ij"
        )

        fx = fx[::step, ::step]
        fy = fy[::step, ::step]
        end_x = x_grid + fx
        end_y = y_grid + fy

        # Filter out points with zero magnitude and where end point is the same as start point
        valid_mask = (magnitude[::step, ::step] > 0) & (
            (end_x != x_grid) | (end_y != y_grid)
        )

        start_points = np.column_stack((x_grid[valid_mask], y_grid[valid_mask]))
        end_points = np.column_stack((end_x[valid_mask], end_y[valid_mask]))

        slopes = (end_points[:, 1] - start_points[:, 1]) / (
            end_points[:, 0] - start_points[:, 0]
        )
        lines = [
            line_from_slope_and_point(slope, start_point)
            for slope, start_point in zip(slopes, start_points)
        ]

        print(f" Number of lines: {len(lines)}")
        return np.array(lines)

    except MemoryError:
        print(
            " MemoryError: Not enough memory to process the data. Consider using smaller data or increasing available memory."
        )
        return None


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


# def find_intersections_within_bounds(lines, width, height):
#     intersections = []
#     num_lines = len(lines)
#     for i in range(num_lines):
#         m1, c1 = lines[i]
#         for j in range(i + 1, num_lines):
#             m2, c2 = lines[j]
#             if m1 != m2:
#                 x = (c2 - c1) / (m1 - m2)
#                 y = m1 * x + c1
#                 x_rounded = round(x)
#                 y_rounded = round(y)
#                 if 0 <= x_rounded < width and 0 <= y_rounded < height:
#                     intersections.append((x_rounded, y_rounded))
#     return np.array(intersections)


def find_intersections_within_bounds(lines, width, height):
    try:
        intersections = []

        # Extract slopes and intercepts
        slopes = np.array([line[0] for line in lines])
        intercepts = np.array([line[1] for line in lines])

        # Create a meshgrid for pairs of lines
        idx1, idx2 = np.triu_indices(len(lines), k=1)

        m1 = slopes[idx1]
        c1 = intercepts[idx1]
        m2 = slopes[idx2]
        c2 = intercepts[idx2]

        # Calculate intersections
        valid = m1 != m2
        x = (c2[valid] - c1[valid]) / (m1[valid] - m2[valid])
        y = m1[valid] * x + c1[valid]

        # Round the coordinates
        x_rounded = np.round(x).astype(int)
        y_rounded = np.round(y).astype(int)

        # Filter intersections within bounds
        in_bounds = (
            (0 <= x_rounded)
            & (x_rounded < width)
            & (0 <= y_rounded)
            & (y_rounded < height)
        )
        intersections = np.vstack((x_rounded[in_bounds], y_rounded[in_bounds])).T

        return intersections

    except MemoryError:
        print(
            " MemoryError: Not enough memory to process the data. Consider using smaller data or increasing available memory."
        )
        return None


# def count_intersections_in_grid(
#     intersections, width, height, grid_size=16, left_bound=None, right_bound=None
# ):
#     mid_point = width // 2
#     left_bound = left_bound if left_bound else mid_point - HORIZONTAL_30_DEGREES
#     right_bound = right_bound if right_bound else mid_point + HORIZONTAL_30_DEGREES

#     grid_height = height // grid_size
#     grid_width = width // grid_size
#     grid = np.zeros((grid_height, grid_width), dtype=int)
#     grid_indices = (intersections // grid_size).astype(int)

#     left_bound_idx = left_bound // grid_size
#     right_bound_idx = right_bound // grid_size

#     for x_idx, y_idx in grid_indices:
#         if left_bound_idx <= x_idx < right_bound_idx and 0 <= y_idx < grid_height:
#             grid[y_idx, x_idx] += 1

#     return grid


def count_intersections_in_grid(
    intersections, width, height, grid_size=16, left_bound=None, right_bound=None
):
    mid_point = width // 2
    left_bound = (
        left_bound if left_bound is not None else mid_point - HORIZONTAL_30_DEGREES
    )
    right_bound = (
        right_bound if right_bound is not None else mid_point + HORIZONTAL_30_DEGREES
    )

    grid_height = height // grid_size
    grid_width = width // grid_size
    grid = np.zeros((grid_height, grid_width), dtype=int)
    grid_indices = (intersections // grid_size).astype(int)

    left_bound_idx = left_bound // grid_size
    right_bound_idx = right_bound // grid_size

    # Filter out indices that are out of bounds
    valid_indices = (
        (grid_indices[:, 0] >= left_bound_idx)
        & (grid_indices[:, 0] < right_bound_idx)
        & (grid_indices[:, 1] >= 0)
        & (grid_indices[:, 1] < grid_height)
    )
    filtered_indices = grid_indices[valid_indices]

    # Use numpy's advanced indexing to count occurrences
    np.add.at(grid, (filtered_indices[:, 1], filtered_indices[:, 0]), 1)

    return grid


# TODO: Handle cases where outliers exist, need to remove them
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


# def get_top_x_sections(grid, top_x, max_distance=5):
#     # Flatten the 2D grid into a 1D array
#     flat_grid = grid.flatten()

#     # Get indices of the top 2*top_x values in the flattened array, sorted in descending order
#     initial_pool_size = top_x * 2
#     top_indices = np.argpartition(flat_grid, -initial_pool_size)[-initial_pool_size:]
#     top_indices = top_indices[np.argsort(flat_grid[top_indices])[::-1]]

#     # Get the top values from the flattened grid
#     top_values = flat_grid[top_indices]

#     # Convert the 1D indices back to 2D coordinates
#     top_coords = np.column_stack(np.unravel_index(top_indices, grid.shape))

#     # Combine the coordinates and values into a list of tuples (row, col, value)
#     top_sections = list(zip(top_coords[:, 0], top_coords[:, 1], top_values))

#     # Initialize the filtered sections with the first top section
#     filtered_sections = [top_sections[0]]

#     # Compute distances once for all top sections
#     distances = cdist(top_coords, top_coords)

#     # Filter out sections that are not close to each other
#     for i in range(1, len(top_sections)):
#         section = top_sections[i]
#         if any(distances[i, j] <= max_distance for j in range(len(filtered_sections))):
#             filtered_sections.append(section)
#         if len(filtered_sections) == top_x:
#             break

#     # If not enough sections found, expand the search
#     if len(filtered_sections) < top_x:
#         for i in range(len(top_sections)):
#             if top_sections[i] not in filtered_sections:
#                 if any(
#                     distances[i, j] <= max_distance
#                     for j in range(len(filtered_sections))
#                 ):
#                     filtered_sections.append(top_sections[i])
#                 if len(filtered_sections) == top_x:
#                     break

#     return filtered_sections


def get_horizontal_range_of_top_sections(top_sections, grid_size):
    # Extract x-coordinates (column indices) and counts of the top sections
    x_coords = np.array([x_idx * grid_size for _, x_idx, _ in top_sections])
    counts = np.array([count for _, _, count in top_sections])

    # Calculate the weighted average of the x-coordinates
    weighted_average_x = np.average(x_coords, weights=counts)
    weighted_average_x = round(weighted_average_x)

    # Find the minimum and maximum x-coordinates
    min_x = np.min(x_coords)
    max_x = np.max(x_coords)

    return int(min_x), int(max_x), int(weighted_average_x)


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
    # Draw dotted off-center lines
    draw_dotted_line(
        image,
        (center_x - HORIZONTAL_20_DEGREES, 0),
        (center_x - HORIZONTAL_20_DEGREES, h),
        (255, 255, 255),
        thickness,
        gap,
    )
    draw_dotted_line(
        image,
        (center_x + HORIZONTAL_20_DEGREES, 0),
        (center_x + HORIZONTAL_20_DEGREES, h),
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


def draw_overlay_flow_map(flow, image):
    h, w = image.shape[:2]
    flow_map = np.zeros_like(image)

    step = 16
    y, x = np.mgrid[step / 2 : h : step, step / 2 : w : step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    for (x1, y1), (x2, y2) in lines:
        cv2.line(flow_map, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(flow_map, (x1, y1), 1, (0, 255, 0), -1)

    overlay = cv2.addWeighted(image, 0.7, flow_map, 0.3, 0)
    return overlay


def draw_angle_map(angle, image):
    h, w = image.shape[:2]
    angle_map = np.zeros((h, w, 3), dtype=np.uint8)

    step = 64
    y, x = (
        np.mgrid[step // 2 : h : step, step // 2 : w : step].reshape(2, -1).astype(int)
    )
    angles = angle[y, x]

    for i, (x1, y1) in enumerate(zip(x, y)):
        angle_text = f"{angles[i] * 180 / np.pi:.1f}"  # Convert radians to degrees
        cv2.putText(
            angle_map,
            angle_text,
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return angle_map
