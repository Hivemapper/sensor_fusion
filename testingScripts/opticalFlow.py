import cv2
import numpy as np
import os
import math


def draw_angle_grid(image, angle):
    h, w = angle.shape[:2]
    grid_image = image.copy()
    step = 64  # Step size for the grid

    for y in range(0, h, step):
        for x in range(0, w, step):
            angle_deg = math.degrees(angle[y, x])
            text = f"{int(angle_deg)}"
            position = (x, y)
            cv2.putText(
                grid_image,
                text,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    return grid_image


def draw_line_to_edge(image, start, angle, color):
    height, width = image.shape[:2]
    x0, y0 = start

    # Convert angle to radians
    angle_rad = math.radians(angle)
    # print(f"Angle: {angle}, Angle (rad): {angle_rad}")

    # Calculate the end points for the boundaries
    if angle == 90:
        end = (x0, 0)
    elif angle == 270:
        end = (x0, height)
    elif angle == 0:
        end = (width, y0)
    elif angle == 180:
        end = (0, y0)
    else:
        x_intercept_top = x0 + (y0 * math.tan(angle_rad))
        x_intercept_bottom = x0 + ((height - y0) * math.tan(angle_rad))
        y_intercept_left = y0 - (x0 / math.tan(angle_rad))
        y_intercept_right = y0 + ((width - x0) * math.tan(angle_rad))

        # Determine which intersection is valid
        if 0 <= x_intercept_top <= width:
            end = (int(x_intercept_top), 0)
        elif 0 <= x_intercept_bottom <= width:
            end = (int(x_intercept_bottom), height)
        elif 0 <= y_intercept_left <= height:
            end = (0, int(y_intercept_left))
        elif 0 <= y_intercept_right <= height:
            end = (width, int(y_intercept_right))
        else:
            return image  # If no valid intersection, do not draw

    # Draw the line on the image
    # cv2.line(image, start, end, (int(color[0]), int(color[1]), int(color[2])), 2)
    cv2.circle(image, start, 5, (0, 255, 0), -1)  # Mark starting point in red
    cv2.circle(image, end, 5, (0, 0, 255), -1)  # Mark ending point in blue

    return image


def draw_line_from_slope(image, start, slope, color, thickness=2):
    height, width = image.shape[:2]
    x0, y0 = start

    # Calculate end points based on the slope
    if slope != 0:
        # For non-horizontal lines
        # Calculate intersection with the top and bottom borders
        y_top = 0
        x_top = int(x0 - y0 / slope)

        y_bottom = height
        x_bottom = int(x0 + (height - y0) / slope)

        # Calculate intersection with the left and right borders
        x_left = 0
        y_left = int(y0 - x0 * slope)

        x_right = width
        y_right = int(y0 + (width - x0) * slope)

        # Collect valid intersection points
        points = []
        if 0 <= x_top < width:
            points.append((x_top, y_top))
        if 0 <= x_bottom < width:
            points.append((x_bottom, y_bottom))
        if 0 <= y_left < height:
            points.append((x_left, y_left))
        if 0 <= y_right < height:
            points.append((x_right, y_right))

        # Choose the points that are furthest apart to draw the line
        if len(points) >= 2:
            cv2.line(image, points[0], points[-1], color, thickness)
    else:
        # For horizontal lines (slope = 0)
        cv2.line(image, (0, y0), (width, y0), color, thickness)

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
    border = 200
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

    # Create an image to display the flow
    h, w = average_flow.shape[:2]
    fx, fy = average_flow[..., 0], average_flow[..., 1]

    flow_img = np.zeros((h, w, 3), dtype=np.uint8)
    angle_img = np.zeros((h, w, 3), dtype=np.uint8)
    step = 16
    for y in range(0, h, step):
        for x in range(0, w, step):
            if x >= border and x < w - border and y >= border and y < h - border:
                start_point = (x, y)
                end_point = (int(x + fx[y, x]), int(y + fy[y, x]))
                cv2.line(flow_img, start_point, end_point, (0, 255, 0), 1)
                cv2.circle(flow_img, start_point, 1, (0, 255, 0), -1)
                if angle[y, x] > 0:
                    slope = (start_point[1] - end_point[1]) / (
                        start_point[0] - end_point[0]
                    )
                    draw_line_from_slope(angle_img, start_point, slope, (0, 255, 0))

    # Read a sample image for overlay
    sample_image = cv2.imread(os.path.join(directory, image_files[0]))

    # Resize images for display
    display_scale = 1.0  # Scale factor for displaying images
    flow_img_resized = cv2.resize(
        flow_img, (int(w * display_scale), int(h * display_scale))
    )
    sample_image_resized = cv2.resize(
        sample_image,
        (
            int(sample_image.shape[1] * display_scale),
            int(sample_image.shape[0] * display_scale),
        ),
    )

    # Overlay the optical flow on the sample image
    overlay = cv2.addWeighted(sample_image_resized, 0.5, flow_img_resized, 0.5, 0)

    # Normalize the magnitude for visualization
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    # Create an HSV image for the angle visualization with magnitude as value
    hsv = np.zeros((angle.shape[0], angle.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue
    hsv[..., 1] = 255  # Saturation
    hsv[..., 2] = magnitude_norm  # Value

    angle_bgr_with_magnitude = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Resize the magnitude and angle maps for display
    magnitude_img_resized = cv2.resize(
        magnitude_norm, (int(w * display_scale), int(h * display_scale))
    )
    angle_img_resized = cv2.resize(
        angle_bgr_with_magnitude, (int(w * display_scale), int(h * display_scale))
    )

    # Overlay the magnitude map on the sample image
    magnitude_overlay = cv2.addWeighted(
        sample_image_resized,
        0.5,
        cv2.cvtColor(magnitude_img_resized, cv2.COLOR_GRAY2BGR),
        0.5,
        0,
    )

    # Draw lines from each angle to the border with color scheme and markers on a black background
    # intersection_img = np.zeros((h, w, 3), dtype=np.uint8)
    # for y in range(0, h, step):
    #     for x in range(0, w, step):
    #         if angle[y, x] > 0:
    #             hue = angle[y, x] * 180 / np.pi / 2
    #             color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[
    #                 0
    #             ][0]
    #             intersection_img = draw_line_to_edge(
    #                 intersection_img, (x, y), math.degrees(angle[y, x]), color
    #             )

    intersection_img_resized = cv2.resize(
        angle_img, (int(w * display_scale), int(h * display_scale))
    )

    angle_grid_img = np.zeros((h, w, 3), dtype=np.uint8)
    angle_grid_img = draw_angle_grid(angle_grid_img, angle)

    # Resize the angle grid image for display
    angle_grid_img_resized = cv2.resize(
        angle_grid_img, (int(w * display_scale), int(h * display_scale))
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
        os.path.join(results_directory, f"{parent_dir_name}_Farneback_magnitude.jpg"),
        magnitude_img_resized,
    )
    cv2.imwrite(
        os.path.join(results_directory, f"{parent_dir_name}_Farneback_angle.jpg"),
        angle_img_resized,
    )
    # cv2.imwrite(
    #     os.path.join(
    #         results_directory, f"{parent_dir_name}_Farneback_magnitude_overlay.jpg"
    #     ),
    #     magnitude_overlay,
    # )
    cv2.imwrite(
        os.path.join(
            results_directory, f"{parent_dir_name}_Farneback_intersections.jpg"
        ),
        intersection_img_resized,
    )
    cv2.imwrite(
        os.path.join(results_directory, f"{parent_dir_name}_Farneback_angle_grid.jpg"),
        angle_grid_img_resized,
    )


# Define the base directory containing subdirectories of images
base_directory = "//Users/rogerberman/Desktop/frameKMImages"
results_base_directory = os.path.join(base_directory, "results")
dirs = [
    d
    for d in os.listdir(base_directory)
    if os.path.isdir(os.path.join(base_directory, d))
]

# Loop over each subdirectory and calculate/save optical flow
for dir_name in dirs:
    full_path = os.path.join(base_directory, dir_name)
    results_directory = os.path.join(results_base_directory, dir_name)
    # calculate_lucas_kanade_optical_flow(full_path, results_directory, dir_name)
    calculate_farneback_optical_flow(full_path, results_directory, dir_name)
    break


## Method did not yield any immediately good results so pushed down here for now

# def calculate_lucas_kanade_optical_flow(directory, results_directory, parent_dir_name):
#     print(f"FrameKM being evaluated: {directory}")

#     # Get a sorted list of image filenames
#     image_files = sorted(
#         [f for f in os.listdir(directory) if f.endswith(".jpg")],
#         key=lambda x: int(x.split(".")[0]),
#     )

#     # Initialize parameters for Lucas-Kanade optical flow
#     feature_params = dict(maxCorners=200, qualityLevel=0.2, minDistance=15, blockSize=7)
#     lk_params = dict(
#         winSize=(21, 21),
#         maxLevel=5,
#         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
#     )

#     # Read the first frame and find corners in it
#     old_frame = cv2.imread(os.path.join(directory, image_files[0]))
#     old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
#     p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

#     # Initialize the optical flow accumulator
#     accumulated_flow = np.zeros(
#         (old_frame.shape[0], old_frame.shape[1], 2), dtype=np.float32
#     )
#     count = 0

#     for i in range(1, len(image_files)):
#         frame = cv2.imread(os.path.join(directory, image_files[i]))
#         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Calculate optical flow
#         p1, st, err = cv2.calcOpticalFlowPyrLK(
#             old_gray, frame_gray, p0, None, **lk_params
#         )

#         # Select good points
#         if p1 is not None:
#             good_new = p1[st == 1]
#             good_old = p0[st == 1]

#             # Accumulate the flow vectors
#             for new, old in zip(good_new, good_old):
#                 a, b = new.ravel()
#                 c, d = old.ravel()
#                 flow_vector = np.array([a - c, b - d])
#                 if (
#                     0 <= c < accumulated_flow.shape[1]
#                     and 0 <= d < accumulated_flow.shape[0]
#                 ):
#                     accumulated_flow[int(d), int(c)] += flow_vector

#             # Update the previous frame and previous points
#             old_gray = frame_gray.copy()
#             p0 = good_new.reshape(-1, 1, 2)
#             count += 1

#     # Calculate the average optical flow
#     average_flow = accumulated_flow / count

#     # Create an image to display the flow
#     h, w = average_flow.shape[:2]
#     flow_img = np.zeros((h, w, 3), dtype=np.uint8)
#     step = 16
#     for y in range(0, h, step):
#         for x in range(0, w, step):
#             fx, fy = average_flow[y, x]
#             start_point = (x, y)
#             end_point = (int(x + fx), int(y + fy))
#             cv2.line(flow_img, start_point, end_point, (0, 255, 0), 1)
#             cv2.circle(flow_img, start_point, 1, (0, 255, 0), -1)

#     # Read a sample image for overlay
#     sample_image = cv2.imread(os.path.join(directory, image_files[0]))

#     # Resize images for display
#     display_scale = 0.5
#     flow_img_resized = cv2.resize(
#         flow_img, (int(w * display_scale), int(h * display_scale))
#     )
#     sample_image_resized = cv2.resize(
#         sample_image,
#         (
#             int(sample_image.shape[1] * display_scale),
#             int(sample_image.shape[0] * display_scale),
#         ),
#     )

#     # Overlay the optical flow on the sample image
#     overlay = cv2.addWeighted(sample_image_resized, 0.5, flow_img_resized, 0.5, 0)

#     # Save the images to the results directory with the parent directory name in the filenames
#     os.makedirs(results_directory, exist_ok=True)
#     # cv2.imwrite(
#     #     os.path.join(
#     #         results_directory, f"{parent_dir_name}_Lucas-Kanade_flow_img_resized.jpg"
#     #     ),
#     #     flow_img_resized,
#     # )
#     # cv2.imwrite(
#     #     os.path.join(
#     #         results_directory,
#     #         f"{parent_dir_name}_Lucas-Kanade_sample_image_resized.jpg",
#     #     ),
#     #     sample_image_resized,
#     # )
#     cv2.imwrite(
#         os.path.join(results_directory, f"{parent_dir_name}_Lucas-Kanade_overlay.jpg"),
#         overlay,
#     )
