import math
import sys
import time

import cv2
import dlib
import numpy as np

# Define the fixed window size
WINDOW_SIZE = (200, 100)

def create_cam():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r'EyesInMotion\shape_predictor_68_face_landmarks.dat')
    cap = cv2.VideoCapture(0)
    # Exit program if we can't open the camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    return detector, predictor, cap

def facial_landmarks(predictor, face, landmarks, lB, uB):
    """
    Get the landmarks/parts for the face in box d.
    Get the coordinates of the eye landmarks.
    Creates list of tuples of x and y coordinates of the left and right eye.
    shape.part is able to grab the point on the facial landmark.
    Calculate the center of each eye by getting its mean.
    """
    shape = predictor(face, landmarks)
    eye_points = np.array([(shape.part(i).x, shape.part(i).y) for i in range(lB, uB)], np.int32)
    eye_center = np.mean(eye_points, axis=0).astype(int)
    return eye_points, eye_center

def draw_landmarks(frame, eye_points):
    # Draw the landmarks on frame
    for (x, y) in eye_points:
        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

def eye_orientation(left_eye_center, right_eye_center):
    # Calculate the angle between the eye centers
    angle = math.degrees(np.arctan2(right_eye_center[1] - left_eye_center[1], right_eye_center[0] - left_eye_center[0]))
    # Calculate the center between the two eyes
    eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2.0, (left_eye_center[1] + right_eye_center[1]) / 2.0)
    return angle, eyes_center

def orient_eyes(frame, detector, eyes_center, angle):
    # Get the rotation matrix for the affine transformation to align the eyes horizontally.
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

    # Apply the affine transformation to rotate the image based on the calculated rotation matrix.
    rotated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    # Convert the rotated frame to grayscale
    gray_rotated = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the rotated image
    dets_rotated = detector(gray_rotated)
    return rotated_frame, gray_rotated, dets_rotated

def calculate_fixed_bounding_box(center, frame_shape, window_size):
    """
    Calculate a fixed-size bounding box around the eye center.
    
    Parameters:
    - center: Tuple representing the center coordinates of the eyes.
    - frame_shape: Tuple representing the dimensions of the frame.
    - window_size: Tuple representing the fixed window size.
    
    Returns:
    - x_min, y_min, x_max, y_max: The coordinates of the fixed-size bounding box.
    """
    half_width = window_size[0] // 2
    half_height = window_size[1] // 2

    x_center, y_center = center

    x_min = int(max(0, x_center - half_width))
    y_min = int(max(0, y_center - half_height))
    x_max = int(min(frame_shape[1], x_center + half_width))
    y_max = int(min(frame_shape[0], y_center + half_height))

    return x_min, y_min, x_max, y_max

def cropped_points(x_min, y_min, x_max, y_max, eye_points_rotated):
    # Calculate the scaling factors
    # Determine the ratio to scale the x and y coordinates to the window size
    scale_x = WINDOW_SIZE[0] / (x_max - x_min)
    scale_y = WINDOW_SIZE[1] / (y_max - y_min)
    eye_cropped_points = []
    for (x, y) in eye_points_rotated:
        new_x = int((x - x_min) * scale_x)
        new_y = int((y - y_min) * scale_y)
        eye_cropped_points.append((new_x, new_y))
    
    return np.array(eye_cropped_points, np.int32)

def draw_eye_crosshair(frame, eye_points, track_blinking=False, opacity=1.0, color=(0, 255, 0)):
    """
    Draws a crosshair on the eye region based on the given eye points.
    
    Parameters:
    - frame: The image frame on which to draw the crosshair.
    - eye_points: List of (x, y) tuples representing the coordinates of the landmarks.
    """
    # Calculate the midpoint of the eye
    x_mid = sum(point[0] for point in eye_points) // len(eye_points)
    y_mid = sum(point[1] for point in eye_points) // len(eye_points)
    
    # Calculate the width and height of the eye
    x_min = min(point[0] for point in eye_points)
    x_max = max(point[0] for point in eye_points)
    y_min = min(point[1] for point in eye_points)
    y_max = max(point[1] for point in eye_points)

    # create overlay
    copy = frame.copy() 
    # Draw the horizontal line
    horizontal_line = cv2.line(copy, (x_min, y_mid), (x_max, y_mid), color, 1)
    
    # Draw the vertical line
    vertical_line = cv2.line(copy, (x_mid, y_min), (x_mid, y_max), color, 1)
    cv2.addWeighted(copy, opacity, frame, 1 - opacity, 0, frame)

    if track_blinking:
        eye_lid_opening = (x_max - x_min) / (y_max - y_min) 
        if (eye_lid_opening > 4.75):
            print("BLINKING")
            cv2.putText(frame, "BLINKING", (0, y_min + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return True, (x_mid, y_mid), eye_lid_opening
        else:
            return False, (x_mid, y_mid), eye_lid_opening
    else:
        return False, (x_mid, y_mid), 0

def mask_eyes(frame, left_eye_points, right_eye_points):
    lowerBound = np.array([0, 0, 0])
    upperBound = np.array([180, 255, 50])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerBound, upperBound)

    # Create a mask to overlay black areas only around the eyes
    eye_mask = np.zeros_like(mask)

    # Create polygons for the eyes and fill them with white color on the mask
    cv2.fillPoly(eye_mask, [left_eye_points], 255)
    cv2.fillPoly(eye_mask, [right_eye_points], 255)
    
    black_around_eyes = cv2.bitwise_and(mask, eye_mask)

    # Combine the white background with the black mask
    white_bg = np.full_like(frame, 255)
    result = cv2.bitwise_and(white_bg, white_bg, mask=cv2.bitwise_not(black_around_eyes))
    white = result.copy()

    result[black_around_eyes == 255] = [0, 0, 0]  # Set black areas near the eyes to black

    # Create a black background
    black_bg = np.zeros_like(frame)

    # Apply the inverse of the eye mask to create the black background
    inverse_eye_mask = cv2.bitwise_not(eye_mask)
    black_result = cv2.bitwise_and(black_bg, black_bg, mask=inverse_eye_mask)

    # Combine the original frame with the eye mask to keep the eye areas in their original colors
    result_with_eyes = cv2.bitwise_and(frame, frame, mask=eye_mask)
    result[inverse_eye_mask == 255] = black_result[inverse_eye_mask == 255]

    # Combine the result with the black background and black around eyes
    final_result = cv2.add(result, result_with_eyes)

    # final_result is the masked eyes, result_with_eyes is just the eyes without the mask 
    return final_result, result_with_eyes, white

def count_black_pixels(image, eye_points, x_mid, y_mid, bottom_weight=1.0):
    """
    Count the number of black pixels in the specified eye region and its subregions.
    
    Parameters:
    - image: The image in which to count black pixels.
    - eye_points: List of (x, y) tuples representing the coordinates of the eye landmarks.
    - x_mid: x-coordinate for the midpoint of the eye.
    - y_mid: y-coordinate for the midpoint of the eye.
    - bottom_weight: Weighting factor for bottom pixels.
    
    Returns:
    - black_pixel_count: The number of black pixels in the specified eye region.
    - subregion_counts: A dictionary with the number of black pixels in each subregion.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    mask = np.zeros(gray_image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [eye_points], 255)

    black_pixel_count = np.sum((gray_image == 0) & (mask == 255))
    
    left_mask = np.zeros_like(mask)
    right_mask = np.zeros_like(mask)
    top_mask = np.zeros_like(mask)
    bottom_mask = np.zeros_like(mask)
    
    left_mask[:, :x_mid] = mask[:, :x_mid]
    right_mask[:, x_mid:] = mask[:, x_mid:]
    top_mask[:y_mid, :] = mask[:y_mid, :]
    bottom_mask[y_mid:, :] = mask[y_mid:, :]
    
    left_pixel_count = np.sum((gray_image == 0) & (left_mask == 255))
    right_pixel_count = np.sum((gray_image == 0) & (right_mask == 255))
    top_pixel_count = np.sum((gray_image == 0) & (top_mask == 255))
    bottom_pixel_count = np.sum((gray_image == 0) & (bottom_mask == 255)) * bottom_weight
    
    subregion_counts = {
        "left": left_pixel_count,
        "right": right_pixel_count,
        "top": top_pixel_count,
        "bottom": bottom_pixel_count
    }
    
    return black_pixel_count, subregion_counts

def determine_gaze_direction(subregion_counts):
    """
    Determine the gaze direction based on the number of black pixels in each subregion.
    
    Parameters:
    - subregion_counts: A dictionary with the number of black pixels in each subregion.
    
    Returns:
    - direction: A string representing the gaze direction ('left', 'right', 'top', 'bottom', 'center').
    """
    left = subregion_counts["left"]
    right = subregion_counts["right"]
    top = subregion_counts["top"]
    bottom = subregion_counts["bottom"]
    
    # Consider center if it's not dominated by any specific direction
    center_threshold = 1.5  # Adjust this threshold as needed

    if left > center_threshold * right and left > center_threshold * top and left > center_threshold * bottom:
        return "left"
    elif right > center_threshold * left and right > center_threshold * top and right > center_threshold * bottom:
        return "right"
    elif top > center_threshold * bottom and top > center_threshold * left and top > center_threshold * right:
        return "top"
    elif bottom > center_threshold * top and bottom > center_threshold * left and bottom > center_threshold * right:
        return "bottom"
    else:
        return "center"

def process_eye(liar, eye_points_cropped, focus_on_eye):
    """
    Process the given eye and determine the gaze direction.

    Parameters:
    - liar: The masked eye image.
    - eye_points_cropped: List of cropped eye points.
    - focus_on_eye: The eye to focus on ("left" or "right").

    Returns:
    - blinking: Boolean indicating if the eye is blinking.
    - direction: The determined gaze direction.
    """
    # Draw crosshair on the eye, track blinking and calculate eye lid opening
    blinking, midpoint, eye_lid_opening = draw_eye_crosshair(liar, eye_points_cropped, True, 0.5)

    # Set weight for bottom pixels based on eye lid opening
    bottom_weight = 1.1 if eye_lid_opening < 1.5 else 1.0

    # Adjust y_mid based on eye lid opening to improve detection accuracy
    y_mid_adjusted = midpoint[1] + int((1 - eye_lid_opening) * 1.1)

    # Count black pixels in the specified eye region and its subregions
    black_pixels, subregions = count_black_pixels(liar, eye_points_cropped, midpoint[0], y_mid_adjusted, bottom_weight)

    # Determine the gaze direction based on subregion counts
    direction = determine_gaze_direction(subregions)

    # Print eye tracking results
    # print(f"{focus_on_eye.capitalize()} eye midpoint: {midpoint}, Blinking: {blinking}, Black pixels: {black_pixels}, Direction: {direction}")

    # Return blinking status and gaze direction
    return blinking, direction

def main():
    try:
        detector, predictor, cap = create_cam()
        focus_on = "right"  # Options: "left", "right", "both"

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Flip the frame horizontally to invert the camera
            frame = cv2.flip(frame, 1)

            # Convert the frame to grayscale as HOG detector works on grayscale images
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            dets = detector(gray)

            for d in dets:
                left_eye_points, left_eye_center = facial_landmarks(predictor, gray, d, 36, 42)
                right_eye_points, right_eye_center = facial_landmarks(predictor, gray, d, 42, 48)

                angle, eyes_center = eye_orientation(left_eye_center, right_eye_center)

                rotated_frame, gray_rotated, dets_rotated = orient_eyes(frame, detector, eyes_center, angle)
                
                for d_rotated in dets_rotated:
                    eye_points_rotated, _ = facial_landmarks(predictor, gray_rotated, d_rotated, 36, 48)
                    x_min, y_min, x_max, y_max = calculate_fixed_bounding_box(eyes_center, rotated_frame.shape, WINDOW_SIZE)

                    # Crop the frame to the bounding box
                    cropped_frame = rotated_frame[y_min:y_max, x_min:x_max]

                    # Resize the cropped frame to fit the fixed window size
                    resized_frame = cv2.resize(cropped_frame, WINDOW_SIZE)
                        
                    eye_points_cropped = cropped_points(x_min, y_min, x_max, y_max, eye_points_rotated)

                    masked_eyes, _, white_mask = mask_eyes(resized_frame, eye_points_cropped[:6], eye_points_cropped[6:])

                    if focus_on in ["left", "both"]:
                        left_blinking, left_direction = process_eye(white_mask, eye_points_cropped[:6], "left")
                    if focus_on in ["right", "both"]:
                        right_blinking, right_direction = process_eye(white_mask, eye_points_cropped[6:], "right")

                    cv2.imshow('Mask Eye', white_mask)

                    if focus_on in ["left", "both"]:
                        draw_eye_crosshair(resized_frame, eye_points_cropped[:6], False, 0.5)
                    if focus_on in ["right", "both"]:
                        draw_eye_crosshair(resized_frame, eye_points_cropped[6:], False, 0.5)

                    # Show the resized frame
                    cv2.imshow('Eyes', resized_frame)
                    
                    time.sleep(0.1)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()