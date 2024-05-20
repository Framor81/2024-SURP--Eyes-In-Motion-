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
    # Get the landmarks/parts for the face in box d

    Get the coordinates of the eye landmarks
    creates list of tuples of x and y coordinates of the left and right eye
    shape.part is able to grab the point on the facial landmark
    Calculate the center of each eye by getting its mean
    """
    shape = predictor(face, landmarks)
    eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(lB, uB)]
    eye_center = np.mean(eye_points, axis=0).astype(int)
    return eye_points, eye_center

def eye_orientation(left_eye_center: list[tuple], right_eye_center: list[tuple]) -> tuple[float, tuple]:
    # Calculate the angle between the eye centers
    angle = math.degrees(np.arctan2(right_eye_center[1] - left_eye_center[1], right_eye_center[0] - left_eye_center[0]))
    # Calculate the center between the two eyes
    eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2.0,
                    (left_eye_center[1] + right_eye_center[1]) / 2.0)
    return angle, eyes_center

def orient_eyes(frame, detector, eyes_center, angle):
    
    # Get the rotation matrix for the affine transformation to align the eyes horizontally.
    # eyes_center: The midpoint between the eye centers, used as the rotation center.
    # angle: The angle to rotate the image, calculated to make the eyes level.
    # 1.0: No scaling, keeping the image size unchanged.
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

    # Apply the affine transformation to rotate the image based on the calculated rotation matrix.
    # cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0])):
    # - frame: The input image to be transformed.
    # - M: The 2x3 affine transformation matrix obtained from cv2.getRotationMatrix2D.
    # - (frame.shape[1], frame.shape[0]): The size of the output image, maintaining the original dimensions.
    rotated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    # Convert the rotated frame to grayscale
    gray_rotated = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the rotated image
    dets_rotated = detector(gray_rotated)
    return rotated_frame, gray_rotated, dets_rotated

def calculate_bounding_box(eye_points_rotated: list[tuple], frame_shape: tuple) -> tuple[int, int, int, int]:
    """
    Calculate the bounding box around the eye landmarks and expand it for better visualization.
    
    Parameters:
    - eye_points_rotated: List of tuples representing the coordinates of the eye landmarks.
    - frame_shape: Tuple representing the dimensions of the frame.
    
    Returns:
    - x_min, y_min, x_max, y_max: The coordinates of the bounding box.
    """
    # Calculate the bounding box around the eye landmarks
    x_min = min(point[0] for point in eye_points_rotated)
    y_min = min(point[1] for point in eye_points_rotated)
    x_max = max(point[0] for point in eye_points_rotated)
    y_max = max(point[1] for point in eye_points_rotated)

    # Expand the bounding box a little for better visualization
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(frame_shape[1], x_max + padding)
    y_max = min(frame_shape[0], y_max + padding)

    return x_min, y_min, x_max, y_max


def draw_eye_crosshair(frame, eye_points, track_blinking: bool):
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

    # Draw the horizontal line
    horizontal_line = cv2.line(frame, (x_min, y_mid), (x_max, y_mid), (0, 255, 0), 1)
    
    # Draw the vertical line
    vertical_line = cv2.line(frame, (x_mid, y_min), (x_mid, y_max), (0, 255, 0), 1)

    if(track_blinking):
        if ((x_max-x_min)/(y_max - y_min) > 4.75):
            print("BLINKING")
            cv2.putText(frame, "BLINKING", (0, y_min+100), cv2.FONT_HERSHEY_SIMPLEX , 3, 200, 3)
            return True
        else:
            return False
    

def cropped_points(x_min, y_min, x_max, y_max, eye_points_rotated: list[tuple]):
    # Calculate the scaling factors
    # Determine the ratio to scale the x and y coordinates to the window size
    scale_x = WINDOW_SIZE[0] / (x_max - x_min)
    scale_y = WINDOW_SIZE[1] / (y_max - y_min)
    eye_cropped_points = []
    for (x, y) in eye_points_rotated:
        new_x = int((x - x_min) * scale_x)
        new_y = int((y - y_min) * scale_y)
        eye_cropped_points.append((new_x, new_y))
    
    return eye_cropped_points

try:
    detector, predictor, cap = create_cam()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

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
                x_min, y_min, x_max, y_max = calculate_bounding_box(eye_points_rotated, rotated_frame.shape)

                # Crop the frame to the bounding box
                cropped_frame = rotated_frame[y_min:y_max, x_min:x_max]

                # Resize the cropped frame to fit the fixed window size
                resized_frame = cv2.resize(cropped_frame, WINDOW_SIZE)
                    
                eye_points_cropped = cropped_points(x_min, y_min, x_max, y_max, eye_points_rotated)

                # Draw the landmarks on the resized frame
                # Scale and draw each landmark on the resized frame to maintain alignment
                for (x, y) in eye_points_cropped:
                    cv2.circle(resized_frame, (x, y), 2, (0, 0, 255), -1)

                draw_eye_crosshair(resized_frame, eye_points_cropped[0:6], False)
                draw_eye_crosshair(resized_frame, eye_points_cropped[6:12], False)

                # Show the resized frame
                cv2.imshow('Eyes', resized_frame)

        # Cross hair for actual image
        draw_eye_crosshair(frame, left_eye_points, True)
        draw_eye_crosshair(frame, right_eye_points, True)
        cv2.imshow('Frame', frame)


        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
