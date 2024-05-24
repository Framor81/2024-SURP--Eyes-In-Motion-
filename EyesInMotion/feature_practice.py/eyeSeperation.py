import time

import cv2
import dlib
import numpy as np

# Load the pre-trained face detector
detector = dlib.get_frontal_face_detector()

# Pre-trained shape predictor file
predictor = dlib.shape_predictor(r'EyesInMotion\shape_predictor_68_face_landmarks.dat')

# Open a connection to the camera
cap = cv2.VideoCapture(0)

# Exit program if we can't open the camera
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Define the fixed window size for each eye
EYE_WINDOW_SIZE = (200, 100)

def overlay_landmarks(frame, eye_points):
    """
    Draws landmarks on the given frame at the specified eye points.
    
    Parameters:
    - frame: The image frame on which to draw the landmarks.
    - eye_points: List of (x, y) tuples representing the coordinates of the landmarks.
    """
    for (x, y) in eye_points:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

def crop_eye_region(frame, eye_points):
    """
    Crops the eye region from the frame based on the given eye points and resizes it to a fixed size.
    
    Parameters:
    - frame: The image frame from which to crop the eye region.
    - eye_points: List of (x, y) tuples representing the coordinates of the landmarks.
    
    Returns:
    - The cropped and resized eye region.
    """
    # Calculate the bounding box around the eye landmarks
    x_min = min(point[0] for point in eye_points)
    y_min = min(point[1] for point in eye_points)
    x_max = max(point[0] for point in eye_points)
    y_max = max(point[1] for point in eye_points)

    # Expand the bounding box a little for better visualization
    padding = 10
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(frame.shape[1], x_max + padding)
    y_max = min(frame.shape[0], y_max + padding)

    # Crop the frame to the bounding box
    cropped_frame = frame[y_min:y_max, x_min:x_max]

    # Resize the cropped frame to fit the fixed window size
    resized_frame = cv2.resize(cropped_frame, EYE_WINDOW_SIZE)
    
    return resized_frame

def draw_eye_crosshair(frame, eye_points):
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
    cv2.line(frame, (x_min, y_mid), (x_max, y_mid), (0, 255, 0), 1)
    
    # Draw the vertical line
    cv2.line(frame, (x_mid, y_min), (x_mid, y_max), (0, 255, 0), 1)

try:
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
            # Get the landmarks/parts for the face in box d
            shape = predictor(gray, d)

            # Get the coordinates of the left and right eye landmarks
            left_eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
            right_eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]

            # Overlay landmarks on the original frame
            overlay_landmarks(frame, left_eye_points)
            overlay_landmarks(frame, right_eye_points)

            # Draw crosshair on left eye
            draw_eye_crosshair(frame, left_eye_points)

            # Draw crosshair on right eye
            draw_eye_crosshair(frame, right_eye_points)

            # Crop and resize each eye regiona
            left_eye = crop_eye_region(frame, left_eye_points)
            right_eye = crop_eye_region(frame, right_eye_points)

            # Concatenate the left and right eye images side by side
            eyes_side_by_side = np.hstack((left_eye, right_eye))

            # Show the concatenated eye images
            cv2.imshow('Eyes', eyes_side_by_side)

        # Show the frame with the crosshairs
        cv2.imshow('Frame with Crosshairs', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
