import time

import cv2
import dlib
import numpy as np


# Function to get the color mask for black
def getColorMask(img):
    lowerBound = np.array([0, 0, 0])
    upperBound = np.array([180, 255, 50])
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerBound, upperBound)
    return mask

# Function to overlay landmarks
def overlay_landmarks(frame, eye_points):
    for (x, y) in eye_points:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

# Function to crop eye region
def crop_eye_region(frame, eye_points):
    x_min = min(point[0] for point in eye_points)
    y_min = min(point[1] for point in eye_points)
    x_max = max(point[0] for point in eye_points)
    y_max = max(point[1] for point in eye_points)

    padding = 10
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(frame.shape[1], x_max + padding)
    y_max = min(frame.shape[0], y_max + padding)

    cropped_frame = frame[y_min:y_max, x_min:x_max]
    resized_frame = cv2.resize(cropped_frame, (200, 100))
    
    return resized_frame

# Load the pre-trained face detector
detector = dlib.get_frontal_face_detector()

# Load the pre-trained shape predictor file
predictor = dlib.shape_predictor(r'EyesInMotion/shape_predictor_68_face_landmarks.dat')

# Open a connection to the camera
cap = cv2.VideoCapture(0)

# Exit program if we can't open the camera
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cv2.namedWindow('mask')  # Create a window for the mask
cv2.namedWindow('frame')  # Create a window for the frame
cv2.namedWindow('Eyes')   # Create a window for the cropped eyes

# Give the camera some time to warm up
time.sleep(2)

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

        # Apply the color mask for black
        mask = getColorMask(frame)

        for d in dets:
            # Get the landmarks/parts for the face in box d.
            shape = predictor(gray, d)

            # Create a mask to overlay black areas only around the eyes
            eye_mask = np.zeros_like(mask)

            # Define the points for the left and right eyes
            left_eye_points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(36, 42)], np.int32)
            right_eye_points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(42, 48)], np.int32)

            # Create polygons for the eyes and fill them with white color on the mask
            cv2.fillPoly(eye_mask, [left_eye_points], 255)
            cv2.fillPoly(eye_mask, [right_eye_points], 255)

            black_around_eyes = cv2.bitwise_and(mask, eye_mask)

            # Create a white background
            white_background = np.full_like(frame, 255)

            # Combine the white background with the black mask
            result = cv2.bitwise_and(white_background, white_background, mask=cv2.bitwise_not(black_around_eyes))
            result[black_around_eyes == 255] = [0, 0, 0]  # Set black areas near the eyes to black

            # Create a black background
            black_background = np.zeros_like(frame)

            # Apply the inverse of the eye mask to create the black background
            inverse_eye_mask = cv2.bitwise_not(eye_mask)
            black_result = cv2.bitwise_and(black_background, black_background, mask=inverse_eye_mask)

            # Combine the original frame with the eye mask to keep the eye areas in their original colors
            result_with_eyes = cv2.bitwise_and(frame, frame, mask=eye_mask)
            result[inverse_eye_mask == 255] = black_result[inverse_eye_mask == 255]

            # Combine the result with the black background and black around eyes
            final_result = cv2.add(result, result_with_eyes)

            # Get the coordinates of the left and right eye landmarks
            left_eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
            right_eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]

            # Crop and resize each eye region
            left_eye = crop_eye_region(final_result, left_eye_points)
            right_eye = crop_eye_region(final_result, right_eye_points)

            # Concatenate the left and right eye images side by side
            eyes_side_by_side = np.hstack((left_eye, right_eye))

            # Show the concatenated eye images
            cv2.imshow('Eyes', eyes_side_by_side)

            # Show the frames
            cv2.imshow("mask", black_around_eyes)  # Display the black areas near the eyes
            cv2.imshow('frame', final_result)  # Display the frame with black background

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
