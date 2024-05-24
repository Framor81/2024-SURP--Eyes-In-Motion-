import time

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist


# Function to get the color mask for black
def getColorMask(img):
    lowerBound = np.array([0, 0, 0])
    upperBound = np.array([180, 255, 50])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerBound, upperBound)
    return mask

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
    
    return resized_frame, x_min, y_min

# Function to draw horizontal lines over eyes
def draw_horizontal_lines(frame, eye_points, is_blinking):
    y_mid = sum(point[1] for point in eye_points) // len(eye_points)
    y_min = min(point[1] for point in eye_points)
    y_max = max(point[1] for point in eye_points)

    for y in range(y_min, y_max, 10):
        color = (0, 0, 255) if is_blinking else (0, 255, 0)
        cv2.line(frame, (eye_points[0][0], y), (eye_points[-1][0], y), color, 1)

# Function to draw crosshair on the eye region
def draw_eye_crosshair(frame, eye_points):
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

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

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
cv2.namedWindow('Eyes_Inverse')   # Create a window for the masked eyes

# Give the camera some time to warm up
time.sleep(2)

# Eye aspect ratio to indicate blink
EYE_AR_THRESH = 0.20  # Adjust this threshold as needed

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

            # Calculate the EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye_points)
            right_ear = eye_aspect_ratio(right_eye_points)
            
            # Check if either eye is closed
            is_blinking = left_ear < EYE_AR_THRESH or right_ear < EYE_AR_THRESH

            if is_blinking:
                print("Blink")

            # Create polygons for the eyes and fill them with white color on the mask
            cv2.fillPoly(eye_mask, [left_eye_points], 255)
            cv2.fillPoly(eye_mask, [right_eye_points], 255)

            black_around_eyes = cv2.bitwise_and(mask, eye_mask)

            # Create a white background
            white_background = np.full_like(frame, 255)

            # Combine the white background with the black mask
            result = cv2.bitwise_and(white_background, white_background, mask=cv2.bitwise_not(black_around_eyes))
            result[black_around_eyes == 255] = [0, 0, 0]  # Set black areas near the eyes to black

            # Create a gray background
            gray_background = np.full_like(frame, 127)

            # Apply the inverse of the eye mask to create the gray background
            inverse_eye_mask = cv2.bitwise_not(eye_mask)
            gray_result = cv2.bitwise_and(gray_background, gray_background, mask=inverse_eye_mask)

            # Combine the original frame with the eye mask to keep the eye areas in their original colors
            result_with_eyes = cv2.bitwise_and(frame, frame, mask=eye_mask)
            result[inverse_eye_mask == 255] = gray_result[inverse_eye_mask == 255]

            # Combine the result with the gray background and black around eyes
            final_result = cv2.add(result, result_with_eyes)

            # Get the coordinates of the left and right eye landmarks
            left_eye_points_list = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
            right_eye_points_list = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]

            # Convert lists to numpy arrays
            left_eye_points_np = np.array(left_eye_points_list, np.int32)
            right_eye_points_np = np.array(right_eye_points_list, np.int32)

            # Crop and resize each eye region
            left_eye, left_x_min, left_y_min = crop_eye_region(final_result, left_eye_points_list)
            right_eye, right_x_min, right_y_min = crop_eye_region(final_result, right_eye_points_list)

            # Adjust eye points to the cropped eye region
            adjusted_left_eye_points = [(x - left_x_min, y - left_y_min) for (x, y) in left_eye_points_list]
            adjusted_right_eye_points = [(x - right_x_min, y - right_y_min) for (x, y) in right_eye_points_list]

            # Draw horizontal lines on cropped eye regions
            draw_horizontal_lines(left_eye, adjusted_left_eye_points, is_blinking)
            draw_horizontal_lines(right_eye, adjusted_right_eye_points, is_blinking)

            # Draw eye crosshairs on cropped eye regions
            draw_eye_crosshair(left_eye, adjusted_left_eye_points)
            draw_eye_crosshair(right_eye, adjusted_right_eye_points)

            # Concatenate the left and right eye images side by side
            eyes_side_by_side = np.hstack((left_eye, right_eye))

            # Show the concatenated eye images
            cv2.imshow('Eyes', eyes_side_by_side)

            # Draw horizontal lines for the final frame to indicate blinking
            if is_blinking:
                draw_horizontal_lines(final_result, left_eye_points_list, is_blinking)
                draw_horizontal_lines(final_result, right_eye_points_list, is_blinking)

            # Draw the eye crosshairs on the final frame
            draw_eye_crosshair(final_result, left_eye_points_list)
            draw_eye_crosshair(final_result, right_eye_points_list)

            # Show the frames
            cv2.imshow("mask", black_around_eyes)  # Display the black areas near the eyes
            cv2.imshow('frame', final_result)  # Display the frame with gray background

            # Crop and orient eyes for the masked image
            left_eye_masked = crop_eye_region(black_around_eyes, left_eye_points_list)[0]
            right_eye_masked = crop_eye_region(black_around_eyes, right_eye_points_list)[0]
            
            # Highlight eye regions in the mask to differentiate blinks
            cv2.polylines(black_around_eyes, [left_eye_points_np], isClosed=True, color=(127, 127, 127), thickness=1)
            cv2.polylines(black_around_eyes, [right_eye_points_np], isClosed=True, color=(127, 127, 127), thickness=1)

            # Concatenate the left and right eye masked images side by side
            eyes_inverse = np.hstack((left_eye_masked, right_eye_masked))
            
            # Show the concatenated eye masked images
            cv2.imshow('Eyes_Inverse', eyes_inverse)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
