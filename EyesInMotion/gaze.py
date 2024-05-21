import math
import queue
import sys
import threading
import time

import cv2
import dlib
import numpy as np

# Define the fixed window size
WINDOW_SIZE = (200, 100)
# Shared queue to hold frames from multiple cameras
frame_queue = queue.Queue()
# Stop Event to signal when to stop the threads.
stop_event = threading.Event()

window_names = {}


def create_cam(camID):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r'EyesInMotion\shape_predictor_68_face_landmarks.dat')
    cap = cv2.VideoCapture(camID)
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
    for (x,y) in eye_points:
        cv2.circle(frame, (x,y), 2, (0, 0, 255), -1)

def eye_orientation(left_eye_center: list[tuple], right_eye_center: list[tuple]) -> tuple[float, tuple]:
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


    if(track_blinking):
        if ((x_max - x_min) / (y_max - y_min) > 4.75):
            print("BLINKING")
            cv2.putText(frame, "BLINKING", (0, y_min + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
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
    
    return np.array(eye_cropped_points, np.int32)

def mask_eyes(frame, left_eye_points, right_eye_points):
    lowerBound = np.array([0,0,0])
    upperBound = np.array([180,255,50])

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
    return final_result, result_with_eyes, black_around_eyes


class camThread(threading.Thread):
    def __init__(self, camID):
        threading.Thread.__init__(self)
        self.camID = camID
        self.detector, self.predictor, self.cam = create_cam(camID)

    def run(self):
        if not self.cam.isOpened():
            print(f"Error Opening Camera {self.camID}")
            return
        
        while not stop_event.is_set():
            eye_tracking(self.detector, self.predictor, self.cam, self.camID)


def displayFrames():
    while not stop_event.is_set():
        if not frame_queue.empty():
            name, camID, frame = frame_queue.get()
            if name not in window_names:
                cv2.namedWindow(name)
                window_names[name] = camID
            cv2.imshow(name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()

    
def eye_tracking(detector, predictor, cap, camID):
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to grab frame from camera {camID}")
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
                    x_min, y_min, x_max, y_max = calculate_fixed_bounding_box(eyes_center, rotated_frame.shape, WINDOW_SIZE)

                    # Crop the frame to the bounding box
                    cropped_frame = rotated_frame[y_min:y_max, x_min:x_max]

                    # Resize the cropped frame to fit the fixed window size
                    resized_frame = cv2.resize(cropped_frame, WINDOW_SIZE)
                        
                    eye_points_cropped = cropped_points(x_min, y_min, x_max, y_max, eye_points_rotated)

                    # draw_landmarks(resized_frame, eye_points_cropped)

                    draw_eye_crosshair(resized_frame, eye_points_cropped[:6], False, 0.5)
                    draw_eye_crosshair(resized_frame, eye_points_cropped[6:], False, 0.5)

                    # Show the resized frame
                    _, masked_eyes, _ = mask_eyes(resized_frame, eye_points_cropped[:6], eye_points_cropped[6:])
                    frame_queue.put(('Eyes', camID, resized_frame))
                    frame_queue.put(('Mask Eye', camID, masked_eyes))

            masked_frame, _, _ = mask_eyes(frame, left_eye_points, right_eye_points)
            _, _, inverse_masked_frame = mask_eyes(frame, left_eye_points, right_eye_points)


            # Crosshair for actual image
            draw_eye_crosshair(frame, left_eye_points, True, 0.5)
            draw_eye_crosshair(frame, right_eye_points, True, 0.5)

            draw_eye_crosshair(masked_frame, left_eye_points, True, 0.8, (0, 0, 255))
            draw_eye_crosshair(masked_frame, right_eye_points, True, 0.8, (0, 0, 255))

            # cv2.imshow('Frame', frame)
            # cv2.imshow('Mask', masked_frame)
            # cv2.imshow('Inverse Mask', inverse_masked_frame)
            frame_queue.put(('Frame', camID, frame))
            frame_queue.put(('Mask', camID, masked_frame))
            frame_queue.put(('Inverse Mask', camID, inverse_masked_frame))

    finally:
        # When everything is done, release the capture
        cap.release()



# Create and start the worker threads
thread1 = camThread(0)
thread2 = camThread(1)
thread1.start()
thread2.start()

# Run the display function in the main thread
displayFrames()

# Wait for the worker threads to finish
thread1.join()
thread2.join()

print("Application closed.")