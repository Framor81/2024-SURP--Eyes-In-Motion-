import math
import sys
import time

import cv2
import dlib
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Load the pre-trained Eye-UNet model
model = EyeUNet()
model.load_state_dict(torch.load('eye_unet_model.pth'))
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 96)),
    transforms.Normalize((0.5,), (0.5,))
])

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

# Define the fixed window size
WINDOW_SIZE = (200, 100)

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

            # Get the coordinates of the eye landmarks (points 36 to 48)
            left_eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
            right_eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]

            # Calculate the center of each eye by getting its mean
            left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
            right_eye_center = np.mean(right_eye_points, axis=0).astype(int)

            # Calculate the angle between the eye centers
            delta_x = right_eye_center[0] - left_eye_center[0]
            delta_y = right_eye_center[1] - left_eye_center[1]
            angle = math.degrees(np.arctan2(delta_y, delta_x))

            # Calculate the center between the two eyes
            eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2.0,
                           (left_eye_center[1] + right_eye_center[1]) / 2.0)

            # Get the rotation matrix for the affine transformation to align the eyes horizontally.
            M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

            # Apply the affine transformation to rotate the image based on the calculated rotation matrix.
            rotated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

            # Convert the rotated frame to grayscale
            gray_rotated = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the rotated image
            dets_rotated = detector(gray_rotated)

            for d_rotated in dets_rotated:
                # Get the landmarks/parts for the face in the rotated image
                shape_rotated = predictor(gray_rotated, d_rotated)

                # Get the coordinates of the eye landmarks (points 36 to 48)
                eye_points_rotated = [(shape_rotated.part(i).x, shape_rotated.part(i).y) for i in range(36, 48)]

                # Calculate the bounding box around the eye landmarks
                x_min = min(point[0] for point in eye_points_rotated)
                y_min = min(point[1] for point in eye_points_rotated)
                x_max = max(point[0] for point in eye_points_rotated)
                y_max = max(point[1] for point in eye_points_rotated)

                # Expand the bounding box a little for better visualization
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(rotated_frame.shape[1], x_max + padding)
                y_max = min(rotated_frame.shape[0], y_max + padding)

                # Crop the frame to the bounding box
                cropped_frame = rotated_frame[y_min:y_max, x_min:x_max]

                # Resize the cropped frame to fit the fixed window size
                resized_frame = cv2.resize(cropped_frame, WINDOW_SIZE)

                # Convert the resized frame to a PIL image
                pil_image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

                # Apply transformations
                input_tensor = transform(pil_image).unsqueeze(0)

                # Perform segmentation using Eye-UNet
                with torch.no_grad():
                    outputs = model(input_tensor)
                    final_output, deep_supervised_output1, deep_supervised_output2, deep_supervised_output3 = outputs
                    predicted_mask = torch.sigmoid(final_output).squeeze().cpu().numpy()

                # Threshold the predicted mask
                predicted_mask = (predicted_mask > 0.5).astype(np.uint8) * 255

                # Convert the predicted mask to BGR for visualization
                predicted_mask_bgr = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2BGR)

                # Draw the landmarks on the resized frame
                for (x, y) in eye_points_rotated:
                    cv2.circle(predicted_mask_bgr, (int((x - x_min) * scale_x), int((y - y_min) * scale_y)), 2, (0, 0, 255), -1)

                # Show the resized frame and predicted mask
                cv2.imshow('Eyes', np.hstack((resized_frame, predicted_mask_bgr)))

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
