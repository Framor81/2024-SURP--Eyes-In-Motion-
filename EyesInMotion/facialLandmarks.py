import sys
import time

import cv2
import dlib

# Load the pre-trained face detector
detector = dlib.get_frontal_face_detector()

# pre-trained shape predictor file 
predictor = dlib.shape_predictor('EyesInMotion\shape_predictor_68_face_landmarks.dat')

# Open a connection to the camera
cap = cv2.VideoCapture(0)

# Exit program if we can't open the camera
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

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
        # print("Number of faces detected: {}".format(len(dets)))

        # Display the frame using OpenCV
        for d in dets:
            # Get the landmarks/parts for the face in box d.
            shape = predictor(gray, d)

            # Draw the rectangle around the face
            # cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (240, 255, 0), 3)

            # Draw the landmarks on the face
            for i in range(36, 48): # There are 68 landmark points but we only care about 36-48 for our eyes
                x = shape.part(i).x
                y = shape.part(i).y
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        # Show the frame
        cv2.imshow('frame', frame)
        time.sleep(0.1)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
