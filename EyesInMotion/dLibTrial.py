#!/usr/bin/python

import sys
import time

# import opencv
import cv2
import dlib

detector = dlib.get_frontal_face_detector()

# Open a connection to the camera
cap = cv2.VideoCapture(0)

# Exit program if we can't open the camera
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

try:
    while True:
        # Capture frame-by-frame
        # cap.read() captures information from our camera ret is a boolean of whether we are able to capture information and
        # frame is the feed of what the camera sees
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to grayscale as HOG detector works on grayscale images
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        dets = detector(gray)
        print("Number of faces detected: {}".format(len(dets)))


        # Display the frame using OpenCV
        # dets are tuples of top left and bottom right I believe
        for d in dets:
            cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (230, 255, 0), 3)
        cv2.imshow('frame', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()