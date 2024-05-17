import queue
import sys
import threading
import time

import cv2
import dlib

# Shared queue to hold frames from both cameras
frame_queue = queue.Queue()
# Stop Event to signal when to stop the threads.
stop_event = threading.Event()

# Worker thread for capturing and processing frames
class camThread(threading.Thread):
    def __init__(self, camID):
        threading.Thread.__init__(self)
        self.camID = camID
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(r'EyesInMotion\\shape_predictor_68_face_landmarks.dat')
        self.cam = cv2.VideoCapture(camID)

    def run(self):
        if not self.cam.isOpened():
            print(f"Error Opening Camera {self.camID}")
            return

        while not stop_event.is_set():
            # Capture frames from the camera
            ret, frame = self.cam.read()
            if not ret:
                print(f"Unable to grab Frame from Camera {self.camID}")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = self.detector(gray)

            for details in dets:
                shape = self.predictor(gray, details)
                for i in range(36, 48):
                    x = shape.part(i).x
                    y = shape.part(i).y
                    cv2.circle(frame, (x, y), 2, (0, 255 - 100 * self.camID, 0 + self.camID * 200), -1)

            # Put the frame in the queue
            frame_queue.put((self.camID, frame))

        self.cam.release()

# Function to display frames in the main thread
def displayFrames():
    cv2.namedWindow("Camera 0")
    cv2.namedWindow("Camera 1")

    while not stop_event.is_set():
        if not frame_queue.empty():
            camID, frame = frame_queue.get()
            # Display the frames in the OpenCV windows
            cv2.imshow(f"Camera {camID}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()

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
