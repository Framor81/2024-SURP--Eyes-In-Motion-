import sys
import threading
import time

import cv2
import dlib


# open face recognition detector
class camThread(threading.Thread):
    def __init__(self, camID):
        threading.Thread.__init__(self)
        self.camID = camID

    def run(self):
        print("Starting Camera " + str(self.camID))
        faceDetector(self.camID)


def faceDetector(camID):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r'EyesInMotion\\shape_predictor_68_face_landmarks.dat')

    cv2.namedWindow("Camera " + str(camID))
    cam = cv2.VideoCapture(camID)

    if not cam.isOpened():
        print("Error Opening Camera")
        return
    
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Unable to grab Frame")
                break   

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            dets = detector(gray)

            for details in dets:
                shape = predictor(gray, details)
                for i in range (36,48):
                    x = shape.part(i).x
                    y = shape.part(i).y
                    cv2.circle(frame, (x,y), 2, (0,255 - 100*camID, 0 + camID*10), -1)

            cv2.imshow('Camera ' + str(camID), frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # When everything is done, release the capture
        cam.release()
        cv2.destroyAllWindows()


thread1 = camThread(0)
thread2 = camThread(1)
thread1.start()
thread2.start()