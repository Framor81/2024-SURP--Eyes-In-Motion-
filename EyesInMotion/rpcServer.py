from xmlrpc.server import SimpleXMLRPCServer

import cv2
import dlib


def whoAmI(name: str, age: int):
    return "My name is " + name + " and I am " + str(age) + " years old."

def open_Camera():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r'shape_predictor_68_face_landmarks.dat')
    cap = cv2.VideoCapture(0)
    
    # Exit program if we can't open the camera
    if not cap.isOpened():
        return "Error: Could not open camera."
    
    while True:
        ret, frame = cap.read()
        if not ret:
            return "Failed to grab frame"
        
        cv2.imshow('Camera Feed', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "Camera feed ended."

server = SimpleXMLRPCServer(("localhost", 8000))
print("Server is running...")
server.register_function(whoAmI, "some_method")
server.register_function(open_Camera, "open_Camera")
server.serve_forever()
