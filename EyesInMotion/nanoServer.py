# server.py
import base64
from xmlrpc.server import SimpleXMLRPCServer

import cv2


def capture_frame():
    ret, frame = cap.read()
    if not ret:
        return None
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

def main():
    global cap
    cap = cv2.VideoCapture(0)
    server = SimpleXMLRPCServer(('0.0.0.0', 8000))
    server.register_function(capture_frame, 'capture_frame')
    print("Server started")
    server.serve_forever()

if __name__ == "__main__":
    main()
