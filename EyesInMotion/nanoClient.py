# client.py
import base64
from xmlrpc.client import ServerProxy

import cv2
import numpy as np


def main():
    server = ServerProxy('http://<server-ip>:8000')
    cv2.namedWindow("Live Feed", cv2.WINDOW_AUTOSIZE)

    while True:
        try:
            jpg_as_text = server.capture_frame()
            if jpg_as_text is None:
                continue
            jpg_as_bytes = base64.b64decode(jpg_as_text)
            np_arr = np.frombuffer(jpg_as_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv2.imshow('Live Feed', frame)
        except Exception as e:
            print(f"Error: {e}")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
