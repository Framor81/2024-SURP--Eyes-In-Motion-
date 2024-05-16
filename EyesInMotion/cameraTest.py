import cv2


def try_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return False, None
    
    ret, frame = cap.read()
    cap.release()
    return ret, frame

for index in range(10):  # Try the first 10 indices
    ret, frame = try_camera(index)
    if ret:
        print(f"Camera found at index {index}")
        # Display the captured frame
        cv2.imshow(f'Captured Frame from Camera Index {index}', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
else:
    print("No camera found.")
