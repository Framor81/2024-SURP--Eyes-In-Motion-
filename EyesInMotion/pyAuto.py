import cv2
import dlib
import numpy as np
import pyautogui


class FacialControl:
    def __init__(self):
        self.detector, self.predictor, self.cap = FacialControl.create_camera()
        self.blinking = False

    def control_mouse(self):
        return

    def detect_face(self):
        try:
            while True:
                # Capture frame-by-frame
                ret, self.frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Flip the frame horizontally to invert the camera
                self.frame = cv2.flip(self.frame, 1)

                # Convert the frame to grayscale as HOG detector works on grayscale images
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the image
                dets = self.detector(gray)

                for d in dets:
                    left_eye_points, left_eye_center = self.facial_landmarks(gray, d, 36, 42)
                    right_eye_points, right_eye_center = self.facial_landmarks(gray, d, 42, 48)

                    # Draw landmarks
                    self.draw_landmarks(self.frame, left_eye_points)
                    self.draw_landmarks(self.frame, right_eye_points)
                    
                    # Draw crosshair
                    self.draw_eye_crosshair(self.frame, left_eye_points, False, 0.5)
                    self.draw_eye_crosshair(self.frame, right_eye_points, False, 0.5)

                    # Show the frame
                    cv2.imshow('Eyes', self.frame)
                    
                # Break the loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # When everything is done, release the capture
            self.cap.release()
            cv2.destroyAllWindows()

    def is_blinking(self):
        return self.blinking

    @staticmethod
    def create_camera():
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(r'EyesInMotion\shape_predictor_68_face_landmarks.dat')
        cap = cv2.VideoCapture(0)

        # Exit program if we can't open the camera
        if not cap.isOpened():
            print("Error: Could not open camera.")
            exit()
        return detector, predictor, cap

    def facial_landmarks(self, face, landmarks, lB, uB):
        """
        Get the landmarks/parts for the face in box d.
        Get the coordinates of the eye landmarks.
        Creates list of tuples of x and y coordinates of the left and right eye.
        shape.part is able to grab the point on the facial landmark.
        Calculate the center of each eye by getting its mean.
        """
        shape = self.predictor(face, landmarks)
        eye_points = np.array([(shape.part(i).x, shape.part(i).y) for i in range(lB, uB)], np.int32)
        eye_center = np.mean(eye_points, axis=0).astype(int)
        return eye_points, eye_center

    @staticmethod
    def draw_landmarks(frame, eye_points):
        # Draw the landmarks on frame
        for (x, y) in eye_points:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    def draw_eye_crosshair(self, frame, eye_points, track_blinking=False, opacity=1.0, color=(0, 255, 0)):
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

        if track_blinking:
            eye_lid_opening = (x_max - x_min) / (y_max - y_min) 
            if eye_lid_opening > 4.75:
                print("BLINKING")
                cv2.putText(frame, "BLINKING", (0, y_min + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.blinking = True
                return self.blinking, (x_mid, y_mid), eye_lid_opening
            else:
                self.blinking = False
                return self.blinking, (x_mid, y_mid), eye_lid_opening
        else:
            self.blinking = False
            return self.blinking, (x_mid, y_mid), 0
        

def main():
    my_face = FacialControl()
    my_face.detect_face() 
    print(my_face.is_blinking())

if __name__ == "__main__":
    main()
