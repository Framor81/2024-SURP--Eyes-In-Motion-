import math
import time

import cv2
import dlib
import numpy as np
from irisHelper import GazeTracking

# Define the fixed window size
WINDOW_SIZE = (200, 100)

def create_cam():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r'EyesInMotion\shape_predictor_68_face_landmarks.dat')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    return detector, predictor, cap

def track_iris(detector, predictor, cap):
    def facial_landmarks(predictor, face, landmarks, lB, uB):
        shape = predictor(face, landmarks)
        eye_points = np.array([(shape.part(i).x, shape.part(i).y) for i in range(lB, uB)], np.int32)
        eye_center = np.mean(eye_points, axis=0).astype(int)
        return eye_points, eye_center

    def eye_orientation(left_eye_center, right_eye_center):
        angle = math.degrees(np.arctan2(right_eye_center[1] - left_eye_center[1], right_eye_center[0] - left_eye_center[0]))
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2.0, (left_eye_center[1] + right_eye_center[1]) / 2.0)
        return angle, eyes_center

    def orient_eyes(frame, detector, eyes_center, angle):
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        rotated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        gray_rotated = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)
        dets_rotated = detector(gray_rotated)
        return rotated_frame, gray_rotated, dets_rotated

    def calculate_fixed_bounding_box(center, frame_shape, window_size):
        half_width = window_size[0] // 3
        half_height = window_size[1] // 3
        x_center, y_center = center
        x_min = int(max(0, x_center - half_width))
        y_min = int(max(0, y_center - half_height))
        x_max = int(min(frame_shape[1], x_center + half_width))
        y_max = int(min(frame_shape[0], y_center + half_height))
        return x_min, y_min, x_max, y_max

    def draw_eye_crosshair(frame, eye_points, track_blinking=False, opacity=1.0, color=(0, 255, 0)):
        if len(eye_points) == 0:
            return False, (0, 0), 0
        x_mid = sum(point[0] for point in eye_points) // len(eye_points)
        y_mid = sum(point[1] for point in eye_points) // len(eye_points)
        x_min = min(point[0] for point in eye_points)
        x_max = max(point[0] for point in eye_points)
        y_min = min(point[1] for point in eye_points)
        y_max = max(point[1] for point in eye_points)
        copy = frame.copy()
        cv2.line(copy, (x_min, y_mid), (x_max, y_mid), color, 1)
        cv2.line(copy, (x_mid, y_min), (x_mid, y_max), color, 1)
        cv2.addWeighted(copy, opacity, frame, 1 - opacity, 0, frame)
        if track_blinking:
            eye_lid_opening = (x_max - x_min) / (y_max - y_min)
            if eye_lid_opening > 4.75:
                print("BLINKING")
                cv2.putText(frame, "BLINKING", (0, y_min + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return True, (x_mid, y_mid), eye_lid_opening
            else:
                return False, (x_mid, y_mid), eye_lid_opening
        else:
            return False, (x_mid, y_mid), 0

    def cropped_points(x_min, y_min, x_max, y_max, eye_points_rotated):
        scale_x = WINDOW_SIZE[0] / (x_max - x_min)
        scale_y = WINDOW_SIZE[1] / (y_max - y_min)
        eye_cropped_points = []
        for (x, y) in eye_points_rotated:
            new_x = int((x - x_min) * scale_x)
            new_y = int((y - y_min) * scale_y)
            eye_cropped_points.append((new_x, new_y))
        return np.array(eye_cropped_points, np.int32)

    def mask_eyes(frame, left_eye_points, right_eye_points):
        lowerBound = np.array([0, 0, 0])
        upperBound = np.array([180, 255, 50])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lowerBound, upperBound)
        eye_mask = np.zeros_like(mask)
        if len(left_eye_points) > 0:
            cv2.fillPoly(eye_mask, [left_eye_points], 255)
        if len(right_eye_points) > 0:
            cv2.fillPoly(eye_mask, [right_eye_points], 255)
        black_around_eyes = cv2.bitwise_and(mask, eye_mask)
        white_bg = np.full_like(frame, 255)
        result = cv2.bitwise_and(white_bg, white_bg, mask=cv2.bitwise_not(black_around_eyes))
        white = result.copy()
        result[black_around_eyes == 255] = [0, 0, 0]
        black_bg = np.zeros_like(frame)
        inverse_eye_mask = cv2.bitwise_not(eye_mask)
        black_result = cv2.bitwise_and(black_bg, black_bg, mask=inverse_eye_mask)
        result_with_eyes = cv2.bitwise_and(frame, frame, mask=eye_mask)
        result[inverse_eye_mask == 255] = black_result[inverse_eye_mask == 255]
        final_result = cv2.add(result, result_with_eyes)
        return final_result, result_with_eyes, white

    def count_black_pixels(image, eye_points, x_mid, y_mid, bottom_weight=1.0):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray_image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [eye_points], 255)
        black_pixel_count = np.sum((gray_image == 0) & (mask == 255))
        left_mask = np.zeros_like(mask)
        right_mask = np.zeros_like(mask)
        top_mask = np.zeros_like(mask)
        bottom_mask = np.zeros_like(mask)
        left_mask[:, :x_mid] = mask[:, :x_mid]
        right_mask[:, x_mid:] = mask[:, x_mid:]
        top_mask[:y_mid, :] = mask[:y_mid, :]
        bottom_mask[y_mid:, :] = mask[y_mid:, :]
        left_pixel_count = np.sum((gray_image == 0) & (left_mask == 255))
        right_pixel_count = np.sum((gray_image == 0) & (right_mask == 255))
        top_pixel_count = np.sum((gray_image == 0) & (top_mask == 255))
        bottom_pixel_count = np.sum((gray_image == 0) & (bottom_mask == 255)) * bottom_weight
        subregion_counts = {
            "left": left_pixel_count,
            "right": right_pixel_count,
            "top": top_pixel_count,
            "bottom": bottom_pixel_count
        }
        return black_pixel_count, subregion_counts

    def determine_gaze_direction(subregion_counts):
        left = subregion_counts["left"]
        right = subregion_counts["right"]
        top = subregion_counts["top"]
        bottom = subregion_counts["bottom"]
        center_threshold = 1.5
        if left > center_threshold * right and left > center_threshold * top and left > center_threshold * bottom:
            return "left"
        elif right > center_threshold * left and right > center_threshold * top and right > center_threshold * bottom:
            return "right"
        elif top > center_threshold * bottom and top > center_threshold * left and top > center_threshold * right:
            return "top"
        elif bottom > center_threshold * top and bottom > center_threshold * left and bottom > center_threshold * right:
            return "bottom"
        else:
            return "center"

    def process_eye(liar, eye_points_cropped, focus_on_eye):
        blinking, midpoint, eye_lid_opening = draw_eye_crosshair(liar, eye_points_cropped, True, 0.5)
        bottom_weight = 1.1 if eye_lid_opening < 1.5 else 1.0
        y_mid_adjusted = midpoint[1] + int((1 - eye_lid_opening) * 1.1)
        black_pixels, subregions = count_black_pixels(liar, eye_points_cropped, midpoint[0], y_mid_adjusted, bottom_weight)
        direction = determine_gaze_direction(subregions)
        return blinking, direction

    gaze = GazeTracking()
    focus_on = "right"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector(gray)

        for d in dets:
            left_eye_points, left_eye_center = facial_landmarks(predictor, gray, d, 36, 42)
            right_eye_points, right_eye_center = facial_landmarks(predictor, gray, d, 42, 48)
            angle, eyes_center = eye_orientation(left_eye_center, right_eye_center)
            rotated_frame, gray_rotated, dets_rotated = orient_eyes(frame, detector, eyes_center, angle)

            for d_rotated in dets_rotated:
                eye_points_rotated, _ = facial_landmarks(predictor, gray_rotated, d_rotated, 36, 48)
                x_min, y_min, x_max, y_max = calculate_fixed_bounding_box(eyes_center, rotated_frame.shape, WINDOW_SIZE)
                cropped_frame = rotated_frame[y_min:y_max, x_min:x_max]
                resized_frame = cv2.resize(cropped_frame, WINDOW_SIZE)
                eye_points_cropped = cropped_points(x_min, y_min, x_max, y_max, eye_points_rotated)
                masked_eyes, _, white_mask = mask_eyes(resized_frame, eye_points_cropped[:6], eye_points_cropped[6:])

                if focus_on in ["left", "both"]:
                    left_blinking, left_direction = process_eye(white_mask, eye_points_cropped[:6], "left")
                if focus_on in ["right", "both"]:
                    right_blinking, right_direction = process_eye(white_mask, eye_points_cropped[6:], "right")

                cv2.imshow('Mask Eye', white_mask)

                if focus_on in ["left", "both"]:
                    draw_eye_crosshair(resized_frame, eye_points_cropped[:6], False, 0.5)
                if focus_on in ["right", "both"]:
                    draw_eye_crosshair(resized_frame, eye_points_cropped[6:], False, 0.5)

                cv2.imshow('Eyes', resized_frame)

        gaze.refresh(frame)
        frame_with_pupil = gaze.annotated_frame()
        cv2.imshow('Pupil Tracking', frame_with_pupil)

        time.sleep(0.1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

try:
    detector, predictor, cap = create_cam()
    track_iris(detector, predictor, cap)
finally:
    cap.release()
    cv2.destroyAllWindows()
