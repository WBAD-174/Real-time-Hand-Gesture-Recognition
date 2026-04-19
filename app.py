import cv2
import numpy as np
import math
from collections import deque

def count_fingers(contour, hull_indices, min_defect_depth=15000):
    """Count extended fingers using convexity defects."""
    if len(hull_indices) < 3:
        return 0

    defects = cv2.convexityDefects(contour, hull_indices)
    if defects is None:
        return 0

    finger_count = 0
    for defect in defects:
        s, e, f, depth = defect[0]
        start = contour[s][0]
        end   = contour[e][0]
        far   = contour[f][0]

        # Angle at the valley point between the two finger sides
        a = math.dist(start, end)
        b = math.dist(far, start)
        c = math.dist(far, end)
        if b * c == 0:
            continue
        cos_angle = (b**2 + c**2 - a**2) / (2 * b * c)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = math.degrees(math.acos(cos_angle))

        # Shallow angle + sufficient depth => finger gap
        if angle < 90 and depth > min_defect_depth:
            finger_count += 1

    return finger_count + 1  # N defects between fingers = N+1 fingers


def classify_sign(fingers, solidity, aspect_ratio):
    """
    Rule-based sign classifier.
      solidity     = contour_area / hull_area  (fist ~high, open ~lower)
      aspect_ratio = w / h
    """
    if fingers == 0 or fingers == 1:
        if solidity > 0.85:
            return "FIST (0)"
        else:
            return "THUMBS UP / 1"

    if fingers == 2:
        return "PEACE / 2"

    if fingers == 3:
        return "3"

    if fingers == 4:
        return "4"

    if fingers >= 5:
        return "OPEN HAND (5)"

    return "UNKNOWN"


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam")
    exit()

bbox_history = deque(maxlen=10)    

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    blured = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blured, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 15, 20])
    upper = np.array([20, 150, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask_open  = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sign_label = ""

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > 2000:
            x, y, w, h = cv2.boundingRect(largest)

            bbox_history.append((x, y, w, h))

            sx = int(np.mean([b[0] for b in bbox_history]))
            sy = int(np.mean([b[1] for b in bbox_history]))
            sw = int(np.mean([b[2] for b in bbox_history]))
            sh = int(np.mean([b[3] for b in bbox_history]))

            # Draw contour and convex hull
            hull_pts = cv2.convexHull(largest)
            cv2.drawContours(frame, [largest],  -1, (0, 255, 0), 2)   # green = contour
            cv2.drawContours(frame, [hull_pts], -1, (0, 0, 255), 2)   # red   = hull
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Geometry features
            hull_area    = cv2.contourArea(hull_pts)
            solidity     = area / hull_area if hull_area > 0 else 0
            aspect_ratio = w / h if h > 0 else 0

            # Finger counting via convexity defects
            hull_indices = cv2.convexHull(largest, returnPoints=False)
            fingers      = count_fingers(largest, hull_indices)

            sign_label = classify_sign(fingers, solidity, aspect_ratio)

            # Debug info on frame
            cv2.putText(frame, f"Fingers: {fingers}  Solidity: {solidity:.2f}",
                        (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Isolated hand shape window
            hand_crop = mask_clean[sy:sy+sh, sx:sx+sw]
            hand_resized = cv2.resize(hand_crop, (200, 200))
            cv2.imshow("Hand Shape", hand_resized)

    cv2.putText(frame, sign_label, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)

    cv2.imshow("Webcam", frame)
    cv2.imshow("Mask", mask_clean)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
