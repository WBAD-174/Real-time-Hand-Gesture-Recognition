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
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None
recording = False

roi_x = 180
roi_y = 100
roi_w = 280
roi_h = 280
move_step = 20

resize_step = 20
min_roi_size = 120

cv2.namedWindow("Trackbars")

cv2.createTrackbar("H_min", "Trackbars", 0, 179, lambda x: None)
cv2.createTrackbar("H_max", "Trackbars", 20, 179, lambda x: None)
cv2.createTrackbar("S_min", "Trackbars", 15, 255, lambda x: None)
cv2.createTrackbar("S_max", "Trackbars", 150, 255, lambda x: None)
cv2.createTrackbar("V_min", "Trackbars", 20, 255, lambda x: None)
cv2.createTrackbar("V_max", "Trackbars", 255, 255, lambda x: None)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))

    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
    cv2.putText(frame, "ROI", (roi_x, max(roi_y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    roi_x = max(0, min(roi_x, 640 - roi_w))
    roi_y = max(0, min(roi_y, 480 - roi_h))

    roi_w = max(min_roi_size, min(roi_w, 640 - roi_x))
    roi_h = max(min_roi_size, min(roi_h, 480 - roi_y))

    roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    blured = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv = cv2.cvtColor(blured, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("H_min", "Trackbars")
    h_max = cv2.getTrackbarPos("H_max", "Trackbars")
    s_min = cv2.getTrackbarPos("S_min", "Trackbars")
    s_max = cv2.getTrackbarPos("S_max", "Trackbars")
    v_min = cv2.getTrackbarPos("V_min", "Trackbars")
    v_max = cv2.getTrackbarPos("V_max", "Trackbars")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
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
            largest_shifted = largest + np.array([[[roi_x, roi_y]]])
            hull_pts = cv2.convexHull(largest)
            hull_pts_shift = hull_pts + np.array([[[roi_x, roi_y]]])
            cv2.drawContours(frame, [largest_shifted],  -1, (0, 255, 0), 2)   # green = contour
            cv2.drawContours(frame, [hull_pts_shift], -1, (0, 0, 255), 2)   # red   = hull
            cv2.rectangle(frame, (roi_x + x,roi_y + y), (roi_x + x + w, roi_y + y + h), (255, 0, 0), 2)

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
    
    cv2.putText(frame, f"H:[{h_min},{h_max}] S:[{s_min},{s_max}] V:[{v_min},{v_max}]",
            (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if recording:
        out.write(frame)
        cv2.circle(frame, (620, 20), 8, (0, 0, 255), -1)  # red dot indicator

    if recording:
        out.write(frame)
        cv2.circle(frame, (620, 20), 8, (0, 0, 255), -1)  # red dot indicator

    cv2.imshow("Webcam", frame)
    cv2.imshow("Mask", mask_clean)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('w'):
        roi_y -= move_step
    elif key == ord('s'):
        roi_y += move_step
    elif key == ord('a'):
        roi_x -= move_step
    elif key == ord('d'):
        roi_x += move_step
    elif key == ord('+') or key == ord('='):
        roi_x -= resize_step // 2
        roi_y -= resize_step // 2
        roi_w += resize_step
        roi_h += resize_step
    elif key == ord('-') or key == ord('_'):
        roi_x += resize_step // 2
        roi_y += resize_step // 2
        roi_w -= resize_step
        roi_h -= resize_step
    elif key == ord('r'):
        recording = not recording
        if recording:
            out = cv2.VideoWriter('output.avi', fourcc, 30, (640, 480))
            print("Recording started: output.avi")
        else:
            out.release()
            out = None
            print("Recording stopped")
    elif key == ord('q'):
        break
    elif key == ord('p'):
        print("Current HSV:")
        print(f"lower = np.array([{h_min}, {s_min}, {v_min}])")
        print(f"upper = np.array([{h_max}, {s_max}, {v_max}])")

if out is not None:
    out.release()
cap.release()
cv2.destroyAllWindows()
