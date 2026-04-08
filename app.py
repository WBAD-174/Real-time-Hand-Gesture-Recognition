import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('captured_video.avi', fourcc, 30, (640, 480), False)  # False = grayscale video

recording = False

while True:
    # Read video 
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame")
        break
    
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))  
    blured = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blured, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 30, 60])
    upper = np.array([20, 150, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("Webcam", frame)
    cv2.imshow("Mask", mask_clean)

    if recording:
        out.write(mask_clean)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        recording = not recording
        print("Recording:", recording)
    if key == ord('q'):
        break
    
    
cap.release()
out.release()
cv2.destroyAllWindows()