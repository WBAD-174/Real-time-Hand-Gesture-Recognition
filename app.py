import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame")
        break

    frame = cv2.flip(frame, 1)  
    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()