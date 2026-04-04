import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('captured_video.mp4', fourcc, 30, (640, 480)) # (Filename, Video codec, fps, size)

while True:
    # Read video 
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame")
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))  
    blured = cv2.GaussianBlur(frame, (5, 5), 0)
    cv2.imshow("Webcam", frame)
    cv2.imshow("Blured", blured)
    out.write(frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()