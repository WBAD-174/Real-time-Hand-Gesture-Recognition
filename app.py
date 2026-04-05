import cv2


    
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('captured_video.mp4', fourcc, 30, (640, 480)) # (Filename, Video codec, fps, size)

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
    gray = cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow("Webcam", frame)
    cv2.imshow("Blured", blured)
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        recording = True
    if key == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()