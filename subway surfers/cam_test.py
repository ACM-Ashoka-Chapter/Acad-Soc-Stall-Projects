import cv2, sys
for i in range(4):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ok = cap.isOpened()
    got = False
    if ok:
        got, f = cap.read()
        if got:
            print(f"camera_index {i}: OPEN, frame {f.shape[1]}x{f.shape[0]}")
    if not got:
        print(f"camera_index {i}: unavailable")
    cap.release()
