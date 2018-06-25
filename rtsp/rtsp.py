import cv2
import rtsp
import time
server = rtsp.Rtsp()
cap = cv2.VideoCapture('11.mp4')
duration = 1/25.
while True:
    ret,frame = cap.read()
    if ret == False:
        print "end"
        break
    else:
        server.Push(frame)
        time.sleep(duration)
    pass