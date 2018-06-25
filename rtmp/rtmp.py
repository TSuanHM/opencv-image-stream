import rtmp
import cv2
import time

cap = cv2.VideoCapture('11.mp4')
fps = 25
client = rtmp.Rtmp('rtmp://192.168.235.129:1935/mytv/1',1920,1080)
duration = 1.0/fps
while True:
	s = time.time()
	ret,frame = cap.read()
	client.Push(frame)
	cost = time.time()-s
	if cost < duration:
		time.sleep(duration-cost)
	else:
		print 'cost:',cost

cap.release()
cv2.destroyAllWindows()
