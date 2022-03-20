import time
import cv2
import numpy as np
from datetime import datetime

cap = cv2.VideoCapture(0)
avg = None
kernel = np.ones((5, 5))

while True: 
    _,frame = cap.read() 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (5, 5))
    
    if avg is None:
        avg = gray.copy().astype("float")
        continue
    
    cv2.accumulateWeighted(gray, avg, 0.05)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    percentage = np.count_nonzero(thresh) / thresh.size

    if percentage > 0.01:
        print(percentage, datetime.utcnow().strftime('%H:%M:%S'))
    
    """ 
    contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 
    if len(contours) > 0:
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]   
        
        if cv2.contourArea(cnt) > 1500:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(1,255,0),2)
            timestamp = datetime.utcnow().strftime('%H:%M:%S')
            percentage = np.count_nonzero(thresh)/thresh.size
            print(percentage, end=' ') #0.005
            print(timestamp, 'Motion detected')
    """ 

    cv2.imshow("Difference", thresh)
    cv2.imshow("Camera", frame)   

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
cv2.destroyAllWindows()
