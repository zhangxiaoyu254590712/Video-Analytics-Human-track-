import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

# lower_blue=np.array([78,43,46])
# upper_blue=np.array([110,255,255])
# lower_blue=np.array([90,220,127])
# upper_blue=np.array([110,241,150])
# lower_blue=np.array([90,150,100])
# upper_blue=np.array([110,241,150])
lower_blue = np.array([100, 150, 46])
upper_blue = np.array([124, 255, 255])
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

def body(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    body = body_cascade.detectMultiScale(gray, 1.05, 3)
    bodyMaxHeight = 0
    bodyMaxArray = [0,0,0,0]
    ifBody = False
    for (bx,by,bw,bh) in body:
        ifBody = True
        body_image=frame[(bx):(bx+bw),(by):(by+bh)]
        cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(0,0,255),2)
        radius=detect_object(frame)
        if bh > bodyMaxHeight:
            bodyMaxHeight = bh
            bodyMaxArray = [bx,by,bw,bh]
        if(radius>0):
            f,inch=convert(radius,bh)
            strr=str(f)+"feet"+str(inch)+"inch"
            cv2.putText(frame,strr,(bx,by - 20),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),2)

def convert(radius,h):
    cm=h/(radius/12.5)
    cm=float(cm)/100
    f=cm/0.3048
    inch=(f-int(f))*12
    inch=int(inch)
    f=int(f)
    print(cm*100)
    return f,inch


def detect_object(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    cnt_blue = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(cnt_blue)>0:
        c_blue = max(cnt_blue, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c_blue)
        cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
        return radius
    else:
        return 0



cap=cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    if(ret != False):
        body(frame)






        cv2.imshow('video',frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
