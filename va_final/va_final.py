import numpy as np
import cv2


MIN_MATCH_COUNT =10
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_mcs_eyepair_big.xml')
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')


detector = cv2.xfeatures2d.SIFT_create(1500)

FLANN_INDEX_KDITREE = 0
flannParam = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)
flann = cv2.FlannBasedMatcher(flannParam, {})

trainImg = cv2.imread("TrainingData/b1.png", 0)
trainKP, trainDesc = detector.detectAndCompute(trainImg, None)


def body(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    body = body_cascade.detectMultiScale(gray, 1.05, 3)
    bodyMaxHeight = 0
    bodyMaxArray = [0,0,0,0]
    ifBody = False
    for (bx,by,bw,bh) in body:
        ifBody = True
        body_gray = gray[by:by+bh, bx:bx+bw]
        body_color = frame[by:by+bh, bx:bx+bw]
        flag = detectMavs(body_color)
        if(flag):
            if bh > bodyMaxHeight:
                bodyMaxHeight = bh
                bodyMaxArray = [bx,by,bw,bh]
    if(ifBody and flag):
        cv2.rectangle(frame, (bodyMaxArray[0],bodyMaxArray[1]), (bodyMaxArray[0]+bodyMaxArray[2],bodyMaxArray[1]+bodyMaxArray[3]), (0, 0, 255), 2)
        body_gray = gray[bodyMaxArray[1]:bodyMaxArray[1]+bodyMaxArray[3],bodyMaxArray[0]:bodyMaxArray[0]+bodyMaxArray[2]]
        body_color = frame[bodyMaxArray[1]:bodyMaxArray[1]+bodyMaxArray[3],bodyMaxArray[0]:bodyMaxArray[0]+bodyMaxArray[2]]
        cv2.putText(frame,'person',(bx,by - 20),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        for (x,y,w,h) in faces:
            if (int(x+w/2))>bx and (int(x+w/2))<(bx+bw) and (int(y+h*2))>by and (int(y+h/2))<(by+bh):
                face_and_eyes(frame,x,y,w,h)





def face_and_eyes(frame,x,y,w,h):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        # To draw a rectangle in a face
        # print(x,y,w,h)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        # print(",,,,,,,,,,,,。。。。。。。。")
        # cv2.imwrite("11111111111.jpg", frame)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # Detects eyes of different sizes in the input image
        eyes = eye_cascade.detectMultiScale(roi_gray)
        #To draw a rectangle in eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
            # print("。。。。。。。。")

def face_flag(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        return x,y,w,h







# def slidewindow(frame,x,y,w):
#     row=int(frame.shape[0])
#     col=int(frame.shape[1])
#     l=int(w/10)
#     # cv2.rectangle(frame,(x,y),(x+l,y+l),(0,0,255),2)
#     for i in range(0,row-l,int(l/5)):
#         for j in range(0,col-l,int(l/5)):
#             # cv2.rectangle(frame,(i,j),(i+l,j+l),(0,0,255),2)
#             # slidewindow=frame(i:i+l,j:j+l)
#             window=frame[i:i+l,j:j+l]
#             if(detectMavs(window)==True):
#                 return True




def detectMavs(frame):
    flag = True
    QueryImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    queryKP, queryDesc = detector.detectAndCompute(QueryImg, None)
    matches = flann.knnMatch(queryDesc, trainDesc, k=2)
    goodMatch = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            goodMatch.append(m)
    if len(goodMatch) > MIN_MATCH_COUNT:
        tp = []
        qp = []
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp, qp = np.float32((tp, qp))
        H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
        h, w = trainImg.shape
        trainBorder = np.float32([[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]])
        queryBorder = cv2.perspectiveTransform(trainBorder, H)
        cv2.polylines(frame, [np.int32(queryBorder)], True, (0, 255, 0), 5)
        flag = True
    else:
        flag = False
    return flag




if __name__ == '__main__':
    # hog = cv2.HOGDescriptor()
    # hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
    cap=cv2.VideoCapture(0)
    while True:
        ret,frame=cap.read()
        body(frame)

        cv2.imshow('frame',frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
