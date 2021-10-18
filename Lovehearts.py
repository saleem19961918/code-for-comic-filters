import dlib
from math import hypot
import cv2
cap=cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("C:\\Users\\SHAIK.RAHEEM\\Downloads\\shape_predictor_68_face_landmarks (2).dat")
img=cv2.imread("C:\\Users\\SHAIK.RAHEEM\\Downloads\\heart.png")
while True:
    ret,frame = cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    for face  in faces:
        
        landmarks=predictor(gray,face)
        lefteyeleft=(landmarks.part(36).x,landmarks.part(36).y)
        
        lefteyeright=(landmarks.part(39).x,landmarks.part(39).y)
        righteyeleft=(landmarks.part(42).x,landmarks.part(42).y)
        righteyeright=(landmarks.part(45).x,landmarks.part(45).y)
        heartwidth=lefteyeright[0]-lefteyeleft[0]
        hearwidth=25*heartwidth
        heartheight=heartwidth
        
        heartimage=cv2.resize(img,(heartwidth,heartheight))
        heartimagegray=cv2.cvtColor(heartimage,cv2.COLOR_BGR2GRAY)
        lefteyecenter=((lefteyeleft[0]+lefteyeright[0])/2,(lefteyeleft[1]+lefteyeright[1])/2)
        righteyecenter=((righteyeleft[0]+righteyeright[0])/2,(righteyeleft[1]+righteyeright[1])/2)
        
        topleft=(int(lefteyecenter[0]-heartwidth/2),int(lefteyecenter[1]-heartheight/2))
        topright=(int(righteyecenter[0]-heartwidth/2),int(righteyecenter[1]-heartheight/2))
        heartarealeft=frame[topleft[1]:topleft[1]+heartheight,
                            topleft[0]:topleft[0]+heartwidth]
        _,heartmask = cv2.threshold(heartimagegray,25,255,cv2.THRESH_BINARY_INV)
        heartareanoheartleft=cv2.bitwise_and(heartarealeft,heartarealeft, mask = heartmask)
        finalheartleft=cv2.add(heartareanoheartleft,heartimage)
        frame[topleft[1]:topleft[1]+heartheight,
                            topleft[0]:topleft[0]+heartwidth]=finalheartleft
        
        heartarearight=frame[topright[1]:topright[1]+heartheight,
                            topright[0]:topright[0]+heartwidth]
        _,heartmask = cv2.threshold(heartimagegray,25,255,cv2.THRESH_BINARY_INV)
        heartareanoheartright=cv2.bitwise_and(heartarearight,heartarearight, mask = heartmask)
        finalheartright=cv2.add(heartareanoheartright,heartimage)
        frame[topright[1]:topright[1]+heartheight,
                            topright[0]:topright[0]+heartwidth]=finalheartright
        
        
        
        
      
        
        
       
        
        
        
        
        
        
      
        
        
        
    cv2.imshow("frame",frame)
    key=cv2.waitKey(1)
    if key ==27:
        cap.release()
        cv2.destroyAllWindows()
        break
        
