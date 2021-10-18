#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dlib
from math import hypot
import cv2


# In[2]:


cap=cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("C:\\Users\\SHAIK.RAHEEM\\Downloads\\shape_predictor_68_face_landmarks (2).dat")
headimage=cv2.imread("C:\\Users\\SHAIK.RAHEEM\\Downloads\\new.png")


# In[3]:


while True:
    ret,frame = cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    for face  in faces:
        
        landmarks=predictor(gray,face)
        lefthead=(landmarks.part(17).x,landmarks.part(17).y)
        
        righthead=(landmarks.part(26).x,landmarks.part(26).y)
        centerhead=(landmarks.part(27).x,landmarks.part(27).y)
        headwidth=int(hypot(lefthead[0]-righthead[0],lefthead[1]-righthead[1])*1.7)
        headheight=headwidth
        topleft=(int(centerhead[0]-headwidth/2),int(centerhead[1]-headheight))
        
        heartimage=cv2.resize(headimage,(headwidth,headheight))
        heartimagegray=cv2.cvtColor(heartimage,cv2.COLOR_BGR2GRAY)
        _,headmask=cv2.threshold(heartimagegray,25,255,cv2.THRESH_BINARY_INV)
        headarea=frame[topleft[1]:topleft[1]+headheight,topleft[0]:topleft[0]+headwidth]
        headareahead=cv2.bitwise_and(headarea,headarea,mask=headmask)
        finalhead=cv2.add(headareahead,heartimage)
        frame[topleft[1]:topleft[1]+headheight,topleft[0]:topleft[0]+headwidth]=finalhead
        
        
       
        
        
        
        
        
        
      
        
        
        
    cv2.imshow("frame",frame)
    key=cv2.waitKey(1)
    if key ==27:
        cap.release()
        cv2.destroyAllWindows()
        break

