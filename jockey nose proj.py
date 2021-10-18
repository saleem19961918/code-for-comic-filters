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
nosepic=cv2.imread("C:\\Users\\SHAIK.RAHEEM\\Pictures\\WhatsApp Image 2021-06-14 at 9.45.49 PM (3).png")


# In[3]:


while True:
    ret,frame = cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    for face  in faces:
        landmarks=predictor(gray,face)
        topnose=(landmarks.part(29).x,landmarks.part(29).y)
        leftnose=(landmarks.part(31).x,landmarks.part(31).y)
        rightnose=(landmarks.part(35).x,landmarks.part(35).y)
        centernose=(landmarks.part(30).x,landmarks.part(30).y)
        
        nosewidth=int(hypot(leftnose[0]-rightnose[0],leftnose[1]-rightnose[1])*1.7)
        noseheight=nosewidth
        topleft=(int(centernose[0]-nosewidth/2),int(centernose[1]-noseheight/2))
        bottomright=(int(centernose[0]+nosewidth/2),int(centernose[0]+noseheight/2))
        #cv2.rectangle(frame,topleft,bottomright,(0,255,0),6)
        nosearea=frame[topleft[1]:topleft[1]+noseheight,topleft[0]:topleft[0]+nosewidth]
        pigimage=cv2.resize(nosepic,(nosewidth,noseheight))
        pigimagegray=cv2.cvtColor(pigimage,cv2.COLOR_BGR2GRAY)
        _,nosemask=cv2.threshold(pigimagegray,25,255,cv2.THRESH_BINARY_INV)
        noseareanose=cv2.bitwise_and(nosearea,nosearea,mask=nosemask)
        finalnose=cv2.add(noseareanose,pigimage)
        frame[topleft[1]:topleft[1]+noseheight,topleft[0]:topleft[0]+nosewidth]=finalnose
        
        
        
      
        
        
        
    cv2.imshow("frame",frame)
    key=cv2.waitKey(1)
    if key ==27:
        cap.release()
        cv2.destroyAllWindows()
        break

