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
nosepic=cv2.imread("C:\\Users\\SHAIK.RAHEEM\\Pictures\\mew (2).png")
hornimage=cv2.imread("C:\\Users\\SHAIK.RAHEEM\\Pictures\\mew (3).png")
righthornimage=cv2.imread("C:\\Users\\SHAIK.RAHEEM\\Pictures\\mew (4).png")


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
        firstpoint=(landmarks.part(17).x,landmarks.part(17).y)
        
        lastpoint=(landmarks.part(21).x,landmarks.part(21).y)
        centerpoint=(landmarks.part(19).x,landmarks.part(19).y)
        
        rightfirstpoint=(landmarks.part(22).x,landmarks.part(22).y)
        rightlastpoint=(landmarks.part(26).x,landmarks.part(26).y)
        rightcenterpoint=(landmarks.part(24).x,landmarks.part(24).y)
        hornwidth=int((lastpoint[0]-firstpoint[0])*1.7)
        hornheight=hornwidth
        
        
        
        topleft=(int(centerpoint[0]-hornwidth/2),int(centerpoint[1]-hornheight))
        topright=(int(rightcenterpoint[0]-hornwidth/2),int(rightcenterpoint[1]-hornheight))
        
        
        heartimage=cv2.resize(hornimage,(hornwidth,hornheight))
        heartimagegray=cv2.cvtColor(heartimage,cv2.COLOR_BGR2GRAY)
       
       
        
        _,headmask=cv2.threshold(heartimagegray,25,255,cv2.THRESH_BINARY_INV)
        headarea=frame[topleft[1]:topleft[1]+hornheight,topleft[0]:topleft[0]+hornwidth]
        headareahead=cv2.bitwise_and(headarea,headarea,mask=headmask)
        finalhead=cv2.add(headareahead,heartimage)
        frame[topleft[1]:topleft[1]+hornheight,topleft[0]:topleft[0]+hornwidth]=finalhead
        rightheartimage=cv2.resize(righthornimage,(hornwidth,hornheight))
        rightheartimagegray=cv2.cvtColor(rightheartimage,cv2.COLOR_BGR2GRAY)
        _,headmask=cv2.threshold(rightheartimagegray,25,255,cv2.THRESH_BINARY_INV)
        rightheadarea=frame[topright[1]:topright[1]+hornheight,topright[0]:topright[0]+hornwidth]
        rightheadareahead=cv2.bitwise_and(rightheadarea,rightheadarea,mask=headmask)
        rightfinalhead=cv2.add(rightheadareahead,rightheartimage)
        frame[topright[1]:topright[1]+hornheight,topright[0]:topright[0]+hornwidth]=rightfinalhead
       
        
        
        
        
        
        
      
        
        
        
    cv2.imshow("frame",frame)
    key=cv2.waitKey(1)
    if key ==27:
        cap.release()
        cv2.destroyAllWindows()
        break
        
         

