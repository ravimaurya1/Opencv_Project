# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 22:57:57 2018

@author: ravim
"""

import cv2            #Import opencv library    
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  #Load the cascade for face.
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')                   #Load the cascade for eye.

cap=cv2.VideoCapture(0)                                                      #For input from webcam

while True:
    ret,img=cap.read()              #Reading frame from webcam
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    #Convert image to gray for analysis of face dtection
    
    
    faces = face_cascade.detectMultiScale(gray,1.3,5)   #calling detecMultiScale method of face_cascade on gray image to detect face
    
    for (x,y,w,h) in faces:   #faces contains the cordinate of detected faces
        cv2.rectangle(img, (x,y), (x+w,y+h),(255,0,0),2) #to draw rectangle around face
        roi_gray = gray[y:y+h, x:x+w]      #Region where face is found in gray
        roi_color = img[y:y+h, x:x+w]      #Region where face is found in color
        eyes = eye_cascade.detectMultiScale(roi_gray)  #To detect multilple eye.
        
        for (ex,ey,ew,eh) in eyes:  #for each detected eye.
            cv2.rectangle(roi_color,(ex,ey), (ex+ew,ey+eh), (0,255,0), 2)  #Draw rectangle around detected area.
    cv2.imshow("gray",img)      #Show orginal capture frame.
    if cv2.waitKey(1) ==27:      #Break Loop when Esc key is pressed.
        break
cv2.destroyAllWindows()         #To destroy all open windows.
cap.release()                   #To release webcam.