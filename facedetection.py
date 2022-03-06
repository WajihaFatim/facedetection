import cv2 as cv
import numpy as np

img=cv.imread('media/gr.jpeg')
cv.imshow('Person',img)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("gray person",gray)

haar_cascade=cv.CascadeClassifier('har_frontface.xml')


faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=2)

print(len(faces_rect))

for (x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,0,0),thickness=2)

cv.imshow("detect faces",img)

cv.waitKey(0)