import cv2 #importing opencv directory
import random
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#load pretrained data on face frontal from opencv

img = cv2.imread('12345.png')
grayscaleimg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinates=trained_face_data.detectMultiScale(grayscaleimg)


for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x,y),(x+w, y+h), (random.randrange(0,255),random.randrange(0,255),random.randrange(0,255)), 4)
#print(face_coordinates)



cv2.imshow('Go Get Some Face',img)
#cv2.imshow('Go Get Some Face',grayscaleimg)
cv2.waitKey() #use to hold the output
