import cv2 #importing opencv directory
import random
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#load pretrained data on face frontal from opencv

webcam=cv2.VideoCapture(0)

#Iterate forever over frames
while True:
    #Read the current frame
    successfull_frame_read, frame = webcam.read()
    grayscaleimg=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates=trained_face_data.detectMultiScale(grayscaleimg)
    #print(face_coordinates)
    for (x,y,w,h) in face_coordinates:
        if x>50:
            cv2.rectangle(frame, (x,y),(x+w, y+h), (random.randrange(0,255),random.randrange(0,255),random.randrange(0,255)), 4)
    cv2.imshow('Go Get Some Face',frame)
    key = cv2.waitKey(1)
    if key==81 or key==113:
        break
webcam.release()