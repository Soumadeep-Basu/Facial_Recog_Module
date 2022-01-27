import cv2
import numpy as np 
import face_recognition

bTrainimg = face_recognition.load_image_file('Images_TestTrain/bTrain.jpg')

bTrainimg = cv2.cvtColor(bTrainimg, cv2.COLOR_BGR2RGB)

sTrainimg = face_recognition.load_image_file('Images_TestTrain/sTrain.jpg')

sTrainimg = cv2.cvtColor(sTrainimg, cv2.COLOR_BGR2RGB)

sTrainimg = cv2.rotate(sTrainimg, cv2.ROTATE_90_COUNTERCLOCKWISE)
bTrainimg = cv2.rotate(bTrainimg, cv2.ROTATE_90_CLOCKWISE)





sfacelocation = face_recognition.face_locations(sTrainimg)[0]
bfacelocation = face_recognition.face_locations(bTrainimg)[0]


sencode = face_recognition.face_encodings(sTrainimg)[0]
bencode = face_recognition.face_encodings(bTrainimg)[0]

cv2.rectangle(sTrainimg, (sfacelocation[3], sfacelocation[0]), (sfacelocation[1], sfacelocation[2]), (255,0,255), 2)
cv2.rectangle(bTrainimg, (bfacelocation[3], bfacelocation[0]), (bfacelocation[1], bfacelocation[2]), (255,0,255), 2)





cv2.imshow('This Image is for Training the Data for Som', sTrainimg)
cv2.imshow('This Image is for Training the Data for Brijesh', bTrainimg)
cv2.waitKey(0)
