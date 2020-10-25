import cv2
import numpy as np
import face_recognition

#Load the images
imageelon = face_recognition.load_image_file('elontrain.jpg')
imageelon = cv2.cvtColor(imageelon,cv2.COLOR_BGR2RGB)

imagetest = face_recognition.load_image_file('elontest.jpg')
imagetest = cv2.cvtColor(imagetest,cv2.COLOR_BGR2RGB)

#Finding faces and their encodings
#As we are giving only one image take the 1 element
#faceloc contains 4 values
faceloc = face_recognition.face_locations(imageelon)[0]
#print(faceloc)
encodeElon = face_recognition.face_encodings(imageelon)[0]
cv2.rectangle(imageelon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,0),2)

faceloctest = face_recognition.face_locations(imagetest)[0]
encodeElontest = face_recognition.face_encodings(imagetest)[0]
cv2.rectangle(imagetest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,0),2)

#Comparing the faces and find distance between them using linearSVM
result = face_recognition.compare_faces([encodeElon],encodeElontest)
#Lower the distance better the match
facedist = face_recognition.face_distance([encodeElon],encodeElontest)
#print(result)
#print(facedist)
cv2.putText(imagetest,f'{result}{round(facedist[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)

cv2.imshow('Elon',imageelon)
cv2.imshow('Elon-1',imagetest)
cv2.waitKey(0)

