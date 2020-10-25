import cv2
import numpy as np
import face_recognition
import os

'''
Given a folder where images are stored make a list and encode all the images and then try to find them in webcam
When a new image is added then its name will be stored in list Name
'''

path = 'trainimages'
images = []
className = []
Names = os.listdir(path)
print(Names)
for cl in Names:
    currImg = cv2.imread(f'{path}/{cl}')
    images.append(currImg)
    #For classname, .jgp is not required hence splittext and take only the first part
    className.append(os.path.splitext(cl)[0])
#print(className)

'''
Now start with the encoding process
Create a function and images should be input
Create a local encodeList and iterate through each image
Convert the color and store encodings
'''
def Encoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return(encodeList)

encodeListknown = Encoding(images)
print('Encoding complete')

#Now take the image from webcam
cap =cv2.VideoCapture(0)
while True:
    ret,img = cap.read()
    #Reduce the size to make the operation easier
    #img = cv2.resize(img,(0,0),None,0.25,0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    '''
    Webcam can have multiple locations
    Hence call the function location of faces
    '''
    facesCurFrame = face_recognition.face_locations(img)
    encodeCurFrame = face_recognition.face_encodings(img,facesCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListknown,encodeFace)
        facedist = face_recognition.face_distance(encodeListknown,encodeFace)
        #print(facedist)
        '''
        FaceDist will be a list having 4 values 
        The one with the least value is the correct image
        '''
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        matchIndex = np.argmin(facedist)
        if matches[matchIndex]:
            name = className[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),1)
            cv2.rectangle(img,(x1,y2-25),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1,y2-6),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1,)


    cv2.imshow("Final",img)
    cv2.waitKey(1)






