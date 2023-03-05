import cv2
import face_recognition
import os
import numpy as np
from cv2 import VideoCapture

path = "ImagesAttendance"
images = []
classNames = []
my_list = os.listdir(path)

print(my_list)
for cl in my_list:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodingLIst = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodingLIst.append(encode)
    return encodingLIst
encodingListknow = findEncodings(images)
print("Encoded Completed")

video_capture: VideoCapture = cv2.VideoCapture(0)

cv2.namedWindow("Face Recognition")

while True:
    ret, frame = video_capture.read()
    imgS = cv2.resize(frame ,(0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodingListknow,encodeFace)
        faceDis = face_recognition.face_distance(encodingListknow,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
    cv2.imshow("Window", frame)
    cv2.waitKey(1)

video_capture.release()
cv2.destroyAllWindows()