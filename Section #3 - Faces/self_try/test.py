import os 
import cv2

DIR = r"C:\Users\Skander\Downloads\opencv-course-master\opencv-course-master\Section #3 - Faces\self_try\Faces\val"

people = []

for subdir in os.listdir(DIR):
    people.append(subdir)

# Testing our trainde model
# we ave to pass a gray roi bcz hat's how we trained our model

img = cv2.imread("Faces/val/ben_afflek/2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Grabbing the region of interest
haar_cascade = cv2.CascadeClassifier('haar_face.xml')


faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
(x,y,w,h) = faces_rect[0]
roi = gray[y:y+h,x:x+w]

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

label, confidence = face_recognizer.predict(roi)
print(label)
print(confidence)
cv2.putText(img, str(people[label]+f"{confidence}"), (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,0), thickness=1)
cv2.imshow("Person", img)

cv2.waitKey(0)
