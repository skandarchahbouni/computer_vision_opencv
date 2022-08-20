import os 

import cv2 

import numpy as np


# create the list of 
DIR = r"C:\Users\Skander\Downloads\opencv-course-master\opencv-course-master\Section #3 - Faces\self_try\Faces\train"


# This two are used to train the model

features = []

labels = []


people = []

def create_train():

    global features, labels
    for subdir in os.listdir(DIR):
        people.append(subdir)
        

    haar_cascade = cv2.CascadeClassifier('haar_face.xml')



    for person in people:

        path = os.path.join(DIR, person)

        # iterate over the images inside the subdir 

        label = people.index(person)

        for img in os.listdir(path):

            img_path = os.path.join(path, img)

            # Reading the image 

            img = cv2.imread(img_path)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Grabbing the face to append it to the features list 

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:

                roi = gray[y:y+h,x:x+w]
                features.append(roi)
                labels.append(label)
    

    features = np.array(features, dtype='object')

    labels = np.array(labels)


create_train()

print("CREATING FEATURES AND LABELS DONE SUCCESFULLY!")


# we have the features and lables now it's time to train our model

face_recognizer = cv2.face.LBPHFaceRecognizer_create()


# Train the Recognizer on the features list and the labels list

face_recognizer.train(features,labels)


# To not redo the train every time we want to test an image 

face_recognizer.save('face_trained.yml')