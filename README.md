# Project Title

Using the GTSRB dataset create a CNN that classify images from the webcam and determine wich of the classes is the corresponded

## Table of Contents
- [Classes](#Classes)
- [Usage](#Usage)
- [Bibliography](#Bibliography)
- [Confusion Matrix](#ConfusionMatrix)

## Classes
There are 43 classes that area classified by the CNN, this could be modify if you need it, rememeber modify the last layer of the cnn to make this possible.

![german_sign](https://github.com/SantiagoLunaMir/GTSRB---Traffic-Sing-Classifier-CNN/assets/111355326/9cd5da23-0940-4344-a77c-d311b0971bf9)

The dataset was originally prupose by the INI Benchmark. Unfortnaly there are not always abailable download but the different people that have use it in kaggle are so much, there are some link that could help.
https://benchmark.ini.rub.de/
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

## Usage

As an Jupiter Notebook with python, there are different use but the principal is the next one:

import cv2
import numpy as np
from keras.models import load_model

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = load_model('traffic_classifier_model.h5')
classNames = ["Speed limit (20km/h)",...
            "End no passing veh > 3.5 tons"
              ]

while True:
    success, orig_img = cap.read()
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (30, 30))  # Resize to 30x30 for model input
    img = np.expand_dims(img, axis=0) 
    class_probs = model.predict(img)[0]
    class_idx = np.argmax(class_probs)
    class_name = classNames[class_idx]
    class_prob = class_probs[class_idx]
    org = (10, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    cv2.putText(orig_img, f"{class_name}: {class_prob:.2f}", org, font, fontScale, color, thickness)
    print(class_name,class_prob)
    cv2.imshow('Webcam', orig_img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

What does it do?
First of all it start to takes video from the webcam, then takes or upload the model of the CNN pretrained (there are two versions, traffic_classifier_model.h5 or traffic_classifier_model_v1.h5, the first one is less complex but with less accuracy an the second one, more complex and more accuracy), create two principal variables from the image, each of the dataframes are resize, expand and prdicted by the model before, then got the most probabilist index, draw the box and classlabel, put the text and show the final resoult.

So now it could classify images:

![image](https://github.com/SantiagoLunaMir/GTSRB---Traffic-Sing-Classifier/assets/111355326/f825b31a-2fd2-487d-8b6e-c709c1470416)

## Confusion Matrix

A good classifier show in their confusion matrix a line that pass from all the classes, that is the next one:
![image](https://github.com/SantiagoLunaMir/GTSRB---Traffic-Sing-Classifier/assets/111355326/9208e347-1e7e-4ac7-90db-4501b27bfa1c)

## Bibliography
As a student a lot of this work or general knowledge is not by me own, so please big thanks to the different tutorial an the principal, the next one:
https://www.analyticsvidhya.com/blog/2021/12/traffic-signs-recognition-using-cnn-and-keras-in-python/#commentModule





