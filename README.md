# GTSRB Classifier

Using the GTSRB dataset, create a CNN that classifies images from the webcam and determines which of the classes is the corresponding one.
## Table of Contents
- [Classes](#Classes)
- [Usage](#Usage)
- [Bibliography](#Bibliography)
- [Confusion Matrix](#ConfusionMatrix)

## Classes
There are 43 classes that are classified by the CNN. This can be modified if needed. Remember to modify the last layer of the CNN to make this possible.

[german_sign](https://github.com/SantiagoLunaMir/GTSRB---Traffic-Sing-Classifier-CNN/assets/111355326/9cd5da23-0940-4344-a77c-d311b0971bf9)

The dataset was originally proposed by the INI Benchmark. Unfortunately, it is not always available for download, but many people have used it on Kaggle. Here are some links that might help:

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
First of all, it starts to take video from the webcam, then takes or uploads the pre-trained CNN model (there are two versions, traffic_classifier_model.h5 or traffic_classifier_model_v1.h5; the first one is less complex but with less accuracy and the second one, more complex and more accurate). It creates two main variables from the image. Each of the dataframes is resized, expanded, and predicted by the model. Then it gets the most probable index, draws the box and class label, puts the text, and shows the final result.

So now it can classify images:

![image](https://github.com/SantiagoLunaMir/GTSRB---Traffic-Sing-Classifier/assets/111355326/f825b31a-2fd2-487d-8b6e-c709c1470416)

## Confusion Matrix

A good classifier shows in its confusion matrix a line that passes through all the classes. That is the following one:

![image](https://github.com/SantiagoLunaMir/GTSRB---Traffic-Sing-Classifier/assets/111355326/9208e347-1e7e-4ac7-90db-4501b27bfa1c)

## Bibliography
As a student, a lot of this work or general knowledge is not solely mine, so please give big thanks to the different tutorials and the main one:

https://www.analyticsvidhya.com/blog/2021/12/traffic-signs-recognition-using-cnn-and-keras-in-python/#commentModule





