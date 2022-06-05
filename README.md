# Emotion Detection using CNN

# Description
It is a deep learning model which detects the emotions through the webcam. It is having 7 different categories emotions :- Angry, Happy, Disgust, Fear, Neutral, Sad, Surprise. Model built using CNN.

# Dataset
I used the FER-2013 dataset to train the model, which is publically available on kaggle.

Dataset download link : https://www.kaggle.com/datasets/msambare/fer2013

After downloading the dataset, keep this data inside the ./data directory 

# Requirements 

- pip install numpy

- pip install opencv-python

- pip install keras

- pip3 install --upgrade tensorflow

- pip install pillow

# To train the model

command : python TrainEmotion.py
 
After training the model we'll get weight file inside the model folder which will be used at the testing time. Here, I provided my model's weight file.

# Testing the model

command : python TestEmotionDetector.py

This the way model work..

Thank you.. 

If any suggestion or query, you can reach out to me via mail id : shailjasengar28@gmail.com

Happy Learning :)
