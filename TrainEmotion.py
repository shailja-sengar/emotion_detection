# Importing libraries

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.optimizers import adam_v2
from keras.preprocessing.image import ImageDataGenerator

#applying Image data generator with rescaling, The ImageDataGenerator class can be used to rescale pixel values from the range of 0-255 to the range 0-1 preferred for neural network models.
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

#Preprocess all train images
train_generator = train_data_gen.flow_from_directory(
                 'data/train',
                  target_size=(48, 48),#bcoz, dataset has 48x48 greyscale images
                 batch_size= 64,
                 color_mode= "grayscale",
                class_mode= "categorical")

#Preprocess all the test images
validation_generator = validation_data_gen.flow_from_directory(
                        'data/test',
                         target_size= (48,48),
                         batch_size= 64,
                         color_mode= "grayscale",
                         class_mode= "categorical")

#Create model structure
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPool2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPool2D(pool_size=(2,2)))
emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPool2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation="relu"))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
opt = adam_v2.Adam(learning_rate=0.0001, decay=0.0001/5)
emotion_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#train the neural_network/model
emotion_model_info = emotion_model.fit_generator(
        train_generator,
        steps_per_epoch= 28709 // 64,
        epochs= 50,
        validation_data= validation_generator,
        validation_steps= 7178 //64
)

#save the model structure in json file
model_json = emotion_model.to_json()
with open ("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

#save trained model weight in .h5 file
emotion_model.save_weights('emotion_model.h5')