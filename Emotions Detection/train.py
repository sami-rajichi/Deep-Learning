import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from archi import *

classes = ("angry", "disgust", "scared", "happy", "sad", "surprised", "neutral")

def load_fer2013(path):
    data = pd.read_csv(path)
    pixels = data['pixels'].tolist()
    faces = []
    for pix in pixels :
        face = [int(pixel) for pixel in pix.split(' ')]
        face = np.asarray(face).reshape(48,48)
        face = cv2.resize(face.astype('uint8'),(48,48))
        faces.append(face)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).values
    return faces, emotions

def preprocess(x):
    x = x.astype('float32')
    x = x/255.0
    return x

model = mini_XCEPTION((48,48,1), 7)
model.compile(optimizer="adam", loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
faces, emotions = load_fer2013("fer2013.csv")
faces = preprocess(faces)

xtrain, xtest, ytrain, ytest = train_test_split(faces, emotions, test_size=0.2, shuffle=True)
     
history = model.fit(xtrain, ytrain, batch_size=32, epochs=100, verbose=1, validation_data=(xtest,ytest))

model.save("model.hdf5")

#evaluation du model
preds = model.predict(xtest, batch_size=32)
print(classification_report(ytest.argmax(axis=1), preds.argmax(axis=1), target_names=classes))
plt.figure()
plt.plot(np.arange(0,100),history.history['loss'], label = "train-loss")
plt.plot(np.arange(0,100),history.history['accuracy'], label = "train_accuracy")
plt.plot(np.arange(0,100),history.history['val_loss'], label = "val_loss")
plt.plot(np.arange(0,100),history.history['val_accuracy'], label = "val_accuracy")

plt.title("result of training")
plt.legend()
plt.savefig("result.png")
plt.show()