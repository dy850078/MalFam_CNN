# -*- coding: utf-8 -*-

# TensorFlow and tf.keras
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


def my_load_data(p):
    with open(p, 'rb') as fd:
        tu = pickle.load(fd)
    return tu[0], tu[1]


train_images, train_labels = my_load_data("/dataset/pefiles.pickle")

# our data includes 11 different malware families
class_names = ['Urausy.C', 'Elkern.B', 'Fareit', 'Hotbar', 'Gepys.A', 'Bulta!rfn', 'Zbot',
               'GameVance', 'Zegost.B', 'Bifrose.AE', 'Zbot!GO']
len_classes = len(class_names)

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
plt.figure(figsize=(10, 10))
for i in range(36):
    plt.subplot(6, 6, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Data preprocessing
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)


def preprocess_images(imgs):
    print(imgs.shape)
    sample_img = imgs if len(imgs.shape) == 2 else imgs[0]
    # make sure images are 28x28 and single-channel (grayscale)
    assert sample_img.shape in [(28, 28, 1), (28, 28)], sample_img.shape
    return imgs / 255.0


train_images = preprocess_images(train_images)

# Build the model
model = keras.Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))  # 32 convolution filters used each of size 3x3
model.add(Conv2D(64, (3, 3), activation='relu'))  # 64 convolution filters used each of size 3x3
model.add(MaxPooling2D(pool_size=(2, 2)))  # choose the best features via pooling
model.add(Dropout(0.25))  # randomly turn neurons on and off to improve convergence
model.add(Flatten())  # flatten since too many dimensions, we only want a classification output
model.add(Dense(128, activation='relu'))  # fully connected to get all relevant data
model.add(Dropout(0.5))  # one more dropout
model.add(Dense(len_classes, activation='softmax'))  # output a softmax to squash the matrix into output probabilities

model.compile(optimizer='rmsprop',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model.fit(train_images, train_labels, validation_split=0.7, epochs=10, callbacks=[callback])
