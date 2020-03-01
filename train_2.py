import os
import os.path
import sys
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from PIL import Image
from model import CNN
from processData import *
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Load images and labels
images, labels = load_images()
p = np.random.permutation(range(len(images)))
images, labels = images[p], labels[p]
NUM_TEST = 800
NUM_VAL = 800


# Separate data into train, val and test datasets
testX, testY = images[0:NUM_TEST].copy(), labels[0:NUM_TEST].copy()
valX, valY = images[NUM_TEST:NUM_VAL +
                    NUM_TEST].copy(), labels[NUM_TEST:NUM_VAL+NUM_TEST].copy()
trainX, trainY = images[NUM_VAL+NUM_TEST:], labels[NUM_VAL+NUM_TEST:]
trainX = trainX.astype('float32')
valX = valX.astype('float32')
testX = testX.astype('float32')


# Compute class weights to deal with unbalanced data
class_weight = class_weight.compute_class_weight(
    'balanced', np.unique(trainY), trainY)


BATCH_SIZE = 64
learning_rate = 0.005
nb_epochs = 400
dropout_rate = 0.7

optimizer = tf.keras.optimizers.Adamax(learning_rate)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()


model = CNN(dropout_rate=dropout_rate)
model.compile(optimizer=optimizer, loss=loss_object,
              metrics=['sparse_categorical_accuracy'])


# Use early stopping to avoid overfitting
callback = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10)


# Train the model on the original dataset
def train():
    model.fit(trainX, trainY, batch_size=BATCH_SIZE,
              epochs=400, validation_data=(valX, valY), class_weight=class_weight, callbacks=[callback])


# Train on augmented data
def train_with_data_augmentation():
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    for e in range(50):
        print('Epochs {}'.format(e))
        batches = 0
        for x_batch, y_batch in datagen.flow(trainX, trainY, batch_size=BATCH_SIZE):
            model.fit(x_batch, y_batch, class_weight=class_weight, verbose=0)
            batches += 1
            if batches >= len(trainX) / BATCH_SIZE:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break
    gty = model.call(valX)
    perf = 100*np.mean(testY == tf.math.argmax(gty, 1))
    print('Val perf: {}'.format(perf))


# Evaluate perf on the test dataset
train()


def test():
    gty = model.call(testX)
    perf = 100*np.mean(testY == tf.math.argmax(gty, 1))
    print('Test perf : {}'.format(perf))
