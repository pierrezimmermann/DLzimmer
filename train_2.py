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


images, labels = load_images()
distinct_labels, labels_count = np.unique(labels, return_counts=true)
class_weights = {}
for k in range(50):
    class_weights[k] = 1/labels_count[distinct_labels.indexOf(k)]*len(labels)
p = np.random.permutation(range(len(images)))
images, labels = images[p], labels[p]
NUM_TEST = 800
testX, testY = images[0:NUM_TEST].copy(), labels[0:NUM_TEST].copy()
trainX, trainY = images[:NUM_TEST], labels[:NUM_TEST]

trainX = trainX.astype('float32')
testX = testX.astype('float32')

train_ds = tf.data.Dataset.from_tensor_slices((trainX, trainY))

BATCH_SIZE = 64
learning_rate = 0.001

optimizer = tf.keras.optimizers.Adagrad(learning_rate)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

model = CNN()
model.compile(optimizer=optimizer, loss=loss_object,
              metrics=['accuracy'])

model.fit(trainX, trainY, batch_size=BATCH_SIZE,
          epochs=1000, validation_split=0.1)


gty = model.call(testX)
perf = 100*np.mean(testY == tf.math.argmax(gty, 1))
print("test perf: ", perf)
