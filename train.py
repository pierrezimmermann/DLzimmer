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
p = np.random.permutation(range(len(images)))
images, labels = images[p], labels[p]
NUM_VAL = 800
NUM_TEST = 800
testX, testY = images[0:NUM_TEST].copy(), labels[0:NUM_TEST].copy()
valX, valY = images[NUM_TEST:NUM_VAL +
                    NUM_TEST].copy(), labels[NUM_TEST:NUM_VAL+NUM_TEST].copy()
trainX, trainY = images[NUM_VAL+NUM_TEST:], labels[NUM_VAL+NUM_TEST:]

trainX = trainX.astype('float32')
valX = valX.astype('float32')
testX = testX.astype('float32')

train_ds = tf.data.Dataset.from_tensor_slices((trainX, trainY))
val_ds = tf.data.Dataset.from_tensor_slices((valX, valY))

BATCH_SIZE = 64
batched_train_ds = train_ds.batch(BATCH_SIZE)
batched_val_ds = val_ds.batch(BATCH_SIZE)

model = CNN()
model.build(input_shape=(BATCH_SIZE, 35, 35, 3))


learning_rate = 0.002
optimizer = tf.keras.optimizers.Adamax(learning_rate)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')
val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')


@tf.function
def train_step(samples, labels):
    with tf.GradientTape() as tape:
        predictions = model(samples)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(samples, labels):
    predictions = model.call(samples)
    t_loss = loss_object(labels, predictions)
    val_loss(t_loss)
    val_accuracy(labels, predictions)


EPOCHS = 400
for epoch in range(EPOCHS):

    for images, labels in batched_train_ds:
        train_step(images, labels)

    for val_images, val_labels in batched_val_ds:
        test_step(val_images, val_labels)

    template = 'Epoch {}, Loss: {:1.4}, Accuracy: {:2.2%}, Val Loss: {:1.4}, Val Accuracy: {:2.2%}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result(),
                          val_loss.result(),
                          val_accuracy.result()))

    # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    val_loss.reset_states()
    val_accuracy.reset_states()

gty = model.call(testX)
perf = 100*np.mean(testY == tf.math.argmax(gty, 1))
print("test perf: ", perf)
