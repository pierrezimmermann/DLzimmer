import numpy as np
import glob
from PIL import Image
import os
from tensorflow.keras.utils import normalize
import tensorflow as tf


def load_images():
    images = []
    labels = []
    artists = os.listdir('data/best-artworks-of-all-time/images/images')
    for k in range(len(artists)):
        for img in glob.glob('data/best-artworks-of-all-time/images/images/{}/*'.format(artists[k])):
            image = tf.io.read_file(img)
            image = tf.image.decode_jpeg(image)
            image = tf.image.resize(image, [35, 35])
            image = tf.cast(image, tf.float32)
            image = image / 256
            if np.array(image).shape == (35, 35, 3):
                images.append(np.array(image))
                labels.append(k)
                print(img)
    return np.array(images), np.array(labels)


def load_few_images():
    images = []
    labels = []
    artists = os.listdir('data/best-artworks-of-all-time/images/images')
    for k in range(len(artists)):
        i = 0
        for img in glob.glob('data/best-artworks-of-all-time/images/images/{}/*'.format(artists[k])):
            if i < 30:
                image = Image.open(img)
                image = image.resize((35, 35))
                if np.array(image).shape == (35, 35, 3):
                    images.append(np.array(image))
                    labels.append(k)
                    print(img)
                image.close()
                i += 1
    return np.array(images), np.array(labels)
