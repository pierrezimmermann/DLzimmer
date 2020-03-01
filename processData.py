import numpy as np
import glob
import os
import tensorflow as tf


# loads the images from the ./data folder and returns an array with iamages
# and another array with labels
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
