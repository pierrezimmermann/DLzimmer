import numpy as np
import glob
from PIL import Image
import os


def load_images():
    images = []
    labels = []
    artists = os.listdir('data/best-artworks-of-all-time/images/images')
    for k in range(len(artists)):
        for img in glob.glob('data/best-artworks-of-all-time/images/images/{}/*'.format(artists[k])):
            image = Image.open(img)
            image = image.resize((30, 30))
            if np.array(image).shape == (30, 30, 3):
                images.append(np.array(image))
                print(img)
            image.close()
            labels.append(k)
    return np.array(images), np.array(labels)


def load_resized_images():
    images = []
    labels = []
    for img in glob.glob('data/best-artworks-of-all-time/resized/resized/*'):
        image = Image.open(img)
        image = image.resize(30, 30)
        artist_name = '_'.join(os.path.basename(img).split('_')[0:2])
        images.append(np.array(image))
        image.close()
        labels.append(artist_name)
        print(img)
    return (np.array(images), np.array(labels))


def load_first_image():
    images = []
    labels = []
    img = glob.glob('data/best-artworks-of-all-time/resized/resized/*')[0]
    image = Image.open(img)
    image = image.resize((30, 30))
    artist_name = '_'.join(os.path.basename(img).split('_')[0:2])
    images.append(np.array(image))
    images.append(np.array(image))

    image.close()
    labels.append(artist_name)
    labels.append(artist_name)
    print(images[0].shape)

    return np.array(images), np.array(labels)


def load_first_image2():
    img = glob.glob(
        'data/best-artworks-of-all-time/images/images/Alfred_Sisley/*')[0]
    image = Image.open(img)
    image = image.resize((30, 30))
    print(np.array(image).shape)
    image.close()
