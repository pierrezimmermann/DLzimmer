import numpy as np
import glob
from PIL import Image
import os


def load_resized_images():
    images = []
    labels = []
    for img in glob.glob('data/best-artworks-of-all-time/resized/resized/*'):
        image = Image.open(img)
        artist_name = '_'.join(os.path.basename(img).split('_')[0:2])
        images.append(np.array(image))
        image.close()
        labels.append(artist_name)
        print(img)
    return (np.array(images), np.array(labels))
