import os
from collections import namedtuple

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import cv2

from src import config


class VGGDerivative:

    def __init__(self):

        self.raw, self.raw_array, self.split_data = None, None, None

    def load_data(self):

        data, targets, filenames = list(), list(), list()
        Raw = namedtuple('Raw', 'data targets filenames')

        print("[INFO] loading dataset...")
        rows = open(config.ANNOTS_PATH).read().strip().split("\n")

        for row in rows:
            # break the row into the filename and bounding box coordinates
            row = row.split(",")
            (filename, start_x, start_y, end_x, end_y) = row

            # derive the path to the input image, load the image (in OpenCV format), and grab its dimensions
            image_path = os.path.sep.join([config.IMAGES_PATH, filename])
            image = cv2.imread(image_path)

            h, w, _ = image.shape

            # scale the bounding box coordinates relative to the spatial dimensions of the input image
            start_x = float(start_x) / w
            start_y = float(start_y) / h
            end_x = float(end_x) / w
            end_y = float(end_y) / h

            # load the image and preprocess it
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image)

            # update our list of data, targets, and filenames
            data.append(image)
            targets.append((start_x, start_y, end_x, end_y))
            filenames.append(filename)

        self.raw = Raw(data, targets, filenames)

    def convert_data(self):

        RawArray = namedtuple('RawArray', 'data_array targets_array')

        # convert the data and targets to NumPy arrays, scaling the input pixel intensities from the range
        # [0, 255] to [0, 1]
        data_array = np.array(self.raw.data, dtype="float32") / 255.0
        targets_array = np.array(self.raw.targets, dtype="float32")

        self.raw_array = RawArray(data_array, targets_array)

    def split_data_train_test(self, test_size=0.10):

        split_data = train_test_split(self.raw_array.data_array, self.raw_array.targets_array, self.raw.filenames, test_size=test_size, random_state=42)

        SplitData = namedtuple('SplitData', 'images targets filenames')
        Images = namedtuple('Images', 'train test')
        Targets = namedtuple('Targets', 'train test')
        Filenames = namedtuple('Filenames', 'train test')

        images = Images(split_data[0], split_data[1])
        targets = Targets(split_data[2], split_data[3])
        filenames = Filenames(split_data[4], split_data[5])

        self.split_data = SplitData(images, targets, filenames)

    def write_test_images(self):

        # write the testing filenames to disk so that we can use then when evaluating/testing our bounding box regressor
        print("[INFO] saving testing filenames...")
        f = open(config.TEST_FILENAMES, "w")
        f.write("\n".join(self.split_data.filenames.test))
        f.close()

    def modify_vgg(self):

        # load the VGG16 network, ensuring the head FC layers are left off
        vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

        # freeze all VGG layers so they will *not* be updated during the training process
        vgg.trainable = False
