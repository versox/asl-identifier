import os
import re  # regular expression

import cv2
import numpy as np

import tensorflow as tf
from tensorflow import keras


def main():
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    train_data, label = loadTrainData()

    print(train_data[0].shape)

    model = keras.Sequential()
    model.add(
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=train_data[0].shape)
    )
    model.add(
        keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
    )
    model.add(
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    )
    model.add(
        keras.layers.Flatten(),
    )
    model.add(
        keras.layers.Dense(units=29, activation='softmax')
    )

    print(model.summary())


def loadTrainData():
    # train_folder = '../data/asl_alphabet_train/'
    train_folder = '../data/small/'

    td = []
    labels = []

    # go through training folders (A - Z)
    for folder in os.listdir(train_folder):
        label = figureOutLabel(folder)
        # go through images for training label
        for i, img_file in enumerate(os.listdir(train_folder + folder)):
            # only do 1st 100 examples
            if __debug__:
                if i > 100:
                    break
            # load
            img = cv2.imread(os.path.join(train_folder, folder,
                                          img_file), cv2.IMREAD_GRAYSCALE)
            # resize to something more managable (50x50)
            resized = cv2.resize(img, (50, 50))
            td.append(np.array(resized))
            labels.append(label)

    return np.array(td), np.array(labels)


def figureOutLabel(name):
    # A - Z will be 0 - 25
    # handle others first
    if name == 'del':
        return 26

    if name == 'space':
        return 27

    if name == 'nothing':
        return 28

    # handle A - Z
    if re.match(r"[A-Z]", name):
        return ord(name[0]) - 65

    # not defined
    # Todo: implement error?


# main() is called if this file is being run directly
# (__name__ contains the name of the code using this file
#       so importing vs running directly is different)
if __name__ == '__main__':
    main()
