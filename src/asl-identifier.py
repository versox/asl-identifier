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

    y_binary = keras.utils.to_categorical(label)

    # print(train_data.shape)
    # print(label.shape)

    model = keras.Sequential()
    model.add(
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(50,50,1))
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
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        x=train_data,
        y=y_binary
    )
    
    test_data, test_labels = loadTestData()
    test_labels_binary = keras.utils.to_categorical(test_labels)
    preds = model.predict(test_data)
    print(preds)


def loadTestData():
    test_folder = "../data/asl_alphabet_test/"

    test_data = []
    labels = []

    for img_file in os.listdir(test_folder):
        img = cv2.imread(os.path.join(test_folder, img_file), cv2.IMREAD_GRAYSCALE)
        resized = np.reshape(cv2.resize(img, (50, 50)), (50,50,1))
        test_data.append(resized)
        label = figureOutLabel(re.findall(r"[^_]*_",img_file)[0][:-1])
        labels.append(label)

    return np.array(test_data), np.array(labels)


def loadTrainData():
    train_folder = '../data/asl_alphabet_train/'
    # train_folder = '../data/small/'

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
            resized = np.reshape(cv2.resize(img, (50, 50)), (50,50,1))
            td.append(resized)
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
