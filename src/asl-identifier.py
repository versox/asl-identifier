import os
import re # regular expression

import cv2
import numpy as np

# train data
train_data = []
# train label
y = []

def main():
    loadTrainData()

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
            if i > 100: break
            # load
            img = cv2.imread(os.path.join(train_folder, folder, img_file), cv2.IMREAD_GRAYSCALE)
            # resize to something more managable (50x50)
            resized = cv2.resize(img, (50, 50))
            cv2.imshow('resize', resized)
            cv2.waitKey(0)
            print(resized.shape)
            td.append(np.array(resized))
            labels.append(label)
    
    train_data = np.array(td)
    y = np.array(labels)
    print(train_data.shape)
    print(y.shape)

            
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