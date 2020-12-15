# Setting Up
## Conda Environment
Use `conda env create -f environment.yml` just like the assignment.
Then activate it with `conda activate aslIdentify`

Ensure you have Pytorch, OpenCV, Mediapipe and TensorFlow installed.

## Data
Get the Kaggle datasets for the alphabet from:\
https://www.kaggle.com/grassknoted/asl-alphabet \
and\
https://www.kaggle.com/kuzivakwashe/significant-asl-sign-language-alphabet-dataset \
then put the datasets in training and testing folders, located in a folder 'data' at the root.\
Each letter should have its own subfolder.

![alt](https://github.com/versox/asl-identifier/blob/master/other/folder_structure.png)


## Training the Model
If you wish to train the model on the data, run ASLTrainer.py in a command terminal after ensuring you have correctly downloaded and separated the images from Kaggle.

## Running the Program
** Before running the identifier, if using the zipped model in this GitHub, extract the model ASL_alphabet.zip to the same directory it is currently in ** \
To run the ASL identifier, run asl-identifier.py in a command terminal.
