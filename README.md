# Setting Up
## Conda Environment
Use `conda env create -f environment.yml` just like the assignment.
Then activate it with `conda activate aslIdentify`

Ensure you have Pytorch, OpenCV, Mediapipe and TensorFlow installed.

## Data
Get the Kaggle datasets for the alphabet
https://www.kaggle.com/grassknoted/asl-alphabet
and
https://www.kaggle.com/kuzivakwashe/significant-asl-sign-language-alphabet-dataset
and put it in a folder called 'data' at the root.
Each letter should have its own subfolder.

![alt](https://github.com/versox/asl-identifier/blob/master/other/folder_structure.png)


## Training the Model
If you wish to train the model on the data, run ASLTrainer.py in a command terminal after ensuring you have correctly downloaded and separated the images from Kaggle.

## Running the Program
To run the ASL identifier, run asl-identifier.py in a command terminal.
