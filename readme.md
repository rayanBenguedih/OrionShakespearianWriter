# Orion, the Shakespear uprising star

This project contains two main files:

1. `mainTrainer.ipynb`: This Jupyter notebook is used to train the model. It handles the entire training process, from data preprocessing to model saving.

2. `loadSavedModel.py`: This Python script is used to load the trained model. It's designed to be used after the model has been trained and saved by `mainTrainer.ipynb`.

## mainTrainer.ipynb

This Jupyter notebook contains the code to train the model. It includes steps for data preprocessing, model training, and model saving. After running this notebook, a trained model will be saved for future use.

## loadSavedModel.py

This Python script is used to load the trained model saved by `mainTrainer.ipynb`. After loading the model, it can be used for making running the AI

## the Project

The Project will create a folder named "history", in which it will include the dialogue our AI wrote, in tmp files.

Meanwhile, the AI will generate a dialogue in the style of Shakespear, and shall then, make an API call to chatGPT to feed it the dialogue, chatGPT shall then, proceed to create a setting, a scene, a context, for the dialogue given. Then, we proceed to print both output in display for the users.

When you run the Jupyter notebook, you may decide to either train the model, or load it and run it, however, in case you lack Jupyter Notebook, you are given a python script, that will allow you to directly run the model loaded, open a command line (cmd) or a shell, and run the program this way:

python3 loadSavedModel.py


Once this is done, you shall have printed output, you can read. In order to exit the program, press "Q"



This AI was created to address the Natural Language Generation, it is able to be trained on different datasets, has a generic tokenizer for text, and has a few layers. 

We wanted to see if it was possible to make a NLG AI, with as few data, and as few layers given, based on what we talked about, with Red and Green AI. This Generative AI is meant to be green with a minimum running time, and a small dataset.

You will require the latest version of Tensorflow and of Numpy, you will require Python 3.11.5 (Latest python version is known to have errors with Numpy)