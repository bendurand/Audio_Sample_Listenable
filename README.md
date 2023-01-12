# Audio_Sample_Listenable

***A convolutional neural network used to predict and sort .wav sound files according to their listenability.***

Authors : CHALOUPE AnhKim, DELAHAYE Matthieu, DURAND Benjamin, FAUSSURIER Yann, FONTEYNE Nathan, FORTE Julia, JUSTE Lucas.

EPF egineering school.

## Install dependencies with

```
$pip install -r requirements.txt
```

# Predict

In the folder `./prediction_streamlit` : 

- Put the files you want to test the listenability in the "Sound_files" file

- Run with python the file Prediction.py

You can find further informations in the report `Guide de l'application prediction.pdf` present in `./prediction_streamlit` folder.

To update the prediction model please refer to the 'Train' part.

# Notebooks

In this folder you can find the differents models and tracks we followed in order to complete the objective. This is basically our proof of concept work.

You can find other good methods we did not have the time to implement (such as transfert learning) and also models we did not select.

# Preprocessing

Preprocessing folder contains the script `utils.py` that create inputs for the neural network from excel and audio files.

**Preprocessing will be applied __automatically__ before the training part. You don't have to run this file.**

Unit tests can be implemented in `test_utils.py` in `./tests` folder

Run unit tests with this bash command at the root of the project : 
```
$python3 -m unittest discover --top-level-directory=. --start-directory=./preprocessing/tests/
```

Data required for tests can be found in `./test_data` 

- `./audio` for .wav files 
- `./marks` for .xlsx file with marks obtained by those audio files

# Train

Train folder contains the script `train.py` that applies preprocessing, instanciates a Convolutional Neural Network, trains it, and save it for prediction module.

You can change hyperparameters or neural layers in this script.

Training data are composed of :
- `.wav` files in the `./training_data/audio` folder
- an `.xlsx` file that contains all marks given to those audios

Excel file must at least contain 3 columns :
- 'Sample 00XX' : name of the audio file -> str
- 'Médiane' : median mark given of each audio -> int

*00XX representing the name of the sample, one sample by sheet.*


Run the training of a new model simply by running this python file. The new model model will be saved at `./prediction_streamlit/models`.

Unit tests can be implemented in `test_train.py` in `/tests` folder.

Run unit tests with this bash command at the root of the project :
```
$python3 -m unittest discover --top-level-directory=. --start-directory=./train/tests/
```

**Disclaimer** : If you run the model with Visual Studio code : chances are that the program has dificulties to import 'preprocessing' module.
You can bypass this error by running an unittest directly on the training data.
Go in test_train.py for further informations.
