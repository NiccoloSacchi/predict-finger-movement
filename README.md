# Predict Finger Movement
In this project we implemented different deep learning models to predict the finger movements from Electroencephalography recordings. Then, we compared them with some common baselines.

Files:
 - `modelWrapper.py`: contains a superclass implementing the general functions adopted by all the proposed models, e.g. fit, cross-validation, score functions.
- `models.py`: contains the implementation of the models.
- `callbacks.py`: callbacks functions that can be passed to the fit() function of the models.
- `test.py`: trains the best model we found and shows both the train and test accuracies.
- `dlc_bci.py`: loads the dataset.
- `helpers.py`: support functions.
- `Report.pdf`: report which explains the problem and our approach to it.

We suggest to read the report for a detalied description.
