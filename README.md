# Predict Finger Movement
In this project we built some different models to predict the finger movements from Electroencephalography recordings. Then, we compared them with some common baselines.

Files:
  - modelWrapper.py: contains the main structure adopted by all the proposed models (fit, cross-validation, score functions etc.)
  - models.py: contains the implementation of the models (that are based on what is specified in modelWrapper.py)
  - callbacks.py: callbacks functions that can be passed to the fit function of the models
  - test.py: shows the train and test accuracy for the best model
  - dlc_bci.py: loads the dataset
  - helpers.py: support functions
  - report.pdf: contains the report

More details can be found in the report.
