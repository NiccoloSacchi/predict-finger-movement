import dlc_bci as bci

import os

import torch
import numpy as np

from models import *
from callbacks import keep_best_model, store_best_model

from types import SimpleNamespace 

import torch 
from torch import optim
from torch import nn

one_khz=False

train = SimpleNamespace()
train.X, train.y = bci.load(root='./data_bci', one_khz=one_khz)
#print(str(type(train.X)), train.X.size())
#print(str(type(train.y)), train.y.size())

test = SimpleNamespace()
test.X, test.y = bci.load(root='./data_bci', train=False, one_khz=one_khz)
#print(str(type(test.X)), test.X.size())
#print(str(type(test.y)), test.y.size())



# CNN 2D
X_tr, y_tr = CNN2D_MaxPool.prepare_data(train)
X_te, y_te = CNN2D_MaxPool.prepare_data(test)

params = {
    'activation': nn.ReLU,
    'dropout': 0.0,
    'nb_hidden': 40,
    'nb_layers': 3,
    'optimizer': torch.optim.Adadelta,
    'weight_decay': 0
}
model = CNN2D_MaxPool(**params)

model.fit(X_tr, y_tr, X_te, y_te, epochs=30, callbacks=[keep_best_model], verbose=False)
print("CNN 2D - Test score:", model.score(X_te, y_te))


# CNN 1D - MAX POOL
X_tr, y_tr = CNN_1D_MaxPool.prepare_data(train)
X_te, y_te = CNN_1D_MaxPool.prepare_data(test)

params = {
    'activation': nn.ELU,
    'dropout': 0.09999999999999999,
    'nb_hidden': 40,
    'nb_layers': 4,
    'optimizer': torch.optim.Adamax,
    'weight_decay': 0.0024787521766663594
}
model = CNN_1D_MaxPool(**params)

model.fit(X_tr, y_tr, X_te, y_te, epochs=30, callbacks=[keep_best_model], verbose=False)
print("CNN 1D MAX POOL - Test score:", model.score(X_te, y_te))


# CNN 1D - Batch Normalization
X_tr, y_tr = CNN_1D_BatchNorm.prepare_data(train)
X_te, y_te = CNN_1D_BatchNorm.prepare_data(test)

params = {
    'activation': nn.ELU,
    'dropout': 0.3,
    'nb_hidden': 160,
    'nb_layers': 5,
    'optimizer': torch.optim.Adamax,
    'weight_decay': 0.0024787521766663594
}
model = CNN_1D_BatchNorm(**params)

model.fit(X_tr, y_tr, X_te, y_te, epochs=40, callbacks=[keep_best_model], verbose=False)
print("CNN 1D BATCH NORM - Test score:", model.score(X_te, y_te))


# CNN 1D - Batch Normalization Dial
X_tr, y_tr = CNN_1D_BatchNorm_Dial.prepare_data(train)
X_te, y_te = CNN_1D_BatchNorm_Dial.prepare_data(test)


params = {
    'activation': nn.ELU,
    'dropout': 0.09999999999999999,
    'nb_hidden': 40,
    'nb_layers': 1,
    'optimizer': torch.optim.Adadelta,
    'weight_decay': 0.1353352832366127}
model = CNN_1D_BatchNorm_Dial(**params)

model.fit(X_tr, y_tr, X_te, y_te, epochs=50, callbacks=[keep_best_model], verbose=False)
print("CNN 1D BATCH NORM DIAL - Test score:", model.score(X_te, y_te))



# CNN 1D - Residual
X_tr, y_tr = CNN_1D_Residual.prepare_data(train)
X_te, y_te = CNN_1D_Residual.prepare_data(test)


params = {
    "nb_layers": 2,
    "nb_hidden": 80,
    "activation": nn.Tanh,
    "weight_decay": 0.0025,
    "dropout": 0.01,
    "optimizer": torch.optim.Adadelta,
}
model = CNN_1D_Residual(**params)

model.fit(X_tr, y_tr, X_te, y_te, epochs=20, callbacks=[keep_best_model], verbose=False)
print("CNN 1D RESIDUAL - Test score:", model.score(X_te, y_te))
