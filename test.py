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

grid_search_on=[
        ("nb_layers", list(range(1, 7))),
        ("nb_hidden", [np.asscalar(n) for n in np.arange(40, 201, 40)]),
        ("activation", [nn.ReLU, nn.Tanh, nn.ELU]),
        ("weight_decay", [0] + [np.asscalar(wd) for wd in np.logspace(-6, -2, 5, base=np.e)]),
        ("dropout", [np.asscalar(d) for d in np.linspace(0, 0.30, 4)]),
        ("optimizer", [optim.Adam, optim.Adadelta, optim.Adamax]),
        ("nb_layers", list(range(1, 7)))
    ]

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
    "nb_layers": 3,
    "nb_hidden": 80,
    "activation": nn.ReLU,
    "weight_decay": 0.0024787521766663594,
    "dropout": 0.0,
    "optimizer": optim.Adam,
}
model = CNN2D_MaxPool(**params)

model.fit(X_tr, y_tr, X_te, y_te, epochs=100, callbacks=[keep_best_model], verbose=False)
print("CNN 2D - Test score:", model.score(X_te, y_te))


# CNN 1D - MAX POOL
X_tr, y_tr = CNN_1D_MaxPool.prepare_data(train)
X_te, y_te = CNN_1D_MaxPool.prepare_data(test)

params = {
    "nb_layers": 2,
    "nb_hidden": 40,
    "activation": nn.ReLU,
    "weight_decay": 0, #0.0024787521766663594
    "dropout": 0.01,
    "optimizer": optim.Adam,
}
model = CNN_1D_MaxPool(**params)

model.fit(X_tr, y_tr, X_te, y_te, epochs=100, callbacks=[keep_best_model], verbose=False)
print("CNN 1D MAX POOL - Test score:", model.score(X_te, y_te))


# CNN 1D - Batch Normalization
X_tr, y_tr = CNN_1D_BatchNorm.prepare_data(train)
X_te, y_te = CNN_1D_BatchNorm.prepare_data(test)

params = {
    "nb_layers": 3,
    "nb_hidden": 40,
    "activation": nn.Tanh,
    "weight_decay": 0.007, #0.006737946999085469
    "dropout": 0.01,
    "optimizer": optim.Adam,
}
model = CNN_1D_BatchNorm(**params)

model.fit(X_tr, y_tr, X_te, y_te, epochs=100, callbacks=[keep_best_model], verbose=False)
print("CNN 1D BATCH NORM - Test score:", model.score(X_te, y_te))


# CNN 1D - Batch Normalization Dial
X_tr, y_tr = CNN_1D_BatchNorm_Dial.prepare_data(train)
X_te, y_te = CNN_1D_BatchNorm_Dial.prepare_data(test)


params = {
    "nb_layers": 1,
    "nb_hidden": 40,
    "activation": nn.Tanh,
    "weight_decay": 0.13, #0.1353352832366127
    "dropout": 0.2,
    "optimizer": optim.Adamax,
}
model = CNN_1D_BatchNorm_Dial(**params)

model.fit(X_tr, y_tr, X_te, y_te, epochs=100, callbacks=[keep_best_model], verbose=False)
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
    "optimizer": optim.Adadelta,
}
model = CNN_1D_Residual(**params)

model.fit(X_tr, y_tr, X_te, y_te, epochs=100, callbacks=[keep_best_model], verbose=False)
print("CNN 1D RESIDUAL - Test score:", model.score(X_te, y_te))
