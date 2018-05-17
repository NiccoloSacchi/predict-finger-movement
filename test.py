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
from torch import manual_seed

one_khz=False

train = SimpleNamespace()
train.X, train.y = bci.load(root='./data_bci', one_khz=one_khz)
#print(str(type(train.X)), train.X.size())
#print(str(type(train.y)), train.y.size())

test = SimpleNamespace()
test.X, test.y = bci.load(root='./data_bci', train=False, one_khz=one_khz)
#print(str(type(test.X)), test.X.size())
#print(str(type(test.y)), test.y.size())



torch.manual_seed(5)
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

model.fit(X_tr, y_tr, X_te, y_te, epochs=40, callbacks=[keep_best_model], verbose=True)
print("CNN 1D BATCH NORM - Getting min train score - Train score:", min(model.history.train_losses), "- Test score:", model.score(X_te, y_te))
