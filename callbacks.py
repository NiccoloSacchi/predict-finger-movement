# -*- coding: utf-8 -*-
""" Definition of callbacks that can be passed to the fit function. """

import torch
from torch import Tensor

from copy import deepcopy

class callback():
    def __call__():
        """ Called at each epoch. """
        raise NotImplementedError
        
    def end():
        """ Called at the end of the training. """
        raise NotImplementedError 

class store_best_model(callback):
    """ Identifies the best model parameters found during training and stores them. """
    def __init__(self, model):
        self.model = model
        self.lowest_train_loss = float("inf")
        
        self.best_model_state = deepcopy(self.model.state_dict())

    def __call__(self):
        # check if the last computed loss is lower that the best seen one
        curr_loss = self.model.history.train_losses[-1]
        if curr_loss < self.lowest_train_loss:
            self.lowest_train_loss = curr_loss
            self.best_model_state = deepcopy(self.model.state_dict())
            
    def end(self):
        self.model.save_model(self.best_model_state)
        
class keep_best_model(callback):
    """ Identifies the best model parameters found during training and loads them 
    in the model at the end of the training. """
    
    def __init__(self, model, use_test_loss=True):
        self.model = model
        self.use_test_loss = use_test_loss
        
        self.lowest_loss = float("inf")
        
        self.best_model_state = deepcopy(self.model.state_dict())

    def __call__(self):
        curr_loss = self.model.history.test_losses[-1] if self.use_test_loss else self.model.history.train_losses[-1]
        # check if the last computed loss is lower that the best seen one
        if curr_loss < self.lowest_loss:
            self.lowest_loss = curr_loss
            self.best_model_state = deepcopy(self.model.state_dict())
            
    def end(self):
        self.model.load_state_dict(self.best_model_state)