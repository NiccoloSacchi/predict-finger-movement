# -*- coding: utf-8 -*-
""" 
Class used to wrap a neural network class used for a classification task. 
Implements utility functions to train, test, predict, cross_validate, etc... 
the neural network. """

import torch
from torch import nn
from torch import optim
from sklearn.model_selection import KFold

import os

from copy import deepcopy

from callbacks import *

class History():
    def __init__(self):
        self.num_epochs = 0
        self.train_losses = []
        self.test_losses = []
       
    def new_epoch(self, train_loss, test_loss=None):
        self.num_epochs += 1
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        
class modelWrapper(nn.Module):
    """ 
    Wrap a neural network class. 
    
    The subclass should specify the following parameters (to be initialized in the __init__):
        - self.features: 
            of class torch.nn.Model (e.g. torch.nn.Sequential(...)) used to preprocess
            the data.
        - self.num_features:
            an integer indicating how many features will be extracted by self.features
            and used to reshape the data before feeding it to the self.classifier.
        - self.classifier:
            after reshaping the data into (#samples, self.num_features) it is fed to 
            self.classifier (of class torch.nn.Model) which should contain fully connected 
            layers and provide the final output of the forward pass.
        - self.criterion: 
            cost function used (e.g. torch.nn.CrossEntropyLoss())
        #- self.optimizer: 
        #    optimizer that will update the parameters based on 
        #    the computed gradients (e.g. torch.optim.Adam(self.parameters()))
    """
        
    def __init__(self, 
                 nb_hidden=50, 
                 activation=nn.ReLU, 
                 optimizer=optim.Adam, 
                 weight_decay=0, 
                 dropout=0.1, 
                 nb_layers=1 # number of additional layers
                ):
        super(modelWrapper, self).__init__()
        self.history = History()
        self.dir_path = "storage/" + self.__class__.__name__
            
        self.setting = {
            "nb_hidden": nb_hidden,
            "activation": activation,
            "optimizer": optimizer,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "nb_layers": nb_layers
        }
        
    def fit(self, X_train, y_train, 
            X_test=None, y_test=None, 
            batch_size=20, 
            epochs=25, 
            verbose=True,
            callbacks=[],
            shuffle=True
           ):
        """ Fit the model on the training data.
        Input:
        - X_train: Variable containing the input of the train data.
                shape=(#train_samples, #dimensions)
        - y_train: Variable containing the target of the train data. 
                shape=(#train_samples) or, if the criterion chosen 
                expects one-hot encoding, shape=(#train_samples, #classes).
        - X_test: Variable containing the input of the test data. 
                shape=(#test_samples, #dimensions)
        - y_test: Variable containing the  the target of the test data.
                shape=(#train_samples) or, if the criterion chosen 
                expects one-hot encoding, shape=(#train_samples, #classes).
                If X_test and y_test are given then then also the test 
                error is computed and printed at each epoch.
        - batch_size: Integer representing the number of samples per 
                gradient update.
        - epochs: Integer representing the number of epochs (#iterations 
                over the entire X_train and y_train data provided) to train 
                the model.
        - verbose: boolean indicating whether or not print a log to the standard
                output.
        - callbacks: list <callback> classes that will be called during training 
                at each epoch and at the end of the training.
        - shuffle: if True. The train set is shuffled at each epoch.
        """
        # ----- initialize the callbacks
        callbacks = [c(self) for c in callbacks]
        
        compute_test_err = X_test is not None and y_test is not None
        
        lowest_loss = float('inf')
        best_model = self.state_dict()
        # use "try" so that if the training stops or gets interrupted I still save the best model 
        # and the intermediary predictions
        try:
            for e in range(1, epochs+1):
                if shuffle:
                    indices_perm = torch.randperm(X_train.shape[0])
                    X_train = X_train[indices_perm]
                    y_train = y_train[indices_perm]
                    
                sum_loss_train = 0
                num_batches = 0
                for b in range(0, X_train.size(0), batch_size):  
                    num_batches += 1
                    output = self(X_train[b : b+batch_size])
                    loss = self.criterion(output, y_train[b : b+batch_size])

                    if torch.__version__ == '0.4.0':
                        sum_loss_train += loss.data[0].item()
                    else:
                        sum_loss_train += loss.data[0]
                    self.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                sum_loss_train = sum_loss_train/num_batches
                
                test_loss = None
                if compute_test_err:
                    test_loss = self.criterion(self(X_test), y_test).data
                    test_loss = test_loss.item() if torch.__version__ == '0.4.0' else test_loss[0]
                self.history.new_epoch(sum_loss_train, test_loss)

                if verbose:
                    print(
                        "Epoch " + str(e) + "/" + str(epochs) + ": " +
                        "Train loss:", str(sum_loss_train) + ". " + 
                        'Train accuracy {:0.2f}%'.format(self.score(X_train, y_train)*100) + ". " +
                        ('Test accuracy {:0.2f}%'.format(self.score(X_test, y_test)*100) if compute_test_err else ""))
                    
                # ----- call the callbacks classes (update their internal state)
                for callback in callbacks:
                    callback()
        finally:
            # ----- finalize the callbacks classes (which may store to file their state) 
            for callback in callbacks:
                    callback.end()
                    
        return self
    
    def compute_nb_errors(self, X, y):
        """ Compute the number of misclassified samples. """
        self.eval()
        
        predicted_classes = self.predict(X)
        true_classes = y.data.max(1)[1] if y.dim() == 2 else y.data # if one-hot encoding then extract the class
        
        nb_errors = (true_classes != predicted_classes).sum()

        self.train()
        return nb_errors

    def predict(self, X):
        """ Predict the label of the samples in X. """
        self.eval()
        
        predictions = self(X).data.max(1)[1]
        
        self.train()
        return predictions
    
    def score(self, X, y):
        """ Compute the accuracy. """
        self.eval()
        
        true_classes = y.data.max(1)[1] if y.dim() == 2 else y.data # if one-hot encoding then extract the class
        pred_clases = self.predict(X)
            
        score = (pred_clases==true_classes).sum()
        
        if torch.__version__ == '0.4.0':
            score = score.item()
            
        score = score/X.shape[0]
        
        self.train()
        return score
    
    def forward(self, x):
        """ Do the forward pass. """
        
        x = self.features(x)
        
        x = x.view(-1, self.num_features)
        
        x = self.classifier(x)
        return x
    
    def cross_validate(self, X, y, n_splits=4, epochs=100, verbose=False):
        """ Run cross validation on the model and return the obtained test and train scores. """
        
        kf = KFold(n_splits=n_splits, random_state=1, shuffle=True)
        tr_scores = []
        va_scores = []

        result = {
            "train_score": [],
            "test_score" : []
        }

        split_n = 1
        i = 0
        for tr_indices, va_indices in kf.split(X):
            i+=1
            if verbose: 
                print("----------------- fold " + str(i) + "/" + str(n_splits) + " -----------------")
            tr_indices = tr_indices.tolist()
            va_indices = va_indices.tolist()
            X_tr, y_tr = X[tr_indices], y[tr_indices]
            X_te, y_te = X[va_indices], y[va_indices]

            self.clear()
            self.fit(X_tr, y_tr, X_te, y_te, epochs=epochs, verbose=verbose, callbacks=[keep_best_model])

            result["train_score"].append(self.score(X_tr, y_tr))
            result["test_score"].append(self.score(X_te, y_te))

        return result
    
    def save_model(self, model_state=None):
        """ Save the model to <self.dir_path>/model. """
            
        if model_state is None:
            model_state = self.state_dict()
            
        self.save_data(model_state, "model")
        return self
    
    def load_model(self):
        """ Load the model parameters from <self.dir_path>/model. """            
        self.load_state_dict(self.load_data("model"))
        return self

    def save_data(self, data, file_path="data", pickle_protocol=2):
        """ Save the passed list of predictions to <self.dir_path>/<file_path>. """
        file_path = self.dir_path + "/" + file_path
        dir_path = os.path.dirname(file_path)
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        torch.save(data, file_path, pickle_protocol=pickle_protocol)
        return self
    
    def load_data(self, file_path="data"):
        """ Load and return the list of predictions from <self.dir_path>/<file_path>. """
        
        file_path = self.dir_path + "/" + file_path
        
        if not os.path.isfile(file_path):
            raise Exception("Could not find the file:" + file_path)
            
        return torch.load(file_path)
    
    def clear(self):
        """ Reinitialize the network (used during cross validation)."""
        device = next(self.parameters()).device
        
        self.__init__(**self.setting)
        self.to(device)
