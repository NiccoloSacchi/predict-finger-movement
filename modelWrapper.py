# -*- coding: utf-8 -*-
""" 
Class used to wrap a neural network class used for a classification task. 
Implements utility functions to train, test, predict, cross_validate, etc... 
the neural network. """

import torch
from torch import nn
from sklearn.model_selection import KFold

class modelWrapper():
    """ 
    Wrap a neural network class. 
    
    The subclass should specify the following parameters:
        - self.criterion: 
            cost function used (e.g. torch.nn.CrossEntropyLoss())
            
        - self.optimizer: 
            optimizer that will update the parameters based on 
            the computed gradients (e.g. torch.optim.Adam(self.parameters()))
            
    The subclass should also implement the following methods: 
        - reset:
            re-initialize the network parameters
            
        - forward:
            given the input X computes the forward pass
    """
    
    def clear(self):
        """ Reinitialize the network (used during cross validation)."""
        raise NotImplementedError
        
    def forward(self, X):
        """ Do the forward pass. """
        raise NotImplementedError
        
        
    def fit(self, X_train, y_train, X_test=None, y_test=None, batch_size=20, epochs=25, verbose=True):
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
        - batch_size: Integer representing the number of epochs (#iterations 
                over the entire X_train and y_train data provided) to train 
                the model.
        """
        
        compute_test_err = X_test is not None and y_test is not None
        
        for e in range(0, epochs):
            sum_loss_train = 0
            for b in range(0, X_train.size(0), batch_size):
                output = self(X_train[b : b+batch_size])
                loss = self.criterion(output, y_train[b : b+batch_size])
                
                sum_loss_train += loss.data[0]
                self.zero_grad()
                loss.backward()
                self.optimizer.step()

            if verbose:
                print(
                    "Epoch " + str(e) + ": " +
                    "Train loss:", str(sum_loss_train) + ". " + 
                    'Train accuracy {:0.2f}%'.format(self.score(X_train, y_train)*100) + ". " +
                    ('Test accuracy {:0.2f}%'.format(self.score(X_test, y_test)*100) if compute_test_err else ""))
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
        score = (self.predict(X)==true_classes).sum()/X.shape[0]
        
        self.train()
        return score
    
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
        for tr_indices, va_indices in kf.split(X):
            if verbose: 
                print("----------------- fold " + str(split_n) + "/" + str(n_splits) + " -----------------")
            tr_indices = tr_indices.tolist()
            va_indices = va_indices.tolist()
            X_tr, y_tr = X[tr_indices], y[tr_indices]
            X_te, y_te = X[va_indices], y[va_indices]

            self.clear()
            self.fit(X_tr, y_tr, epochs=epochs, verbose=verbose)

            result["train_score"].append(self.score(X_tr, y_tr))
            result["test_score"].append(self.score(X_te, y_te))

        return result