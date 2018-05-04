# -*- coding: utf-8 -*-
""" Tries Models. """

from torch import nn 
from torch import optim

class CNN2D(modelWrapper):
    def __init__(self, nb_hidden=50, activation=nn.ReLU, optimizer=optim.Adam):
        super(CNN2D, self).__init__()    
        
        self.nb_hidden = nb_hidden
        self.activation = activation
        
        self.activation = activation
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1, 3)),
            nn.MaxPool2d(2),
            self.activation(),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 5)),
            self.activation(),
            
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2, padding=(1, 1)),
            self.activation(),
        )
        
        self.num_features = 384
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, nb_hidden),
            self.activation(),
            nn.Linear(nb_hidden, 2)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer(self.parameters())
    
    def clear(self):
        self.__init__(self.nb_hidden, self.activation)
        
