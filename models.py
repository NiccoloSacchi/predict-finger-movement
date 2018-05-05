# -*- coding: utf-8 -*-
""" Tried Models. """

from torch import nn 
from torch import optim

from torch.autograd import Variable

from modelWrapper import modelWrapper

def prepare_data(data):
    X, y = Variable(data.X), Variable(data.y)
    return X, y
    
# ------------------- 2D convolution + MaxPool -------------------
class CNN2D_MaxPool(modelWrapper):
    def __init__(self, **kwargs):
        super(CNN2D_MaxPool, self).__init__(**kwargs)    
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1, 3)),
            nn.MaxPool2d(2),
            self.setting["activation"](),
            nn.Dropout(self.setting["dropout"]),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 5)),
            self.setting["activation"](),
            nn.Dropout(self.setting["dropout"]),
            
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2, padding=(1, 1)),
            self.setting["activation"](),
            nn.Dropout(self.setting["dropout"]),
        )
        
        self.num_features = 384
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, self.setting["nb_hidden"]),
            self.setting["activation"](),
            nn.Linear(self.setting["nb_hidden"], 2)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.setting["optimizer"](self.parameters(), weight_decay=self.setting["weight_decay"])
        
    def prepare_data(data):
        X, y = Variable(data.X.clone().unsqueeze(1)), Variable(data.y)
        return X, y
# ---------------------------------------------------------

# ------------------- 1D convolution + dropout + MaxPool -------------------
class CNN_1D_MaxPool(modelWrapper):
    def __init__(self, **kwargs):
        super(CNN_1D_MaxPool, self).__init__(**kwargs)
        
        self.features = nn.Sequential(
            nn.Conv1d(28, 64, kernel_size=5, padding=2),
            self.setting["activation"](),
            nn.Dropout(self.setting["dropout"]),
            
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.MaxPool1d(2, padding=1),
            self.setting["activation"](),
            nn.Dropout(self.setting["dropout"]),
            
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.MaxPool1d(2, padding=1),
            self.setting["activation"](),
            nn.Dropout(self.setting["dropout"]),
        )
        
        self.num_features = 448
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, nb_hidden),
            self.setting["activation"](),
            nn.Linear(nb_hidden, 2)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.setting["optimizer"](self.parameters(), weight_decay=self.setting["weight_decay"])
        
    def prepare_data(data):
        return prepare_data(data)
# --------------------------------------------------------- 
    
# ------------------- 1D convoution + dropout + batchnorm -------------------
class CNN_1D_BatchNorm(modelWrapper):
    def __init__(self, **kwargs):
        super(CNN_1D_BatchNorm, self).__init__(**kwargs)
        
        self.features = nn.Sequential(
            nn.BatchNorm1d(28),
            nn.Conv1d(28, 32, kernel_size=5, padding=2),
            self.setting["activation"](),
            nn.Dropout(self.setting["dropout"]),            
            
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            self.setting["activation"](),
            nn.Dropout(self.setting["dropout"]), 
            
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            self.setting["activation"](),
            nn.Dropout(self.setting["dropout"]),            
            
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.MaxPool1d(2, padding=1),
            self.setting["activation"](),
            nn.Dropout(self.setting["dropout"]),            
            
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.MaxPool1d(2),
            self.setting["activation"](),
            
            nn.BatchNorm1d(32),
        )
        
        self.num_features = 32*13
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, self.setting["nb_hidden"]),
            self.setting["activation"](),
            nn.Linear(self.setting["nb_hidden"], 2)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.setting["optimizer"](self.parameters(), weight_decay=self.setting["weight_decay"])
        
    def prepare_data(data):
        return prepare_data(data)
# ---------------------------------------------------------

# ------------------- 1D dialated convolution + dropout + batch norm -------------------
class CNN_1D_BatchNorm_Dial(modelWrapper):
    def __init__(self, **kwargs):
        super(CNN_1D_BatchNorm_Dial, self).__init__(**kwargs)
        
        n_filters = 32
        self.features = nn.Sequential(
            nn.BatchNorm1d(28),
            nn.Conv1d(28, n_filters, kernel_size=3, padding=2, dilation=2),
            self.setting["activation"](),
            nn.Dropout(self.setting["dropout"]),            
            
            nn.BatchNorm1d(n_filters),
            nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=2, dilation=2),
            self.setting["activation"](),
            nn.Dropout(self.setting["dropout"]),            
            
            nn.BatchNorm1d(n_filters),
            nn.Conv1d(n_filters, n_filters, kernel_size=5, padding=2),
            self.setting["activation"](),
            nn.Dropout(self.setting["dropout"]),            
            
            nn.BatchNorm1d(n_filters),
            nn.Conv1d(n_filters, n_filters, kernel_size=5, padding=4, dilation=2),
            self.setting["activation"](),
            nn.Dropout(self.setting["dropout"]),            

            nn.BatchNorm1d(n_filters),
            nn.Conv1d(n_filters, 16, kernel_size=3, padding=1),
            self.setting["activation"](),
            nn.Dropout(self.setting["dropout"]),            
        )
        
        self.num_features = 16*50
        
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, self.setting["nb_hidden"]),
            self.setting["activation"](),
            nn.Linear(self.setting["nb_hidden"], 2)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.setting["optimizer"](self.parameters(), weight_decay=self.setting["weight_decay"])
        
    def prepare_data(data):
        return prepare_data(data)
# ---------------------------------------------------------

# # ------------------- 1D convolution + MaxPool1d + batchnorm
# class CNN_1D_BatchNorm_Big(modelWrapper):
#     def __init__(self, **kwargs):
#         super(CNN_1D_BatchNorm_Big, self).__init__(**kwargs)
        
#         self.features = nn.Sequential(
#             nn.BatchNorm1d(28),
#             nn.Conv1d(28, 32, kernel_size=5, padding=2),
#             self.setting["activation"](),
            
#             nn.BatchNorm1d(32),
#             nn.Conv1d(32, 64, kernel_size=5, padding=2),
#             self.setting["activation"](),
            
#             nn.BatchNorm1d(64),
#             nn.Conv1d(64, 64, kernel_size=5, padding=2),
#             nn.MaxPool1d(2, padding=1),
#             self.setting["activation"](),
            
#             nn.BatchNorm1d(64),
#             nn.Conv1d(64, 32, kernel_size=5, padding=2),
#             nn.MaxPool1d(2, padding=1),
#             self.setting["activation"](),
            
#             nn.BatchNorm1d(32),
#             nn.Conv1d(32, 32, kernel_size=5, padding=2),
#             self.setting["activation"](),
            
#             nn.BatchNorm1d(32),
#             nn.Conv1d(32, 16, kernel_size=5, padding=2),
#             self.setting["activation"](),
            
#             nn.BatchNorm1d(16),
#         )
        
#         self.num_features = 16*14
#         self.classifier = nn.Sequential(
#             nn.Linear(self.num_features, self.setting["nb_hidden"]),
#             self.setting["activation"](),
#             nn.Linear(self.setting["nb_hidden"], 2)
#         )

#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = self.setting["optimizer"](self.parameters(), weight_decay=self.setting["weight_decay"])
        
#     def prepare_data(data):
#         return prepare_data(data)
    
# --------------- 1D convolution residual network with aggregated modules + batchnorm ---------------
class residual_block(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super(residual_block, self).__init__()
        self.activation = activation
        
        num_filters = 32
        self.features = nn.Sequential(
            nn.BatchNorm1d(num_filters),
            nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=2, dilation=2),
            self.setting["activation"](),
            
            nn.BatchNorm1d(num_filters),
            nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1),
            self.setting["activation"](),
            
            nn.BatchNorm1d(num_filters),
            nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1),
            self.setting["activation"](),
        )
    
    def forward(self, x):        
        return x+self.features(x)

class aggregated_residual_blocks(nn.Module):
    def __init__(self, n_residual_blocks=2, activation=nn.ReLU):
        super(aggregated_residual_blocks, self).__init__()
        self.activation = activation
        
        self.residual_blocks = nn.ModuleList()
        for i in range(n_residual_blocks):
            self.residual_blocks.append(residual_block(activation=activation))
    
    def forward(self, x):
        out = []
        
        for block in self.residual_blocks:
            out.append(block(x))
            
        return sum(out)+x
    
class CNN_1D_BatchNorm_Residual(modelWrapper):
    def __init__(self, nb_hidden=50, activation=nn.ReLU, optimizer=optim.Adam, weight_decay=0, 
                 n_aggregated_residual_blocks=2, n_residual_blocks=2):
        # n_aggregated_residual_blocks: number of aggregated residual blocks (aggregated_residual_blocks)
        # n_residual_blocks: number of residual blocks per aggregated residual block
        
        super(CNN_1D_BatchNorm_Residual, self).__init__()
        
        self.activation = activation
        
        self.features = [
            nn.BatchNorm1d(28),
            nn.Conv1d(28, 32, kernel_size=3, padding=2, dilation=2),
            self.setting["activation"](),
        ]
        for i in range(n_aggregated_residual_blocks):
            self.features.append(aggregated_residual_blocks(n_residual_blocks))
            self.features.append(nn.Dropout(0.15))
        
        self.features += [            
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.Dropout(0.1),
            nn.MaxPool1d(2),
            self.setting["activation"](),
        ]
        self.features = nn.Sequential(*self.features)
        
        self.num_features = 16*25
        
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, self.setting["nb_hidden"]),
            self.setting["activation"](),
            nn.Linear(self.setting["nb_hidden"], 2)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.setting["optimizer"](self.parameters(), weight_decay=self.setting["weight_decay"])
        
    def prepare_data(data):
        return prepare_data(data)
# ---------------------------------------------------------------------------