
from torch import nn
import torch
from torchvision import transforms
import torch.nn.functional as F



class FFNN_network(nn.Module):
    def __init__(self, hidden_dims = [13], input_size = 13, out_size=1):
        super(FFNN_network, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append( nn.Linear(input_size, hidden_dims[0]))
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # hidden layers generation
        cur_dim = hidden_dims[0]
        for l in hidden_dims:
            self.layers.append(nn.Linear(cur_dim, l))  
            cur_dim =l # save this so that the 
        
        self.layers.append(nn.Linear(cur_dim, out_size))
         
    def forward(self, x):
        for layer in self.layers[:-1]: #loop over all but last, ReLu not applied to last
            x = self.act(layer(x))
        out = self.sigmoid(self.layers[-1](x))
        return out