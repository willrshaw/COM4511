from torch import nn
import torch
from torchvision import transforms




class LSTM_network(nn.Module):
    def __init__(self, hidden_size = 13, hidden_layers=1, bidirectional = False):
        super(LSTM_network,self).__init__()
        self.lstm = nn.LSTM(input_size=13,hidden_size=hidden_size,num_layers=hidden_layers,batch_first=True)
        self.fc1 = nn.Linear(in_features=hidden_size,out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.bidirectional = bidirectional

    def forward(self,x):
       # print(f"x :{x.size}")
        output,_status = self.lstm(x)
        #print(f"after lstm: {output.size()}")
        output = self.fc1(output)
        #print(f"after lin layer: {output.size()}")
        output = self.sigmoid(output)
       # print(f"after sigmoid: {output.size()}")
        return output