import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(42)
print('Imports Successful')

# importing data and labels
with open("/Users/will/Documents/COM4511/ass/COM4511/task4VAD/audio/CMU-0E07000-CM00x-R7.npy", 'rb') as f:
    A = np.load(f)

A.shape
A[:,0].shape
A[:,1].shape
plt.figure(figsize=(16, 12))

print(A.T[:,:100].shape)
plt.imshow(A.T[:,:100], interpolation='nearest', aspect='auto')
# blach blach Imports

# model


lstm = nn.LSTM(13, 1) # imput dim is 13 (size of our MFCCs), output dim is 1
