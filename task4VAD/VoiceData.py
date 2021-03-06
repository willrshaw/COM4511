import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os, random
import numpy as np
import torchvision 




data_folder = "/Users/will/Documents/COM4511/ass/COM4511/task4VAD/audio"
labs_folder = "/Users/will/Documents/COM4511/ass/COM4511/task4VAD/labels"

training_prefixes = ["N", "V"]
validation_prefixes = ["E"]
testing_prefixes = ["C"]


class VoiceDataSet(Dataset):
    def __init__(self, dataset, transform = None, window_length = 512):
        self.prefix = ["C"]
        if dataset.lower() == "train":
            self.prefix = training_prefixes
        if dataset.lower() == "val":
            self.prefix = validation_prefixes
        if dataset.lower() == "test":
            print(dataset.lower())
            self.prefix == ["C"]

        self.paths = self.gen_paths(self.prefix)
        self.len = len(self.paths[0])
        self.win = window_length
        self.transform = transform
           
           
    
    def __getitem__(self, idx):
        X, y = self.read_data_from_path(self.paths[0][idx], self.paths[1][idx])
        l = X.size(dim = 0)
        i = random.randint(0, l - self.win)
         
        if self.win == 0:
            if self.transform:
                temp = X.view(1, -1, 13).permute(2, 0, 1)
            
                X = self.transform(temp)
                X = X.view(13, -1).permute(1, 0)
                return X, y.type(torch.float32)
            else:
                return X, y.type(torch.float32)
        
        X = X[i:i+self.win]
        y = y[i:i+self.win]
        
        
        if self.transform:
            temp = X.view(1, -1, 13).permute(2, 0, 1)
            
            X = self.transform(temp)
            X = X.view(13, -1).permute(1, 0)
           
            
        
        return X, y.type(torch.float32) 
    
    def __len__(self):
        return self.len
    

    def read_data_from_path(self, d_path, l_path):

        with open(d_path, 'rb') as f:
            X = torch.from_numpy(np.load(f))
        
        with open(l_path, 'rb') as f:
            y = torch.from_numpy(np.load(f))
        
        return X, y
        
    def gen_paths(self, prefixes):
        """gen_paths given a prefix, generate paths to files

        Parameters
        ----------
        prefixes : string
            Prefix for train, dev or test

        Returns
        -------
        (str, str)
            (data path, labels path))
        """
        os.chdir(data_folder)
        data_paths = [f"{data_folder}/{file}" for file in os.listdir() if file[0] in prefixes]
        
        os.chdir(labs_folder)
        labs_paths = [f"{labs_folder}/{file}" for file in os.listdir() if file[0] in prefixes]
        
        return (data_paths, labs_paths)
    
     
    def calc_means_std(self, prefixes = None):
        if not prefixes:
            prefixes = self.prefix
        a_paths, l_paths = self.gen_paths(self,prefixes=prefixes)
        a_data, l_data = [], []
        
        for a, l in zip(a_paths, l_paths):
            temp_d, temp_l = self.read_data_from_path(self, a, l)
            a_data.append(temp_d)
        
        total_data = torch.cat(a_data)
        
        stds, means = torch.std_mean(total_data, dim=0)
        
        return stds, means
        
            
